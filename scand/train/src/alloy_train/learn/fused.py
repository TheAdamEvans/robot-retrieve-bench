"""Stage C — FUSED: one vector per window that answers structured + visual queries in a single lookup.

Window encoder f(w) = head([SigLIP2 image mean (frozen); s(w) signal features + masks]) → L2-normalised vector in the
SigLIP2 *text* space, so the query side is unchanged (frozen text tower + dot product: no program, no LLM tokens).

Training pairs (no human grades are ever used for training):
  (a) program pseudo-labels — grammar-sampled programs rendered to text and EXECUTED on the training recordings
      with the same executor PROGRAM uses; windows containing a match's primary anchor are positives. This is the
      "usage becomes a dataset" loop, and it distils PROGRAM into the embedding.
  (b) L1 labeller captions and attribute sentences for their segment windows (semantic content).
Pairings used by compose_test are excluded from (a).

Loss: SigLIP-style pairwise sigmoid over a multi-positive window x text matrix.
Evaluation: leave-one-recording-out. Every window is embedded by the fold model that never saw its recording, and
the out-of-fold vectors form one corpus index (bundle/index/fused_v1_windows.parquet), evaluated on the same harness.
Controls: C1a concat (image ⊕ standardised signals, query ⊕ 0) and C1b linear head, next to the C3 MLP head.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from google.protobuf import json_format

from alloy_server.bundle import Bundle
from alloy_server.catalog.windows import window_span_s
from alloy_server.gen.alloy.v1 import query_pb2 as q
from alloy_server.models.siglip import SPACE_ID, SPEC, SiglipEncoder
from alloy_train.eval.pooling import l1_labels
from alloy_index.recordings import SCAND_ROOT

NS = 1_000_000_000
SIGNALS = [  # (feature, reducer) over the window
    ("speed_mps", "mean"), ("speed_mps", "min"), ("speed_mps", "max"), ("speed_mps", "delta"), ("speed_mps", "drop"),
    ("yaw_rate_dps", "absmax"), ("heading_deg", "delta"), ("accel_mps2", "min"), ("accel_mps2", "max"),
    ("min_clearance_front_m", "min"), ("min_clearance_any_m", "min"), ("lateral_clearance_left_m", "min"),
    ("lateral_clearance_right_m", "min"), ("gap_width_m", "min"), ("doorway_active", "max"),
    ("persons_visible_front", "mean"), ("persons_visible_front", "max"), ("persons_in_corridor", "max"),
    ("vehicles_visible_front", "max"), ("vehicle_box_frac", "max"), ("bicycles_visible_front", "max"),
]


# ---------------- features ----------------

def window_signals(bundle: Bundle, wid: str) -> tuple[np.ndarray, np.ndarray]:
    rec_id, t0, t1 = window_span_s(wid)
    rec = bundle.recordings[rec_id]
    lo, hi = rec.t_abs(t0), rec.t_abs(t1)
    vals, mask = [], []
    for name, red in SIGNALS:
        s = bundle.features.series(name, rec_id)
        if s is None:
            vals.append(0.0); mask.append(0.0)
            continue
        m = (s.t_ns >= lo) & (s.t_ns <= hi)
        y = s.value[m]
        y = y[np.isfinite(y)]
        if not len(y):
            vals.append(0.0); mask.append(0.0)
            continue
        v = {"mean": y.mean(), "min": y.min(), "max": y.max(), "delta": y[-1] - y[0], "absmax": np.abs(y).max(),
             "drop": (np.maximum.accumulate(y) - y).max()}[red]
        vals.append(float(v)); mask.append(1.0)
    tr = bundle.features.tracks("person_tracks_front", rec_id)
    new = float(((tr.first_ns >= lo) & (tr.first_ns <= hi)).sum()) if tr is not None else 0.0
    vals.append(new); mask.append(float(tr is not None))
    vals.append(float(bundle.robots[rec_id] == "spot")); mask.append(1.0)
    return np.array(vals, np.float32), np.array(mask, np.float32)


def image_vectors(bundle: Bundle) -> dict[str, np.ndarray]:
    idx = bundle.embedding_index("siglip2")
    return {w: idx.vector(w) for w in idx.ids}


# ---------------- pseudo-label programs (grammar samples) ----------------

def P(primary, events, relations=(), before=4, after=6):
    return {"primaryEvent": primary, "events": events, "relations": list(relations), "selection": {"quantifier": "ALL"},
            "contextBefore": {"value": before, "unit": "S"}, "contextAfter": {"value": after, "unit": "S"}}


def E(name, kind, feature, **kw):
    return {"name": name, "kind": kind, "feature": feature, "required": True, **kw}


def Q(v, u):
    return {"value": v, "unit": u}


def R(p_, pp_, c_, cp_, kind, mx, mn=None):
    d = {"parent": {"event": p_, "point": pp_}, "child": {"event": c_, "point": cp_}, "kind": kind,
         "maxGap": Q(mx, "S"), "required": True}
    if mn is not None:
        d["minGap"] = Q(mn, "S")
    return d


SLOW = E("slow", "CHANGE", "speed_mps", change=Q(30, "PERCENT"), direction="DOWN", within=Q(3, "S"))
TURN = E("turn", "CHANGE", "heading_deg", change=Q(30, "DEG"), within=Q(3, "S"))
LEFT = E("turn", "CHANGE", "heading_deg", change=Q(30, "DEG"), direction="UP", within=Q(3, "S"))
RIGHT = E("turn", "CHANGE", "heading_deg", change=Q(30, "DEG"), direction="DOWN", within=Q(3, "S"))
GO = E("go", "ONSET", "speed_mps", fromBelow=Q(0.05, "MPS"), minDuration=Q(1, "S"), threshold=Q(0.1, "MPS"),
       sustain=Q(0.5, "S"))
STOP = E("stop", "THRESHOLD", "speed_mps", comparator="LTE", threshold=Q(0.05, "MPS"), minDuration=Q(1, "S"))
PERSON = E("person", "TRACK_APPEAR", "person_tracks_front")
CROWD = E("crowd", "THRESHOLD", "persons_visible_front", comparator="GTE", threshold=Q(4, "DIMENSIONLESS"))
CLOSE = E("close", "THRESHOLD", "min_clearance_front_m", comparator="LT", threshold=Q(1.0, "M"))
DOOR = E("door", "THRESHOLD", "doorway_active", comparator="EQ", threshold=Q(1, "DIMENSIONLESS"), minDuration=Q(0.3, "S"))
CAR = E("car", "THRESHOLD", "vehicles_visible_front", comparator="GTE", threshold=Q(1, "DIMENSIONLESS"), minDuration=Q(0.5, "S"))
BIKE = E("bike", "THRESHOLD", "bicycles_visible_front", comparator="GTE", threshold=Q(1, "DIMENSIONLESS"))
FAST = E("fast", "THRESHOLD", "speed_mps", comparator="GT", threshold=Q(1.3, "MPS"), minDuration=Q(1, "S"))
NARROW = E("narrow", "THRESHOLD", "gap_width_m", comparator="LT", threshold=Q(2.0, "M"), minDuration=Q(0.5, "S"))

# (texts, program). Pairings used by compose_test are deliberately absent: turn→brake, person→speed-up,
# tight right-side clearance, close car while fast.
PSEUDO = [
    (["the robot slows down sharply", "robot brakes", "a sudden slowdown"], P("slow", [SLOW])),
    (["the robot turns", "robot changes direction"], P("turn", [TURN])),
    (["the robot turns left", "a left turn"], P("turn", [LEFT])),
    (["the robot turns right", "a right turn"], P("turn", [RIGHT])),
    (["the robot starts moving after standing still", "robot accelerates from a stop"], P("go", [GO])),
    (["the robot is standing still", "the robot stops"], P("stop", [STOP])),
    (["a new person appears in front of the robot", "someone comes into view ahead"], P("person", [PERSON])),
    (["a crowd of people in front of the robot", "many pedestrians ahead"], P("crowd", [CROWD])),
    (["an obstacle very close in front of the robot", "something within a metre ahead"], P("close", [CLOSE])),
    (["the robot passes through a doorway", "going through a narrow door"], P("door", [DOOR])),
    (["a car in front of the robot", "a vehicle ahead"], P("car", [CAR])),
    (["a bicycle in view", "a bike near the robot"], P("bike", [BIKE])),
    (["the robot moving fast", "walking quickly"], P("fast", [FAST])),
    (["a narrow passage", "the robot squeezes through a tight space"], P("narrow", [NARROW])),
    (["the robot slows down because of people ahead", "a crowd makes the robot slow down"],
     P("crowd", [CROWD, SLOW], [R("crowd", "START", "slow", "START", "WITHIN", 3)])),
    (["the robot slows down as a person appears", "a pedestrian appears and the robot brakes"],
     P("person", [PERSON, SLOW], [R("person", "START", "slow", "START", "AFTER", 4, 0)])),
    (["the robot stops near an obstacle", "stops close to something"],
     P("stop", [STOP, CLOSE], [R("stop", "START", "close", "START", "WITHIN", 2)])),
    (["a person appears while the robot is turning", "someone comes into view during a turn"],
     P("turn", [TURN, PERSON], [R("turn", "START", "person", "START", "AFTER", 4, 0)])),
    (["the robot turns in a narrow passage", "turning in a tight corridor"],
     P("turn", [TURN, NARROW], [R("turn", "START", "narrow", "START", "WITHIN", 2)])),
    (["a car nearby while the robot is standing still", "waiting while a vehicle is in view"],
     P("stop", [STOP, CAR], [R("stop", "START", "car", "START", "WITHIN", 2)])),
]

ATTR_TEXT = {
    "stationaryGroup": "a group of people standing together near the robot's path",
    "doorwayTraversal": "the robot passes through a doorway",
    "vehiclePresent": "a vehicle is visible near the robot",
    "vehicleInteraction": "a vehicle moves across or near the robot's path",
    "bicycle": "a bicycle near the robot",
    "indoor": "the robot is indoors",
    "turnVisible": "the robot turns",
}
PERSON_TEXT = {"ONE_TWO": "one or two people in front of the robot", "THREE_FIVE": "several people in the robot's path"}


def build_pairs(bundle: Bundle, recs: list[str], ann: Path) -> tuple[list[str], dict[str, set[str]]]:
    """→ (texts, text → positive window ids) restricted to `recs` (the training recordings of a fold)."""
    pos: dict[str, set[str]] = {}
    ex = bundle.executor
    for texts, pd in PSEUDO:
        prog = json_format.ParseDict(pd, q.QueryProgram())
        assert not bundle.validate(prog), texts[0]
        wins = set()
        for rec in recs:
            ms, _ = ex.matches(prog, rec)
            for m in ms:
                if m.ordinal >= 2:  # definite matches only
                    cd = SimpleCand(m.recording_id, m.primary_time)
                    w = bundle.window_for_interval(cd)
                    if w:
                        wins.add(w)
        if wins:
            for t in texts:
                pos.setdefault(t, set()).update(wins)
    for sid, lab in l1_labels(ann).items():
        if lab["recordingId"] not in recs:
            continue
        if lab.get("caption"):
            pos.setdefault(lab["caption"], set()).add(sid)
        for field, text in ATTR_TEXT.items():
            if lab.get(field) == "TRUTH_TRUE":
                pos.setdefault(text, set()).add(sid)
        if lab.get("personsInCorridor") in PERSON_TEXT:
            pos.setdefault(PERSON_TEXT[lab["personsInCorridor"]], set()).add(sid)
    return sorted(pos), pos


@dataclass
class SimpleCand:
    recording_id: str
    t: int

    @property
    def anchors(self):
        return [type("A", (), {"t_ns": self.t})()]

    @property
    def seed(self):
        return type("S", (), {"start_ns": self.t})()


# ---------------- model ----------------

class Head(nn.Module):
    def __init__(self, d_sig: int, kind: str, dim: int = SPEC["dim"]):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.net = nn.Linear(dim + d_sig, dim)
            last = self.net
        else:
            self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim + d_sig, 512), nn.GELU(), nn.Dropout(0.2),
                                     nn.Linear(512, dim))
            last = self.net[-1]
        nn.init.zeros_(last.weight)  # start exactly at the image vector (= EMBED) and learn a correction
        nn.init.zeros_(last.bias)
        self.t = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(-10.0))

    def forward(self, img, sig):
        x = self.net(torch.cat([img, sig], -1)) + img  # residual: start from the image vector
        return x / x.norm(dim=-1, keepdim=True)


def train_fold(img, sig, text_vecs, labels, kind, epochs=150, lr=5e-4, seed=0):
    torch.manual_seed(seed)
    model = Head(sig.shape[1], kind)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    T = torch.tensor(text_vecs)
    Y = torch.tensor(labels) * 2 - 1  # +1 positive, -1 negative
    I, S = torch.tensor(img), torch.tensor(sig)
    for _ in range(epochs):
        model.train()
        z = model(I, S)
        logits = z @ T.T * model.t.exp() + model.b
        loss = -torch.nn.functional.logsigmoid(Y * logits).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, float(loss.detach())


def roc(scores, y) -> float:
    s = np.asarray(scores); y = np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def run(bundle: Bundle, ann: Path, kind: str = "mlp", out_name: str = "fused_v1") -> dict:
    wins = bundle.embedding_index("siglip2").ids
    imgs = image_vectors(bundle)
    sig_raw = {w: window_signals(bundle, w) for w in wins}
    enc = SiglipEncoder()
    recs = sorted(bundle.recordings)
    out_vecs, report = {}, {"folds": {}}
    for held in recs:
        train_recs = [r for r in recs if r != held]
        texts, pos = build_pairs(bundle, train_recs, ann)
        tw = [w for w in wins if window_span_s(w)[0] in train_recs]
        X = np.stack([sig_raw[w][0] for w in tw]); Mk = np.stack([sig_raw[w][1] for w in tw])
        mu, sd = X.mean(0), X.std(0) + 1e-6  # standardise with training-fold statistics only
        prep = lambda w: np.concatenate([(sig_raw[w][0] - mu) / sd * sig_raw[w][1], sig_raw[w][1]]).astype(np.float32)
        tv = np.concatenate([enc.encode_texts(texts[k:k + 64]) for k in range(0, len(texts), 64)]).astype(np.float32)
        labels = np.array([[w in pos[t] for t in texts] for w in tw], np.float32)
        img_tr = np.stack([imgs[w] for w in tw]).astype(np.float32)
        sig_tr = np.stack([prep(w) for w in tw])
        if kind == "concat":  # C1a negative control: no learning, query side gets zeros
            hw = [w for w in wins if window_span_s(w)[0] == held]
            for w in hw:
                v = np.concatenate([imgs[w], prep(w)])
                out_vecs[w] = v[: SPEC["dim"]] / np.linalg.norm(v)  # a text query [q; 0] only sees the image part
            report["folds"][held] = {"texts": len(texts)}
            continue
        model, loss = train_fold(img_tr, sig_tr, tv, labels, kind)
        hw = [w for w in wins if window_span_s(w)[0] == held]
        with torch.no_grad():
            z = model(torch.tensor(np.stack([imgs[w] for w in hw]).astype(np.float32)),
                      torch.tensor(np.stack([prep(w) for w in hw])))
        for w, v in zip(hw, z.numpy()):
            out_vecs[w] = v
        htexts, hpos = build_pairs(bundle, [held], ann)
        pseudo = [t for t in htexts if any(t in x[0] for x in PSEUDO) or t in {tt for x in PSEUDO for tt in x[0]}]
        ptv = enc.encode_texts(pseudo) if pseudo else np.zeros((0, SPEC["dim"]), np.float32)
        Ximg = np.stack([imgs[w] for w in hw]); Xf = z.numpy()
        auc_e, auc_f = [], []
        for t, v in zip(pseudo, ptv):
            y = [int(w in hpos[t]) for w in hw]
            if 0 < sum(y) < len(y):
                auc_e.append(roc(Ximg @ v, y)); auc_f.append(roc(Xf @ v, y))
        report["folds"][held] = {"texts": len(texts), "train_windows": len(tw), "positives": int(labels.sum()),
                                 "final_loss": round(loss, 4), "oof_pseudo_auc_embed": round(float(np.mean(auc_e)), 3) if auc_e else None,
                                 "oof_pseudo_auc_fused": round(float(np.mean(auc_f)), 3) if auc_f else None,
                                 "n_pseudo_texts": len(auc_e)}
        print(json.dumps({"held_out": held, **report["folds"][held]}), flush=True)
    arr = np.stack([out_vecs[w] for w in wins]).astype(np.float16)
    # hubness: how concentrated are top-5 results over a fixed probe set (lower = healthier)
    probes = enc.encode_texts([t for x in PSEUDO for t in x[0]])
    for name, M in (("embed", np.stack([imgs[w] for w in wins])), (out_name, arr.astype(np.float32))):
        top = np.argsort(-(M @ probes.T), axis=0)[:5].ravel()
        counts = np.bincount(top, minlength=len(wins))
        report[f"hubness_top5_max_{name}"] = int(counts.max())
        report[f"hubness_top5_distinct_{name}"] = int((counts > 0).sum())
    table = pa.table({"window_id": wins, "recording_id": [window_span_s(w)[0] for w in wins],
                      "n_frames": [0] * len(wins),
                      "vec": pa.FixedSizeListArray.from_arrays(pa.array(arr.ravel(), pa.float16()), arr.shape[1])}
                     ).replace_schema_metadata({"space_id": SPACE_ID, "spec": json.dumps(SPEC),
                                                "recipe": f"{out_name}: {kind} head, LORO out-of-fold vectors"})
    pq.write_table(table, bundle.root / "index" / f"{out_name}_windows.parquet")
    report.update({"kind": kind, "windows": len(wins), "signals": len(SIGNALS) + 2})
    (bundle.root / "index" / f"{out_name}.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=SCAND_ROOT / "bundles" / "dev")
    ap.add_argument("--kind", choices=["mlp", "linear", "concat"], default="mlp")
    ap.add_argument("--name", default="fused_v1")
    a_ = ap.parse_args()
    print(json.dumps(run(Bundle(a_.bundle), SCAND_ROOT / "annotations", a_.kind, a_.name), indent=1))


if __name__ == "__main__":
    main()
