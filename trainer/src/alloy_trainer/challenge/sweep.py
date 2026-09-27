"""Exhaustive numeric sweeps: every span in a question's scope that could satisfy its numeric clauses.

A question's clauses are part numeric (speed loss, heading excursion, net displacement, time below half speed) and
part semantic (an oncoming pedestrian, a visibly clear route, the same person). Code checks the numeric part over
every sample; the judge then settles the semantic part on every candidate. The judged positives are then a complete
ground-truth set for the scope, which is what recall needs.

**Superset by construction.** Thresholds are loosened by a stated tolerance (e.g. >=25% for a >=30% loss) and
maneuver windows are searched on a fine grid (onsets every 0.25 s, lengths 1-10 s), so filtering choices and window
boundaries cannot drop a qualifying span. Every episode already judged relevant must fall inside the sweep (the
sweep-recall check); a miss fails loudly. Signals are computed like `scandq signals` (the judges' tool): profile
twist frame, profile speed smoother, heading integrated from yaw rate, plus raw odometry pose for displacement.

    uv run python -m alloy_trainer.challenge.sweep --set l1_compositions_v1 --split dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alloy_server.catalog import embodiment as E
from alloy_server.timeline.store import Recording
from alloy_index import embodiment as emb
from alloy_index.annotate.store import load_labels
from alloy_index.decode import Decoder, odom_arrays
from alloy_index.providers.common import smooth_speed
from alloy_index.recordings import SCAND_ROOT, bundle_root, robot

VERSION = "sweep@1"
DT = 0.1                                   # resampling grid (s)
ONSET_STEP = 0.25
WINDOWS = np.arange(1.0, 10.01, 0.5)       # maneuver lengths searched (s)
MIN_BASELINE = 0.2                         # contract: a baseline below 0.2 m/s makes relative clauses UNKNOWN


@dataclass
class Signals:
    t: np.ndarray        # recording-relative seconds, regular grid
    v: np.ndarray        # smoothed speed (m/s)
    v_raw: np.ndarray
    hdg: np.ndarray      # integrated heading (deg, + = left)
    xy: np.ndarray       # raw odometry pose (m)

    def at(self, t0: float, t1: float) -> slice:
        i, j = np.searchsorted(self.t, [t0, t1])
        return slice(i, j)

    def baseline(self, t: float) -> float:
        """Median smoothed speed over the 2 s before t (the contract's default baseline)."""
        s = self.at(t - 2.0, t)
        return float(np.median(self.v[s])) if s.stop - s.start >= 10 else float("nan")


def signals(bundle: Path, rec_id: str) -> Signals:
    rec = Recording(bundle, rec_id)
    rb = robot(rec_id)
    prof = emb.prof(rb)
    dec = Decoder(bundle, rec)
    topic = emb.odom_topic(rb)
    raw = odom_arrays(dec, topic, prof)
    smooth, _ = smooth_speed(raw["t_ns"], raw["speed_mps"], prof, E.load_intake(bundle, rec_id))
    t = (raw["t_ns"] - rec.start_ns) / 1e9
    hdg = np.concatenate([[0.0], np.cumsum(raw["yaw_rate_dps"][1:] * np.diff(t))])
    pose = np.array([(m.pose.pose.position.x, m.pose.pose.position.y)
                     for m in (dec.msg(topic, i) for i in range(len(rec.topics[topic])))])
    g = np.arange(t[0], t[-1], DT)
    lerp = lambda y: np.interp(g, t, y)
    return Signals(g, lerp(smooth), lerp(raw["speed_mps"]), lerp(hdg), np.stack([lerp(pose[:, 0]), lerp(pose[:, 1])], 1))


def _onsets(sg: Signals, lead: float = 2.0, tail: float = 1.0):
    return np.arange(sg.t[0] + lead, sg.t[-1] - tail, ONSET_STEP)


def _window_stats(sg: Signals, t: float, w: float, b: float) -> tuple[float, float, float]:
    s = sg.at(t, t + w)
    if s.stop - s.start < 2:
        return 0.0, 0.0, 0.0
    v, h = sg.v[s], sg.hdg[s]
    return 1 - v.min() / b, float(h.max() - h.min()), float(h[-1] - h[0])


def speed_heading(sg: Signals, loss_min=None, loss_max=None, exc_min=None, exc_max=None, net_abs_min=None) -> list[dict]:
    """Maneuver windows [t, t+w] whose loss / excursion / net heading pass the (loosened) bounds."""
    hits = []
    for t in _onsets(sg):
        b = sg.baseline(t)
        if not b >= MIN_BASELINE:
            continue
        for w in WINDOWS:
            if t + w > sg.t[-1]:
                break
            loss, exc, net = _window_stats(sg, t, w, b)
            if ((loss_min is None or loss >= loss_min) and (loss_max is None or loss < loss_max)
                    and (exc_min is None or exc >= exc_min) and (exc_max is None or exc < exc_max)
                    and (net_abs_min is None or abs(net) >= net_abs_min)):
                hits.append({"start_s": t, "end_s": t + w, "baseline_mps": b, "loss": loss, "excursion_deg": exc,
                             "net_heading_deg": net})
    return hits


def slowdown_events(sg: Signals, loss_min: float, recover_frac: float | None = None, horizon: float = 20.0) -> list[dict]:
    """Distinct slowdowns (deduplicated by their speed minimum), optionally only those that recover."""
    events = {}
    for t in _onsets(sg):
        b = sg.baseline(t)
        if not b >= MIN_BASELINE:
            continue
        s = sg.at(t, t + 8.0)
        if s.stop - s.start < 2 or 1 - sg.v[s].min() / b < loss_min:
            continue
        i_min = s.start + int(np.argmin(sg.v[s]))
        rec_t = None
        if recover_frac is not None:
            after = sg.at(sg.t[i_min], sg.t[i_min] + horizon)
            ok = np.nonzero(sg.v[after] >= recover_frac * b)[0]
            if not len(ok):
                continue
            rec_t = float(sg.t[after.start + ok[0]])
        key = round(float(sg.t[i_min]) * 2) / 2
        if key not in events or t < events[key]["onset_s"]:
            events[key] = {"onset_s": float(t), "min_s": float(sg.t[i_min]), "recovery_s": rec_t, "baseline_mps": b,
                           "loss": 1 - float(sg.v[i_min]) / b,
                           "start_s": float(t) - 2.0, "end_s": (rec_t or float(sg.t[i_min])) + 2.0}
    return sorted(events.values(), key=lambda e: e["start_s"])


def stays_slow(sg: Signals, frac: float, dur: float, lookback: float = 10.0) -> list[dict]:
    """Spans where speed stays below frac x the highest recent baseline for dur seconds."""
    hits = []
    for t in _onsets(sg):
        bs = [sg.baseline(x) for x in np.arange(max(sg.t[0] + 2, t - lookback), t + 1e-9, 0.5)]
        bs = [b for b in bs if b >= MIN_BASELINE]
        if not bs:
            continue
        s = sg.at(t, t + dur)
        if s.stop - s.start >= 2 and (sg.v[s] < frac * max(bs)).all():
            hits.append({"start_s": float(t) - 6.0, "end_s": float(t) + dur + 2.0, "baseline_mps": max(bs),
                         "slow_from_s": float(t)})
    return hits


def little_progress(sg: Signals, window: float, disp_max: float, moving_min: float) -> list[dict]:
    hits = []
    for t in np.arange(sg.t[0], sg.t[-1] - window, ONSET_STEP):
        s = sg.at(t, t + window)
        d = float(np.hypot(*(sg.xy[s.stop - 1] - sg.xy[s.start])))
        if d < disp_max and sg.v_raw[s].max() >= moving_min:
            hits.append({"start_s": float(t), "end_s": float(t) + window, "net_displacement_m": d})
    return hits


def holds_motion_with_people(sg: Signals, rec_id: str, range_max: float, exc_max: float) -> list[dict]:
    """4 s label segments where the robot holds its motion and a person is present by L1 labels OR the detector."""
    import pyarrow.parquet as pq
    labels = load_labels(SCAND_ROOT / "annotations", "attributes")
    det_p = bundle_root() / "features" / "detections" / f"{rec_id}.parquet"
    det = pq.read_table(det_p).to_pydict() if det_p.exists() else None
    rec = Recording(bundle_root(), rec_id)
    hits = []
    for end in range(4, int(sg.t[-1]) + 1, 4):
        s = sg.at(end - 4, end)
        if s.stop - s.start < 2:
            continue
        rng, exc = float(sg.v[s].max() - sg.v[s].min()), float(sg.hdg[s].max() - sg.hdg[s].min())
        if rng >= range_max or exc >= exc_max:
            continue
        lab = labels.get(f"{rec_id}:{end:04d}", {})
        l1_person = lab.get("personsInCorridor", "ZERO") not in ("ZERO", "BUCKET_UNSPECIFIED")
        det_person = False
        if det is not None:
            tt = (np.array(det["t_ns"]) - rec.start_ns) / 1e9
            m = (tt >= end - 4) & (tt < end)
            det_person = bool(m.any() and np.array(det["persons_visible_front"])[m].max() >= 1)
        if l1_person or det_person:
            hits.append({"start_s": end - 4.0, "end_s": float(end), "speed_range_mps": rng, "excursion_deg": exc,
                         "person_source": "+".join(x for x, ok in (("l1", l1_person), ("detector", det_person)) if ok)})
    return hits


def merge(hits: list[dict], gap: float = 0.0) -> list[dict]:
    out = []
    for h in sorted(hits, key=lambda h: h["start_s"]):
        if out and h["start_s"] <= out[-1]["end_s"] + gap:
            o = out[-1]
            o["end_s"] = max(o["end_s"], h["end_s"])
            o["n_windows"] += 1
            for k, v in h.items():
                if isinstance(v, float) and k not in ("start_s", "end_s"):
                    o.setdefault("max_" + k, v)
                    o["max_" + k] = max(o["max_" + k], v)
        else:
            out.append({"start_s": h["start_s"], "end_s": h["end_s"], "n_windows": 1,
                        **{"max_" + k: v for k, v in h.items() if isinstance(v, float) and k not in ("start_s", "end_s")},
                        **{k: v for k, v in h.items() if isinstance(v, str)}})
    return out


# Per intent: members (a comparison has two), each a candidate generator with loosened thresholds.
# `exact` records the question's own numeric clause next to the tolerance used, for the record.
SPECS = {
    "l1x_dream_steer_or_brake": {
        "slowing": {"exact": "loss >= 30%, excursion < 15 deg", "loose": "loss >= 25%, excursion < 20 deg",
                    "fn": lambda sg, r: merge(speed_heading(sg, loss_min=0.25, exc_max=20.0))},
        "steering": {"exact": "excursion >= 30 deg, loss < 20%", "loose": "excursion >= 25 deg, loss < 25%",
                     "fn": lambda sg, r: merge(speed_heading(sg, exc_min=25.0, loss_max=0.25))},
    },
    "l1x_dream_opportunity_to_go": {
        "episode": {"exact": "speed < 50% of pre-encounter baseline for 2 s",
                    "loose": "speed < 55% of the highest baseline in the previous 10 s for 1.75 s",
                    "fn": lambda sg, r: merge(stays_slow(sg, frac=0.55, dur=1.75))},
    },
    "l1x_dream_recover_person_present": {
        "episode": {"exact": "loss >= 30%, then recovery to 90% of baseline",
                    "loose": "loss >= 25% within 8 s, then recovery to 85% of baseline within 20 s (one per speed minimum)",
                    "fn": lambda sg, r: slowdown_events(sg, loss_min=0.25, recover_frac=0.85)},
    },
    "l1x_dream_turn_creates_exposure": {
        "episode": {"exact": "turn >= 45 deg", "loose": "|net heading change| >= 40 deg within 1-10 s",
                    "fn": lambda sg, r: merge([{**h, "start_s": h["start_s"] - 4, "end_s": h["end_s"] + 2}
                                               for h in speed_heading(sg, net_abs_min=40.0)])},
    },
    "l1x_dream_multiple_attempts": {
        "episode": {"exact": "8 s, net planar displacement < 2 m from raw pose, two speed dips >= 0.2 m/s",
                    "loose": "8 s, net displacement < 2.5 m, raw speed reaches >= 0.15 m/s",
                    "fn": lambda sg, r: merge(little_progress(sg, window=8.0, disp_max=2.5, moving_min=0.15))},
    },
    "l1x_dream_person_adjusts": {
        "episode": {"exact": "speed range < 0.2 m/s, heading excursion < 15 deg, a person changes course",
                    "loose": "4 s segments with speed range < 0.25 m/s, excursion < 20 deg, and a person present by "
                             "L1 labels or the detector (recall is relative to that person coverage)",
                    "fn": lambda sg, r: merge(holds_motion_with_people(sg, r, range_max=0.25, exc_max=20.0))},
    },
}


def sweep_recall(intent: str, cands: dict[str, list[dict]]) -> list[dict]:
    """Every judged episode, and whether a candidate overlaps it; grade >= 1 must be covered."""
    out = []
    for r in load_labels(SCAND_ROOT / "annotations", "episode").values():
        if r["intentGroupId"] != intent:
            continue
        spans = [c for c in cands.get(r["recordingId"], []) if c["start_s"] < r["endS"] and c["end_s"] > r["startS"]]
        out.append({"episode": f'{r["jobId"]}:{r["episodeId"]}', "recording": r["recordingId"],
                    "span": [r["startS"], r["endS"]], "grade": r.get("grade", 0), "covered": bool(spans)})
    return out


def run(set_name: str, split: str, scope_all: bool = False) -> dict:
    set_dir = SCAND_ROOT / "benchmark" / "challenges" / set_name
    qs = [json.loads(l) for l in (set_dir / f"queries_{split}.jsonl").read_text().splitlines() if l.strip()]
    bundle = bundle_root()
    cache: dict[str, Signals] = {}
    out_dir = SCAND_ROOT / "annotations" / "exhaustive" / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    summary = {}
    for q in qs:
        intent = q["intentGroupId"]
        if intent not in SPECS:
            continue
        scope = sorted(bundle.joinpath("mcap").glob("*.json")) if scope_all else q["scope"]["recordingIds"]
        scope = [p.stem for p in scope] if scope_all else scope
        members, all_cands = {}, {}
        for name, spec in SPECS[intent].items():
            per = {}
            for r in scope:
                if r not in cache:
                    cache[r] = signals(bundle, r)
                per[r] = spec["fn"](cache[r], r)
                all_cands.setdefault(r, []).extend(per[r])
            members[name] = {"exact": spec["exact"], "loose": spec["loose"], "candidates": per,
                             "n": sum(len(v) for v in per.values()),
                             "seconds": round(sum(c["end_s"] - c["start_s"] for v in per.values() for c in v), 1)}
        check = sweep_recall(intent, all_cands)
        missed = [c for c in check if c["grade"] >= 1 and not c["covered"]]
        doc = {"version": VERSION, "code_sha": code, "intent": intent, "split": split, "scope": scope,
               "members": members, "sweep_recall_check": check, "sweep_recall_ok": not missed}
        (out_dir / f"{intent}.json").write_text(json.dumps(doc, indent=1))
        summary[intent] = {m: (v["n"], v["seconds"]) for m, v in members.items()}
        summary[intent]["check"] = f"{sum(c['covered'] for c in check)}/{len(check)} judged episodes covered" + \
                                   ("" if not missed else f"; MISSED relevant: {missed}")
        if missed:
            raise SystemExit(f"{intent}: the sweep misses judged-relevant episodes {missed}; it is not a superset")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--scope-all", action="store_true", help="every indexed recording, not the question's scope")
    a = ap.parse_args()
    for k, v in run(a.set, a.split, a.scope_all).items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
