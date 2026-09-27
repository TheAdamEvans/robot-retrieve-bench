"""FUSED v2: balanced program supervision with uniform or importance sampling.

The importance regressor is used only to select Train examples. The serving
artifact is a separate full-Train model/index; evaluation uses out-of-fold
vectors for every Train recording.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from alloy_index.models.fused import Head, encode, image_vectors, save, standardise, window_signals
from alloy_index.recordings import SCAND_ROOT, held_out
from alloy_server.bundle import Bundle
from alloy_server.catalog.windows import window_span_s
from alloy_server.models.siglip import SPACE_ID, SPEC, SiglipEncoder
from alloy_trainer.eval.pooling import l1_labels
from alloy_trainer.learn import importance
from alloy_trainer.learn.fused import ATTR_TEXT, PERSON_TEXT
from alloy_trainer.learn.fused_supervision import ProgramText, Supervision, build as build_programs


def attribute_groups(root: Path, recs: list[str]) -> list[Supervision]:
    rows = [r for r in l1_labels(root).values() if r["recordingId"] in recs]
    out = []
    for field, text in ATTR_TEXT.items():
        pos = {r["segmentId"] for r in rows if r.get(field) == "TRUTH_TRUE"}
        neg = {r["segmentId"] for r in rows if r.get(field) == "TRUTH_FALSE"}
        out.append(Supervision(ProgramText(f"l1_{field}", text, {}), pos, set(), neg))
    for bucket, text in PERSON_TEXT.items():
        pos = {r["segmentId"] for r in rows if r.get("personsInCorridor") == bucket}
        neg = {r["segmentId"] for r in rows if r.get("personsInCorridor") in
               {"ZERO", "ONE_TWO", "THREE_FIVE", "SIX_PLUS"} - {bucket}}
        out.append(Supervision(ProgramText(f"l1_people_{bucket}", text, {}), pos, set(), neg))
    return out


def caption_pairs(root: Path, recs: list[str], available: set[str]) -> list[tuple[str, str]]:
    return sorted((r["segmentId"], r["caption"]) for r in l1_labels(root).values()
                  if r["recordingId"] in recs and r["segmentId"] in available and r.get("caption"))


def probabilities(wins: list[str], scores: dict[str, float], mode: str) -> np.ndarray:
    """Equal recording mass, then 50/50 uniform and score-proportional within each recording."""
    out = np.zeros(len(wins), np.float64)
    by_rec = defaultdict(list)
    for i, wid in enumerate(wins):
        by_rec[window_span_s(wid)[0]].append(i)
    for indices in by_rec.values():
        n = len(indices)
        if mode == "uniform":
            local = np.full(n, 1 / n)
        else:
            raw = np.maximum([scores[wins[i]] for i in indices], 0)
            # Quantile over positive scores: a rare useful peak must survive
            # clipping even when over 95% of windows score zero.
            top = np.quantile(raw[raw > 0], 0.95) if np.any(raw > 0) else 0.0
            mass = np.minimum(raw, top) + 0.1
            local = 0.5 / n + 0.5 * mass / mass.sum()
            # Water-fill the excess so no window receives more than 4x uniform.
            cap = 4 / n
            while np.any(local > cap + 1e-12):
                capped = local > cap
                excess = float((local[capped] - cap).sum())
                local[capped] = cap
                free = ~capped
                if not free.any():
                    break
                local[free] += excess * local[free] / local[free].sum()
        out[indices] = local / len(by_rec)
    return out / out.sum()


def _encode_texts(enc: SiglipEncoder, texts: list[str]) -> dict[str, np.ndarray]:
    unique = sorted(set(texts))
    vectors = np.concatenate([enc.encode_texts(unique[i:i + 64]) for i in range(0, len(unique), 64)])
    return dict(zip(unique, vectors.astype(np.float32)))


def train_model(wins: list[str], img: np.ndarray, sig: np.ndarray, groups: list[Supervision],
                captions: list[tuple[str, str]], vectors: dict[str, np.ndarray], scores: dict[str, float], mode: str,
                *, epochs: int = 150, batch_size: int = 256, hidden_dim: int = 512, dropout: float = 0.2,
                seed: int = 0) -> tuple[Head, dict]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    row = {w: i for i, w in enumerate(wins)}
    p = probabilities(wins, scores, mode)
    by_family = defaultdict(list)
    for g in groups:
        pos = [row[w] for w in sorted(g.positive) if w in row]
        hard = [row[w] for w in sorted(g.hard) if w in row]
        easy = [row[w] for w in sorted(g.easy) if w in row]
        if pos and (hard or easy):
            by_family[g.spec.family].append((g.spec.text, pos, hard, easy))
    if not by_family:
        raise ValueError("no positive and verified negative program groups in the training split")
    families = sorted(by_family)
    caption_rows = defaultdict(list)
    for w, caption in captions:
        if w in row:
            caption_rows[caption].append(row[w])
    caption_texts = sorted(caption_rows)
    text_list = sorted({g[0] for gs in by_family.values() for g in gs} | set(caption_texts))
    text_row = {t: i for i, t in enumerate(text_list)}
    T = torch.tensor(np.stack([vectors[t] for t in text_list]), dtype=torch.float32, device=device)
    I = torch.tensor(img, dtype=torch.float32, device=device)
    S = torch.tensor(sig, dtype=torch.float32, device=device)
    model = Head(sig.shape[1], "mlp", hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    rec = [window_span_s(w)[0] for w in wins]
    # Pre-bin by recording and predicted-score decile. The same near-miss pool is
    # used in both sampling arms; only positive-window probabilities differ.
    decile = np.zeros(len(wins), np.int8)
    for r in set(rec):
        ix = np.array([i for i, x in enumerate(rec) if x == r])
        values = np.array([scores[wins[i]] for i in ix])
        decile[ix] = np.searchsorted(np.quantile(values, np.arange(1, 10) / 10), values).astype(np.int8)

    def choose_negative(pool: list[int], positive: int) -> int:
        if not pool:
            return positive
        matched = [i for i in pool if rec[i] == rec[positive] and abs(int(decile[i]) - int(decile[positive])) <= 1]
        same_rec = [i for i in pool if rec[i] == rec[positive]]
        return rng.choice(matched or same_rec or pool)

    last = 0.0
    for _ in range(epochs):
        pairs_w, pairs_t, signs, weights = [], [], [], []
        for _ in range(batch_size):
            family = rng.choice(families)
            text, pos, hard, easy = rng.choice(by_family[family])
            # Give each recording an equal chance when a program occurs there.
            pos_recs = sorted({rec[i] for i in pos})
            chosen_rec = rng.choice(pos_recs)
            candidates = [i for i in pos if rec[i] == chosen_rec]
            k = rng.choices(candidates, weights=[p[i] for i in candidates])[0]
            h = choose_negative(hard or easy, k)
            e = choose_negative(easy or hard, k)
            for wi, sign, weight in ((k, 1, 0.5), (h, -1, 0.25), (e, -1, 0.25)):
                pairs_w.append(wi); pairs_t.append(text_row[text]); signs.append(sign); weights.append(weight)
        ix, inv = np.unique(pairs_w, return_inverse=True)
        model.train()
        z = model(I[torch.tensor(ix, device=device)], S[torch.tensor(ix, device=device)])
        logits = (z[torch.tensor(inv, device=device)] * T[torch.tensor(pairs_t, device=device)]).sum(-1)
        logits = logits * model.t.exp() + model.b
        loss = (F.softplus(-torch.tensor(signs, dtype=torch.float32, device=device) * logits) *
                torch.tensor(weights, dtype=torch.float32, device=device)).sum() / batch_size
        # Each triplet shares a text. Make the positive beat both verified
        # negatives directly, in addition to the balanced sigmoid objective.
        similarity = (z[torch.tensor(inv, device=device)] * T[torch.tensor(pairs_t, device=device)]).sum(-1)
        triplets = similarity.reshape(batch_size, 3)
        loss = loss + 0.1 * F.softplus(10 * (triplets[:, 1:] - triplets[:, :1])).mean()
        if caption_texts:
            chosen = rng.sample(caption_texts, min(64, len(caption_texts)))
            cwin = torch.tensor([rng.choice(caption_rows[t]) for t in chosen], device=device)
            ctext = torch.tensor([text_row[t] for t in chosen], device=device)
            cz = model(I[cwin], S[cwin])
            caption_logits = 10 * (cz @ T[ctext].T)
            loss = loss + 0.05 * F.cross_entropy(caption_logits, torch.arange(len(chosen), device=device))
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach().cpu())
    model.eval()
    return model.cpu(), {"final_loss": round(last, 5), "groups": sum(len(v) for v in by_family.values()),
                         "families": len(families), "captions": sum(map(len, caption_rows.values())),
                         "unique_captions": len(caption_texts), "device": device}


def _write_index(path: Path, wins: list[str], vectors: dict[str, np.ndarray], source: list[str],
                 recipe: str, model_hash: str) -> None:
    arr = np.stack([vectors[w] for w in wins]).astype(np.float16)
    table = pa.table({"window_id": wins, "recording_id": [window_span_s(w)[0] for w in wins],
                      "n_frames": [0] * len(wins), "source": source,
                      "vec": pa.FixedSizeListArray.from_arrays(pa.array(arr.ravel(), pa.float16()), arr.shape[1])}
                     ).replace_schema_metadata({"space_id": SPACE_ID, "spec": json.dumps(SPEC),
                                                "recipe": recipe, "model_hash": model_hash})
    pq.write_table(table, path)


def diagnose(bundle: Bundle, groups: list[Supervision], text_vectors: dict[str, np.ndarray],
             mode: str, train_recs: list[str]) -> dict:
    """Within-recording OOF AUC and one-clause near-miss ranking by program family."""
    from alloy_trainer.learn.fused import roc

    index = bundle.embedding_index(f"fused_v2_{mode}_oof")
    by_family = defaultdict(lambda: {"auc": [], "hard_win": [], "positives": 0,
                                     "hard": 0, "easy": 0, "comparisons": 0})
    for rec in train_recs:
        mine = {w for w in index.ids if window_span_s(w)[0] == rec}
        for group in groups:
            pos, hard, easy = (sorted(s & mine) for s in (group.positive, group.hard, group.easy))
            if not pos:
                continue
            dest = by_family[group.spec.family]
            dest["positives"] += len(pos)
            dest["hard"] += len(hard)
            dest["easy"] += len(easy)
            vec = text_vectors[group.spec.text]
            ps = np.array([index.vector(w) @ vec for w in pos])
            hs = np.array([index.vector(w) @ vec for w in hard])
            es = np.array([index.vector(w) @ vec for w in easy])
            if len(hs) + len(es):
                ns = np.concatenate([hs, es])
                dest["auc"].append(roc(np.concatenate([ps, ns]),
                                       np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])))
                dest["comparisons"] += 1
            if len(hs):
                dest["hard_win"].append(float(((ps[:, None] > hs[None, :]).mean() +
                                               0.5 * (ps[:, None] == hs[None, :]).mean())))
    families = {name: {"auc": round(float(np.mean(d["auc"])), 4) if d["auc"] else None,
                       "hard_pair_win_rate": round(float(np.mean(d["hard_win"])), 4) if d["hard_win"] else None,
                       "positive_pairs": d["positives"], "hard_pairs": d["hard"],
                       "easy_pairs": d["easy"], "recording_text_comparisons": d["comparisons"]}
                for name, d in sorted(by_family.items())}
    summary = {"mode": mode, "families": families,
               "macro_family_auc": round(float(np.mean([d["auc"] for d in families.values()
                                                        if d["auc"] is not None])), 4),
               "macro_family_hard_win": round(float(np.mean([d["hard_pair_win_rate"] for d in families.values()
                                                             if d["hard_pair_win_rate"] is not None])), 4)}
    (bundle.root / "index" / f"fused_v2_{mode}_diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run(bundle: Bundle, root: Path, mode: str, all_groups: list[Supervision], text_vectors: dict[str, np.ndarray],
        fold_salience: dict[str, dict[str, float]], *, epochs: int = 150, seed: int = 0, hidden_dim: int = 512,
        dropout: float = 0.2) -> dict:
    name = f"fused_v2_{mode}"
    wins = bundle.embedding_index("siglip2").ids
    imgs = image_vectors(bundle)
    sig_raw = {w: window_signals(bundle, w) for w in wins}
    train_recs = sorted(set(bundle.recordings) - held_out())
    outputs = {}
    report = {"name": name, "mode": mode, "folds": {}, "program_texts": len(all_groups)}
    for held in train_recs:
        recs = [r for r in train_recs if r != held]
        tr = [w for w in wins if window_span_s(w)[0] in recs]
        hw = [w for w in wins if window_span_s(w)[0] == held]
        x = np.stack([sig_raw[w][0] for w in tr]); mu, sd = x.mean(0), x.std(0) + 1e-6
        salience = fold_salience[held]
        model, stats = train_model(tr, np.stack([imgs[w] for w in tr]),
                                   np.stack([standardise(sig_raw[w], mu, sd) for w in tr]),
                                   all_groups, caption_pairs(root, recs, set(tr)), text_vectors, salience, mode,
                                   epochs=epochs, hidden_dim=hidden_dim, dropout=dropout, seed=seed)
        embedded = encode("mlp", model, np.stack([imgs[w] for w in hw]),
                          np.stack([standardise(sig_raw[w], mu, sd) for w in hw]))
        outputs.update(zip(hw, embedded))
        report["folds"][held] = stats
        print(json.dumps({"mode": mode, "held_out": held, **stats}), flush=True)
    tr = [w for w in wins if window_span_s(w)[0] in train_recs]
    x = np.stack([sig_raw[w][0] for w in tr]); mu, sd = x.mean(0), x.std(0) + 1e-6
    salience = fold_salience["full"]
    model, stats = train_model(tr, np.stack([imgs[w] for w in tr]),
                               np.stack([standardise(sig_raw[w], mu, sd) for w in tr]),
                               all_groups, caption_pairs(root, train_recs, set(tr)), text_vectors, salience, mode,
                               epochs=epochs, hidden_dim=hidden_dim, dropout=dropout, seed=seed)
    model_dir = bundle.root / "models" / name
    save(model_dir, "mlp", model, mu, sd,
         {"trained_on": train_recs, "recipe": f"{name}: balanced verified supervision",
          "epochs": epochs, "seed": seed, "sampling": mode})
    model_hash = hashlib.sha256((model_dir / "head.safetensors").read_bytes()).hexdigest()
    report["full_model"] = stats
    val = [w for w in wins if w not in outputs]
    if val:
        outputs.update(zip(val, encode("mlp", model, np.stack([imgs[w] for w in val]),
                                        np.stack([standardise(sig_raw[w], mu, sd) for w in val]))))
    oof_source = ["loro_out_of_fold" if window_span_s(w)[0] in train_recs else "full_model" for w in wins]
    _write_index(bundle.root / "index" / f"{name}_oof_windows.parquet", wins, outputs, oof_source,
                 f"{name}: LORO evaluation", model_hash)
    served = dict(zip(wins, encode("mlp", model, np.stack([imgs[w] for w in wins]),
                                    np.stack([standardise(sig_raw[w], mu, sd) for w in wins]))))
    _write_index(bundle.root / "index" / f"{name}_windows.parquet", wins, served, ["full_model"] * len(wins),
                 f"{name}: full Train model serving", model_hash)
    report.update({"windows": len(wins), "model_hash": model_hash,
                   "positive_pairs": sum(len(g.positive) for g in all_groups),
                   "hard_negative_pairs": sum(len(g.hard) for g in all_groups),
                   "other_negative_pairs": sum(len(g.easy) for g in all_groups)})
    (bundle.root / "index" / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=SCAND_ROOT / "bundles" / "dev")
    ap.add_argument("--labels", type=Path, default=SCAND_ROOT / "labels")
    ap.add_argument("--sampler", choices=("uniform", "importance", "both"), default="both")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--diagnose-only", action="store_true",
                    help="score program families on existing OOF indexes without fitting a model")
    ap.add_argument("--salience-scores", type=Path,
                    default=SCAND_ROOT / "benchmark" / "importance" / "fused_v2_scores.jsonl")
    a = ap.parse_args()
    bundle = Bundle(a.bundle)
    train = sorted(set(bundle.recordings) - held_out())
    groups = build_programs(bundle, train) + attribute_groups(a.labels, train)
    texts = [g.spec.text for g in groups] + [t for _, t in caption_pairs(a.labels, train,
                                                                          set(bundle.embedding_index("siglip2").ids))]
    enc = SiglipEncoder()
    vectors = _encode_texts(enc, texts)
    del enc
    if a.diagnose_only:
        for mode in (("uniform", "importance") if a.sampler == "both" else (a.sampler,)):
            print(json.dumps({k: v for k, v in diagnose(bundle, groups, vectors, mode, train).items()
                              if k != "families"}))
        return
    labels = importance.targets(a.labels, set(train))
    # XGBoost and PyTorch link different OpenMP runtimes on some CPU machines.
    # Precompute nested scores in a separate process and cache them in the
    # ignored bundle. Each fold's scores are out of its own Train recordings.
    cache_path = bundle.root / "index" / "fused_v2_fold_salience.json"
    signature = importance.cache_signature(labels, train, a.seed)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get("signature") != signature:
        subprocess.run([sys.executable, "-m", "alloy_trainer.learn.importance",
                        "--bundle", str(a.bundle), "--labels", str(a.labels),
                        "--fold-cache", str(cache_path), "--seed", str(a.seed)], check=True)
        cache = json.loads(cache_path.read_text())
    fold_salience = cache["folds"]
    # The full serving model consumes the public LORO scores.
    manifest_path = a.salience_scores.with_name(a.salience_scores.stem + ".manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    label_hash = hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()
    if (not a.salience_scores.exists() or manifest.get("interval_sha256") != label_hash or
            manifest.get("params") != importance.PARAMS or manifest.get("rounds") != importance.ROUNDS):
        subprocess.run([sys.executable, "-m", "alloy_trainer.learn.importance",
                        "--bundle", str(a.bundle), "--labels", str(a.labels),
                        "--out", str(a.salience_scores)], check=True)
    fold_salience["full"] = {
        r["window_id"]: float(r["score"]) for r in
        map(json.loads, a.salience_scores.read_text().splitlines()) if r["source"] == "train_oof"}
    for mode in (("uniform", "importance") if a.sampler == "both" else (a.sampler,)):
        report = run(bundle, a.labels, mode, groups, vectors, fold_salience, epochs=a.epochs, seed=a.seed,
                     hidden_dim=a.hidden_dim, dropout=a.dropout)
        print(json.dumps({"completed": mode, "windows": report["windows"],
                          "positive_pairs": report["positive_pairs"],
                          "hard_negative_pairs": report["hard_negative_pairs"]}), flush=True)


if __name__ == "__main__":
    main()
