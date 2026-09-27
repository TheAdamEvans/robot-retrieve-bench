"""Weak salience from Train answer-interval coverage of raw 1152D SigLIP2 frames.

Only positive Train episodes contribute to the target. A zero means no known
answer interval covers the frame, not a verified irrelevant frame. Scores are
out of recording for training and are used only as a bounded sampling hint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from alloy_index.annotate.store import load_labels
from alloy_index.recordings import SCAND_ROOT, held_out
from alloy_server.bundle import Bundle
from alloy_server.catalog.windows import window_span_s

PARAMS = {"objective": "count:poisson", "tree_method": "hist", "device": "cpu", "max_depth": 2,
          "eta": 0.05, "min_child_weight": 10, "colsample_bytree": 0.5, "subsample": 0.8,
          "lambda": 10, "nthread": 4, "seed": 0}
ROUNDS = 100


def targets(root: Path, recs: set[str]) -> dict[str, list[tuple[str, float, float]]]:
    """Count distinct answer intents covering a frame, using Train episodes only."""
    out: dict[str, list[tuple[str, float, float]]] = {r: [] for r in recs - held_out()}
    seen = set()
    for row in load_labels("episode", ("train",), root).values():
        rec = row["recordingId"]
        if rec not in out or int(row.get("grade", 0)) < 1:
            continue
        key = (rec, row["intentGroupId"], float(row["startS"]), float(row["endS"]))
        if key not in seen:
            out[rec].append(key[1:])
            seen.add(key)
    return out


@lru_cache(maxsize=32)
def _frames(bundle_root: str, rec: str, start_ns: int) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(Path(bundle_root) / "features" / "siglip2_frames" / f"{rec}.parquet",
                          columns=["log_time_ns", "vec"])
    t = (table.column("log_time_ns").to_numpy().astype(np.float64) - start_ns) / 1e9
    vec = table.column("vec").combine_chunks()
    x = np.asarray(vec.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(len(t), -1)
    if x.shape[1] != 1152:
        raise ValueError(f"expected raw 1152D SigLIP2 frames, got {x.shape[1]}D")
    order = np.argsort(t, kind="stable")
    return t[order], x[order]


def frames(bundle: Bundle, rec: str) -> tuple[np.ndarray, np.ndarray]:
    return _frames(str(bundle.root.resolve()), rec, bundle.recordings[rec].start_ns)


def counts(times: np.ndarray, intervals: list[tuple[str, float, float]]) -> np.ndarray:
    """Multiple intervals for one answer intent count once at each frame."""
    active = {}
    for intent, lo, hi in intervals:
        mask = active.setdefault(intent, np.zeros(len(times), bool))
        mask |= (times >= lo) & (times <= hi)
    return sum(active.values(), np.zeros(len(times), np.float32))


def fit(bundle: Bundle, labels: dict[str, list[tuple[str, float, float]]], recs: set[str]):
    import xgboost as xgb

    rng = np.random.default_rng(0)
    xs, ys, ws = [], [], []
    for rec in sorted(recs):
        t, x = frames(bundle, rec)
        y = counts(t, labels.get(rec, []))
        positive = np.flatnonzero(y > 0)
        background = np.flatnonzero(y == 0)
        take = min(len(background), max(500, 10 * len(positive)))
        sampled = rng.choice(background, take, replace=False) if take else np.array([], int)
        ix = np.concatenate([positive, sampled])
        if len(ix):
            xs.append(x[ix]); ys.append(y[ix])
            ws.append(np.where(y[ix] > 0, 1.0, len(background) / max(take, 1)))
    if not xs or not any(np.any(y > 0) for y in ys):
        raise ValueError("no positive Train answer intervals in the fit split")
    d = xgb.DMatrix(np.concatenate(xs), label=np.concatenate(ys), weight=np.concatenate(ws))
    return xgb.train(PARAMS, d, num_boost_round=ROUNDS, verbose_eval=False)


def predict_windows(bundle: Bundle, model, rec: str) -> dict[str, float]:
    import xgboost as xgb

    t, x = frames(bundle, rec)
    pred = model.predict(xgb.DMatrix(x)).astype(np.float32)
    cumulative = np.concatenate(([0.0], np.cumsum(pred, dtype=np.float64)))
    out = {}
    for wid in bundle.embedding_index("siglip2").ids:
        r, lo, hi = window_span_s(wid)
        if r != rec:
            continue
        first, last = np.searchsorted(t, [lo, hi], side="right")
        out[wid] = float((cumulative[last] - cumulative[first]) / (last - first)) if last > first else float(pred.mean())
    return out


def crossfit(bundle: Bundle, labels: dict[str, list[tuple[str, float, float]]], recs: list[str],
             groups: int = 3, seed: int = 0) -> dict[str, float]:
    """Scores for FUSED's training recordings, without fitting on each scored recording."""
    shuffled = sorted(recs)
    random.Random(seed).shuffle(shuffled)
    folds = [set(shuffled[i::groups]) for i in range(min(groups, len(recs)))]
    out = {}
    for fold in folds:
        model = fit(bundle, labels, set(recs) - fold)
        for rec in sorted(fold):
            out.update(predict_windows(bundle, model, rec))
    return out


def cache_signature(labels: dict, train: list[str], seed: int) -> str:
    payload = {"labels": labels, "train": train, "seed": seed, "params": PARAMS, "rounds": ROUNDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_fold_cache(bundle: Bundle, root: Path, out: Path, seed: int = 0) -> dict:
    """Precompute nested salience outside the PyTorch training process."""
    train = sorted(set(bundle.recordings) - held_out())
    labels = targets(root, set(train))
    scores = {held: crossfit(bundle, labels, [r for r in train if r != held], seed=seed)
              for held in train}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"signature": cache_signature(labels, train, seed),
                               "folds": scores}) + "\n")
    return {"folds": len(scores), "signature": cache_signature(labels, train, seed)}


def publish(bundle: Bundle, root: Path, out: Path) -> dict:
    import xgboost as xgb

    train = sorted(set(bundle.recordings) - held_out())
    labels = targets(root, set(train))
    rows = []
    for rec in train:
        model = fit(bundle, labels, set(train) - {rec})
        rows.extend({"window_id": w, "score": round(s, 6), "source": "train_oof"}
                    for w, s in predict_windows(bundle, model, rec).items())
    val = sorted(set(bundle.recordings) & held_out())
    if val:
        full = fit(bundle, labels, set(train))
        for rec in val:
            rows.extend({"window_id": w, "score": round(s, 6), "source": "val_full"}
                        for w, s in predict_windows(bundle, full, rec).items())
    rows.sort(key=lambda r: r["window_id"])
    truth, score = [], []
    for row in rows:
        rec, lo, hi = window_span_s(row["window_id"])
        if rec in labels:
            truth.append(sum(any(end >= lo and start <= hi for intent, start, end in labels[rec] if intent == name)
                             for name in {intent for intent, _, _ in labels[rec]}))
            score.append(row["score"])
    truth, score = np.asarray(truth, np.float32), np.asarray(score, np.float32)
    threshold = float(np.quantile(score, 0.9))
    top_lift = float(truth[score >= threshold].mean() / max(truth.mean(), 1e-8))
    meta = {"recipe": "train_episode_coverage_v1", "input": "raw 1152D SigLIP2 frame vector; no PCA",
            "target": "distinct positive Train answer intents active at frame; zero means unlabeled",
            "pooling": "mean frame prediction over trailing four-second window", "params": PARAMS,
            "rounds": ROUNDS, "xgboost": xgb.__version__, "train_recordings": train,
            "held_out_recordings": val, "positive_intervals": sum(map(len, labels.values())),
            "windows": len(rows), "loro_top_decile_interval_lift": round(top_lift, 5),
            "interval_sha256": hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    (out.parent / (out.stem + ".manifest.json")).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=SCAND_ROOT / "bundles" / "dev")
    ap.add_argument("--labels", type=Path, default=SCAND_ROOT / "labels")
    ap.add_argument("--out", type=Path, default=SCAND_ROOT / "benchmark" / "importance" / "fused_v2_scores.jsonl")
    ap.add_argument("--fold-cache", type=Path, help="write nested LORO salience for the retriever trainer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    bundle = Bundle(args.bundle)
    if args.fold_cache:
        print(json.dumps(write_fold_cache(bundle, args.labels, args.fold_cache, args.seed), indent=2))
    else:
        print(json.dumps(publish(bundle, args.labels, args.out), indent=2))


if __name__ == "__main__":
    main()
