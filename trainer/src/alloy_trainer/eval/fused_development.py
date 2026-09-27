"""Train-only retrieval selection: held-out programs and utterances on LORO vectors.

Only executor-verifiable windows are judged. Unknown windows count towards judged
coverage, but never as negatives. The frozen Val and compose_test sets stay out of
this development measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from alloy_index.recordings import SCAND_ROOT, held_out
from alloy_server.bundle import Bundle
from alloy_server.catalog.windows import window_span_s
from alloy_server.models.siglip import SiglipEncoder
from alloy_trainer.learn.fused_supervision import Supervision, build, development_programs


def ranked_metrics(ranked: list[str], positive: set[str], negative: set[str]) -> dict:
    """Top ranks measure retrieval; coverage makes unknown-heavy lists visible."""
    if not positive:
        raise ValueError("ranked metrics require at least one verified positive")
    top10, top50 = ranked[:10], ranked[:50]
    judged10 = [w for w in top10 if w in positive or w in negative]
    # Do not backfill top ranks with judged windows from deep in the list.
    # Unknowns receive no credit, and judged_at_10 shows how much is unjudged.
    dcg = sum((w in positive) / np.log2(i + 2) for i, w in enumerate(top10))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(positive))))
    return {
        "hit_at_10": float(any(w in positive for w in top10)),
        "recall_at_50": sum(w in positive for w in top50) / len(positive),
        "precision_at_10_judged": sum(w in positive for w in judged10) / len(judged10) if judged10 else None,
        "ndcg_at_10_judged": dcg / ideal if ideal else None,
        "judged_at_10": len(judged10) / len(top10) if top10 else None,
        "positives": len(positive), "negatives": len(negative),
    }


def _mean(rows: list[dict]) -> dict:
    keys = ("hit_at_10", "recall_at_50", "precision_at_10_judged", "ndcg_at_10_judged", "judged_at_10")
    return {k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 4)
            if any(r[k] is not None for r in rows) else None for k in keys} | {"n": len(rows)}


def measure(index, groups: list[Supervision], vectors: dict[str, np.ndarray], train_recs: list[str]) -> dict:
    global_rows, rec_rows = [], []
    by_rec = defaultdict(list)
    by_family = defaultdict(list)
    train = set(train_recs)
    indices = {r: np.flatnonzero(index.recs == r) for r in train_recs}
    all_ix = np.flatnonzero(np.isin(index.recs, train_recs))
    for group in groups:
        pos, neg = group.positive, group.hard | group.easy
        if not pos or not neg:
            continue
        sims = index.vecs @ vectors[group.spec.text]
        ranked = [index.ids[i] for i in all_ix[np.argsort(-sims[all_ix], kind="stable")]]
        row = ranked_metrics(ranked, pos, neg) | {"family": group.spec.family, "query": group.spec.text}
        global_rows.append(row)
        for rec, ix in indices.items():
            rpos = {w for w in pos if window_span_s(w)[0] == rec}
            rneg = {w for w in neg if window_span_s(w)[0] == rec}
            if not rpos or not rneg:
                continue
            ranked = [index.ids[i] for i in ix[np.argsort(-sims[ix], kind="stable")]]
            metric = ranked_metrics(ranked, rpos, rneg) | {"recording": rec, "family": group.spec.family,
                                                         "query": group.spec.text}
            rec_rows.append(metric)
            by_rec[rec].append(metric)
            by_family[group.spec.family].append(metric)
    return {"global": _mean(global_rows), "within_recording": _mean(rec_rows),
            "by_recording": {r: _mean(rows) for r, rows in sorted(by_rec.items())},
            "by_family": {f: _mean(rows) for f, rows in sorted(by_family.items())},
            "queries": global_rows, "recording_queries": rec_rows}


def require_out_of_fold(bundle: Bundle, space: str, train_recs: list[str]) -> None:
    """Refuse a serving index that has embedded its own training recordings."""
    path = bundle.root / "index" / f"{space}_windows.parquet"
    table = pq.read_table(path, columns=["recording_id", "source"])
    bad = [(rec, source) for rec, source in zip(table["recording_id"].to_pylist(),
                                                 table["source"].to_pylist())
           if rec in train_recs and source != "loro_out_of_fold"]
    if bad:
        raise ValueError(f"{space} is not out of fold on Train: {bad[:3]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=SCAND_ROOT / "bundles" / "dev")
    ap.add_argument("--spaces", nargs="+", default=["fused_v1", "fused_v2_uniform_oof", "fused_v2_importance_oof"])
    ap.add_argument("--output", type=Path, default=SCAND_ROOT / "results" / "eval" / "fused_development.json")
    a = ap.parse_args()
    bundle = Bundle(a.bundle)
    train = sorted(set(bundle.recordings) - held_out())
    specs = development_programs()
    query_sha256 = hashlib.sha256(json.dumps(
        [(s.family, s.text, s.program) for s in specs], sort_keys=True).encode()).hexdigest()
    groups = build(bundle, train, specs)
    texts = [g.spec.text for g in groups]
    encoder = SiglipEncoder()
    vectors = dict(zip(texts, encoder.encode_texts(texts).astype(np.float32)))
    for space in a.spaces:
        require_out_of_fold(bundle, space, train)
    spaces = {}
    for space in a.spaces:
        meta = pq.read_metadata(bundle.root / "index" / f"{space}_windows.parquet").metadata or {}
        spaces[space] = measure(bundle.embedding_index(space), groups, vectors, train) | {
            "index_model_sha256": meta.get(b"model_hash", b"").decode(),
            "index_recipe": meta.get(b"recipe", b"").decode()}
    report = {"protocol": "Train recordings only; disjoint program settings and exact utterances; LORO indexes",
              "query_sha256": query_sha256, "spaces": spaces}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({space: {k: result[k] for k in ("global", "within_recording")}
                      for space, result in report["spaces"].items()}, indent=2))


if __name__ == "__main__":
    main()
