"""Benchmark metrics. Judgments are per intent group; unjudged windows are NEVER silently counted as non-relevant.

  * Eligibility fixed from judgments before any scoring: AUC-eligible = >=1 judged positive and >=1 judged negative;
    ranking-eligible = >=1 judged positive. Positive = grade 2 (grade >=1 reported as secondary).
  * Ranking: nDCG@10, R@10, R@50 on condensed lists (unjudged removed); judged@10 per config; unjudged-as-0 is a
    labelled sensitivity analysis only. Recall is pooled-reference recall.
  * AUC (ROC, PR): per query over the judged set S_q, scored by SCORE_ALL rank (filtered last), macro-averaged.
    Reported three ways: coverage, conditional AUC (scored queries only), penalised AUC (abstained/failed = 0.5 ROC,
    = prevalence PR). Penalised is the headline: abstention is not free.
  * Stage diagnostics from the StageReport tree: generator recall/precision, filter precision.
  * Uncertainty: bootstrap 95% CIs resampling intent groups; every table carries n.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np

POS = 2


def dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at(ranked: list[str], qrels: dict[str, int], k: int, unjudged_as_zero: bool = False) -> float | None:
    rel = [g for g in qrels.values() if g >= 0]
    if not any(g >= 1 for g in rel):
        return None
    lst = [qrels.get(w, 0 if unjudged_as_zero else None) for w in ranked]
    lst = [g for g in lst if g is not None and g >= 0][:k]
    ideal = sorted(rel, reverse=True)[:k]
    return dcg(lst) / dcg(ideal) if dcg(ideal) > 0 else None


def recall_at(ranked: list[str], qrels: dict[str, int], k: int) -> float | None:
    pos = {w for w, g in qrels.items() if g >= POS}
    if not pos:
        return None
    judged = [w for w in ranked if w in qrels and qrels[w] >= 0][:k]
    return len(pos & set(judged)) / len(pos)


def judged_at(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    top = ranked[:k]
    return sum(1 for w in top if w in qrels and qrels[w] >= 0) / max(len(top), 1)


def roc_auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def pr_auc(scores: list[float], labels: list[int]) -> float | None:
    if not any(labels):
        return None
    order = np.argsort(-np.asarray(scores), kind="stable")
    y = np.asarray(labels)[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())  # average precision


def score_all_scores(ranked_all: list[str], judged: list[str]) -> list[float]:
    """Score = negative final position under the config's own order (filtered windows are already last)."""
    pos = {w: i for i, w in enumerate(ranked_all)}
    n = len(ranked_all)
    return [-(pos.get(w, n)) for w in judged]


def bootstrap_ci(per_group: dict[str, float], iters: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    vals = [v for v in per_group.values() if v is not None]
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    keys = list(per_group)
    means = []
    for _ in range(iters):
        sample = [per_group[rng.choice(keys)] for _ in keys]
        sample = [v for v in sample if v is not None]
        if sample:
            means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]


def group_mean(per_query: dict[str, float | None], group_of: dict[str, str]) -> dict[str, float]:
    """Average queries within an intent group first (paraphrases are not independent)."""
    acc: dict[str, list[float]] = defaultdict(list)
    for qid, v in per_query.items():
        if v is not None:
            acc[group_of[qid]].append(v)
    return {g: sum(v) / len(v) for g, v in acc.items()}


def summarise(per_query: dict[str, float | None], group_of: dict[str, str]) -> dict:
    groups = group_mean(per_query, group_of)
    if not groups:
        return {"mean": None, "ci": None, "n_groups": 0, "n_queries": 0}
    return {"mean": sum(groups.values()) / len(groups), "ci": bootstrap_ci(groups), "n_groups": len(groups),
            "n_queries": sum(v is not None for v in per_query.values())}
