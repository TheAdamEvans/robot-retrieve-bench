"""Judgment pools per intent group (TREC-style), built BEFORE grading and blind to which config retrieved what.

Pool(intent) = union over the group's paraphrases of
  * the top-POOL_DEPTH windows of every config,
  * attribute-derived candidates from the L1 labels (so relevant windows no config found can still be judged),
  * RANDOM_PER_INTENT stratified random windows from the intent's scope (a floor on negatives / pool-bias check).
Windows outside the pool stay UNJUDGED and are never counted as non-relevant by default.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from alloy_server.catalog.windows import parse_window_id, window_span_s, windows
from alloy_trainer.eval import querysets
from alloy_index.annotate.store import LABELS, load_labels

POOL_DEPTH = 20
RANDOM_PER_INTENT = 10
MAX_ATTRIBUTE = 25  # cap attribute-derived additions (seeded sample) so broad rules don't swamp the pool


def l1_labels(root: Path = LABELS) -> dict[str, dict]:
    """L1 attributes: training data (labels/train/), read here only as a recall aid for the pool."""
    return load_labels("attributes", ("train",), root)


def attribute_candidates(intent: str, labels: dict[str, dict]) -> list[str]:
    """Segments whose L1 attributes make them plausibly relevant (recall aid for the pool, not a grade)."""
    t = lambda r, f: r.get(f) == "TRUTH_TRUE"
    many = lambda r: r.get("personsInCorridor") in ("THREE_FIVE", "SIX_PLUS")
    some = lambda r: r.get("personsInCorridor") in ("ONE_TWO", "THREE_FIVE", "SIX_PLUS")
    rules = {
        "crowd_hesitation": lambda r: many(r),
        "doorway_crossing": lambda r: t(r, "doorwayTraversal"),
        "vehicle_interaction_gdc": lambda r: t(r, "vehicleInteraction") or t(r, "vehiclePresent"),
        "chained_turn_person": lambda r: t(r, "turnVisible") and some(r),
        "test_speedup_after_person": lambda r: some(r),
        "test_turn_then_brake": lambda r: t(r, "turnVisible"),
        "test_car_close_fast": lambda r: t(r, "vehiclePresent"),
        "test_body_cam_person": lambda r: r.get("personsInCorridor") == "ZERO",
    }
    rule = rules.get(intent)
    return sorted(sid for sid, r in labels.items() if rule and rule(r))


def build(runs: Path, root: Path, bundle_windows: dict[str, list[str]], seed: int = 7) -> dict[str, dict]:
    rows = [json.loads(x) for x in runs.read_text().splitlines()]
    labels = l1_labels(root)
    queries = {q.query_id: q for q in querysets.load()}
    by_intent: dict[str, dict] = defaultdict(lambda: {"windows": set(), "sources": defaultdict(set)})
    for r in rows:
        pool = by_intent[r["intent_group_id"]]
        for w in r["windows"][:POOL_DEPTH]:
            pool["windows"].add(w)
            pool["sources"][w].add(r["config"])
    rng = random.Random(seed)
    out = {}
    for intent, pool in by_intent.items():
        q = next(q for q in queries.values() if q.intent_group_id == intent)
        if q.expect_abstain:  # nothing in the data can be relevant: scored on status only, not judged
            continue
        scope = list(q.scope.recording_ids) or sorted(bundle_windows)
        attr = [w for w in attribute_candidates(intent, labels) if parse_window_id(w)[0] in scope]
        if len(attr) > MAX_ATTRIBUTE:
            attr = sorted(rng.sample(attr, MAX_ATTRIBUTE))
        for w in attr:
            pool["windows"].add(w)
            pool["sources"][w].add("L1_ATTRIBUTES")
        universe = [w for rec in scope for w in bundle_windows[rec]]
        for w in rng.sample(universe, min(RANDOM_PER_INTENT, len(universe))):
            pool["windows"].add(w)
            pool["sources"][w].add("RANDOM")
        out[intent] = {"intent": q.intent, "scope": scope, "windows": sorted(pool["windows"]),
                       "n": len(pool["windows"]), "sources": {w: sorted(s) for w, s in pool["sources"].items()}}
    return out


def write(pools: dict[str, dict], out: Path) -> None:
    out.write_text(json.dumps(pools, indent=1))
