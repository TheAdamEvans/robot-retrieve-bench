"""Export prompt assets into the bundle: the program generator's few-shot pool (compose_dev only).

Reads the frozen benchmark query files directly (data, not the trainer), and refuses a pool that overlaps any
held-out utterance.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from alloy_index.recordings import SCAND_ROOT

QUERIES = SCAND_ROOT / "benchmark" / "queries"
HELD_OUT = ("demo5", "demo5_para", "compose_test")


def _norm(u: str) -> str:
    return re.sub(r"\s+", " ", u.strip().lower())


def _load(name: str) -> list[dict]:
    return json.loads((QUERIES / f"{name}.json").read_text())


def build(bundle: Path) -> dict:
    shots = [{"utterance": q["utterance"], "program": q["oracleProgram"]} for q in _load("compose_dev")]
    held_out = {_norm(q["utterance"]) for n in HELD_OUT for q in _load(n)}
    assert not held_out & {_norm(s["utterance"]) for s in shots}, "few-shot pool leaks held-out text"
    blob = json.dumps(shots, indent=1, sort_keys=True)
    p = bundle / "prompts" / "fewshots.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(blob)
    return {"fewshots": len(shots), "sha256": hashlib.sha256(blob.encode()).hexdigest()[:16]}
