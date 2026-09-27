"""Export prompt assets into the bundle: the program generator's few-shot pool (compose_dev only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from google.protobuf import json_format

from alloy_train.eval import querysets


def build(bundle: Path) -> dict:
    shots = [{"utterance": q.utterance, "program": json_format.MessageToDict(q.oracle_program)}
             for q in querysets.load(["compose_dev"])]
    held_out = {querysets.normalise_utt(q.utterance) for q in querysets.load(["demo5", "demo5_para", "compose_test"])}
    assert not held_out & {querysets.normalise_utt(s["utterance"]) for s in shots}, "few-shot pool leaks held-out text"
    blob = json.dumps(shots, indent=1, sort_keys=True)
    p = bundle / "prompts" / "fewshots.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(blob)
    return {"fewshots": len(shots), "sha256": hashlib.sha256(blob.encode()).hexdigest()[:16]}
