"""ONSET timing. Seeded by the labeller's finding that stop→go onsets came ~0.6 s late on Spot (gait oscillation in
the raw speed dipped below threshold after the true start); the fix times onsets on the raw signal."""
import numpy as np
from google.protobuf import json_format

from alloy_server.catalog.features import Series
from alloy_server.catalog.registry import REGISTRY
from alloy_server.evalkit import Dataset, Gate, abs_error, exact
from alloy_server.gen.alloy.v1 import answer_pb2 as a, query_pb2 as q
from alloy_server.verify.executor import onset_instances

DATASETS = [Dataset("synthetic", "cases_synthetic.jsonl"),
            Dataset("recorded", "cases_recorded.jsonl", requires=["bundle"])]
SCORERS = [abs_error("onset_s"), exact("n_onsets")]
GATES = [Gate("onset_s.abs_error.max", "<=", 0.10), Gate("n_onsets.accuracy", ">=", 1.0)]

GO = {"name": "go", "kind": "ONSET", "feature": "speed_mps", "fromBelow": {"value": 0.05, "unit": "MPS"},
      "minDuration": {"value": 1, "unit": "S"}, "threshold": {"value": 0.1, "unit": "MPS"},
      "sustain": {"value": 0.5, "unit": "S"}, "required": True}


def _synthetic(i):
    t = (np.arange(0, i["duration_s"], 0.06) * 1e9).astype(np.int64)
    ts = t / 1e9 - i["start_s"]
    raw = np.where(ts < 0, 0.0, i["mean_mps"] + i["osc_mps"] * np.sin(2 * np.pi * i["gait_hz"] * ts))
    smooth = np.convolve(raw, np.ones(15) / 15)[: len(raw)]
    s = Series(REGISTRY["speed_mps"], t, smooth, None, None, t, 16.0, True, [])
    inst = onset_instances(s, json_format.ParseDict(GO, q.EventSpec()), raw)
    return [x.start / 1e9 for x in inst]


def _recorded(i, ctx):
    from alloy_server.pipeline.runner import run
    b = ctx.bundle
    p = json_format.ParseDict({"primaryEvent": "go", "selection": {"quantifier": "ALL"},
                               "scope": {"recordingIds": [i["recording"]]}, "events": [GO],
                               "contextBefore": {"value": 4, "unit": "S"}, "contextAfter": {"value": 4, "unit": "S"}},
                              q.QueryProgram())
    r = run(b, b.specs["PROGRAM"], a.SearchRequest(utterance="starts moving from a stop", pipeline_id="PROGRAM",
                                                    program=p, k=50))
    t0 = b.recordings[i["recording"]].start_ns
    return sorted((an.t_ns - t0) / 1e9 for it in r.results for an in it.candidate.anchors if an.name == "go.START")


def run_case(case, ctx):
    i = case["input"]
    onsets = _recorded(i, ctx) if "recording" in i else _synthetic(i)
    want = case["expect"].get("onset_s")
    nearest = min(onsets, key=lambda x: abs(x - want)) if onsets and want is not None else float("inf")
    return {"onset_s": nearest, "n_onsets": len(onsets)}
