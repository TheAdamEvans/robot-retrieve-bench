"""When may an unmatched clause be FALSE rather than UNKNOWN? Context around a candidate that runs beyond the log's
start or end keeps FALSE false (the log is the world). A relation's window (the time the question waits for) that
runs past the end cannot be FALSE: the event may come after recording stopped. A sensor dropout inside the log makes
the clause UNKNOWN either way."""
from types import SimpleNamespace

import numpy as np

from alloy_server.evalkit import Dataset, Gate, exact
from alloy_server.verify.executor import EventEval, Executor

DATASETS = [Dataset("synthetic", "cases.jsonl")]
SCORERS = [exact("verdict")]
GATES = [Gate("verdict.accuracy", ">=", 1.0)]
S = 1_000_000_000


def run_case(case, ctx):
    i = case["input"]
    rec = SimpleNamespace(start_ns=0, end_ns=int(i["log_s"] * S))
    t = np.arange(i["feature_from_s"], i["feature_to_s"], 1 / i["hz"])
    if "gap_s" in i:  # remove samples: the sensor stopped reporting for a while
        t = t[(t < i["gap_s"][0]) | (t >= i["gap_s"][1])]
    t_ns = (t * S).astype(np.int64)
    series = SimpleNamespace(t_ns=t_ns, coverage_hz=i["hz"])
    ee = EventEval([], None, int(t_ns[0]), int(t_ns[-1]), True, series)
    covered = Executor._covered(ee, int(i["window_s"][0] * S), int(i["window_s"][1] * S), rec,
                                hard_edges=i.get("window") != "relation")
    return {"verdict": "FALSE" if covered else "UNKNOWN"}
