"""MergeAdjacent on hand-built ranked lists: which results fold together, the span shown, and that weak results
never widen a span. Input windows are [recording, start_s, end_s] in rank order."""
from types import SimpleNamespace

from alloy_server.evalkit import Dataset, Gate, exact
from alloy_server.gen.alloy.v1 import pipeline_pb2 as pp
from alloy_server.pipeline.stages import MergeAdjacent

DATASETS = [Dataset("lists", "cases.jsonl")]
SCORERS = [exact("groups"), exact("spans"), exact("all_members_accounted")]
GATES = [Gate("groups.accuracy", ">=", 1.0), Gate("spans.accuracy", ">=", 1.0),
         Gate("all_members_accounted.accuracy", ">=", 1.0)]


def run_case(case, ctx):
    i = case["input"]
    cands = []
    for n, (rec, s, e) in enumerate(i["windows"]):
        cd = pp.Candidate(candidate_id=f"w{n}", recording_id=rec, kind=pp.WINDOW, window_id=f"{rec}:{n}")
        cd.seed.start_ns, cd.seed.end_ns = int(s * 1e9), int(e * 1e9)
        cands.append(cd)
    st = MergeAdjacent("merge", {k: str(v) for k, v in i.get("params", {}).items()})
    res = st.run(SimpleNamespace(k=i.get("k", 10)), cands)
    groups = [sorted(int(m[1:]) for m in cd.member_ids) if cd.kind == pp.MERGED else [int(cd.candidate_id[1:])]
              for cd in res.ordered]
    spans = [[cd.seed.start_ns / 1e9, cd.seed.end_ns / 1e9] for cd in res.ordered]
    into = {f.candidate_id for f in res.filtered if f.reason == pp.MERGED_INTO}
    members = {m for cd in res.ordered if cd.kind == pp.MERGED for m in cd.member_ids}
    return {"groups": groups, "spans": spans, "all_members_accounted": into == members}
