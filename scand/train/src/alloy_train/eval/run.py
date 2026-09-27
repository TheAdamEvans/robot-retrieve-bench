"""Benchmark runner: every config x every query through alloy_server.pipeline.run (the live search path).

Configs = (pipeline spec, program source). Program sources: none, ORACLE (the frozen Claude-authored program,
ProgramState=PROVIDED) or LUNA (gpt-6-luna, generated lazily only when a stage needs it). One generator instance with
a per-run cache is shared, so PROGRAM_LUNA and HYBRID_LUNA see the same generated program; its generation cost is
recorded once per query and attributed to every LUNA config in the report.

Output: results/eval/<run>/runs.jsonl — one row per (query, config) with ranked window ids, status, completeness,
per-stage report, cost and program diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from google.protobuf import json_format

from alloy_server.bundle import Bundle
from alloy_server.gen.alloy.v1 import answer_pb2 as a
from alloy_server.gen.alloy.v1 import pipeline_pb2 as pp
from alloy_server.pipeline.runner import run as pipeline_run
from alloy_server.programs.openai_generator import OpenAIProgramGenerator
from alloy_train.eval import querysets
from alloy_train.recordings import SCAND_ROOT

CONFIGS = {
    "TAGS": ("TAGS", None),
    "EMBED": ("EMBED", None),
    "PROGRAM_ORACLE": ("PROGRAM", "oracle"),
    "PROGRAM_LUNA": ("PROGRAM", "luna"),
    "HYBRID_ORACLE": ("HYBRID", "oracle"),
    "HYBRID_LUNA": ("HYBRID", "luna"),
    "FUSED": ("FUSED", None),
    "FUSED_LINEAR": ("FUSED_LINEAR", None),
    "FUSED_CONCAT": ("FUSED_CONCAT", None),
    "FUSED_V_LUNA": ("FUSED_V", "luna"),
}
EVAL_SETS = ["demo5", "demo5_para", "compose_test"]


def interval_windows(bundle: Bundle, cd) -> list[str]:
    """An INTERVAL result stands for every window within 1 s of its primary anchor — the same unit the judges grade
    (grade 2 = contains or is within 1 s of an anchor). Mapping a match to one window would cap PROGRAM's recall."""
    rec = bundle.recordings[cd.recording_id]
    t = rec.t_rel(cd.anchors[0].t_ns if cd.anchors else cd.seed.start_ns)
    first = bundle.window_for_interval(cd)
    ids = set(bundle.windows[cd.recording_id])
    near = [f"{cd.recording_id}:{e:04d}" for e in range(int(math.ceil(t - 1)), int(math.floor(t + 1)) + 5)
            if f"{cd.recording_id}:{e:04d}" in ids and e - 4 <= t + 1 and e >= t - 1]
    return ([first] if first else []) + [w for w in near if w != first]


def ranked_windows(bundle: Bundle, resp: a.SearchResponse) -> list[str]:
    """Candidates → window ids in rank order; INTERVAL candidates expand to their anchor's windows."""
    out, seen = [], set()
    for item in resp.results:
        cd = item.candidate
        for wid in ([cd.window_id] if cd.window_id else interval_windows(bundle, cd)):
            if wid and wid not in seen:
                seen.add(wid)
                out.append(wid)
    return out


def flat_report(r: pp.StageReport) -> list[dict]:
    out = [{"stage_id": r.stage_id, "impl": r.impl, "in": r.candidates_in, "out": r.candidates_out,
            "filtered": dict(r.filtered_by_reason), "truncated": r.truncated, "outcome": pp.StageOutcome.Name(r.outcome),
            "detail": r.outcome_detail, "triggered_by": r.triggered_by, "wall_ms": r.cost.wall_ms,
            "bytes": r.cost.bytes_read, "tokens": r.cost.prompt_tokens + r.cost.completion_tokens}]
    for ch in r.children:
        out += flat_report(ch)
    return out


def request_for(q, spec_id: str, source: str | None, k: int, mode=pp.SEARCH, window_ids=()) -> a.SearchRequest:
    req = a.SearchRequest(utterance=q.utterance, pipeline_id=spec_id, k=k, mode=mode)
    req.scope.CopyFrom(q.scope)
    req.window_ids.extend(window_ids)
    if source == "oracle":
        req.program.CopyFrom(q.oracle_program)
    return req


def run_all(bundle: Bundle, out_dir: Path, configs: list[str], sets: list[str], k: int = 50) -> Path:
    dev = [(q.utterance, json_format.MessageToDict(q.oracle_program)) for q in querysets.load(["compose_dev"])]
    gen = OpenAIProgramGenerator(bundle, dev, cache_dir=out_dir / "program_cache")
    assert not gen.fewshot_utterances & {querysets.normalise_utt(q.utterance) for q in querysets.load(sets)}, \
        "few-shot pool overlaps an evaluated utterance"
    out_dir.mkdir(parents=True, exist_ok=True)
    from alloy_server.models.siglip import SPACE_ID
    bundle.encode_query(SPACE_ID, "warm-up")  # model load is startup cost, never charged to the first query
    rows_path = out_dir / "runs.jsonl"
    done = set()
    if rows_path.exists():
        done = {(r["query_id"], r["config"]) for r in map(json.loads, rows_path.read_text().splitlines())}
    gen_cost: dict[str, dict] = {}
    with open(rows_path, "a") as f:
        for q in querysets.load(sets):
            for name in configs:
                if (q.query_id, name) in done:
                    continue
                spec_id, source = CONFIGS[name]
                t0 = time.perf_counter()
                resp = pipeline_run(bundle, bundle.specs[spec_id], request_for(q, spec_id, source, k),
                                    generator=gen if source == "luna" else None)
                wall = (time.perf_counter() - t0) * 1000
                stages = flat_report(resp.report)
                pg = next((s for s in stages if s["stage_id"] == "program_generation"), None)
                if pg is not None and not resp.diagnostics.cache_hit:
                    gen_cost[q.query_id] = {"wall_ms": pg["wall_ms"], "tokens": pg["tokens"]}
                gen_spec = pp.PipelineSpec()
                gen_spec.CopyFrom(bundle.specs[spec_id])
                del gen_spec.rankers[:]
                gen_spec.final_k = 100000  # generators only: what reached the rankers (for generator recall)
                gen_resp = pipeline_run(bundle, gen_spec, request_for(q, spec_id, source, 100000),
                                        generator=gen if source == "luna" else None)
                row = {
                    "generated": ranked_windows(bundle, gen_resp),
                    "query_id": q.query_id, "intent_group_id": q.intent_group_id, "query_set": q.query_set,
                    "config": name, "windows": ranked_windows(bundle, resp),
                    "status": a.AnswerStatus.Name(resp.status), "completeness": pp.Completeness.Name(resp.completeness),
                    "expected_status": a.AnswerStatus.Name(q.expected_status),
                    "program_state": a.ProgramState.Name(resp.diagnostics.state),
                    "program_errors": list(resp.diagnostics.errors), "program_attempts": resp.diagnostics.attempts,
                    "program_cache_hit": resp.diagnostics.cache_hit,
                    "program": json_format.MessageToDict(resp.program_used) if resp.HasField("program_used") else None,
                    "stages": stages, "cost": json_format.MessageToDict(resp.cost), "wall_ms": wall,
                    "filtered": [json_format.MessageToDict(x) for x in resp.filtered],
                    "unsupported_top": list(resp.results[0].unsupported) if resp.results else [],
                    "generation_cost": gen_cost.get(q.query_id) if source == "luna" else None,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{q.query_id:40s} {name:15s} {row['status']:22s} n={len(row['windows']):3d} {wall:8.0f} ms",
                      flush=True)
    return rows_path


def score_all(bundle: Bundle, out_dir: Path, judged: dict[str, list[str]], configs: list[str], sets: list[str]) -> Path:
    """SCORE_ALL over each intent's judged windows: every window gets a position under the config's own order,
    filtered ones last and flagged. Costs here are eval-only and never mixed into search cost."""
    dev = [(q.utterance, json_format.MessageToDict(q.oracle_program)) for q in querysets.load(["compose_dev"])]
    gen = OpenAIProgramGenerator(bundle, dev, cache_dir=out_dir / "program_cache")  # cache hits from the search run
    path = out_dir / "score_all.jsonl"
    with open(path, "w") as f:
        for q in querysets.load(sets):
            wins = judged.get(q.intent_group_id, [])
            if not wins:
                continue
            for name in configs:
                spec_id, source = CONFIGS[name]
                resp = pipeline_run(bundle, bundle.specs[spec_id],
                                    request_for(q, spec_id, source, len(wins), mode=pp.SCORE_ALL, window_ids=wins),
                                    generator=gen if source == "luna" else None)
                order = [it.candidate.window_id for it in resp.results]
                filtered = [it.candidate.window_id for it in resp.results if it.candidate.filtered]
                abstained = resp.status == a.INSUFFICIENT_EVIDENCE and not order or \
                    any(s["outcome"] == "ABSTAINED" for s in flat_report(resp.report))
                f.write(json.dumps({"query_id": q.query_id, "intent_group_id": q.intent_group_id,
                                    "query_set": q.query_set, "config": name, "order": order, "filtered": filtered,
                                    "abstained": bool(abstained),
                                    "program_state": a.ProgramState.Name(resp.diagnostics.state)}) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=SCAND_ROOT / "bundles" / "dev")
    ap.add_argument("--run", required=True)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--sets", default=",".join(EVAL_SETS))
    ap.add_argument("--score-all", action="store_true", help="SCORE_ALL over each intent's judged windows")
    a_ = ap.parse_args()
    bundle = Bundle(a_.bundle)
    out = SCAND_ROOT / "results" / "eval" / a_.run
    if a_.score_all:
        from alloy_train.eval.report import load_judgments
        judged = {k: sorted(w for w, g in v.items() if g >= 0) for k, v in load_judgments(SCAND_ROOT / "annotations").items()}
        print(score_all(bundle, out, judged, a_.configs.split(","), a_.sets.split(",")))
        return
    run_all(bundle, out, a_.configs.split(","), a_.sets.split(","))


if __name__ == "__main__":
    main()
