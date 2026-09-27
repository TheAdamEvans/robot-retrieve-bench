"""The single execution path for live search, ScoreCandidates and the benchmark: run(bundle, spec, request)."""
from __future__ import annotations

import time
from typing import Callable

from ..gen.alloy.v1 import answer_pb2 as a
from ..gen.alloy.v1 import common_pb2 as c
from ..gen.alloy.v1 import pipeline_pb2 as pp
from ..gen.alloy.v1 import query_pb2 as q
from ..io import cost
from ..verify.receipts import receipt
from .context import ProgramUnavailable, QueryContext
from .stages import REGISTRY, FixedCandidates, Stage, StageResult


class InvariantViolation(AssertionError):
    pass


def build(spec_stage: pp.StageSpec) -> Stage:
    return REGISTRY[spec_stage.impl](spec_stage.stage_id, dict(spec_stage.params))


def _snapshot(cands: list[pp.Candidate]) -> dict[str, tuple]:
    return {cd.candidate_id: (cd.seed.start_ns, cd.seed.end_ns, dict((k, v.value) for k, v in cd.scores.items()),
                              len(cd.lineage), len(cd.clauses), len(cd.evidence)) for cd in cands}


def _check(stage: Stage, before: dict, after: list[pp.Candidate], res: StageResult) -> None:
    removed = set(before) - {cd.candidate_id for cd in after}
    declared = {f.candidate_id for f in res.filtered}
    if removed - declared and not stage.truncates:
        raise InvariantViolation(f"{stage.stage_id} removed candidates without FilterRecords")
    if declared and not stage.filters:
        raise InvariantViolation(f"{stage.stage_id} filtered but does not declare filtering")
    for cd in after:
        if cd.candidate_id not in before:
            continue
        s0, e0, scores0, nl, nc, ne = before[cd.candidate_id]
        if (cd.seed.start_ns, cd.seed.end_ns) != (s0, e0):
            raise InvariantViolation(f"{stage.stage_id} rewrote a seed")
        for k, v in scores0.items():
            if k != stage.stage_id and cd.scores[k].value != v:
                raise InvariantViolation(f"{stage.stage_id} modified {k}'s score")
        if len(cd.lineage) < nl or len(cd.clauses) < nc or len(cd.evidence) < ne:
            raise InvariantViolation(f"{stage.stage_id} shrank an append-only field")


def _stage_report(stage: Stage, role: int, n_in: int, res: StageResult, scope, bundle, recs, wall_ms) -> pp.StageReport:
    rep = pp.StageReport(stage_id=stage.stage_id, impl=stage.impl, role=role, candidates_in=n_in,
                         candidates_out=len(res.ordered), outcome=res.outcome, outcome_detail=res.detail)
    for f in res.filtered:
        key = pp.FilterReason.Name(f.reason)
        rep.filtered_by_reason[key] = rep.filtered_by_reason.get(key, 0) + 1
    if stage.truncates:
        rep.truncated = max(0, n_in - len(res.ordered))
    gen = next((c for c in scope.children if c.name == "program_generation"), None)
    rep.cost.CopyFrom(bundle.cost_report(scope, recs, wall_ms - (gen.wall_ms if gen else 0.0),
                                         exclude=("program_generation",)))
    return rep


def run(bundle, spec: pp.PipelineSpec, req: a.SearchRequest, generator: Callable | None = None) -> a.SearchResponse:
    t_start = time.perf_counter()
    resp = a.SearchResponse(bundle_id=bundle.bundle_id)
    root = pp.StageReport(stage_id=spec.pipeline_id, impl="Pipeline", role=pp.TOP_LEVEL, outcome=pp.RAN)
    if spec.stub:
        root.outcome, root.outcome_detail = pp.UNAVAILABLE, "stub: specified, not yet implemented"
        resp.report.CopyFrom(root)
        resp.status, resp.completeness = a.INSUFFICIENT_EVIDENCE, pp.COMPLETENESS_UNKNOWN
        resp.notes.append(f"{spec.pipeline_id} is a stub")
        return resp
    with cost.scope("request") as req_scope:
        ctx = QueryContext(bundle, req, generator)
        ctx.k = req.k or spec.final_k or 10  # request overrides the pipeline's default everywhere
        score_all = ctx.mode == pp.SCORE_ALL
        gens = [FixedCandidates("fixed", {})] if score_all else [build(g) for g in spec.generators]
        cands: list[pp.Candidate] = []
        filtered_all: list[pp.FilterRecord] = []
        filtered_cands: list[pp.Candidate] = []
        abstained, verified, completeness = False, False, pp.CANDIDATE_LIMITED
        seen: dict[str, pp.Candidate] = {}
        for g in gens:
            t0 = time.perf_counter()
            with cost.scope(g.stage_id) as s:
                try:
                    res = g.run(ctx, [])
                except ProgramUnavailable as ex:
                    res = StageResult([], outcome=pp.ABSTAINED if g.on_program_failure == "ABSTAIN" else pp.SKIPPED,
                                      detail=f"program_unavailable: {ex}")
                    abstained |= g.on_program_failure == "ABSTAIN"
            if ctx.generation_report is not None and not any(ch.stage_id == "program_generation" for ch in root.children):
                root.children.append(ctx.generation_report)
            root.children.append(_stage_report(g, pp.GENERATE, 0, res, s, bundle, ctx.scope_recordings,
                                               (time.perf_counter() - t0) * 1000))
            verified |= res.verified
            completeness = res.completeness if res.verified else completeness
            for cd in res.ordered:  # merge BY_ID: first generator's seed wins, scores kept per stage
                if cd.candidate_id in seen:
                    for k, v in cd.scores.items():
                        seen[cd.candidate_id].scores[k].CopyFrom(v)
                    seen[cd.candidate_id].lineage.extend(cd.lineage)
                else:
                    seen[cd.candidate_id] = cd
                    cands.append(cd)
        for spec_r in spec.rankers:
            st = build(spec_r)
            if score_all and st.truncates:
                continue
            if getattr(st, "display_only", False) and (score_all or not req.presentation):
                continue  # presentation-only stages never touch what the benchmark scores
            n_in = len(cands)
            before = _snapshot(cands)
            t0 = time.perf_counter()
            with cost.scope(st.stage_id) as s:
                try:
                    res = st.run(ctx, cands)
                except ProgramUnavailable as ex:
                    res = StageResult(cands, outcome=pp.SKIPPED, detail=f"program_unavailable: {ex}; results unverified")
            if ctx.generation_report is not None and not any(ch.stage_id == "program_generation" for ch in root.children):
                root.children.append(ctx.generation_report)
            _check(st, before, res.ordered, res)
            root.children.append(_stage_report(st, pp.RERANK, n_in, res, s, bundle, ctx.scope_recordings,
                                               (time.perf_counter() - t0) * 1000))
            verified |= res.verified and res.outcome == pp.RAN
            by_id = {cd.candidate_id: cd for cd in cands}
            for f in res.filtered:
                filtered_all.append(f)
                if f.candidate_id in by_id:
                    by_id[f.candidate_id].filtered = True
                    filtered_cands.append(by_id[f.candidate_id])
            cands = res.ordered
        final = cands + filtered_cands if score_all else cands[:ctx.k]
        prog = ctx.program
        for cd in final:
            item = resp.results.add()
            item.candidate.CopyFrom(cd)
            item.unsupported.extend(f"{cl.clause_id} ({c.UnknownReason.Name(cl.reason)}){': ' + cl.doc if cl.doc else ''}"
                                    for cl in cd.clauses if cl.required and cl.truth == c.TRUTH_UNKNOWN)
            if prog is not None and prog.HasField("receipt") and cd.anchors and not score_all:
                r = prog.receipt
                cut = next((x.t_ns for x in cd.anchors if x.name == f"{r.cutoff.event}.{q.AnchorPoint.Name(r.cutoff.point)}"),
                           cd.anchors[0].t_ns)
                cut = min(cut, next((x.lo_ns for x in cd.anchors if x.HasField("lo_ns")), cut))  # onset band: use lo
                item.receipt.CopyFrom(receipt(bundle.recordings[cd.recording_id], bundle.sensors(cd.recording_id),
                                              list(r.sensors), cut, int(r.max_age.value * 1e9), r.boundary))
        for ch in root.children:  # say why verification did not happen
            if ch.outcome in (pp.SKIPPED, pp.ABSTAINED) and "program_unavailable: " in ch.outcome_detail:
                note = ch.outcome_detail.split("program_unavailable: ", 1)[-1].split("; results unverified")[0]
                if note not in resp.notes:
                    resp.notes.append(note)
        if not score_all:
            bundle.verify_evidence(resp)  # hash-check cited payloads (reads MCAP chunks: counted bytes)
        resp.filtered.extend(filtered_all)
        resp.diagnostics.CopyFrom(ctx.diagnostics)
        if prog is not None:
            resp.program_used.CopyFrom(prog)
        resp.completeness = completeness
        resp.status = _status(resp, verified, abstained, prog, completeness, score_all)
        if prog is not None and prog.selection.quantifier in (q.FIRST, q.ARGMIN, q.ARGMAX) and completeness != pp.EXHAUSTIVE:
            resp.notes.append("selection is among retrieved candidates, not exhaustive over the scope")
    root.cost.CopyFrom(bundle.cost_report(req_scope, ctx.scope_recordings, (time.perf_counter() - t_start) * 1000))
    root.candidates_out = len(resp.results)
    resp.report.CopyFrom(root)
    resp.cost.CopyFrom(root.cost)
    return resp


def _status(resp: a.SearchResponse, verified: bool, abstained: bool, prog, completeness, score_all: bool) -> int:
    if abstained:
        return a.INSUFFICIENT_EVIDENCE
    if not verified:
        return a.ANSWERED_UNVERIFIED
    live = [r for r in resp.results if not r.candidate.filtered]
    if not live:
        return a.NONE_FOUND_EXHAUSTIVE if completeness == pp.EXHAUSTIVE else a.INSUFFICIENT_EVIDENCE
    top = live[0].candidate
    ordinal = next((s.value for s in top.scores.values() if s.kind == pp.TRUTH_ORDINAL), 0.0)
    if ordinal >= 2.0:
        return a.ANSWERED
    if prog is not None and prog.abstain_if_insufficient:
        # nothing definite: the question asked to abstain rather than be offered possibilities
        return a.INSUFFICIENT_EVIDENCE
    return a.ANSWERED_PARTIAL
