"""CandidateGenerators and Rankers. Every search config is a PipelineSpec composed from these; none is an engine.

Contracts (checked by the runner after every stage):
  * candidate_id, seed and other stages' scores are never modified; lineage/clauses/evidence only grow;
  * a stage writes only scores[its stage_id];
  * removals only by stages that declare filtering, each recorded as a FilterRecord.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

import numpy as np

from ..catalog.windows import parse_window_id, window_span_s
from ..gen.alloy.v1 import common_pb2 as c
from ..gen.alloy.v1 import pipeline_pb2 as pp
from ..gen.alloy.v1 import query_pb2 as q
from .context import ProgramUnavailable, QueryContext

NS = 1_000_000_000


def candidate_id(rec: str, kind: int, start: int, end: int) -> str:
    return hashlib.sha256(f"{rec}|{kind}|{start}|{end}".encode()).hexdigest()[:16]


def window_candidate(ctx: QueryContext, wid: str, stage_id: str) -> pp.Candidate:
    rec, t0, t1 = window_span_s(wid)
    r = ctx.bundle.recordings[rec]
    s, e = r.t_abs(t0), r.t_abs(t1)
    cand = pp.Candidate(candidate_id=candidate_id(rec, pp.WINDOW, s, e), kind=pp.WINDOW, recording_id=rec,
                        window_id=wid, completeness=pp.CANDIDATE_LIMITED, spatial_basis=c.SPATIAL_NONE)
    cand.seed.start_ns, cand.seed.end_ns = s, e
    cand.lineage.add(stage_id=stage_id, action="generated")
    return cand


@dataclass
class StageResult:
    ordered: list[pp.Candidate]
    filtered: list[pp.FilterRecord] = field(default_factory=list)
    outcome: int = pp.RAN
    detail: str = ""
    completeness: int = pp.CANDIDATE_LIMITED
    verified: bool = False


class Stage:
    impl = "Stage"
    needs_program = False
    filters = False
    truncates = False
    on_program_failure = "ABSTAIN"   # or SKIP_MARK_UNVERIFIED

    def __init__(self, stage_id: str, params: dict[str, str]):
        self.stage_id, self.params = stage_id, params

    def score(self, cand: pp.Candidate, value: float, kind: int) -> None:
        cand.scores[self.stage_id].CopyFrom(pp.StageScore(stage_id=self.stage_id, impl=self.impl, value=value, kind=kind))
        cand.lineage.add(stage_id=self.stage_id, action="scored")


# ---------------- generators ----------------

class TagCandidates(Stage):
    """BM25 over recording-level tags (SCAND_index.csv). Every window of a matching recording inherits the score:
    the tags say nothing about *where* in the recording something happens."""
    impl = "TagCandidates"

    def run(self, ctx: QueryContext, cands: list[pp.Candidate]) -> StageResult:
        scores = ctx.bundle.tags.scores(ctx.utterance)
        out = []
        for rec in ctx.scope_recordings:
            if scores.get(rec, 0.0) <= 0:
                continue
            for wid in ctx.bundle.windows[rec]:
                cand = window_candidate(ctx, wid, self.stage_id)
                self.score(cand, scores[rec], pp.BM25)
                out.append(cand)
        out.sort(key=lambda x: (-x.scores[self.stage_id].value, x.candidate_id))
        return StageResult(out)


class EmbeddingCandidates(Stage):
    """Encode the utterance with the space's text tower; exact dot product over the window index."""
    impl = "EmbeddingCandidates"

    def run(self, ctx: QueryContext, cands: list[pp.Candidate]) -> StageResult:
        index = ctx.bundle.embedding_index(self.params.get("space", "siglip2"))
        qv = ctx.query_vector(index.space_id)
        k = int(self.params.get("top_k", 100))
        ids, sims = index.search(qv, ctx.scope_recordings, k)
        out = []
        for wid, sim in zip(ids, sims):
            cand = window_candidate(ctx, wid, self.stage_id)
            self.score(cand, float(sim), pp.COSINE)
            out.append(cand)
        return StageResult(out)


class ProgramCandidates(Stage):
    """Execute the program's primary event over the full scope → INTERVAL candidates, EXHAUSTIVE for the structured
    clauses whose features are indexed at native rate."""
    impl = "ProgramCandidates"
    needs_program = True

    def run(self, ctx: QueryContext, cands: list[pp.Candidate]) -> StageResult:
        prog = ctx.require_program(self)
        ex = ctx.bundle.executor
        out, exhaustive = [], True
        for rec in ctx.program_recordings(prog):
            ms, pe = ex.matches(prog, rec)
            exhaustive &= pe.unknown_reason is None and pe.exhaustive
            for m in ms:
                if m.ordinal <= 0:  # a required clause is definitively FALSE: not a candidate
                    continue
                s, e = m.interval
                cand = pp.Candidate(candidate_id=candidate_id(rec, pp.INTERVAL, s, e), kind=pp.INTERVAL,
                                    recording_id=rec)
                cand.seed.start_ns, cand.seed.end_ns = s, e
                cand.lineage.add(stage_id=self.stage_id, action="generated")
                ctx.attach_match(cand, m, self.stage_id)
                self.score(cand, m.ordinal, pp.TRUTH_ORDINAL)
                out.append(cand)
        out = ctx.order_by_selection(prog, out, self.stage_id)
        comp = pp.EXHAUSTIVE if exhaustive else pp.COMPLETENESS_UNKNOWN
        return StageResult(out, completeness=comp, verified=True)


class FixedCandidates(Stage):
    """The given windows (SCORE_ALL / eval only)."""
    impl = "FixedCandidates"

    def run(self, ctx: QueryContext, cands: list[pp.Candidate]) -> StageResult:
        return StageResult([window_candidate(ctx, w, self.stage_id) for w in ctx.window_ids])


# ---------------- rankers ----------------

class BM25Ranker(Stage):
    impl = "BM25Ranker"

    def run(self, ctx: QueryContext, cands: list[pp.Candidate]) -> StageResult:
        scores = ctx.bundle.tags.scores(ctx.utterance)
        for cand in cands:
            self.score(cand, scores.get(cand.recording_id, 0.0), pp.BM25)
        return StageResult(ctx.sort(cands, self.order_by()))

    def order_by(self):
        return [(self.stage_id, True)]


class SimilarityRanker(Stage):
    """Cosine similarity computed here (never silently reused from a generator). Optional `order_by`: a
    lexicographic key list like 'PredicateRanker:desc,self:desc' (explicit fusion, visible in the spec)."""
    impl = "SimilarityRanker"

    def run(self, ctx: QueryContext, cands: list[pp.Candidate]) -> StageResult:
        index = ctx.bundle.embedding_index(self.params.get("space", "siglip2"))
        qv = ctx.query_vector(index.space_id)
        for cand in cands:
            wid = cand.window_id or ctx.bundle.window_for_interval(cand)
            self.score(cand, float(index.vector(wid) @ qv) if wid else -1.0, pp.COSINE)
        return StageResult(ctx.sort(cands, self.order_by()))

    def order_by(self):
        spec = self.params.get("order_by", "self:desc")
        keys = []
        for part in spec.split(","):
            sid, direction = part.split(":")
            keys.append((self.stage_id if sid == "self" else sid, direction == "desc"))
        return keys


class PredicateRanker(Stage):
    """Deterministic verification. Resolve the program's primary anchor within seed ± slack, evaluate every clause
    in Kleene logic over the program's declared context, write a TRUTH_ORDINAL. Filters ONLY on a definitively FALSE
    required clause; UNKNOWN keeps the candidate and is listed with its reason."""
    impl = "PredicateRanker"
    needs_program = True
    filters = True
    on_program_failure = "SKIP_MARK_UNVERIFIED"

    def run(self, ctx: QueryContext, cands: list[pp.Candidate]) -> StageResult:
        prog = ctx.require_program(self)
        ex = ctx.bundle.executor
        slack = int(float(self.params.get("slack_s", 4)) * NS)
        evals: dict[str, dict] = {}
        kept, filtered = [], []
        allowed = set(ctx.program_recordings(prog))
        for cand in cands:
            if cand.recording_id not in allowed:
                filtered.append(pp.FilterRecord(candidate_id=cand.candidate_id, stage_id=self.stage_id,
                                                reason=pp.REQUIRED_CLAUSE_FALSE, detail="outside program scope"))
                continue
            if cand.recording_id not in evals:
                evals[cand.recording_id] = ex.evals(prog, cand.recording_id)
            if cand.kind == pp.INTERVAL and cand.anchors:  # already bound by ProgramCandidates
                self.score(cand, cand.scores[next(iter(cand.scores))].value, pp.TRUTH_ORDINAL)
                kept.append(cand)
                continue
            m, verdict = ex.best_near(prog, cand.recording_id, (cand.seed.start_ns, cand.seed.end_ns), slack,
                                      evals[cand.recording_id])
            if m is None:
                ctx.attach_clause(cand, verdict, self.stage_id)
                if verdict.truth == c.TRUTH_FALSE:
                    filtered.append(pp.FilterRecord(candidate_id=cand.candidate_id, stage_id=self.stage_id,
                                                    reason=pp.REQUIRED_CLAUSE_FALSE,
                                                    detail=f"{verdict.clause_id}: no instance near the candidate"))
                    self.score(cand, 0.0, pp.TRUTH_ORDINAL)
                    continue
                self.score(cand, 1.0, pp.TRUTH_ORDINAL)
                kept.append(cand)
                continue
            ctx.attach_match(cand, m, self.stage_id)
            if m.ordinal <= 0:
                bad = next(cl for cl in m.clauses if cl.required and cl.truth == c.TRUTH_FALSE)
                filtered.append(pp.FilterRecord(candidate_id=cand.candidate_id, stage_id=self.stage_id,
                                                reason=pp.REQUIRED_CLAUSE_FALSE, detail=f"{bad.clause_id} is FALSE"))
                self.score(cand, 0.0, pp.TRUTH_ORDINAL)
                continue
            self.score(cand, m.ordinal, pp.TRUTH_ORDINAL)
            kept.append(cand)
        return StageResult(ctx.sort(kept, [(self.stage_id, True)]), filtered, verified=True)


class Truncate(Stage):
    impl = "Truncate"
    truncates = True

    def run(self, ctx: QueryContext, cands: list[pp.Candidate]) -> StageResult:
        return StageResult(cands[: int(self.params.get("k", ctx.k))])


REGISTRY = {cls.impl: cls for cls in (TagCandidates, EmbeddingCandidates, ProgramCandidates, FixedCandidates,
                                      BM25Ranker, SimilarityRanker, PredicateRanker, Truncate)}
