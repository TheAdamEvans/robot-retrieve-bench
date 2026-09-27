"""QueryContext: one per request. Immutable except the lazy program slot and the memoized query vector."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..catalog.registry import REGISTRY
from ..gen.alloy.v1 import answer_pb2 as a
from ..gen.alloy.v1 import common_pb2 as c
from ..gen.alloy.v1 import pipeline_pb2 as pp
from ..gen.alloy.v1 import query_pb2 as q
from ..io import cost

if TYPE_CHECKING:
    from ..bundle import Bundle


class ProgramUnavailable(Exception):
    pass


class UndeclaredProgramAccess(Exception):
    pass


class QueryContext:
    def __init__(self, bundle: "Bundle", req: a.SearchRequest, generator: Callable | None):
        self.bundle = bundle
        self.utterance = req.utterance
        self.k = req.k or 10
        self.mode = req.mode or pp.SEARCH
        self.window_ids = list(req.window_ids)
        self.availability_config_id = req.availability_config_id or "all"
        scope = set(req.scope.recording_ids) or set(bundle.recordings)
        robots = set(req.scope.robots)
        self.scope_recordings = sorted(r for r in scope if r in bundle.recordings and
                                       (not robots or bundle.robots[r] in robots))
        self._generator = generator
        self.diagnostics = a.ProgramDiagnostics(state=a.NOT_REQUESTED)
        self.program: q.QueryProgram | None = None
        if req.HasField("program"):
            prog = q.QueryProgram()
            prog.CopyFrom(req.program)
            errs = bundle.validate(prog)
            if errs:
                self.diagnostics = a.ProgramDiagnostics(state=a.FAILED, errors=[str(e) for e in errs])
            else:
                self.program = prog
                self.diagnostics = a.ProgramDiagnostics(state=a.PROVIDED, attempts=0)
        self._qv: dict[str, np.ndarray] = {}
        self.generation_report: pp.StageReport | None = None
        self._generation_attempted = False

    # ---- lazy program slot ----
    def require_program(self, stage) -> q.QueryProgram:
        if not stage.needs_program:
            raise UndeclaredProgramAccess(f"{stage.stage_id} did not declare needs_program")
        if self.program is not None:
            return self.program
        if self.diagnostics.state == a.FAILED or self._generation_attempted or self._generator is None:
            if self._generator is None and self.diagnostics.state == a.NOT_REQUESTED:
                self.diagnostics = a.ProgramDiagnostics(state=a.FAILED, errors=["NO_GENERATOR"])
            raise ProgramUnavailable(";".join(self.diagnostics.errors) or "program unavailable")
        self._generation_attempted = True
        t0 = time.perf_counter()
        with cost.scope("program_generation") as s:
            prog, diag = self._generator(self.utterance, self.bundle)
        rep = pp.StageReport(stage_id="program_generation", impl=diag.model or "generator", role=pp.GENERATE,
                             triggered_by=stage.stage_id, outcome=pp.RAN if prog is not None else pp.ABSTAINED,
                             outcome_detail=";".join(diag.errors))
        rep.cost.CopyFrom(self.bundle.cost_report(s, self.scope_recordings, (time.perf_counter() - t0) * 1000))
        self.generation_report = rep
        self.diagnostics = diag
        if prog is None:
            raise ProgramUnavailable(";".join(diag.errors))
        self.program = prog
        return prog

    def program_recordings(self, prog: q.QueryProgram) -> list[str]:
        ids = set(prog.scope.recording_ids) or set(self.scope_recordings)
        robots = set(prog.scope.robots)
        return [r for r in self.scope_recordings if r in ids and (not robots or self.bundle.robots[r] in robots)]

    # ---- query encoding (memoized; cost lands in the calling stage's scope) ----
    def query_vector(self, space_id: str) -> np.ndarray:
        if space_id not in self._qv:
            self._qv[space_id] = self.bundle.encode_query(space_id, self.utterance)
        return self._qv[space_id]

    # ---- append-only annotation of candidates ----
    def attach_clause(self, cand: pp.Candidate, cl, stage_id: str) -> None:
        r = cand.clauses.add(clause_id=cl.clause_id, doc=cl.doc, required=cl.required, truth=cl.truth,
                             reason=cl.reason if cl.truth == c.TRUTH_UNKNOWN else c.UNKNOWN_REASON_UNSPECIFIED,
                             unit=cl.unit)
        if cl.value is not None and np.isfinite(cl.value):
            r.value = float(cl.value)
        cand.lineage.add(stage_id=stage_id, action=f"clause:{cl.clause_id}")

    def attach_match(self, cand: pp.Candidate, m, stage_id: str) -> None:
        for cl in m.clauses:
            self.attach_clause(cand, cl, stage_id)
        for name, t, lo, hi in m.anchors:
            na = cand.anchors.add(name=name, t_ns=int(t))
            if lo is not None:
                na.lo_ns, na.hi_ns = int(lo), int(hi)
        s, e = m.interval
        cand.resolved.start_ns, cand.resolved.end_ns = int(s), int(e)
        cand.completeness = pp.EXHAUSTIVE if m.exhaustive else pp.COMPLETENESS_UNKNOWN
        cand.spatial_basis = c.SpatialBasis.Value(m.spatial_basis)
        prog = self.program
        feats = {e.name: e.feature for e in prog.events} if prog else {}
        rec = self.bundle.recordings[m.recording_id]
        seen = {(x.topic, x.topic_ordinal) for x in cand.evidence}
        for name, inst in m.bindings.items():
            if inst is None or name not in feats:
                continue
            topics = self.bundle.features.source_topics(feats[name], m.recording_id)
            if not topics or topics[0] not in rec.topics:
                continue
            i = rec.topics[topics[0]].last_before(inst.start, inclusive=True)
            if i is None:
                continue
            mid = rec.message_id(topics[0], i)
            if (mid.topic, mid.topic_ordinal) not in seen:
                cand.evidence.append(mid)
                seen.add((mid.topic, mid.topic_ordinal))

    # ---- ordering ----
    @staticmethod
    def sort(cands: list[pp.Candidate], keys: list[tuple[str, bool]]) -> list[pp.Candidate]:
        def key(cd):
            out = []
            for sid, desc in keys:
                v = cd.scores[sid].value if sid in cd.scores else -np.inf
                out.append(-v if desc else v)
            return (*out, cd.candidate_id)
        return sorted(cands, key=key)

    def order_by_selection(self, prog: q.QueryProgram, cands: list[pp.Candidate], stage_id: str) -> list[pp.Candidate]:
        sel = prog.selection
        truth = lambda cd: -cd.scores[stage_id].value
        start = lambda cd: cd.anchors[0].t_ns if cd.anchors else cd.seed.start_ns
        if sel.quantifier == q.FIRST:
            return sorted(cands, key=lambda cd: (truth(cd), self.bundle.recordings[cd.recording_id].start_ns * 0 + start(cd)))
        if sel.quantifier in (q.ARGMIN, q.ARGMAX):
            def val(cd):
                s = self.bundle.features.series(sel.by_feature, cd.recording_id)
                if s is None:
                    return np.inf
                lo, hi = cd.resolved.start_ns or cd.seed.start_ns, cd.resolved.end_ns or cd.seed.end_ns
                m = (s.t_ns >= lo) & (s.t_ns <= hi)
                if not m.any():
                    return np.inf
                v = s.value[m]
                return float(v.min()) if sel.quantifier == q.ARGMIN else -float(v.max())
            return sorted(cands, key=lambda cd: (truth(cd), val(cd), cd.candidate_id))
        return sorted(cands, key=lambda cd: (truth(cd), start(cd), cd.candidate_id))
