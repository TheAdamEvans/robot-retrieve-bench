"""QueryProgram execution in Kleene logic, over a whole recording or around a candidate seed.

One executor serves both uses:
  * ProgramCandidates: `matches(program, rec)` enumerates every binding of the primary event over the recording;
  * PredicateRanker:  `best_near(program, rec, seed, slack)` binds the primary event near a candidate's seed.

A clause is FALSE only when its feature is indexed, applicable to the robot, and covers the searched window with no
satisfying instance. Not indexed / not applicable / outside coverage / calibration band → UNKNOWN with a reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..catalog.features import FeatureStore, Series
from ..catalog.registry import REGISTRY
from ..gen.alloy.v1 import common_pb2 as c
from ..gen.alloy.v1 import query_pb2 as q

T, F, U = c.TRUTH_TRUE, c.TRUTH_FALSE, c.TRUTH_UNKNOWN
NS = 1_000_000_000
BASIS_RANK = {"SPATIAL_NONE": 0, "RECORDED_TF": 1, "NOMINAL": 2, "ESTIMATED": 3}
EXTREMUM_HALF_S = 1.0
REL_CHANGE_FLOOR = 0.25   # relative changes are only measured from values >= 25% of the feature's max


@dataclass
class Instance:
    start: int
    end: int
    ext: int
    truth: int                   # T or U
    value: float | None = None
    track_id: int | None = None
    lo: int | None = None        # anchor uncertainty band (onsets)
    hi: int | None = None

    def at(self, point: int) -> int:
        return {q.START: self.start, q.END: self.end, q.EXTREMUM_POINT: self.ext}.get(point, self.start)


@dataclass
class EventEval:
    instances: list[Instance]
    unknown_reason: int | None = None      # event can never be FALSE here (not indexed / not applicable)
    t0: int = 0
    t1: int = 0
    exhaustive: bool = True
    series: Series | None = None


@dataclass
class Clause:
    clause_id: str
    doc: str
    required: bool
    truth: int
    reason: int = c.UNKNOWN_REASON_UNSPECIFIED
    value: float | None = None
    unit: str = ""


@dataclass
class Match:
    recording_id: str
    bindings: dict[str, Instance | None]
    clauses: list[Clause]
    ordinal: float
    definite: bool
    spatial_basis: str
    exhaustive: bool
    anchors: list[tuple[str, int, int | None, int | None]] = field(default_factory=list)

    @property
    def interval(self) -> tuple[int, int]:
        ts = [x for b in self.bindings.values() if b for x in (b.start, b.end)]
        return min(ts), max(ts)

    @property
    def primary_time(self) -> int:
        return self.anchors[0][1] if self.anchors else self.interval[0]


# ---------- per-event instance computation ----------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _cmp(y: np.ndarray, op: int, x: float) -> np.ndarray:
    return {q.GTE: y >= x, q.GT: y > x, q.LTE: y <= x, q.LT: y < x, q.EQ: np.isclose(y, x)}[op]


def _period(t: np.ndarray) -> int:
    return int(np.median(np.diff(t))) if len(t) > 1 else 0


def threshold_instances(s: Series, ev: q.EventSpec) -> list[Instance]:
    t, y = s.t_ns, s.value
    x = ev.threshold.value
    op = ev.comparator
    if s.lo is not None and s.hi is not None:
        lo, hi = s.lo.astype(float), s.hi.astype(float)
        definite = _cmp(lo, op, x) if op in (q.GTE, q.GT) else _cmp(hi, op, x) if op in (q.LTE, q.LT) else _cmp(y, op, x)
        possible = _cmp(hi, op, x) if op in (q.GTE, q.GT) else _cmp(lo, op, x) if op in (q.LTE, q.LT) else definite
    else:
        definite = possible = _cmp(y, op, x) & np.isfinite(y)
    min_ns = int(ev.min_duration.value * NS) if ev.HasField("min_duration") else 0
    per = _period(t)
    out: list[Instance] = []
    for a, b in _runs(possible):
        sub = definite[a:b + 1]
        d_runs = [(a + i, a + j) for i, j in _runs(sub) if t[a + j] - t[a + i] + per >= min_ns]
        for i, j in d_runs:
            k = i + int(np.argmax(y[i:j + 1]) if op in (q.GTE, q.GT, q.EQ) else np.argmin(y[i:j + 1]))
            out.append(Instance(int(t[i]), int(t[j]), int(t[k]), T, float(y[k])))
        if not d_runs and t[b] - t[a] + per >= min_ns:
            k = a + int(np.argmax(y[a:b + 1]) if op in (q.GTE, q.GT, q.EQ) else np.argmin(y[a:b + 1]))
            out.append(Instance(int(t[a]), int(t[b]), int(t[k]), U, float(y[k])))
    return out


def change_instances(s: Series, ev: q.EventSpec) -> list[Instance]:
    t, y = s.t_ns, s.value
    rel = ev.change.unit == q.RATIO
    thr = ev.change.value
    direction = ev.direction if ev.HasField("direction") else q.DIRECTION_UNSPECIFIED
    w = int(ev.within.value * NS)
    j_end = np.searchsorted(t, t + w, side="right")
    floor = REL_CHANGE_FLOOR * np.nanmax(np.abs(y)) if rel else 0.0
    qual, delta, ext_idx = np.zeros(len(y), bool), np.zeros(len(y)), np.zeros(len(y), int)
    for i in range(len(y)):
        seg = y[i + 1:j_end[i]]
        if not len(seg) or not np.isfinite(y[i]):
            continue
        kd, ku = int(np.nanargmin(seg)), int(np.nanargmax(seg))
        down, up = y[i] - seg[kd], seg[ku] - y[i]
        if direction == q.DOWN or (direction == q.DIRECTION_UNSPECIFIED and down >= up):
            d, k = down, kd
        else:
            d, k = up, ku
        if rel:
            if abs(y[i]) < floor:
                continue
            d = d / abs(y[i])
        if d >= thr:
            qual[i], delta[i], ext_idx[i] = True, d, i + 1 + k
    out = []
    i = 0
    while i < len(y):
        if not qual[i]:
            i += 1
            continue
        j, reach = i, ext_idx[i]
        while j + 1 < len(y) and qual[j + 1] and j + 1 <= reach:
            j += 1
            reach = max(reach, ext_idx[j])
        grp = np.arange(i, j + 1)
        # episode start = the peak (for a drop) / trough (for a rise) among qualifying starts
        pick = grp[np.argmax(y[grp])] if direction == q.DOWN else grp[np.argmin(y[grp])] if direction == q.UP \
            else grp[np.argmax(delta[grp])]
        k = ext_idx[pick]
        out.append(Instance(int(t[pick]), int(t[k]), int(t[k]), T, float(delta[pick])))
        i = max(j + 1, k)
    return out


def onset_instances(s: Series, ev: q.EventSpec, raw: np.ndarray | None) -> list[Instance]:
    t, y = s.t_ns, s.value
    stop = y < ev.from_below.value
    min_ns, sus_ns = int(ev.min_duration.value * NS), int(ev.sustain.value * NS)
    yr = raw if raw is not None else y
    out = []
    for a, b in _runs(stop):
        if t[b] - t[a] < min_ns:
            continue
        # refine on the raw signal: first sample after the stop began that stays above threshold for `sustain`
        k = None
        for m in range(a, len(t)):
            if yr[m] <= ev.threshold.value:
                continue
            end = np.searchsorted(t, t[m] + sus_ns, side="right")
            if end >= len(t):
                break
            if np.all(yr[m:end] > ev.threshold.value) and t[m] - t[a] >= min_ns:
                k = m
                break
        if k is None:
            continue
        lo = int(t[k - 1]) if k > 0 else int(t[k])
        out.append(Instance(int(t[k]), int(t[k]), int(t[k]), T, float(yr[k]), lo=lo, hi=int(t[k])))
    # de-duplicate onsets reached from overlapping stop runs
    uniq: dict[int, Instance] = {}
    for inst in out:
        uniq.setdefault(inst.start, inst)
    return sorted(uniq.values(), key=lambda z: z.start)


def extremum_instances(s: Series, ev: q.EventSpec) -> list[Instance]:
    t, y = s.t_ns, s.value
    h = int(EXTREMUM_HALF_S * NS)
    lo = np.searchsorted(t, t - h)
    hi = np.searchsorted(t, t + h, side="right")
    out = []
    for i in range(len(y)):
        seg = y[lo[i]:hi[i]]
        if not np.isfinite(y[i]) or not len(seg):
            continue
        if (ev.direction == q.DOWN and y[i] <= np.nanmin(seg)) or (ev.direction == q.UP and y[i] >= np.nanmax(seg)):
            if out and t[i] - out[-1].ext < h:
                continue
            out.append(Instance(int(t[i]), int(t[i]), int(t[i]), T, float(y[i])))
    return out


class Executor:
    def __init__(self, store: FeatureStore, recordings: dict[str, "object"], robots: dict[str, str]):
        self.store = store
        self.recordings = recordings    # id → Recording
        self.robots = robots            # id → robot

    def _series(self, name: str, rec_id: str) -> Series | None:
        return self.store.series(name, rec_id)

    def event_eval(self, ev: q.EventSpec, rec_id: str) -> EventEval:
        f = REGISTRY[ev.feature]
        rec = self.recordings[rec_id]
        if self.robots[rec_id] not in f.robots:
            return EventEval([], c.NOT_APPLICABLE, rec.start_ns, rec.end_ns)
        if not f.indexed:
            return EventEval([], c.NOT_INDEXED, rec.start_ns, rec.end_ns, exhaustive=False)
        if ev.kind in (q.TRACK_APPEAR, q.TRACK_DISAPPEAR):
            tr = self.store.tracks(ev.feature, rec_id)
            if tr is None:
                return EventEval([], c.NOT_INDEXED, rec.start_ns, rec.end_ns, exhaustive=False)
            insts = [Instance(int(a), int(a), int(a), T, float(n), track_id=int(i))
                     for i, a, n in zip(tr.track_id, tr.first_ns, tr.n_obs)] if ev.kind == q.TRACK_APPEAR else []
            return EventEval(sorted(insts, key=lambda z: z.start), None, rec.start_ns, rec.end_ns)
        s = self._series(ev.feature, rec_id)
        if s is None or not len(s.t_ns):
            return EventEval([], c.NOT_INDEXED, rec.start_ns, rec.end_ns, exhaustive=False)
        ee = EventEval([], None, int(s.t_ns[0]), int(s.t_ns[-1]), s.exhaustive, s)
        if ev.kind == q.THRESHOLD:
            ee.instances = threshold_instances(s, ev)
        elif ev.kind == q.CHANGE:
            ee.instances = change_instances(s, ev)
        elif ev.kind == q.ONSET:
            raw = self.store.series_column(f.provider, rec_id, "speed_raw_mps") if ev.feature == "speed_mps" else None
            ee.instances = onset_instances(s, ev, raw)
        elif ev.kind == q.EXTREMUM:
            ee.instances = extremum_instances(s, ev)
        return ee   # RETURN_TO is dependent: computed during binding

    # ---------- binding ----------

    def _dependent(self, ev: q.EventSpec, ee: EventEval, bound: dict[str, Instance | None], search_from: int,
                   search_to: int, rec_id: str) -> tuple[list[Instance], int | None]:
        ref = bound.get(ev.reference.event)
        if ref is None:
            return [], c.NO_BINDING
        if ee.unknown_reason is not None:
            return [], ee.unknown_reason
        if ev.kind == q.TRACK_DISAPPEAR:
            tr = self.store.tracks(ev.feature, rec_id)
            if ref.track_id is None or tr is None:
                return [], c.NO_BINDING
            k = np.where(tr.track_id == ref.track_id)[0]
            if not len(k):
                return [], c.NO_BINDING
            last = int(tr.last_ns[k[0]])
            return [Instance(last, last, last, T)], None
        s = ee.series  # RETURN_TO
        t, y = s.t_ns, s.value
        i_ref = max(int(np.searchsorted(t, ref.at(ev.reference.point), side="right")) - 1, 0)
        target = ev.fraction * y[i_ref]
        a, b = int(np.searchsorted(t, search_from)), int(np.searchsorted(t, search_to, side="right"))
        hit = np.where(y[a:b] >= target)[0]
        if len(hit):
            k = a + int(hit[0])
            return [Instance(int(t[k]), int(t[k]), int(t[k]), T, float(y[k]))], None
        return [], None

    @staticmethod
    def _window(rel: q.Relation, p: int, parent: Instance, ctx_after: int) -> tuple[int, int]:
        mn = int(rel.min_gap.value * NS) if rel.HasField("min_gap") else 0
        mx = int(rel.max_gap.value * NS) if rel.HasField("max_gap") else ctx_after
        if rel.kind == q.AFTER:
            return p + mn, p + mx
        if rel.kind == q.BEFORE:
            return p - mx, p - mn
        if rel.kind == q.WITHIN:
            return p - mx, p + mx
        return parent.start, parent.end  # DURING

    def bind(self, program: q.QueryProgram, rec_id: str, primary: Instance, evals: dict[str, EventEval]) -> Match:
        events = {e.name: e for e in program.events}
        ctx_after = int(program.context_after.value * NS)
        rec = self.recordings[rec_id]
        bound: dict[str, Instance | None] = {program.primary_event: primary}
        pe = events[program.primary_event]
        clauses = [Clause(pe.name, pe.doc, pe.required or True, primary.truth,
                          c.CALIBRATION_UNCERTAIN if primary.truth == U else c.UNKNOWN_REASON_UNSPECIFIED,
                          primary.value, REGISTRY[pe.feature].unit)]
        order = self._bfs(program)
        for rel in order:
            ev = events[rel.child.event]
            ee = evals[ev.name]
            parent = bound.get(rel.parent.event)
            required = ev.required or rel.required
            doc = " / ".join(x for x in (ev.doc, rel.doc) if x)
            if parent is None:
                bound[ev.name] = None
                clauses.append(Clause(ev.name, doc, required, U, c.NO_BINDING))
                continue
            p = parent.at(rel.parent.point)
            lo, hi = self._window(rel, p, parent, ctx_after)
            if ev.kind in (q.RETURN_TO, q.TRACK_DISAPPEAR):
                cands, why = self._dependent(ev, ee, bound, lo, hi, rec_id)
                if why is not None:
                    bound[ev.name] = None
                    clauses.append(Clause(ev.name, doc, required, U, why))
                    continue
            else:
                if ee.unknown_reason is not None:
                    bound[ev.name] = None
                    clauses.append(Clause(ev.name, doc, required, U, ee.unknown_reason))
                    continue
                cands = ee.instances
            ok = [x for x in cands if lo <= x.at(rel.child.point) <= hi]
            if ok:
                if rel.kind == q.BEFORE:
                    ok.sort(key=lambda x: (x.truth != T, -x.at(rel.child.point)))
                elif rel.kind == q.WITHIN:
                    ok.sort(key=lambda x: (x.truth != T, abs(x.at(rel.child.point) - p)))
                else:
                    ok.sort(key=lambda x: (x.truth != T, x.at(rel.child.point)))
                best = ok[0]
                bound[ev.name] = best
                clauses.append(Clause(ev.name, doc, required, best.truth,
                                      c.CALIBRATION_UNCERTAIN if best.truth == U else c.UNKNOWN_REASON_UNSPECIFIED,
                                      best.value, REGISTRY[ev.feature].unit))
            else:
                bound[ev.name] = None
                covered = ee.t0 <= max(lo, rec.start_ns) and min(hi, rec.end_ns) <= ee.t1 and hi <= rec.end_ns
                clauses.append(Clause(ev.name, doc, required, F if covered else U,
                                      c.UNKNOWN_REASON_UNSPECIFIED if covered else c.OUTSIDE_COVERAGE))
        for i, text in enumerate(program.unexpressible):
            clauses.append(Clause(f"unexpressible[{i}]", text, True, U, c.UNEXPRESSIBLE))
        req = [cl for cl in clauses if cl.required]
        n_true = sum(cl.truth == T for cl in req)
        if any(cl.truth == F for cl in req):
            ordinal, definite = 0.0, False
        elif n_true == len(req):
            ordinal, definite = 2.0, True
        else:
            ordinal, definite = 1.0 + 0.99 * n_true / max(len(req), 1), False
        basis = max((REGISTRY[events[n].feature].spatial_basis for n, b in bound.items() if b is not None),
                    key=lambda z: BASIS_RANK[z], default="SPATIAL_NONE")
        exhaustive = all(evals[n].exhaustive and evals[n].unknown_reason is None for n in events) and \
            not program.unexpressible
        m = Match(rec_id, bound, clauses, ordinal, definite, basis, exhaustive)
        refs = list(program.return_anchors) or [q.AnchorRef(event=program.primary_event, point=q.START)]
        for a in refs:
            b = bound.get(a.event)
            if b is not None:
                m.anchors.append((f"{a.event}.{q.AnchorPoint.Name(a.point)}", b.at(a.point), b.lo, b.hi))
        return m

    @staticmethod
    def _bfs(program: q.QueryProgram) -> list[q.Relation]:
        out, frontier = [], [program.primary_event]
        rels = list(program.relations)
        while frontier:
            nxt = []
            for name in frontier:
                for r in rels:
                    if r.parent.event == name:
                        out.append(r)
                        nxt.append(r.child.event)
            frontier = nxt
        return out

    def evals(self, program: q.QueryProgram, rec_id: str) -> dict[str, EventEval]:
        return {e.name: self.event_eval(e, rec_id) for e in program.events}

    def matches(self, program: q.QueryProgram, rec_id: str) -> tuple[list[Match], EventEval]:
        ev = self.evals(program, rec_id)
        pe = ev[program.primary_event]
        return [self.bind(program, rec_id, inst, ev) for inst in pe.instances], pe

    def best_near(self, program: q.QueryProgram, rec_id: str, seed: tuple[int, int], slack_ns: int,
                  evals: dict[str, EventEval] | None = None) -> tuple[Match | None, Clause | None]:
        """Bind the primary event near `seed`. Returns (best match, or None with the primary clause's verdict)."""
        ev = evals or self.evals(program, rec_id)
        pe_spec = next(e for e in program.events if e.name == program.primary_event)
        pe = ev[program.primary_event]
        lo, hi = seed[0] - slack_ns, seed[1] + slack_ns
        if pe.unknown_reason is not None:
            return None, Clause(pe_spec.name, pe_spec.doc, True, U, pe.unknown_reason)
        near = [i for i in pe.instances if i.start <= hi and i.end >= lo]
        if not near:
            covered = pe.t0 <= lo and hi <= pe.t1
            return None, Clause(pe_spec.name, pe_spec.doc, True, F if covered else U,
                                c.UNKNOWN_REASON_UNSPECIFIED if covered else c.OUTSIDE_COVERAGE)
        ms = [self.bind(program, rec_id, i, ev) for i in near]
        ms.sort(key=lambda m: (-m.ordinal, abs(m.primary_time - seed[1])))
        return ms[0], None
