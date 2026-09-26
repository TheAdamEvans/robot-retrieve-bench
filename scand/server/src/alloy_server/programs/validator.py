"""Semantic validation of a QueryProgram against the FeatureRegistry, plus unit canonicalisation.

Shape is checked by json_format (unknown fields rejected). This pass checks meaning. Error codes are the execution
failure taxonomy and the repair feedback for the program generator.
"""
from __future__ import annotations

from dataclasses import dataclass

from google.protobuf import json_format

from ..catalog.registry import DIMENSION, REGISTRY, SENSORS, TO_CANONICAL
from ..gen.alloy.v1 import common_pb2, query_pb2 as q

CANONICAL_UNIT = {"count": "DIMENSIONLESS", "ratio": "RATIO", "length": "M", "speed": "MPS", "accel": "MPS2",
                  "angle": "DEG", "angular_rate": "DEG_PER_S", "time": "S"}
TRACK_KINDS = {q.TRACK_APPEAR, q.TRACK_DISAPPEAR}
DEPENDENT_KINDS = {q.RETURN_TO, q.TRACK_DISAPPEAR}


@dataclass
class ValidationError:
    code: str
    where: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code} at {self.where}: {self.detail}"


def _unit_name(u: int) -> str:
    return q.Unit.Name(u)


def canon(qty: q.Quantity, want_dim: str, where: str, errors: list[ValidationError]) -> None:
    """Convert in place to the canonical unit of `want_dim`, or record SEM_UNIT_MISMATCH."""
    if qty.unit == q.UNIT_UNSPECIFIED:
        errors.append(ValidationError("SEM_UNIT_MISSING", where, f"expected a {want_dim} unit"))
        return
    name = _unit_name(qty.unit)
    dim = DIMENSION[name]
    if dim != want_dim:
        errors.append(ValidationError("SEM_UNIT_MISMATCH", where, f"{name} is a {dim}; expected a {want_dim}"))
        return
    qty.value *= TO_CANONICAL.get(name, 1.0)
    qty.unit = q.Unit.Value(CANONICAL_UNIT[dim])


def validate(program: q.QueryProgram, recordings: dict[str, str]) -> list[ValidationError]:
    """Validates and canonicalises `program` in place. recordings: {recording_id: robot}."""
    e: list[ValidationError] = []
    events = {ev.name: ev for ev in program.events}
    if len(events) != len(program.events):
        e.append(ValidationError("SEM_DUPLICATE_EVENT", "events", "event names must be unique"))
    if not program.events:
        e.append(ValidationError("SEM_NO_EVENTS", "events", "a program needs at least one event"))
    if program.primary_event not in events:
        e.append(ValidationError("SEM_PRIMARY_MISSING", "primary_event", f"{program.primary_event!r} is not an event"))

    for rid in program.scope.recording_ids:
        if rid not in recordings:
            e.append(ValidationError("SEM_SCOPE", "scope.recording_ids", f"unknown recording {rid!r}"))
    for rb in program.scope.robots:
        if rb not in SENSORS:
            e.append(ValidationError("SEM_SCOPE", "scope.robots", f"unknown robot {rb!r}"))

    for ev in program.events:
        w = f"events[{ev.name}]"
        f = REGISTRY.get(ev.feature)
        if f is None:
            e.append(ValidationError("SEM_UNKNOWN_FEATURE", w, f"{ev.feature!r} is not in the registry"))
            continue
        k = ev.kind
        if k == q.EVENT_KIND_UNSPECIFIED:
            e.append(ValidationError("SEM_MISSING_PARAM", w, "kind is required"))
            continue
        if (k in TRACK_KINDS) != (f.kind == "track"):
            e.append(ValidationError("SEM_BAD_KIND_FEATURE", w,
                                     f"{q.EventKind.Name(k)} needs a {'track' if k in TRACK_KINDS else 'non-track'} "
                                     f"feature; {ev.feature} is {f.kind}"))
        need = {q.THRESHOLD: ["comparator", "threshold"], q.CHANGE: ["change", "within"],
                q.ONSET: ["from_below", "threshold", "min_duration", "sustain"], q.EXTREMUM: ["direction"],
                q.RETURN_TO: ["fraction", "reference"], q.TRACK_APPEAR: [], q.TRACK_DISAPPEAR: ["reference"]}[k]
        for p in need:
            if not ev.HasField(p):
                e.append(ValidationError("SEM_MISSING_PARAM", w, f"{q.EventKind.Name(k)} needs `{p}`"))
        if ev.HasField("threshold"):
            canon(ev.threshold, f.dimension, f"{w}.threshold", e)
        if ev.HasField("from_below"):
            canon(ev.from_below, f.dimension, f"{w}.from_below", e)
        if ev.HasField("change"):
            if ev.change.unit in (q.RATIO, q.PERCENT):
                canon(ev.change, "ratio", f"{w}.change", e)
            else:
                canon(ev.change, f.dimension, f"{w}.change", e)
        for p in ("within", "min_duration", "sustain"):
            if ev.HasField(p):
                canon(getattr(ev, p), "time", f"{w}.{p}", e)
        if ev.HasField("fraction") and not 0 < ev.fraction <= 2:
            e.append(ValidationError("SEM_RANGE", f"{w}.fraction", "fraction must be in (0, 2]"))
        if ev.HasField("reference") and ev.reference.event not in events:
            e.append(ValidationError("SEM_UNKNOWN_EVENT", f"{w}.reference", f"{ev.reference.event!r} is not an event"))
        if k == q.TRACK_DISAPPEAR and ev.reference.event in events and events[ev.reference.event].kind != q.TRACK_APPEAR:
            e.append(ValidationError("SEM_BAD_REFERENCE", f"{w}.reference", "TRACK_DISAPPEAR must reference a TRACK_APPEAR"))

    # relations: a tree rooted at the primary event; each non-primary event is the child of exactly one relation
    parent_of: dict[str, str] = {}
    for i, rel in enumerate(program.relations):
        w = f"relations[{i}]"
        for side in ("parent", "child"):
            ref = getattr(rel, side)
            if ref.event not in events:
                e.append(ValidationError("SEM_UNKNOWN_EVENT", f"{w}.{side}", f"{ref.event!r} is not an event"))
            if ref.point == q.ANCHOR_POINT_UNSPECIFIED:
                e.append(ValidationError("SEM_MISSING_PARAM", f"{w}.{side}.point", "anchor point is required"))
        if rel.kind == q.RELATION_KIND_UNSPECIFIED:
            e.append(ValidationError("SEM_MISSING_PARAM", w, "relation kind is required"))
        if rel.kind in (q.AFTER, q.BEFORE, q.WITHIN) and not rel.HasField("max_gap"):
            e.append(ValidationError("SEM_MISSING_PARAM", w, f"{q.RelationKind.Name(rel.kind)} needs max_gap"))
        for p in ("min_gap", "max_gap"):
            if rel.HasField(p):
                canon(getattr(rel, p), "time", f"{w}.{p}", e)
        if rel.child.event in parent_of:
            e.append(ValidationError("SEM_NOT_TREE", w, f"{rel.child.event!r} has two parents"))
        parent_of[rel.child.event] = rel.parent.event
    for name in events:
        if name != program.primary_event and name not in parent_of:
            e.append(ValidationError("SEM_NOT_TREE", f"events[{name}]", "not connected to the primary event by a relation"))
    for name in parent_of:  # reach the root without cycles
        seen, cur = set(), name
        while cur in parent_of:
            if cur in seen:
                e.append(ValidationError("SEM_RELATION_CYCLE", f"events[{name}]", "relations form a cycle"))
                break
            seen.add(cur)
            cur = parent_of[cur]
        else:
            if cur != program.primary_event and name in events:
                e.append(ValidationError("SEM_NOT_TREE", f"events[{name}]", "does not descend from the primary event"))
    for ev in program.events:  # a dependent event's reference must already be bound: an ancestor
        if ev.kind in DEPENDENT_KINDS and ev.HasField("reference"):
            anc, cur = set(), ev.name
            while cur in parent_of:
                cur = parent_of[cur]
                anc.add(cur)
            if ev.reference.event not in anc:
                e.append(ValidationError("SEM_BAD_REFERENCE", f"events[{ev.name}].reference",
                                         f"{ev.reference.event!r} must be an ancestor of {ev.name!r} in the relation tree"))

    sel = program.selection
    if sel.quantifier == q.QUANTIFIER_UNSPECIFIED:
        e.append(ValidationError("SEM_MISSING_PARAM", "selection.quantifier", "quantifier is required (ALL if unsure)"))
    if sel.quantifier in (q.ARGMIN, q.ARGMAX):
        if not sel.HasField("by_feature") or sel.by_feature not in REGISTRY:
            e.append(ValidationError("SEM_SELECTION", "selection.by_feature", "ARGMIN/ARGMAX need a registry feature"))
        if not sel.HasField("by_event") or sel.by_event not in events:
            e.append(ValidationError("SEM_SELECTION", "selection.by_event", "ARGMIN/ARGMAX need an event"))
    for i, a in enumerate(program.return_anchors):
        if a.event not in events:
            e.append(ValidationError("SEM_UNKNOWN_EVENT", f"return_anchors[{i}]", f"{a.event!r} is not an event"))
    if program.HasField("receipt"):
        r = program.receipt
        if r.cutoff.event not in events:
            e.append(ValidationError("SEM_UNKNOWN_EVENT", "receipt.cutoff", f"{r.cutoff.event!r} is not an event"))
        if r.boundary == common_pb2.BOUNDARY_UNSPECIFIED:
            e.append(ValidationError("SEM_MISSING_PARAM", "receipt.boundary", "declare STRICT_BEFORE or INCLUSIVE"))
        known = {s for m in SENSORS.values() for s in m}
        for s in r.sensors:
            if s not in known:
                e.append(ValidationError("SEM_UNKNOWN_SENSOR", "receipt.sensors", f"{s!r}; known: {sorted(known)}"))
        canon(r.max_age, "time", "receipt.max_age", e)
    for p in ("context_before", "context_after"):
        if not program.HasField(p):
            e.append(ValidationError("SEM_CONTEXT", p, "declare the evaluation context around the primary anchor"))
        else:
            canon(getattr(program, p), "time", p, e)
    return e


def parse_and_validate(d: dict, recordings: dict[str, str]) -> tuple[q.QueryProgram | None, list[ValidationError]]:
    try:
        prog = json_format.ParseDict(d, q.QueryProgram())
    except json_format.ParseError as ex:
        return None, [ValidationError("PARSE", "program", str(ex))]
    errs = validate(prog, recordings)
    return (prog if not errs else None), errs
