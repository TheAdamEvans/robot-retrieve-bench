"""Strict JSON Schema for QueryProgram, generated from the proto descriptor (the proto stays the single source).

OpenAI strict structured outputs need: every property required, additionalProperties false, optional values as
nullable types. Field names are proto-JSON (lowerCamel); enums exclude *_UNSPECIFIED. Domain enums (feature names,
sensors, recordings) come from the registry and embodiment profiles, so an unknown feature cannot even be emitted.
"""
from __future__ import annotations

from google.protobuf.descriptor import Descriptor, FieldDescriptor as FD

from ..catalog.registry import ALL_SENSORS, REGISTRY
from ..gen.alloy.v1 import query_pb2

SCALAR = {FD.TYPE_DOUBLE: "number", FD.TYPE_FLOAT: "number", FD.TYPE_INT32: "integer", FD.TYPE_INT64: "integer",
          FD.TYPE_UINT32: "integer", FD.TYPE_UINT64: "integer", FD.TYPE_BOOL: "boolean", FD.TYPE_STRING: "string"}


def _domain(msg: str, field: str, recordings: list[str], robots: list[str]) -> list[str] | None:
    if field in ("feature", "by_feature"):
        return sorted(REGISTRY)
    if msg == "ReceiptSpec" and field == "sensors":
        return ALL_SENSORS
    if msg == "Scope" and field == "recording_ids":
        return recordings
    if msg == "Scope" and field == "robots":
        return robots
    return None


def schema(recordings: list[str], robots: list[str]) -> dict:
    defs: dict[str, dict] = {}

    def field_schema(md: Descriptor, f) -> dict:
        dom = _domain(md.name, f.name, recordings, robots)
        if f.type == FD.TYPE_MESSAGE:
            visit(f.message_type)
            s = {"$ref": f"#/$defs/{f.message_type.name}"}
        elif f.type == FD.TYPE_ENUM:
            s = {"type": "string", "enum": [v.name for v in f.enum_type.values if not v.name.endswith("_UNSPECIFIED")]}
        else:
            s = {"type": SCALAR[f.type]}
            if dom:
                s["enum"] = dom
        if f.is_repeated:
            return {"type": "array", "items": s}
        if f.type == FD.TYPE_MESSAGE:  # messages are nullable only where the grammar makes them optional
            return {"anyOf": [s, {"type": "null"}]} if f.name in _NULLABLE_MESSAGES.get(md.name, ()) else s
        if f.has_presence:  # proto3 `optional` scalar / enum
            out = {**s, "type": [s["type"], "null"]}
            if "enum" in out:
                out["enum"] = [*out["enum"], None]
            return out
        return s

    def visit(md: Descriptor) -> None:
        if md.name in defs:
            return
        defs[md.name] = {}
        props = {f.json_name: field_schema(md, f) for f in md.fields}
        defs[md.name] = {"type": "object", "additionalProperties": False, "properties": props,
                         "required": list(props)}

    visit(query_pb2.QueryProgram.DESCRIPTOR)
    root = defs.pop("QueryProgram")
    return {**root, "$defs": defs}


# message-typed fields the model may leave null (proto3 `optional` on messages is expressed via this table)
_NULLABLE_MESSAGES = {
    "EventSpec": ("threshold", "change", "within", "min_duration", "from_below", "sustain", "reference"),
    "Relation": ("min_gap", "max_gap"),
    "QueryProgram": ("receipt",),
}


def strip_nulls(d):
    """Remove nulls (proto-JSON treats absence as unset; ParseDict rejects null for some types)."""
    if isinstance(d, dict):
        return {k: strip_nulls(v) for k, v in d.items() if v is not None}
    if isinstance(d, list):
        return [strip_nulls(v) for v in d]
    return d
