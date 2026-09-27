"""`scandq label put|get` — validated label records, sharded per job so parallel jobs never share a file.

Input is proto-JSON (enum names as strings) with refs given as MessageId strings (`Rec/topic#ordinal`).
Validation, beyond the proto shape:
  * every Truth / Bucket field is set (never *_UNSPECIFIED; use TRUTH_UNKNOWN / UNSURE instead);
  * every cited ref exists, lies near the labelled span, and was opened by THIS job (audit log);
  * a positive semantic claim (a TRUE attribute, persons >= 1, or grade >= 1) cites at least one ref the job
    opened at native resolution.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from google.protobuf import json_format

from alloy_server.catalog.windows import WINDOW_S, parse_window_id, window_span_s
from alloy_server.gen.alloy.v1 import common_pb2, eval_pb2
from alloy_server.timeline.store import parse_mid
from alloy_index.scandq import cli
from alloy_index.annotate.store import key_of, load_labels

JUDGE = os.environ.get("SCANDQ_JUDGE", "claude-opus-5-5")
TRUTH_FIELDS = ["stationary_group", "doorway_traversal", "vehicle_present", "vehicle_interaction", "bicycle",
                "indoor", "turn_visible"]
CONTEXT_SLACK_S = 20.0  # judgments may cite evidence from the intent's context around the window


def add_parser(sp) -> None:
    p = sp.add_parser("label", help="put/get validated labels (attributes | judgment)")
    p.add_argument("op", choices=["put", "get"])
    p.add_argument("kind", choices=["attributes", "judgment"])
    p.add_argument("--json", help="put: a JSON object, or @path to a file holding one object or a JSON list")
    p.add_argument("--rec")
    p.add_argument("--intent")


def shard(kind: str) -> Path:
    d = cli.ANN / "labels" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{cli.JOB}.jsonl"


def opened_refs() -> dict[str, bool]:
    """mid → opened natively at least once, for this job."""
    seen: dict[str, bool] = {}
    p = cli.ANN / "audit.jsonl"
    if not p.exists():
        return seen
    for line in p.read_text().splitlines():
        e = json.loads(line)
        if e["job"] != cli.JOB:
            continue
        for r in e.get("refs", []):
            seen[r["mid"]] = seen.get(r["mid"], False) or r["native"]
    return seen


def resolve_refs(mids: list[str], span: tuple[str, float, float], slack: float, errors: list[str]):
    rec_id, t0, t1 = span
    opened = opened_refs()
    out, native_any = [], False
    for s in mids:
        try:
            rec, topic, ordinal = parse_mid(s)
            r = cli.rec_(rec)
            i = r.index_of(topic, ordinal)
        except (KeyError, ValueError, SystemExit):
            errors.append(f"ref {s!r} does not resolve to a recorded message")
            continue
        if rec != rec_id:
            errors.append(f"ref {s!r} is from {rec}, not {rec_id}")
        t = r.t_rel(int(r.topics[topic].log_ns[i]))
        if not (t0 - slack <= t <= t1 + slack):
            errors.append(f"ref {s!r} at {t:.2f}s is outside [{t0 - slack:.1f}, {t1 + slack:.1f}]s")
        if s not in opened:
            errors.append(f"ref {s!r} was never opened by job {cli.JOB!r} (see audit log)")
        native_any |= opened.get(s, False)
        out.append(r.message_id(topic, i))
    return out, native_any


def validate_attributes(d: dict) -> tuple[eval_pb2.WindowAttributes, list[str]]:
    errors: list[str] = []
    mids = d.pop("refs", [])
    d.setdefault("judge", JUDGE)
    d["job_id"] = cli.JOB
    msg = json_format.ParseDict(d, eval_pb2.WindowAttributes())
    rec, end = parse_window_id(msg.segment_id)
    if end % WINDOW_S:
        errors.append(f"segment_id {msg.segment_id!r} is not a segment (end second must be a multiple of {WINDOW_S})")
    span = window_span_s(msg.segment_id)
    r = cli.rec_(rec)
    msg.recording_id = rec
    msg.span.start_ns, msg.span.end_ns = r.t_abs(span[1]), r.t_abs(span[2])
    for f in TRUTH_FIELDS:
        if getattr(msg, f) == common_pb2.TRUTH_UNSPECIFIED:
            errors.append(f"{f} unset: use TRUTH_TRUE / TRUTH_FALSE / TRUTH_UNKNOWN")
    if msg.persons_in_corridor == eval_pb2.BUCKET_UNSPECIFIED:
        errors.append("persons_in_corridor unset: use ZERO / ONE_TWO / THREE_FIVE / SIX_PLUS / UNSURE")
    if not msg.caption.strip():
        errors.append("caption is empty")
    refs, native = resolve_refs(mids, span, 1.0, errors)
    positive = any(getattr(msg, f) == common_pb2.TRUTH_TRUE for f in TRUTH_FIELDS) or \
        msg.persons_in_corridor in (eval_pb2.ONE_TWO, eval_pb2.THREE_FIVE, eval_pb2.SIX_PLUS)
    if positive and not native:
        errors.append("positive semantic claims need >= 1 ref opened at native resolution (frame / step n<=4 / "
                      "sync body cams / lidar)")
    if not refs:
        errors.append("no refs cited")
    msg.refs.extend(refs)
    return msg, errors


def validate_judgment(d: dict) -> tuple[eval_pb2.Judgment, list[str]]:
    errors: list[str] = []
    mids = d.pop("refs", [])
    d.setdefault("judge", JUDGE)
    d.setdefault("source", "AGENT_PROVISIONAL")
    d["job_id"] = cli.JOB
    msg = json_format.ParseDict(d, eval_pb2.Judgment())
    if msg.grade not in (-1, 0, 1, 2):
        errors.append("grade must be 0, 1, 2, or -1 (UNJUDGED: unsure)")
    if not msg.intent_group_id:
        errors.append("intent_group_id missing")
    if not msg.rationale.strip():
        errors.append("rationale is empty")
    span = window_span_s(msg.window_id)
    refs, native = resolve_refs(mids, span, CONTEXT_SLACK_S, errors)
    if msg.grade >= 1 and not native:
        errors.append("grade >= 1 needs >= 1 ref opened at native resolution")
    if msg.grade >= 0 and not refs:
        errors.append("no refs cited")
    msg.refs.extend(refs)
    return msg, errors


def cmd_label(a) -> None:
    if a.op == "get":
        latest = load_labels(cli.ANN, a.kind)
        for v in latest.values():  # show refs as MessageId strings, as submitted
            v["refs"] = [f'{x["recordingId"]}{x["topic"]}#{x.get("topicOrdinal", 0)}' for x in v.get("refs", [])]
        rows = [v for v in latest.values()
                if (not a.rec or v.get("recordingId", v.get("windowId", "")).startswith(a.rec))
                and (not a.intent or v.get("intentGroupId") == a.intent)]
        print(json.dumps({"kind": a.kind, "count": len(rows), "labels": rows}, indent=1))
        return
    if not a.json:
        raise SystemExit("put needs --json '{...}' or --json @file.json")
    raw = Path(a.json[1:]).read_text() if a.json.startswith("@") else a.json
    items = json.loads(raw)
    items = items if isinstance(items, list) else [items]
    accepted, rejected = [], []
    validate = validate_attributes if a.kind == "attributes" else validate_judgment
    for item in items:
        try:
            msg, errors = validate(dict(item))
        except json_format.ParseError as e:
            msg, errors = None, [f"shape: {e}"]
        ident = item.get("segment_id") or item.get("window_id")
        if errors:
            rejected.append({"id": ident, "errors": errors})
            continue
        accepted.append(json_format.MessageToDict(msg))
    if accepted:
        path = shard(a.kind)
        existing = {}
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                existing[key_of(a.kind, r)] = r
        for r in accepted:
            existing[key_of(a.kind, r)] = r
        path.write_text("".join(json.dumps(r) + "\n" for r in existing.values()))
    cli.emit("label_put", {"kind": a.kind, "n": len(items)},
             {"accepted": len(accepted), "rejected": rejected, "shard": str(shard(a.kind))})
