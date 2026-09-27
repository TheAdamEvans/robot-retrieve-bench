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
from alloy_index.annotate.store import USES, key_of, load_labels, shard_path, use_for

JUDGE = os.environ.get("SCANDQ_JUDGE", "claude-opus-5-5")
TRUTH_FIELDS = ["stationary_group", "doorway_traversal", "vehicle_present", "vehicle_interaction", "bicycle",
                "indoor", "turn_visible"]
CONTEXT_SLACK_S = 20.0  # judgments may cite evidence from the intent's context around the window


def add_parser(sp) -> None:
    p = sp.add_parser("label", help="put/get validated labels (attributes | judgment | episode)")
    p.add_argument("op", choices=["put", "get"])
    p.add_argument("kind", choices=["attributes", "judgment", "episode"])
    p.add_argument("--json", help="put: a JSON object, or @path to a file holding one object or a JSON list")
    p.add_argument("--rec")
    p.add_argument("--intent")


def shard(kind: str, use: str) -> Path:
    p = shard_path(use, kind, cli.CAMPAIGN, cli.JOB, cli.LABELS)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def opened_refs() -> dict[str, bool]:
    """mid → opened natively at least once, for this job."""
    seen: dict[str, bool] = {}
    p = cli.audit_path()
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


VERDICTS = ("TRUE", "FALSE", "UNKNOWN")


def challenge_query(intent: str) -> dict | None:
    """The episode-judging spec for a challenge intent (benchmark/challenges/*/queries_*.jsonl)."""
    for f in sorted((cli.SCAND_ROOT / "benchmark" / "challenges").glob("*/queries_*.jsonl")):
        for line in f.read_text().splitlines():
            q = json.loads(line) if line.strip() else {}
            if q.get("intentGroupId") == intent:
                return q
    return None


def validate_episode(d: dict) -> tuple[dict, list[str]]:
    """One intent judged against one episode: a verdict per required clause, anchors, a grade and refs."""
    errors: list[str] = []
    q = challenge_query(d.get("intent_group_id", ""))
    if q is None:
        return d, [f"unknown challenge intent {d.get('intent_group_id')!r}"]
    rec = d.get("recording_id", "")
    if rec not in q["scope"]["recordingIds"]:
        errors.append(f"recording {rec!r} is outside the intent's scope {q['scope']['recordingIds']}")
    try:
        t0, t1 = float(d["start_s"]), float(d["end_s"])
    except (KeyError, TypeError, ValueError):
        return d, errors + ["start_s and end_s (recording-relative seconds) are required"]
    if not 0 <= t0 < t1:
        errors.append("need 0 <= start_s < end_s")
    if not str(d.get("episode_id", "")).strip():
        errors.append("episode_id missing")
    clauses = d.get("clauses", [])
    want = q["requiredClauses"]
    if len(clauses) != len(want):
        errors.append(f"give exactly {len(want)} clause verdicts, in order: {want}")
    mids = []
    for i, c in enumerate(clauses):
        if c.get("verdict") not in VERDICTS:
            errors.append(f"clauses[{i}].verdict must be one of {VERDICTS}")
        if not str(c.get("evidence", "")).strip():
            errors.append(f"clauses[{i}].evidence is empty")
        mids += c.get("refs", [])
        c["clause"] = want[i] if i < len(want) else c.get("clause", "")
    grade = d.get("grade")
    verdicts = [c.get("verdict") for c in clauses]
    if grade not in (0, 1, 2):
        errors.append("grade must be 2 (complete match), 1 (relevant, incomplete) or 0 (contradicted / off-topic)")
    elif grade == 2 and any(v != "TRUE" for v in verdicts):
        errors.append("grade 2 needs every clause TRUE")
    elif grade == 1 and "FALSE" in verdicts and not d.get("grade_note"):
        errors.append("grade 1 with a FALSE clause is a near miss (grade 0) unless grade_note explains why")
    for a_ in d.get("anchors", []):
        if not (t0 - CONTEXT_SLACK_S <= float(a_.get("t_s", -1e9)) <= t1 + CONTEXT_SLACK_S) or not a_.get("name"):
            errors.append(f"anchor {a_} needs a name and a t_s near the episode")
    refs, native = resolve_refs(mids + d.pop("refs", []), (rec, t0, t1), CONTEXT_SLACK_S, errors)
    if not refs:
        errors.append("no refs cited")
    if "TRUE" in verdicts and not native:
        errors.append("a TRUE clause needs >= 1 ref opened at native resolution")
    for c in clauses:
        c["refs"] = [str(x) for x in c.get("refs", [])]
    out = {"intentGroupId": q["intentGroupId"], "querySet": q["querySet"], "episodeId": d["episode_id"],
           "recordingId": rec, "startS": t0, "endS": t1, "grade": grade, "clauses": clauses,
           "anchors": d.get("anchors", []), "comparisonRole": d.get("comparison_role", ""),
           "gradeNote": d.get("grade_note", ""), "coverage": d.get("coverage", ""),
           "windowIds": d.get("window_ids", []), "chunkId": d.get("chunk_id", ""), "judge": JUDGE, "jobId": cli.JOB,
           "refs": [json_format.MessageToDict(r) for r in refs]}
    return out, errors


def cmd_label(a) -> None:
    if a.op == "get":
        conflicts: list = []
        latest = load_labels(a.kind, USES, cli.LABELS, conflicts=conflicts)
        for v in latest.values():  # show refs as MessageId strings, as submitted
            v["refs"] = [f'{x["recordingId"]}{x["topic"]}#{x.get("topicOrdinal", 0)}' for x in v.get("refs", [])]
        rows = [v for v in latest.values()
                if (not a.rec or v.get("recordingId", v.get("windowId", "")).startswith(a.rec))
                and (not a.intent or v.get("intentGroupId") == a.intent)]
        out = {"kind": a.kind, "count": len(rows), "labels": rows}
        if conflicts:
            out["unresolved_conflicts"] = conflicts[:20]
            out["note"] = ("equal-priority campaigns disagree on these keys; training and evaluation refuse to load them "
                           "until one campaign.json gets a higher priority")
        print(json.dumps(out, indent=1))
        return
    if not a.json:
        raise SystemExit("put needs --json '{...}' or --json @file.json")
    raw = Path(a.json[1:]).read_text() if a.json.startswith("@") else a.json
    items = json.loads(raw)
    items = items if isinstance(items, list) else [items]
    accepted, rejected = [], []
    validate = {"attributes": validate_attributes, "judgment": validate_judgment, "episode": validate_episode}[a.kind]
    for item in items:
        try:
            msg, errors = validate(json.loads(json.dumps(item)))
        except json_format.ParseError as e:
            msg, errors = None, [f"shape: {e}"]
        ident = item.get("segment_id") or item.get("window_id") or item.get("episode_id")
        if errors:
            rejected.append({"id": ident, "errors": errors})
            continue
        accepted.append(msg if isinstance(msg, dict) else json_format.MessageToDict(msg))
    shards = set()
    for r in accepted:  # each record goes to the folder of its use: train / eval / agreement
        r["campaign"] = cli.CAMPAIGN
        path = shard(a.kind, use_for(a.kind, r, cli.CAMPAIGN, cli.LABELS))
        existing = {}
        if path.exists():
            for line in path.read_text().splitlines():
                old = json.loads(line)
                existing[key_of(a.kind, old)] = old
        existing[key_of(a.kind, r)] = r
        path.write_text("".join(json.dumps(x) + "\n" for x in existing.values()))
        shards.add(str(path))
    cli.emit("label_put", {"kind": a.kind, "n": len(items)},
             {"accepted": len(accepted), "rejected": rejected, "shards": sorted(shards)})
