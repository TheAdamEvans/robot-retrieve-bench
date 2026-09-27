"""text → QueryProgram with an OpenAI model under a strict JSON Schema, then semantic validation and one repair turn.

The prompt's static prefix (grammar, registry, embodiments, recordings, few-shots) comes first and is byte-stable,
so provider-side prompt caching applies; the utterance comes last. Programs are cached by
(normalised utterance, registry hash, prompt hash, model) — a cache hit costs zero tokens.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from google.protobuf import json_format

from ..catalog import embodiment as E
from ..catalog.registry import FEATURES, SENSORS, registry_hash
from ..gen.alloy.v1 import answer_pb2 as a
from ..gen.alloy.v1 import query_pb2 as q
from ..io import cost
from .schema import schema, strip_nulls

MODEL = os.environ.get("ALLOY_PROGRAM_MODEL", "gpt-6-luna")
REASONING = os.environ.get("ALLOY_PROGRAM_REASONING", "low")


def _price() -> tuple[float, float, float] | None:
    """USD per 1M tokens (input, cached input, output) from ALLOY_PRICE_USD_PER_MTOK="in,cached,out"; unset → None."""
    v = os.environ.get("ALLOY_PRICE_USD_PER_MTOK")
    return tuple(float(x) for x in v.split(",")) if v else None


GRAMMAR = """You translate a question about archived robot recordings into a QueryProgram (JSON, strict schema).

A program is a set of named EVENTS over registry FEATURES, linked by temporal RELATIONS into a tree rooted at the
PRIMARY event, plus a selection, return anchors, an evaluation context and optionally a causal receipt.

EVENT kinds (each needs the listed params; leave others null):
- THRESHOLD: intervals where `feature comparator threshold` holds, optionally for >= minDuration.
- CHANGE: the feature drops (direction DOWN) / rises (UP) by `change` within `within`; direction null = either way
  (use for turns on heading_deg). START = just before the change, EXTREMUM_POINT = min/max reached, END = same.
  `change` may be absolute (unit of the feature) or relative (PERCENT or RATIO).
- ONSET: feature below `fromBelow` for >= minDuration, then above `threshold` sustained for `sustain`
  (e.g. accelerating from a stop).
- EXTREMUM: local minimum (direction DOWN) or maximum (UP) of the feature.
- RETURN_TO: first time the feature returns to `fraction` x its value at the `reference` anchor (recovery).
- TRACK_APPEAR: a new track (e.g. a person) first seen in a track feature.
- TRACK_DISAPPEAR: last observation of the track bound by `reference` (a TRACK_APPEAR event), in a track feature.
Anchor points: START, END, EXTREMUM_POINT.

RELATIONS bind a child event relative to a parent anchor: AFTER (child in [parent+minGap, parent+maxGap]),
BEFORE (child in [parent-maxGap, parent-minGap]), WITHIN (|child-parent| <= maxGap), DURING (child anchor inside
the parent event's interval). Every non-primary event is the child of exactly one relation; no cycles. RETURN_TO and
TRACK_DISAPPEAR must reference an ANCESTOR event in that tree.

SELECTION: ALL (default); FIRST for first/earliest; ARGMIN / ARGMAX with byFeature + byEvent for
narrowest/smallest/closest or strongest/largest/biggest.
RECEIPT (only when the question asks what evidence was available before a moment): cutoff anchor, sensor names,
maxAge, boundary STRICT_BEFORE ("before", "immediately before") or INCLUSIVE ("at or before").
CONTEXT: contextBefore/contextAfter (durations) around the primary anchor; cover everything the question needs
(e.g. a recovery up to 20 s later).

RULES
1. Encode every requirement of the question as an event or relation, with `doc` quoting the phrase it encodes.
2. Never approximate a requirement with an unrelated feature. If it cannot be expressed with these features and event
   kinds, add a short phrase to `unexpressible` instead. Features marked NOT INDEXED may still be used: the system
   will report them as unknown rather than guess.
3. Set abstainIfInsufficient when the question says to abstain / return insufficient evidence when unsure.
4. Scope: set recordingIds only when the question names a specific run; otherwise leave it empty (all recordings).
5. Always give units. Counts are DIMENSIONLESS. Use the feature's own unit family (e.g. CM is fine for a length).
6. Mark every event and relation required unless the question makes it optional.
"""


def _registry_block() -> str:
    lines = ["FEATURES (name | unit | kind | robots | notes)"]
    for f in FEATURES:
        flag = "" if f.indexed else " NOT INDEXED."
        lines.append(f"- {f.name} | {f.unit} | {f.kind} | {','.join(f.robots)} | {f.doc}{flag}")
    return "\n".join(lines)


def _embodiment_block() -> str:
    lines = ["ROBOTS"]
    for pid, p in E.profiles().items():
        sensors = ", ".join(sorted(SENSORS[pid]))
        lines.append(f"- {pid}: {p.robot}; {p.locomotion}. Receipt sensor names: {sensors}. "
                     f"Does not have: {'; '.join(p.absent)}.")
    return "\n".join(lines)


def _recordings_block(bundle) -> str:
    tags_p = bundle.root / "index" / "tags.json"
    tags = json.loads(tags_p.read_text()) if tags_p.exists() else {}
    lines = ["RECORDINGS (id | robot | duration | description)"]
    for r, rec in bundle.recordings.items():
        lines.append(f"- {r} | {bundle.robots[r]} | {(rec.end_ns - rec.start_ns) / 1e9:.0f} s | {tags.get(r, '')}")
    return "\n".join(lines)


def _fewshot_block(fewshots: list[tuple[str, dict]]) -> str:
    out = ["EXAMPLES"]
    for utt, prog in fewshots:
        out.append(f"Question: {utt}\nProgram: {json.dumps(prog, separators=(',', ':'))}")
    return "\n\n".join(out)


def normalise(utt: str) -> str:
    return re.sub(r"\s+", " ", utt.strip().lower())


class OpenAIProgramGenerator:
    def __init__(self, bundle, fewshots: list[tuple[str, dict]], cache_dir: Path | None = None,
                 model: str = MODEL, reasoning: str = REASONING):
        from openai import OpenAI
        self.client = OpenAI()
        self.model, self.reasoning = model, reasoning
        self.system = "\n\n".join([GRAMMAR, _registry_block(), _embodiment_block(), _recordings_block(bundle),
                                   _fewshot_block(fewshots)])
        self.schema = schema(sorted(bundle.recordings), sorted(E.profiles()))
        self.prompt_hash = hashlib.sha256((self.system + json.dumps(self.schema, sort_keys=True)).encode()).hexdigest()[:12]
        self.fewshot_utterances = {normalise(u) for u, _ in fewshots}
        self.cache_dir = cache_dir
        self._mem: dict[str, dict] = {}

    def _key(self, utt: str) -> str:
        return hashlib.sha256(f"{normalise(utt)}|{registry_hash()}|{self.prompt_hash}|{self.model}|{self.reasoning}"
                              .encode()).hexdigest()[:24]

    def _cache_get(self, key: str) -> dict | None:
        if key in self._mem:
            return self._mem[key]
        p = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if p and p.exists():
            self._mem[key] = json.loads(p.read_text())
            return self._mem[key]
        return None

    def _cache_put(self, key: str, value: dict) -> None:
        self._mem[key] = value
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps(value))

    def _call(self, messages: list[dict]) -> tuple[dict | None, str]:
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, reasoning_effort=self.reasoning,
            response_format={"type": "json_schema", "json_schema": {"name": "QueryProgram", "strict": True,
                                                                    "schema": self.schema}})
        s = cost.current()
        u = r.usage
        if s is not None and u is not None:
            cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            reasoning = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
            s.llm_calls += 1
            s.prompt_tokens += u.prompt_tokens
            s.cached_prompt_tokens += cached
            s.completion_tokens += u.completion_tokens
            s.reasoning_tokens += reasoning
            price = _price()
            if price:
                s.usd_est += ((u.prompt_tokens - cached) * price[0] + cached * price[1] + u.completion_tokens * price[2]) / 1e6
        text = r.choices[0].message.content or ""
        try:
            return json.loads(text), text
        except json.JSONDecodeError:
            return None, text

    def __call__(self, utterance: str, bundle) -> tuple[q.QueryProgram | None, a.ProgramDiagnostics]:
        diag = a.ProgramDiagnostics(state=a.GENERATED, model=f"{self.model}:{self.reasoning}")
        key = self._key(utterance)
        hit = self._cache_get(key)
        if hit is not None:
            diag.cache_hit, diag.attempts = True, 0
            if hit.get("program") is None:
                diag.state = a.FAILED
                diag.errors.extend(hit.get("errors", []))
                return None, diag
            prog = json_format.ParseDict(hit["program"], q.QueryProgram())
            bundle.validate(prog)  # canonicalise units in place
            return prog, diag
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": utterance}]
        errors: list[str] = []
        for attempt in (1, 2):
            diag.attempts = attempt
            raw, text = self._call(messages)
            if raw is None:
                errors = ["PARSE: model output is not JSON"]
            else:
                raw = strip_nulls(raw)
                try:
                    prog = json_format.ParseDict(raw, q.QueryProgram())
                    errs = bundle.validate(prog)
                    errors = [str(e) for e in errs]
                except json_format.ParseError as ex:
                    errors = [f"PARSE: {ex}"]
                if not errors:
                    self._cache_put(key, {"program": raw, "utterance": utterance, "attempts": attempt,
                                          "repaired_from": list(diag.errors)})
                    return prog, diag
                diag.errors.extend(f"attempt{attempt}: {e}" for e in errors)  # kept on success too: repair reasons
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "The program failed validation:\n- " + "\n- ".join(errors) +
                          "\nReturn a corrected program for the same question."}]
        diag.state = a.FAILED
        self._cache_put(key, {"program": None, "errors": list(diag.errors), "utterance": utterance})
        return None, diag
