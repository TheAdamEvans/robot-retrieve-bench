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
from pathlib import Path

from google.protobuf import json_format
from openai import APIError, OpenAI

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


PROMPTS = Path(__file__).parent / "prompts"
PROMPT = os.environ.get("ALLOY_PROGRAM_PROMPT", "v3")  # versioned grammar/rules text: prompts/<version>.md


def grammar(version: str = PROMPT) -> str:
    return (PROMPTS / f"{version}.md").read_text()


GRAMMAR = grammar()


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
                 model: str = MODEL, reasoning: str = REASONING, cache_only: bool = False, prompt: str = PROMPT):
        self.cache_only = cache_only
        self.client = None if cache_only else OpenAI(timeout=float(os.environ.get("ALLOY_PROGRAM_TIMEOUT_S", "20")), max_retries=0)
        self.model, self.reasoning, self.prompt_version = model, reasoning, prompt
        self.system = "\n\n".join([grammar(prompt), _registry_block(), _embodiment_block(), _recordings_block(bundle),
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
        s = cost.current()
        if s is not None:
            s.llm_calls += 1  # a failed request is still an attempted call
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, reasoning_effort=self.reasoning,
            response_format={"type": "json_schema", "json_schema": {"name": "QueryProgram", "strict": True,
                                                                    "schema": self.schema}})
        u = r.usage
        if s is not None and u is not None:
            cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            reasoning = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
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

    @staticmethod
    def _usage(since: dict | None) -> dict:
        """Tokens spent in the current cost scope since `since`; stored with the cached program so replays and
        module evals can report what the original generation cost."""
        s = cost.current()
        now = {k: getattr(s, k, 0) if s is not None else 0
               for k in ("prompt_tokens", "cached_prompt_tokens", "completion_tokens", "reasoning_tokens")}
        return now if since is None else {k: now[k] - since[k] for k in now}

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
        if self.cache_only:
            raise RuntimeError(f"No cached program for {utterance!r}; refusing a new API call during replay")
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": utterance}]
        errors: list[str] = []
        start = self._usage(None)
        for attempt in (1, 2):
            diag.attempts = attempt
            try:
                raw, text = self._call(messages)
            except (APIError, TimeoutError) as ex:
                # Return through the runner's abstain/unverified path. Transient failures must never poison the cache.
                diag.state = a.FAILED
                diag.errors.append(f"Program generation unavailable ({type(ex).__name__}); retry the search.")
                return None, diag
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
                                          "repaired_from": list(diag.errors), "usage": self._usage(start)})
                    return prog, diag
            diag.errors.extend(f"attempt{attempt}: {e}" for e in errors)  # kept on success too: repair reasons
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "The program failed validation:\n- " + "\n- ".join(errors) +
                          "\nReturn a corrected program for the same question."}]
        diag.state = a.FAILED
        self._cache_put(key, {"program": None, "errors": list(diag.errors), "utterance": utterance, "attempts": 2,
                              "usage": self._usage(start)})
        return None, diag
