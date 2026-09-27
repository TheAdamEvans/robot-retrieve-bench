"""Per-request cost accounting.

Readers emit byte events into the active CostScope (a contextvar). The pipeline runner opens a child scope per
stage, so bytes, tokens and time are attributed to the stage that caused them and roll up to the request.
"""
from __future__ import annotations

import contextlib
import contextvars
import time
from dataclasses import dataclass, field


@dataclass
class ReadEvent:
    layer: str
    path: str
    offset: int
    length: int
    cache_hit: bool


@dataclass
class CostScope:
    name: str
    events: list[ReadEvent] = field(default_factory=list)
    children: list[CostScope] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0
    completion_tokens: int = 0
    usd_est: float = 0.0
    encoder_forward_ms: float = 0.0
    wall_ms: float = 0.0
    _chunks_seen: set[tuple[str, int]] = field(default_factory=set)

    def iter_all(self, exclude: tuple[str, ...] = ()):
        yield self
        for c in self.children:
            if c.name not in exclude:
                yield from c.iter_all(exclude)

    def totals(self, exclude: tuple[str, ...] = ()) -> dict:
        """exclude: child scope names reported as their own stage (e.g. program_generation) so a stage tree never
        counts them twice."""
        scopes = list(self.iter_all(exclude))
        evs = [e for s in scopes for e in s.events]
        by_layer: dict[str, int] = {}
        for e in evs:
            if not e.cache_hit:
                by_layer[e.layer] = by_layer.get(e.layer, 0) + e.length
        return {
            "bytes_read": sum(by_layer.values()),
            "bytes_by_layer": by_layer,
            "bytes_served_from_cache": sum(e.length for e in evs if e.cache_hit),
            "llm_calls": sum(s.llm_calls for s in scopes),
            "prompt_tokens": sum(s.prompt_tokens for s in scopes),
            "cached_prompt_tokens": sum(s.cached_prompt_tokens for s in scopes),
            "reasoning_tokens": sum(s.reasoning_tokens for s in scopes),
            "completion_tokens": sum(s.completion_tokens for s in scopes),
            "usd_est": sum(s.usd_est for s in scopes),
            "encoder_forward_ms": sum(s.encoder_forward_ms for s in scopes),
        }


_current: contextvars.ContextVar[CostScope | None] = contextvars.ContextVar("cost_scope", default=None)


def current() -> CostScope | None:
    return _current.get()


def root_scope() -> CostScope | None:
    """Session-level loads (timelines, models) are recorded here when no request scope is active."""
    return _current.get()


@contextlib.contextmanager
def scope(name: str):
    parent = _current.get()
    s = CostScope(name)
    if parent is not None:
        parent.children.append(s)
        s._chunks_seen = parent._chunks_seen  # a chunk is counted once per request
    token = _current.set(s)
    t0 = time.perf_counter()
    try:
        yield s
    finally:
        s.wall_ms = (time.perf_counter() - t0) * 1000
        _current.reset(token)


def record_read(layer: str, path: str, offset: int, length: int, cache_hit: bool,
                scope: CostScope | None = None) -> None:
    """`scope` pins the destination for reads issued from worker threads (contextvars don't cross into them)."""
    s = scope if scope is not None else _current.get()
    if s is None:
        return
    if not cache_hit:
        key = (path, offset)
        if key in s._chunks_seen:
            cache_hit = True
        else:
            s._chunks_seen.add(key)
    s.events.append(ReadEvent(layer, path, offset, length, cache_hit))
