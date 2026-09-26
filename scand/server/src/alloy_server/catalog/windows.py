"""The shared window grid: 4 s trailing windows at a 1 s stride, keyed by end second.

Windows are retrieval units and candidate anchors only, never the scope of verification.
Segments (non-overlapping, for per-window attribute labels) are the windows whose end is a multiple of 4.
"""
from __future__ import annotations

WINDOW_S = 4
STRIDE_S = 1


def window_id(rec: str, end_s: int) -> str:
    return f"{rec}:{end_s:04d}"


def parse_window_id(wid: str) -> tuple[str, int]:
    rec, end = wid.rsplit(":", 1)
    return rec, int(end)


def window_span_s(wid: str) -> tuple[str, float, float]:
    rec, end = parse_window_id(wid)
    return rec, float(end - WINDOW_S), float(end)


def windows(rec: str, duration_s: float) -> list[str]:
    return [window_id(rec, e) for e in range(WINDOW_S, int(duration_s) + 1, STRIDE_S)]


def segments(rec: str, duration_s: float) -> list[str]:
    return [window_id(rec, e) for e in range(WINDOW_S, int(duration_s) + 1, WINDOW_S)]
