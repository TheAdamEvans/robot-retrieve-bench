"""Provider output conventions (see alloy_server.catalog.features)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def trailing(t_ns: np.ndarray, y: np.ndarray, window_s: float, fn) -> np.ndarray:
    """Causal trailing-window reduction: out[i] = fn(y[t > t_i - w and t <= t_i])."""
    lo = np.searchsorted(t_ns, t_ns - int(window_s * 1e9), side="right")
    return np.array([fn(y[a:i + 1]) for i, a in enumerate(lo)])


def trailing_time_mean(t_ns: np.ndarray, y: np.ndarray, window_s: float) -> np.ndarray:
    """Exact causal time-average of the piecewise-linear signal over [t - W, t]. With W = k x gait period this puts
    spectral nulls at the gait frequency and its harmonics (a median does not)."""
    t = t_ns / 1e9
    integral = np.concatenate([[0.0], np.cumsum(np.diff(t) * (y[1:] + y[:-1]) / 2)])
    lo = np.clip(t - window_s, t[0], None)
    span = t - lo
    out = (integral - np.interp(lo, t, integral)) / np.where(span > 0, span, 1.0)
    return np.where(span > 0, out, y)


def smooth_speed(t_ns: np.ndarray, y: np.ndarray, prof, intake) -> tuple[np.ndarray, str]:
    from alloy_server.catalog.embodiment import speed_window_s
    kind, w = speed_window_s(prof, intake)
    if kind == "mean":
        return trailing_time_mean(t_ns, y, w), f"trailing mean {w:.3f}s"
    return trailing(t_ns, y, w, np.median), f"trailing median {w:.2f}s"


def write(bundle: Path, provider: str, rec: str, cols: dict[str, np.ndarray], meta: dict) -> Path:
    p = bundle / "features" / provider / f"{rec}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(cols).replace_schema_metadata({"provider": json.dumps(meta)})
    pq.write_table(table, p, row_group_size=4096)
    return p
