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


def write(bundle: Path, provider: str, rec: str, cols: dict[str, np.ndarray], meta: dict) -> Path:
    p = bundle / "features" / provider / f"{rec}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(cols).replace_schema_metadata({"provider": json.dumps(meta)})
    pq.write_table(table, p, row_group_size=4096)
    return p
