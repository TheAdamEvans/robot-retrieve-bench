"""Server-side read of provider outputs: bundle/features/<provider>/<rec>.parquet (+ <provider>_tracks).

Each provider table has `t_ns` (measurement time), `available_at_ns` (max availability over its inputs) and one
column per feature (plus `<col>_lo` / `<col>_hi` for uncertain features). Loaded once per session; the load's bytes are
counted in whichever CostScope is active (session scope at startup).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import parquet_reader
from .registry import REGISTRY, Feature


@dataclass
class Series:
    feature: Feature
    t_ns: np.ndarray
    value: np.ndarray
    lo: np.ndarray | None
    hi: np.ndarray | None
    available_at_ns: np.ndarray
    coverage_hz: float
    exhaustive: bool          # computed over the full recording at the provider's declared rate


@dataclass
class Tracks:
    track_id: np.ndarray
    first_ns: np.ndarray
    last_ns: np.ndarray
    n_obs: np.ndarray


class FeatureStore:
    def __init__(self, bundle: Path):
        self.bundle = bundle
        self._tables: dict[tuple[str, str], dict] = {}
        self._tracks: dict[str, Tracks | None] = {}

    def _table(self, provider: str, rec: str) -> dict | None:
        key = (provider, rec)
        if key not in self._tables:
            p = self.bundle / "features" / provider / f"{rec}.parquet"
            if not p.exists():
                self._tables[key] = None
            else:
                t = parquet_reader.read_table(str(p), layer="features")
                meta = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
                self._tables[key] = {"cols": {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names},
                                     "meta": json.loads(meta.get("provider", "{}"))}
        return self._tables[key]

    def series(self, name: str, rec: str) -> Series | None:
        f = REGISTRY[name]
        tab = self._table(f.provider, rec) if f.indexed else None
        if tab is None:
            return None
        c = tab["cols"]
        col = f.column or f.name
        if col not in c:
            return None
        return Series(f, c["t_ns"].astype(np.int64), c[col].astype(float),
                      c.get(f"{col}_lo"), c.get(f"{col}_hi"), c["available_at_ns"].astype(np.int64),
                      float(tab["meta"].get("hz", 0)), bool(tab["meta"].get("exhaustive", True)))

    def tracks(self, name: str, rec: str) -> Tracks | None:
        f = REGISTRY[name]
        if not f.indexed:
            return None
        if rec not in self._tracks:
            tab = self._table(f"{f.provider}_tracks", rec)
            self._tracks[rec] = None if tab is None else Tracks(
                tab["cols"]["track_id"], tab["cols"]["first_ns"].astype(np.int64),
                tab["cols"]["last_ns"].astype(np.int64), tab["cols"]["n_obs"])
        return self._tracks[rec]
