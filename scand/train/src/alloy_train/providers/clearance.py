"""clearance@1: sector minimum ranges, lateral clearances, gap width and a geometric doorway flag from the 2D scan.

Spot uses /scan (from the Velodyne, RECORDED_TF mount); Jackal uses /velodyne_2dscan (NOMINAL mount, no /tf).
Ranges are from the sensor origin; lateral clearances use points alongside the NOMINAL footprint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from alloy_train import embodiment as emb
from alloy_train.decode import Decoder
from alloy_train.providers.common import write
from alloy_train.recordings import robot

VERSION = "clearance@1"
GAP_DOORWAY_M = 1.6
DOOR_MIN_S, DOOR_MAX_S = 0.3, 5.0


def flag_runs(t_ns: np.ndarray, mask: np.ndarray, min_s: float, max_s: float) -> np.ndarray:
    out = np.zeros(len(mask), dtype=bool)
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        dur = (t_ns[j] - t_ns[i]) / 1e9
        if min_s <= dur <= max_s:
            out[i:j + 1] = True
        i = j + 1
    return out


def build(bundle: Path, rec) -> dict:
    rb = robot(rec.id)
    dec = Decoder(bundle, rec)
    topic = emb.scan_topic(rb)
    tl = rec.topics[topic]
    half_len = emb.CARDS[rb]["footprint_m"]["length"] / 2
    n = len(tl)
    front, anyr, left, right = (np.full(n, np.nan) for _ in range(4))
    for i in range(n):
        xy, a = dec.scan_xy(topic, i, rmin=0.5)
        if not len(xy):
            continue
        r = np.hypot(xy[:, 0], xy[:, 1])
        anyr[i] = r.min()
        sel = np.abs(a) < np.radians(30)
        if sel.any():
            front[i] = r[sel].min()
        beside = (np.abs(xy[:, 0]) <= half_len + 0.3) & (np.abs(xy[:, 1]) < 5.0)
        lp = xy[beside & (xy[:, 1] > 0), 1]
        rp = -xy[beside & (xy[:, 1] < 0), 1]
        left[i] = lp.min() if len(lp) else 5.0
        right[i] = rp.min() if len(rp) else 5.0
    t = tl.log_ns.copy()
    gap = left + right
    door = flag_runs(t, gap < GAP_DOORWAY_M, DOOR_MIN_S, DOOR_MAX_S)
    cols = {"t_ns": t, "available_at_ns": t, "min_clearance_front_m": front, "min_clearance_any_m": anyr,
            "lateral_clearance_left_m": left, "lateral_clearance_right_m": right, "gap_width_m": gap,
            "doorway_active": door.astype(float)}
    hz = n / ((t[-1] - t[0]) / 1e9)
    # doorway_active looks up to 5 s ahead to decide a run's length → not causal; everything else is per-scan.
    write(bundle, "clearance", rec.id, cols, {"version": VERSION, "hz": round(hz, 2), "exhaustive": True,
                                              "source_topics": [topic], "causal": True,
                                              "non_causal_columns": ["doorway_active"]})
    return {"provider": VERSION, "rows": n, "hz": round(hz, 1), "doorway_samples": int(door.sum())}
