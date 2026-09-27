"""clearance@2: sector minimum ranges, lateral clearances, gap width and a geometric doorway flag.

Returns come from the profile's clearance sensor: Spot's 3D Velodyne cut to a body-height band (a single scan plane
misses legs, crutches and low obstacles), Jackal's flattened 2D scan. Ranges are from the sensor origin; lateral
clearances use points alongside the NOMINAL footprint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from alloy_index.decode import Decoder
from alloy_index.providers.common import write

VERSION = "clearance@4"
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


def build(bundle: Path, rec, ctx=None) -> dict:
    from alloy_server.catalog.embodiment import EmbodimentContext
    ctx = ctx or EmbodimentContext.for_recording(bundle, rec.id)
    dec = Decoder(bundle, rec)
    prof = ctx.profile
    topic = ctx.clearance.topics[0]
    tl = rec.topics[topic]
    n = len(tl)
    front, anyr, left, right, margin = (np.full(n, np.nan) for _ in range(5))
    for i in range(n):
        xy, a, _, _ = dec.obstacle_xy(prof, i, by_time=False)
        if not len(xy):
            continue
        r = np.hypot(xy[:, 0], xy[:, 1])
        anyr[i] = r.min()
        sel = np.abs(a) < np.radians(30)
        if sel.any():
            front[i] = r[sel].min()
        left[i], right[i] = ctx.body_side_room(xy)  # alongside the body, from each side (labeller findings)
        margin[i] = ctx.front_margin(xy)
    t = tl.log_ns.copy()
    gap = left + right + prof.footprint.width_m  # full width of the gap the body passes through
    door = flag_runs(t, gap < GAP_DOORWAY_M, DOOR_MIN_S, DOOR_MAX_S)
    cols = {"t_ns": t, "available_at_ns": t, "min_clearance_front_m": front, "min_clearance_any_m": anyr,
            "lateral_clearance_left_m": left, "lateral_clearance_right_m": right, "gap_width_m": gap,
            "doorway_active": door.astype(float), "clearance_margin_front_m": margin}
    hz = n / ((t[-1] - t[0]) / 1e9)
    # doorway_active looks up to 5 s ahead to decide a run's length → not causal; everything else is per-scan.
    write(bundle, "clearance", rec.id, cols, {"version": VERSION, "hz": round(hz, 2), "exhaustive": True,
                                              "source_topics": [topic], "causal": True,
                                              "non_causal_columns": ["doorway_active"]})
    return {"provider": VERSION, "rows": n, "hz": round(hz, 1), "doorway_samples": int(door.sum())}
