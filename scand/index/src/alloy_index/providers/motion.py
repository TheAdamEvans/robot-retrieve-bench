"""motion@2: speed smoothed per the embodiment profile (legged: gait-synchronous mean over k x the intake-measured
gait period; wheeled: trailing median), yaw rate (trailing 0.5 s mean), integrated heading, acceleration.

All causal: every output at t uses only odometry messages received at or before t.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from alloy_index import embodiment as emb
from alloy_index.decode import Decoder, odom_arrays
from alloy_server.catalog import embodiment as E
from alloy_index.providers.common import smooth_speed, trailing, write
from alloy_index.recordings import robot

VERSION = "motion@2"


def build(bundle: Path, rec) -> dict:
    rb = robot(rec.id)
    prof = E.profile(rb)
    raw = odom_arrays(Decoder(bundle, rec), emb.odom_topic(rb), prof)
    t = raw["t_ns"]
    speed, smoother = smooth_speed(t, raw["speed_mps"], prof, E.load_intake(bundle, rec.id))
    yaw = trailing(t, raw["yaw_rate_dps"], 0.5, np.mean)
    dt = np.diff(t, prepend=t[0]) / 1e9
    heading = np.cumsum(raw["yaw_rate_dps"] * dt)
    lag = np.searchsorted(t, t - int(0.5e9), side="right")
    lag = np.maximum(lag - 1, 0)
    span = np.maximum((t - t[lag]) / 1e9, 1e-3)
    accel = np.where(t - t[lag] > 0, (speed - speed[lag]) / span, 0.0)
    cols = {"t_ns": t, "available_at_ns": t, "speed_mps": speed, "speed_raw_mps": raw["speed_mps"],
            "yaw_rate_dps": yaw, "heading_deg": heading, "accel_mps2": accel}
    hz = len(t) / ((t[-1] - t[0]) / 1e9)
    write(bundle, "motion", rec.id, cols, {"version": VERSION, "hz": round(hz, 2), "exhaustive": True,
                                           "source_topics": [emb.odom_topic(rb)], "causal": True,
                                           "speed_smoother": smoother})
    return {"provider": VERSION, "rows": len(t), "hz": round(hz, 1), "smoother": smoother}
