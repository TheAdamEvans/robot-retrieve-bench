"""imu@2: vibration from the accelerometer (terrain roughness, standstill evidence) and gyro yaw rate.

imu_vibration_rms: |a| high-passed by subtracting its trailing 0.5 s mean, then RMS over a trailing 0.5 s window
(m/s^2). Gravity and slow attitude changes cancel; wheel/terrain vibration remains, and it collapses at a true standstill.
imu_yaw_rate_dps: gyro z, trailing 0.2 s mean (+ = left), an odometry-independent turn measurement.
odom_gyro_yaw_disagreement_dps: |wheel-odometry yaw rate - gyro yaw rate|, both as trailing 0.5 s time-means on the
same clock (odometry sampled causally: the last message at or before each IMU sample). Wheel slip, skid or an
odometry glitch shows up here; on SCAND's Jackal logs the two agree (r ~0.99), so it stays small everywhere.

All causal (receipt time). SCAND's Jackal bags record the IMU without a message definition; the decoder repairs it
only by md5 match (see intake's schema check).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from alloy_index.decode import Decoder, odom_arrays
from alloy_server.catalog import embodiment as E
from alloy_index.providers.common import trailing_time_mean, write

VERSION = "imu@2"  # @2: odom_gyro_yaw_disagreement_dps
HP_WINDOW_S = 0.5
RMS_WINDOW_S = 0.5
YAW_WINDOW_S = 0.2
AGREE_WINDOW_S = 0.5


def build(bundle: Path, rec, ctx: E.EmbodimentContext | None = None) -> dict:
    ctx = ctx or E.EmbodimentContext.for_recording(bundle, rec.id)
    topic = ctx.topic("imu")
    dec = Decoder(bundle, rec)
    tl = rec.topics[topic]
    acc, gz = np.empty((len(tl), 3)), np.empty(len(tl))
    for i in range(len(tl)):
        m = dec.msg(topic, i)
        a = m.linear_acceleration
        acc[i] = (a.x, a.y, a.z)
        gz[i] = m.angular_velocity.z
    t = tl.log_ns.copy()
    mag = np.linalg.norm(acc, axis=1)
    hp = mag - trailing_time_mean(t, mag, HP_WINDOW_S)
    vib = np.sqrt(np.maximum(trailing_time_mean(t, hp ** 2, RMS_WINDOW_S), 0.0))
    yaw = trailing_time_mean(t, np.degrees(gz), YAW_WINDOW_S)
    odom = odom_arrays(dec, ctx.topic("odom"), ctx.profile)
    to = odom["t_ns"]
    o = trailing_time_mean(to, odom["yaw_rate_dps"], AGREE_WINDOW_S)
    last = np.clip(np.searchsorted(to, t, side="right") - 1, 0, len(to) - 1)  # causal: latest odometry at or before t
    disagree = np.abs(np.where(to[last] <= t, o[last], np.nan) - trailing_time_mean(t, np.degrees(gz), AGREE_WINDOW_S))
    valid = ~np.isnan(disagree)
    hz = len(t) / ((t[-1] - t[0]) / 1e9)
    write(bundle, "imu", rec.id, {"t_ns": t[valid], "available_at_ns": t[valid], "imu_vibration_rms": vib[valid],
                                  "imu_yaw_rate_dps": yaw[valid], "odom_gyro_yaw_disagreement_dps": disagree[valid]},
          {"version": VERSION, "hz": round(hz, 2), "exhaustive": True, "source_topics": [topic, ctx.topic("odom")], "causal": True,
           "trailing_bytes": dec.trailing.get(topic, 0), "schema_repaired": topic in dec.repairs})
    return {"provider": VERSION, "rows": len(t), "hz": round(hz, 1),
            "vib_p10": round(float(np.percentile(vib, 10)), 3), "vib_p90": round(float(np.percentile(vib, 90)), 3),
            "disagree_p99": round(float(np.percentile(disagree[valid], 99)), 1),
            "disagree_max": round(float(disagree[valid].max()), 1)}
