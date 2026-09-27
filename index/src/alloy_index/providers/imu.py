"""imu@1: vibration from the accelerometer (terrain roughness, standstill evidence) and gyro yaw rate.

imu_vibration_rms: |a| high-passed by subtracting its trailing 0.5 s mean, then RMS over a trailing 0.5 s window
(m/s^2). Gravity and slow attitude changes cancel; wheel/terrain vibration remains, and it collapses at a true standstill.
imu_yaw_rate_dps: gyro z, trailing 0.2 s mean (+ = left), an odometry-independent turn measurement.

All causal (receipt time). SCAND's Jackal bags record the IMU without a message definition; the decoder repairs it
only by md5 match (see intake's schema check).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from alloy_index.decode import Decoder
from alloy_server.catalog import embodiment as E
from alloy_index.providers.common import trailing_time_mean, write

VERSION = "imu@1"
HP_WINDOW_S = 0.5
RMS_WINDOW_S = 0.5
YAW_WINDOW_S = 0.2


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
    hz = len(t) / ((t[-1] - t[0]) / 1e9)
    write(bundle, "imu", rec.id, {"t_ns": t, "available_at_ns": t, "imu_vibration_rms": vib, "imu_yaw_rate_dps": yaw},
          {"version": VERSION, "hz": round(hz, 2), "exhaustive": True, "source_topics": [topic], "causal": True,
           "trailing_bytes": dec.trailing.get(topic, 0), "schema_repaired": topic in dec.repairs})
    return {"provider": VERSION, "rows": len(t), "hz": round(hz, 1),
            "vib_p10": round(float(np.percentile(vib, 10)), 3), "vib_p90": round(float(np.percentile(vib, 90)), 3)}
