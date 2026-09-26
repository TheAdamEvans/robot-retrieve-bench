"""Train-side decoding of raw MCAP payloads with a typestore built from the MCAP's own schemas."""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from alloy_server.timeline.store import Recording


@functools.lru_cache(maxsize=None)
def typestore_for(mcap_path: str):
    with open(mcap_path, "rb") as f:
        summary = make_reader(f).get_summary()
    ts = get_typestore(Stores.EMPTY)  # only the MCAP's own schemas: the bag is the source of truth
    types, topic_type = {}, {}
    for ch in summary.channels.values():
        schema = summary.schemas[ch.schema_id]
        topic_type[ch.topic] = schema.name
        types.update(get_types_from_msg(schema.data.decode(), schema.name))
    ts.register(types)
    return ts, topic_type


class Decoder:
    def __init__(self, bundle: Path, rec: Recording):
        self.rec = rec
        self.ts, self.types = typestore_for(str(bundle / "mcap" / f"{rec.id}.mcap"))

    def msg(self, topic: str, i: int):
        return self.ts.deserialize_ros1(self.rec.read(topic, i).data, self.types[topic])

    def points_xyz(self, topic: str, i: int) -> np.ndarray:
        m = self.msg(topic, i)
        n = m.width * m.height
        buf = np.frombuffer(m.data, dtype=np.uint8).reshape(n, m.point_step)
        off = {f.name: f.offset for f in m.fields}
        return np.stack([buf[:, off[k] : off[k] + 4].copy().view(np.float32)[:, 0] for k in "xyz"], axis=1)

    def scan_xy(self, topic: str, i: int, rmin: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
        """→ (xy points in sensor frame, per-beam angle) with invalid ranges dropped."""
        m = self.msg(topic, i)
        r = np.asarray(m.ranges, dtype=np.float32)
        a = m.angle_min + np.arange(len(r), dtype=np.float32) * m.angle_increment
        ok = np.isfinite(r) & (r > max(rmin, m.range_min)) & (r < m.range_max)
        return np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])], axis=1), a[ok]


def odom_arrays(dec: Decoder, topic: str, robot: str) -> dict[str, np.ndarray]:
    """speed_mps (planar speed), yaw_rate_dps, header-free times; decoded natively for every message."""
    tl = dec.rec.topics[topic]
    speed = np.empty(len(tl))
    yaw = np.empty(len(tl))
    for i in range(len(tl)):
        m = dec.msg(topic, i)
        v = m.twist.twist.linear
        speed[i] = np.hypot(v.x, v.y) if robot == "spot" else abs(v.x)
        yaw[i] = np.degrees(m.twist.twist.angular.z)
    return {"t_ns": tl.log_ns.copy(), "speed_mps": speed, "yaw_rate_dps": yaw}
