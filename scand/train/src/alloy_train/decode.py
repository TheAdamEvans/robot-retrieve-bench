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

    def obstacle_xy(self, prof, i_or_t: int, by_time: bool = True) -> tuple[np.ndarray, np.ndarray, str, int]:
        """Obstacle returns in the sensor plane from the profile's clearance sensor → (xy, angle, topic, index).
        3D clouds are cut to a body-height band (ground and overhead removed)."""
        from alloy_server.gen.alloy.v1 import embodiment_pb2 as e
        s = next(x for x in prof.sensors if x.name == prof.clearance_sensor)
        topic = s.topics[0]
        tl = self.rec.topics[topic]
        i = (tl.last_before(i_or_t, inclusive=True) if by_time else i_or_t)
        if i is None:
            return np.zeros((0, 2)), np.zeros(0), topic, -1
        if s.kind == e.LIDAR_3D:
            p = self.points_xyz(topic, i)
            p = p[np.isfinite(p).all(1) & (p[:, 2] > -0.35) & (p[:, 2] < 1.5)]
            r = np.hypot(p[:, 0], p[:, 1])
            p = p[r > 0.5]
            return p[:, :2], np.arctan2(p[:, 1], p[:, 0]), topic, i
        xy, a = self.scan_xy(topic, i, rmin=0.5)
        return xy, a, topic, i

    def scan_xy(self, topic: str, i: int, rmin: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
        """→ (xy points in sensor frame, per-beam angle) with invalid ranges dropped."""
        m = self.msg(topic, i)
        r = np.asarray(m.ranges, dtype=np.float32)
        a = m.angle_min + np.arange(len(r), dtype=np.float32) * m.angle_increment
        ok = np.isfinite(r) & (r > max(rmin, m.range_min)) & (r < m.range_max)
        return np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])], axis=1), a[ok]


def odom_arrays(dec: Decoder, topic: str, prof, sl: slice | None = None) -> dict[str, np.ndarray]:
    """speed_mps (planar speed), yaw_rate_dps per odom message. The twist frame comes from the embodiment profile:
    in the odom frame planar speed is |(vx, vy)|; in the body frame it is |vx|."""
    from alloy_server.gen.alloy.v1 import embodiment_pb2 as e
    odom = next(s for s in prof.sensors if s.kind == e.ODOMETRY)
    in_odom = odom.twist_frame == e.TWIST_IN_ODOM
    tl = dec.rec.topics[topic]
    idx = range(len(tl)) if sl is None else range(sl.start, sl.stop)
    speed = np.empty(len(idx))
    yaw = np.empty(len(idx))
    for k, i in enumerate(idx):
        m = dec.msg(topic, i)
        v = m.twist.twist.linear
        speed[k] = np.hypot(v.x, v.y) if in_odom else abs(v.x)
        yaw[k] = np.degrees(m.twist.twist.angular.z)
    return {"t_ns": tl.log_ns[list(idx)].copy() if len(idx) else np.zeros(0, np.int64), "speed_mps": speed,
            "yaw_rate_dps": yaw}
