"""Bytes read per operation, interleaved vs by_sensor MCAP layout (same messages, counted by the same readers)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alloy_server.catalog.embodiment import EmbodimentContext, load_intake, profile
from alloy_server.io import cost
from alloy_server.timeline.store import Recording
from alloy_index.recordings import SCAND_ROOT


def ops(rec: Recording, ctx: EmbodimentContext, t_s: float) -> dict:
    t = rec.t_abs(t_s)
    front = ctx.topic("front_camera")
    lidar = (ctx.topic("lidar") or ctx.topic("lidar_2d"))
    odom = ctx.topic("odom")
    out = {}

    def measure(name, fn):
        rec._readers.clear()  # cold: no chunk cache carried between operations
        with cost.scope(name) as s:
            fn()
        out[name] = s.totals()["bytes_read"]

    def clip():
        tl = rec.topics[front]
        for i in range(*tl.range(t - int(4e9), t).indices(len(tl))):
            rec.read(front, i)

    def telemetry():
        rec.read(odom, rec.topics[odom].last_before(t))

    def receipt():
        for s_ in ("front_camera", "body_cameras", "lidar", "odom", "tf"):
            for tp in ctx.topics(s_):
                i = rec.topics[tp].last_before(t)
                if i is not None:
                    rec.read(tp, i)

    def lidar_frame():
        rec.read(lidar, rec.topics[lidar].last_before(t))

    measure("clip_4s_front", clip)
    measure("telemetry_evidence", telemetry)
    measure("causal_receipt_all_sensors", receipt)
    measure("lidar_frame", lidar_frame)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interleaved", type=Path, default=SCAND_ROOT / "bundles" / "bench-interleaved")
    ap.add_argument("--by-sensor", type=Path, default=SCAND_ROOT / "bundles" / "dev")
    ap.add_argument("--recordings", default="Butler,Sanjac")
    ap.add_argument("--t", type=float, default=50.0)
    a = ap.parse_args()
    rows = []
    for rid in a.recordings.split(","):
        intake = load_intake(a.by_sensor, rid)
        ctx = EmbodimentContext(profile(intake.embodiment_id), intake)
        il = ops(Recording(a.interleaved, rid), ctx, a.t)
        bs = ops(Recording(a.by_sensor, rid), ctx, a.t)
        for k in il:
            rows.append({"recording": rid, "operation": k, "interleaved_bytes": il[k], "by_sensor_bytes": bs[k],
                         "reduction_x": round(il[k] / max(bs[k], 1), 1)})
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
