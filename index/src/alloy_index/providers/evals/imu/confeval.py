"""imu@1 on Jackal: vibration separates standstill, slow and fast driving; the gyro agrees with wheel odometry on the
same 0.5 s window (r >= 0.95, scale 0.9-1.1); the intake records the schema repair. Seeded by the Brackenridge "stop" question: the operator only released turbo (0.5 m/s)."""
import numpy as np
import pyarrow.parquet as pq

from alloy_server.catalog import embodiment as E
from alloy_server.evalkit import Dataset, Gate, at_least, at_most, exact
from alloy_index.providers.common import trailing_time_mean

DATASETS = [Dataset("jackal", "cases.jsonl", requires=["bundle"])]
SCORERS = [at_most("vib_max"), at_least("vib_min"), at_least("yaw_r"), at_least("scale_min"), at_most("scale_max"),
           exact("repair")]
GATES = [Gate("vib_max.at_most", ">=", 1.0), Gate("vib_min.at_least", ">=", 1.0), Gate("yaw_r.at_least", ">=", 1.0),
         Gate("scale_min.at_least", ">=", 1.0), Gate("scale_max.at_most", ">=", 1.0), Gate("repair.accuracy", ">=", 1.0)]


def _series(ctx, rid, provider, col):
    t = pq.read_table(ctx.bundle_root / "features" / provider / f"{rid}.parquet").to_pydict()
    start = ctx.bundle.recordings[rid].start_ns
    return (np.array(t["t_ns"]) - start) / 1e9, np.array(t[col])


def run_case(case, ctx):
    i = case["input"]
    rid = i["recording"]
    if i["op"] == "vibration":
        t, v = _series(ctx, rid, "imu", "imu_vibration_rms")
        m = (t >= i["t0"]) & (t < i["t1"])
        med = float(np.median(v[m]))
        return {"vib_max": med, "vib_min": med}
    if i["op"] == "yaw_agreement":
        ti, g = _series(ctx, rid, "imu", "imu_yaw_rate_dps")
        tm, y = _series(ctx, rid, "motion", "yaw_rate_dps")
        g = trailing_time_mean((ti * 1e9).astype(np.int64), g, 0.5)  # the odometry yaw rate's own 0.5 s window
        o = np.interp(ti, tm, y)
        scale = float(np.polyfit(g, o, 1)[0])
        return {"yaw_r": float(np.corrcoef(g, o)[0, 1]), "scale_min": scale, "scale_max": scale}
    rep = E.load_intake(ctx.bundle_root, rid)
    r = next((x for x in rep.schema_repairs if x.topic == i["topic"]), None)
    return {"repair": None if r is None else {"msgtype": r.msgtype, "trailing_bytes": r.trailing_bytes}}
