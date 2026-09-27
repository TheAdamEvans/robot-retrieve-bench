"""alloy-train intake: a new log in → recognised, converted, measured, flagged.

1. Recognise the robot from the log's topics against the embodiment profiles (no match → fail loudly, naming the
   closest profile and its missing signature topics: a new robot needs a reviewed profile, not new code).
2. Convert to MCAP + timeline (payload hashes checked against the source).
3. Measure every topic (rate, gaps, capture-to-receipt latency, future-stamped headers) and, for legged robots, the
   gait period that the speed smoother is synchronised to.
4. Flag deviations from the profile as anomalies for a human to look at.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from alloy_server.catalog import embodiment as E
from alloy_server.gen.alloy.v1 import embodiment_pb2 as e
from alloy_server.timeline.store import NO_HEADER, Recording
from alloy_index.convert.bag_to_mcap import convert, file_sha256
from alloy_index.decode import Decoder, odom_arrays
from alloy_index.recordings import RECORDINGS, bag_path

VERSION = "intake@1"
LATENCY_NOTE_MS = 200.0
RATE_TOL = 0.2


def recognise(topics: set[str]) -> str:
    pid, missing = E.match(topics)
    if pid is None:
        closest = min(missing, key=lambda k: len(missing[k]))
        raise SystemExit(f"no embodiment profile matches this log. Closest: {closest!r}, missing {missing[closest]}. "
                         f"Add a reviewed profile under alloy_server/embodiments/.")
    return pid


def estimate_gait(t_ns: np.ndarray, speed: np.ndarray) -> e.GaitEstimate | None:
    """Dominant 1-6 Hz oscillation of raw speed while walking (Welch-averaged, parabolic peak refinement)."""
    ts = t_ns / 1e9
    dt = float(np.median(np.diff(ts)))
    g = np.arange(ts[0], ts[-1], dt)
    y = np.interp(g, ts, speed)
    k = max(int(round(1.0 / dt)), 1)
    walk = np.convolve(y, np.ones(k) / k, "same") > 0.5
    n = int(round(8 / dt))
    specs = []
    for s in range(0, len(y) - n, n // 2):
        if walk[s:s + n].mean() < 0.95:
            continue
        m = int(2 / dt)
        seg = y[s:s + n] - np.convolve(y[s:s + n], np.ones(m) / m, "same")
        specs.append(np.abs(np.fft.rfft(seg * np.hanning(n), 8 * n)) ** 2)
    if not specs:
        return None
    p = np.mean(specs, 0)
    f = np.fft.rfftfreq(8 * n, dt)
    band = (f > 1.0) & (f < min(6.0, 0.45 / dt))
    i = int(np.argmax(np.where(band, p, 0)))
    a, b, c = np.log(p[i - 1:i + 2])
    fk = f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])
    return e.GaitEstimate(period_s=1 / fk, frequency_hz=fk, peak_to_median=float(p[i] / np.median(p[band])),
                          windows=len(specs))


def measure(bundle: Path, rec_id: str, prof: e.EmbodimentProfile, src: Path) -> e.IntakeReport:
    r = Recording(bundle, rec_id)
    types = Decoder(bundle, r).types
    rep = e.IntakeReport(recording_id=rec_id, source=src.name, source_sha256=r.info["bag_sha256"],
                         source_bytes=r.info["bag_bytes"], embodiment_id=prof.embodiment_id,
                         profile_version=prof.profile_version, duration_s=(r.end_ns - r.start_ns) / 1e9,
                         intake_version=VERSION)
    profiled = {t: s for s in prof.sensors for t in s.topics}
    for topic, tl in sorted(r.topics.items()):
        tp = e.TopicProfile(topic=topic, msgtype=types.get(topic, ""), count=len(tl))
        if len(tl) > 1:
            gaps = np.diff(tl.log_ns) / 1e9
            period = (tl.log_ns[-1] - tl.log_ns[0]) / 1e9 / (len(tl) - 1)  # mean rate: robust to bursty delivery
            tp.hz, tp.max_gap_s = 1 / period, float(gaps.max())
            tp.gaps_over_2x_period = int((gaps > 2 * period).sum())
        has_h = tl.header_ns != NO_HEADER
        tp.header_present = bool(has_h.all())
        if has_h.any():
            lat = (tl.log_ns[has_h] - tl.header_ns[has_h]) / 1e6
            tp.median_capture_to_receipt_ms = float(np.median(lat))
            tp.p99_capture_to_receipt_ms = float(np.percentile(lat, 99))
            tp.future_stamped_frac = float((lat < 0).mean())
            tp.max_future_ms = float(max(0.0, -lat.min()))
        rep.topics.append(tp)
        s = profiled.get(topic)
        if s is None:
            rep.unprofiled_topics.append(topic)
            continue
        if s.nominal_hz and tp.hz and abs(tp.hz - s.nominal_hz) / s.nominal_hz > RATE_TOL:
            rep.anomalies.append(f"{topic}: {tp.hz:.1f} Hz vs profile {s.nominal_hz:g} Hz")
        if tp.max_gap_s > max(1.0, 5 / max(tp.hz, 1e-9)):
            rep.anomalies.append(f"{topic}: gap of {tp.max_gap_s:.2f} s")
        if tp.HasField("median_capture_to_receipt_ms") and tp.median_capture_to_receipt_ms > LATENCY_NOTE_MS:
            rep.anomalies.append(f"{topic}: arrives {tp.median_capture_to_receipt_ms:.0f} ms after capture (median)")
        if tp.HasField("future_stamped_frac") and tp.future_stamped_frac > 0.005:
            rep.anomalies.append(f"{topic}: {100 * tp.future_stamped_frac:.0f}% of headers stamped after receipt "
                                 f"(up to {tp.max_future_ms:.0f} ms)")
    for sp in prof.sensors:  # buffer flush: a few frames, then a gap > 5 periods within the first 2 s
        if sp.kind != e.CAMERA:
            continue
        for t in sp.topics:
            tl = r.topics.get(t)
            if tl is None or len(tl) < 10:
                continue
            gaps = np.diff(tl.log_ns[:20]) / 1e9
            period = float(np.median(np.diff(tl.log_ns)) / 1e9)
            big = np.where(gaps > 5 * period)[0]
            if len(big) and big[0] < 5 and (tl.log_ns[big[0] + 1] - r.start_ns) / 1e9 < 2.0:
                rep.stale_leading_frames[t] = int(big[0] + 1)
                rep.anomalies.append(f"{t}: first {big[0] + 1} frame(s) precede a {gaps[big[0]]:.2f} s gap at the "
                                     f"start (likely a stale buffer flush; ordinals 0..{big[0]})")
    for s in prof.sensors:
        present = [t for t in s.topics if t in r.topics]
        if not present:  # a missing profiled sensor degrades the log; it does not reject it
            rep.absent_sensors.append(s.name)
            rep.anomalies.append(f"sensor {s.name} absent from this log: its features resolve to UNKNOWN "
                                 f"(SENSOR_ABSENT_IN_LOG), never FALSE")
        for t in s.topics:
            if present and t not in r.topics:
                rep.anomalies.append(f"{t}: in profile ({s.name}) but absent from this log")
    odom = E.sensor(prof, "odom")
    raw = odom_arrays(Decoder(bundle, r), odom.topics[0], prof)
    rep.speed_floor_mps = float(np.percentile(raw["speed_mps"], 1))
    if rep.speed_floor_mps > 0.2:
        rep.anomalies.append(f"slowest odometry speed (1st percentile) is {rep.speed_floor_mps:.2f} m/s: either the robot "
                             f"never stopped or odometry cannot report standstill; review before trusting stop clauses")
    if prof.speed_smoothing.method == e.GAIT_SYNC_MEAN:
        g = estimate_gait(raw["t_ns"], raw["speed_mps"])
        if g is None:
            rep.anomalies.append("no sustained walking found: gait period not measured (smoother falls back)")
        else:
            rep.gait.CopyFrom(g)
            if g.peak_to_median < 50:
                rep.anomalies.append(f"weak gait peak ({g.frequency_hz:.2f} Hz, peak/median {g.peak_to_median:.0f})")
    return rep


def intake(bundle: Path, rec_id: str, src: Path, skip_convert: bool = False) -> e.IntakeReport:
    with AnyReader([src]) as reader:
        topics = {c.topic for c in reader.connections}
    prof = E.profile(recognise(topics))
    from alloy_index.recordings import robot
    assert robot(rec_id) == prof.embodiment_id, f"{rec_id}: file name says {robot(rec_id)}, topics say {prof.embodiment_id}"
    if not skip_convert:
        convert(rec_id, bundle)
    rep = measure(bundle, rec_id, prof, src)
    E.save_intake(bundle, rep)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--skip-convert", action="store_true", help="MCAP + timeline already built")
    ap.add_argument("recs", nargs="*", default=list(RECORDINGS))
    a = ap.parse_args()
    for rec in a.recs:
        rep = intake(a.bundle, rec, bag_path(rec), a.skip_convert)
        gait = f"gait {rep.gait.frequency_hz:.2f} Hz (T={rep.gait.period_s:.3f}s)" if rep.HasField("gait") else "no gait"
        print(json.dumps({"recording": rec, "embodiment": rep.embodiment_id, "speed_floor": round(rep.speed_floor_mps, 3), "topics": len(rep.topics),
                          "unprofiled": list(rep.unprofiled_topics), "gait": gait, "anomalies": list(rep.anomalies)}))


if __name__ == "__main__":
    main()
