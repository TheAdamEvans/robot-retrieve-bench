"""scandq — anchored, audited views over the source-of-truth MCAP bags, for the labeller.

Every view is addressed by (rec, t seconds from recording start), a window_id (`Rec:EEEE`, a 4 s window ending at
second EEEE) or a MessageId string (`Rec/topic#ordinal`). Every view prints JSON listing the refs it rendered and
appends to labels/metadata/<campaign>/audit.jsonl. Images are written under labels/.work/renders/<job>/ for the agent to
open with Read. A ref counts as native evidence only when the view rendered it at full resolution (`native: true`).

Ref fields: t = when the recorder received the message (s from recording start); header_t = when the sensor
stamped it; dt_ms = t minus the requested time. Spot body cameras arrive ~0.4-0.7 s after capture, so a body frame
received at t shows the scene at header_t. `warnings` flags stale or gap-adjacent frames.

Set SCANDQ_CAMPAIGN=<campaign> and SCANDQ_JOB=<job_id> so the audit log and label shards are attributed to them.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from alloy_server.catalog import embodiment as E
from alloy_server.catalog.windows import WINDOW_S, segments, window_span_s, windows
from alloy_server.io.ros1 import compressed_image
from alloy_server.timeline.store import NO_HEADER, Recording, mid_str, parse_mid
from alloy_index import embodiment as emb
from alloy_index.decode import Decoder, odom_arrays
from alloy_index.providers.common import smooth_speed
from alloy_index.recordings import RECORDINGS, SCAND_ROOT, robot

BUNDLE = Path(os.environ.get("SCANDQ_BUNDLE", SCAND_ROOT / "bundles" / "dev"))
LABELS = Path(os.environ.get("SCANDQ_LABELS", SCAND_ROOT / "labels"))
CAMPAIGN = os.environ.get("SCANDQ_CAMPAIGN", "adhoc")
JOB = os.environ.get("SCANDQ_JOB", "adhoc")
FONT = ImageFont.load_default(size=18)
SMALL = ImageFont.load_default(size=14)
STALE_MS = 100.0

# Display rotation and naming quirks (e.g. Spot's cross-eyed front stereo pair) come from the embodiment profile.


# ---------- plumbing ----------

@functools.lru_cache(maxsize=None)
def rec_(name: str) -> Recording:
    if name not in RECORDINGS:
        raise SystemExit(f"unknown recording {name!r}; one of {sorted(RECORDINGS)}")
    return Recording(BUNDLE, name)


@functools.lru_cache(maxsize=None)
def dec_(name: str) -> Decoder:
    return Decoder(BUNDLE, rec_(name))


def out_dir() -> Path:
    d = LABELS / ".work" / "renders" / JOB
    d.mkdir(parents=True, exist_ok=True)
    return d


def audit_path() -> Path:
    return LABELS / "metadata" / CAMPAIGN / "audit.jsonl"


def ref(r: Recording, topic: str, i: int, t_req_ns: int | None, native: bool, warnings: list | None = None) -> dict:
    tl = r.topics[topic]
    d = {"mid": mid_str(r.message_id(topic, i)), "t": round(r.t_rel(int(tl.log_ns[i])), 3), "native": native}
    if tl.header_ns[i] != NO_HEADER:
        d["header_t"] = round(r.t_rel(int(tl.header_ns[i])), 3)
    if t_req_ns is not None:
        d["dt_ms"] = round((int(tl.log_ns[i]) - t_req_ns) / 1e6, 1)
        if warnings is not None and abs(d["dt_ms"]) > STALE_MS and "image" in topic:
            warnings.append(f"{d['mid']} is {d['dt_ms']:+.0f} ms from the requested time (gap or edge of recording)")
    return d


def emit(call: str, args: dict, result: dict) -> None:
    refs = result.get("refs", [])
    audit_path().parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path(), "a") as f:
        f.write(json.dumps({"ts": time.time(), "job": JOB, "call": call, "args": args,
                            "refs": [{"mid": x["mid"], "native": x["native"]} for x in refs]}) + "\n")
    if not result.get("warnings"):
        result.pop("warnings", None)
    # compact: one line per ref
    body = {k: v for k, v in result.items() if k != "refs"}
    lines = json.dumps(body, indent=1)
    if refs:
        lines = lines[:-2] + ',\n "refs": [\n' + ",\n".join("  " + json.dumps(x) for x in refs) + "\n ]\n}"
    print(lines)


def pick(r: Recording, topic: str, t_ns: int, mode: str) -> int | None:
    tl = r.topics.get(topic)
    if tl is None:
        return None  # sensor absent in this log
    i = tl.last_before(t_ns, inclusive=True) if mode == "last_before" else tl.nearest(t_ns)
    stale = E.EmbodimentContext.for_recording(BUNDLE, r.id).stale_leading(topic)
    if i is not None and i < stale:  # buffer-flushed frames show another moment: never show them
        i = stale if stale < len(tl) and mode != "last_before" else None
    return i


def rotation_for(r: Recording, topic: str) -> int:
    return E.topic_display(emb.prof(robot(r.id))).get(topic, (0, topic))[0]


def cam_name(r: Recording, topic: str) -> str:
    return E.topic_display(emb.prof(robot(r.id))).get(topic, (0, topic))[1]


def image(r: Recording, topic: str, i: int) -> Image.Image:
    _, data = compressed_image(r.read(topic, i).data)
    im = Image.open(io.BytesIO(data)).convert("RGB")
    rot = rotation_for(r, topic)
    return im.rotate(rot, expand=True) if rot else im


def cam_topic(r: Recording, cam: str) -> str:
    rb = robot(r.id)
    if cam == "front":
        return emb.front_camera_topic(rb)
    topic = f"/spot/camera/{cam.removeprefix('body_')}/image/compressed"
    if topic not in r.topics:
        raise SystemExit(f"camera {cam!r} not on {rb}; cameras: front" +
                         "".join(f", body_{c}" for c in emb.body_camera_names(rb)))
    return topic


def label(img: Image.Image, text: str, font=FONT) -> Image.Image:
    d = ImageDraw.Draw(img)
    w = d.textlength(text, font=font)
    d.rectangle([0, 0, 8 + w, font.size + 8], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 0), font=font)
    return img


def placeholder(size: tuple[int, int], text: str) -> Image.Image:
    im = Image.new("RGB", size, (40, 20, 20))
    ImageDraw.Draw(im).text((10, size[1] // 2), text, fill=(255, 150, 150), font=FONT)
    return im


def cam_tile(r: Recording, topic: str, t_ns: int, mode: str, size: tuple[int, int], refs: list, warnings: list,
             native: bool) -> Image.Image:
    i = pick(r, topic, t_ns, mode)
    if i is None:
        return placeholder(size, f"{cam_name(r, topic)}: no message {'before' if mode == 'last_before' else 'near'} t")
    rf = ref(r, topic, i, t_ns, native, warnings)
    refs.append(rf)
    lag = f" age {rf['t'] - rf['header_t']:.2f}s" if "header_t" in rf and rf["t"] - rf["header_t"] > 0.1 else ""
    return label(image(r, topic, i).resize(size), f"{cam_name(r, topic)} t={rf['t']:.2f}s{lag}", SMALL)


def grid(tiles: list[Image.Image], cols: int) -> Image.Image:
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    g = Image.new("RGB", (w * cols, h * rows), (30, 30, 30))
    for k, t in enumerate(tiles):
        g.paste(t.resize((w, h)), ((k % cols) * w, (k // cols) * h))
    return g


def stack(parts: list[Image.Image]) -> Image.Image:
    W = max(p.width for p in parts)
    parts = [p if p.width == W else p.resize((W, int(p.height * W / p.width))) for p in parts]
    canvas = Image.new("RGB", (W, sum(p.height for p in parts)), (20, 20, 20))
    y = 0
    for p in parts:
        canvas.paste(p, (0, y))
        y += p.height
    return canvas


def save(img: Image.Image, name: str) -> str:
    p = out_dir() / name
    img.save(p, quality=90)
    return str(p)


# ---------- renders ----------

def front_corridor_overlay(img: Image.Image, rb: str) -> Image.Image:
    """Draw the NOMINAL corridor (0-5 m ahead, ±half-width) on a front frame (ESTIMATED: no intrinsics recorded)."""
    from alloy_index.providers.detections import camera_bands
    b, cor = camera_bands(rb), emb.corridor(rb)
    c = {"half_width_m": cor.half_width_m, "length_m": cor.length_m}
    w, h = img.size
    fx = (w / 2) / np.tan(np.radians(b["hfov"][0] / 2))
    pts = []
    for x, z in ((-c["half_width_m"], 1.0), (-c["half_width_m"], c["length_m"]),
                 (c["half_width_m"], c["length_m"]), (c["half_width_m"], 1.0)):
        u = w / 2 + fx * x / z
        v = h / 2 + fx * b["h"][0] / z
        pts.append((u, v))
    d = ImageDraw.Draw(img)
    d.line(pts, fill=(80, 255, 80), width=3)
    d.text((pts[1][0], pts[1][1] - 18), "corridor 5 m (ESTIMATED)", fill=(80, 255, 80), font=SMALL)
    for z in (10.0, 20.0):  # ground-distance marks from the nominal camera model
        v = h / 2 + fx * b["h"][0] / z
        d.line([(w * 0.3, v), (w * 0.7, v)], fill=(255, 220, 80), width=1)
        d.text((w * 0.7 + 4, v - 8), f"~{z:.0f} m", fill=(255, 220, 80), font=SMALL)
    return img


def bev(r: Recording, t_ns: int, rng: float = 10.0, px: int = 600) -> tuple[Image.Image, dict | None]:
    """Top-down lidar in the sensor frame (x forward = up). Spot points coloured by height; ground-level dropped."""
    rb = robot(r.id)
    dec = dec_(r.id)
    topic = emb.lidar3d_topic(rb) or emb.scan_topic(rb)
    three_d = topic == emb.lidar3d_topic(rb)
    i = pick(r, topic, t_ns, "last_before")
    if i is None:
        return placeholder((px, px), "lidar: no scan before t"), None
    if three_d:
        p = dec.points_xyz(topic, i)
        p = p[np.isfinite(p).all(1) & (p[:, 2] > -0.35) & (p[:, 2] < 1.5)]  # sensor ~0.5 m up: drop ground returns
        xy, z = p[:, :2], p[:, 2]
    else:
        xy, _ = dec.scan_xy(topic, i)
        z = np.zeros(len(xy))
    s = px / (2 * rng)
    img = Image.new("RGB", (px, px), (12, 12, 20))
    d = ImageDraw.Draw(img)
    for ring in range(2, int(rng) + 1, 2):
        d.ellipse([px / 2 - ring * s, px / 2 - ring * s, px / 2 + ring * s, px / 2 + ring * s], outline=(45, 45, 60))
    u = (px / 2 - xy[:, 1] * s).astype(int)
    v = (px / 2 - xy[:, 0] * s).astype(int)
    ok = (u >= 0) & (u < px) & (v >= 0) & (v < px)
    zc = np.clip((z + 0.35) / 1.85, 0, 1)
    colour = np.stack([80 + 175 * zc, 220 - 60 * zc, 255 - 200 * zc], 1).astype(np.uint8)
    arr = np.asarray(img).copy()
    arr[v[ok], u[ok]] = colour[ok]
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    fp = emb.footprint(rb)
    d.rectangle([px / 2 - fp.width_m / 2 * s, px / 2 - fp.length_m / 2 * s,
                 px / 2 + fp.width_m / 2 * s, px / 2 + fp.length_m / 2 * s], outline=(255, 180, 60), width=2)
    c = emb.corridor(rb)
    fe = fp.length_m / 2
    d.rectangle([px / 2 - c.half_width_m * s, px / 2 - (fe + c.length_m) * s,
                 px / 2 + c.half_width_m * s, px / 2 - fe * s], outline=(120, 255, 120))
    rf = ref(r, topic, i, t_ns, native=True)
    d.text((6, px - 40), f"lidar t={rf['t']:.2f}s  rings 2 m  ahead = up", fill=(220, 220, 220), font=SMALL)
    d.text((6, px - 20), "footprint NOMINAL, corridor ESTIMATED" + ("; colour = height" if three_d else ""),
           fill=(160, 160, 160), font=SMALL)
    return img, rf


def signal_series(r: Recording, t0_ns: int, t1_ns: int) -> tuple[dict, list[dict]]:
    """COMPUTED from raw odom and lidar messages with the same conventions as the motion/clearance providers
    (profile twist frame, profile speed smoother with the intake-measured gait period, profile clearance sensor)."""
    rb = robot(r.id)
    prof = emb.prof(rb)
    dec = dec_(r.id)
    otopic = emb.odom_topic(rb)
    tl = r.topics[otopic]
    # smooth over a lead-in so the trailing window is full at t0
    sl_all = tl.range(t0_ns - int(2e9), t1_ns)
    sl = tl.range(t0_ns, t1_ns)
    raw = odom_arrays(dec, otopic, prof, sl_all)
    smooth, how = smooth_speed(raw["t_ns"], raw["speed_mps"], prof, E.load_intake(BUNDLE, r.id))
    keep = raw["t_ns"] >= t0_ns
    t_od = np.array([r.t_rel(int(x)) for x in raw["t_ns"][keep]])
    speed, yaw, smooth = raw["speed_mps"][keep], raw["yaw_rate_dps"][keep], smooth[keep]
    heading = np.concatenate([[0.0], np.cumsum(yaw[1:] * np.diff(t_od))]) if len(yaw) else yaw
    cs = next(x for x in prof.sensors if x.name == prof.clearance_sensor)
    ctopic = cs.topics[0]
    ssl = r.topics[ctopic].range(t0_ns, t1_ns)
    c = emb.corridor(rb)
    front, corr, t_sc = [], [], []
    for i in range(ssl.start, ssl.stop):
        xy, a, _, _ = dec.obstacle_xy(prof, i, by_time=False)
        rr = np.hypot(xy[:, 0], xy[:, 1])
        sel = np.abs(a) < np.radians(30)
        front.append(float(rr[sel].min()) if sel.any() else np.nan)
        front_edge = emb.footprint(rb).length_m / 2
        box = (xy[:, 0] > front_edge) & (xy[:, 0] < front_edge + c.length_m) & (np.abs(xy[:, 1]) < c.half_width_m)
        corr.append(float(xy[box, 0].min() - front_edge) if box.any() else c.length_m)  # length = corridor clear
        t_sc.append(r.t_rel(int(r.topics[ctopic].log_ns[i])))
    refs = []
    for topic, s_ in ((otopic, sl), (ctopic, ssl)):
        if s_.stop > s_.start:
            refs += [ref(r, topic, s_.start, None, native=True), ref(r, topic, s_.stop - 1, None, native=True)]
    return {"speed_mps": (t_od, speed), "speed_smooth_mps": (t_od, smooth), "yaw_rate_dps": (t_od, yaw),
            "heading_change_deg": (t_od, heading), "min_range_front_m": (np.array(t_sc), np.array(front)),
            "min_range_corridor_m": (np.array(t_sc), np.array(corr)), "_smoother": how}, refs


def heading_summary(t: np.ndarray, hdg: np.ndarray, t0: float, t1: float) -> dict:
    m = (t >= t0) & (t <= t1)
    if m.sum() < 2:
        return {}
    tt, hh = t[m], hdg[m]
    lo = np.searchsorted(tt, tt - 1.0)
    one_s = np.abs(hh - hh[lo])
    return {"net_deg": round(float(hh[-1] - hh[0]), 1), "max_abs_excursion_deg": round(float(np.abs(hh - hh[0]).max()), 1),
            "max_change_in_any_1s_deg": round(float(one_s.max()), 1)}


def signal_plot(series: dict, fields: list[str], span: tuple[float, float] | None = None, mark_t: float | None = None,
                size=(9, 3.2)) -> Image.Image:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(fields), 1, figsize=(size[0], size[1] * len(fields) / 2), sharex=True, dpi=90)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, fields):
        t, y = series[f]
        if f == "speed_mps" and "speed_smooth_mps" in series:
            ax.plot(t, y, lw=0.8, alpha=0.5, label="raw odometry")
            ax.plot(*series["speed_smooth_mps"], lw=1.8, label=series.get("_smoother", "smoothed"))
            ax.legend(fontsize=7, loc="lower left")
        else:
            ax.plot(t, y, lw=1.4)
        ax.set_ylabel(f, fontsize=8)
        ax.grid(alpha=0.3)
        if span is not None:
            ax.axvspan(*span, color="orange", alpha=0.15)
        if mark_t is not None:
            ax.axvline(mark_t, color="r", lw=0.8)
    axes[-1].set_xlabel("t (s from recording start); shaded = window;  COMPUTED from raw odom/scan")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return Image.open(buf).convert("RGB")


# ---------- commands ----------

def cmd_recordings(a) -> None:
    tags = {}
    with open(SCAND_ROOT / "SCAND_index.csv", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tags[row.get("FileName", "").strip()] = row.get("Tags", "")
    out = []
    for name, stem in RECORDINGS.items():
        r = rec_(name)
        out.append({
            "recording": name, "robot": robot(name), "duration_s": round((r.end_ns - r.start_ns) / 1e9, 1),
            "windows": len(windows(name, (r.end_ns - r.start_ns) / 1e9)),
            "topics": {t: {"n": len(tl), "hz": round(len(tl) / max(1e-9, (tl.log_ns[-1] - tl.log_ns[0]) / 1e9), 1)}
                       for t, tl in r.topics.items() if len(tl) > 1},
            "tags_recording_level_not_evidence": tags.get(stem, ""),
        })
    emit("recordings", {}, {"recordings": out, "embodiment_profiles": emb.cards()})


def cmd_coverage(a) -> None:
    r = rec_(a.rec)
    t0, t1 = r.t_abs(a.t0 or 0), r.t_abs(a.t1) if a.t1 is not None else r.end_ns
    out = {}
    for t, tl in r.topics.items():
        sl = tl.range(t0, t1)
        ts = tl.log_ns[sl]
        if len(ts) < 2:
            out[t] = {"n": int(len(ts))}
            continue
        gaps = np.diff(ts) / 1e9
        period = float(np.median(gaps))
        big = np.where(gaps > 2 * period)[0]
        entry = {"n": int(len(ts)), "hz": round(1 / period, 2), "max_gap_s": round(float(gaps.max()), 3),
                 "gaps_gt_2x_period": [[round(r.t_rel(int(ts[k])), 2), round(float(gaps[k]), 3)] for k in big[:20]]}
        h = tl.header_ns[sl]
        if (h != NO_HEADER).all():
            entry["median_capture_to_receipt_ms"] = round(float(np.median(ts - h)) / 1e6, 1)
        out[t] = entry
    emit("coverage", vars(a), {"recording": a.rec, "t0": a.t0, "t1": a.t1, "topics": out})


def cmd_frame(a) -> None:
    r = rec_(a.rec)
    topic = cam_topic(r, a.cam)
    t = r.t_abs(a.t)
    i = pick(r, topic, t, a.mode)
    if i is None:
        raise SystemExit(f"no {topic} message before t={a.t}")
    _, data = compressed_image(r.read(topic, i).data)
    rot = rotation_for(r, topic)
    ordn = r.topics[topic].ordinal[i]
    if a.crop:
        x0, y0, x1, y1 = (int(v) for v in a.crop.split(","))
        im = image(r, topic, i).crop((x0, y0, x1, y1))
        scale = max(1, 640 // max(1, x1 - x0))
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)  # pixel-replicated zoom: no new detail
        p = out_dir() / f"frame_{a.rec}_{a.cam}_{ordn}_crop_{x0}_{y0}_{x1}_{y1}.png"
        im.save(p)
    elif rot:  # upright for viewing; PNG keeps the decoded pixels exactly
        p = out_dir() / f"frame_{a.rec}_{a.cam}_{ordn}.png"
        image(r, topic, i).save(p)
    else:  # original encoded bytes: native resolution, no re-encode
        p = out_dir() / f"frame_{a.rec}_{a.cam}_{ordn}.jpg"
        p.write_bytes(data)
    warnings: list[str] = []
    rf = ref(r, topic, i, t, native=True, warnings=warnings)
    emit("frame", vars(a), {"path": str(p), "camera": cam_name(r, topic), "size": Image.open(p).size,
                            "display_rotation_deg": rot, "warnings": warnings, "refs": [rf]})


def cmd_step(a) -> None:
    r = rec_(a.rec)
    topic = cam_topic(r, a.cam)
    tl = r.topics[topic]
    t = r.t_abs(a.t)
    i = pick(r, topic, t, "nearest")
    if a.every:
        sign = 1 if a.dir == "fwd" else -1
        idx = sorted({tl.nearest(int(tl.log_ns[i] + sign * k * a.every * 1e9)) for k in range(a.n)})
    else:
        idx = [i + k * (1 if a.dir == "fwd" else -1) for k in range(a.n)]
        idx = sorted(j for j in idx if 0 <= j < len(tl))
    native = a.n <= 4
    tiles, warnings = [], []
    period = float(np.median(np.diff(tl.log_ns))) if len(tl) > 1 else 0
    for j, k in zip(idx, idx[1:]):
        if tl.log_ns[k] - tl.log_ns[j] > 2 * period:
            warnings.append(f"gap of {(tl.log_ns[k] - tl.log_ns[j]) / 1e9:.2f}s between {r.t_rel(int(tl.log_ns[j])):.2f}s "
                            f"and {r.t_rel(int(tl.log_ns[k])):.2f}s")
    for j in idx:
        im = image(r, topic, j)
        if not native:
            im = im.resize((im.width // 2, im.height // 2))
        tiles.append(label(im, f"{r.t_rel(int(tl.log_ns[j])):.3f}s #{tl.ordinal[j]}"))
    path = save(grid(tiles, cols=2 if native else 4), f"step_{a.rec}_{a.cam}_{a.t:.2f}_{a.dir}{a.n}.jpg")
    emit("step", vars(a), {"path": path, "native": native, "warnings": warnings,
                           "note": "n<=4 renders full-resolution tiles (native evidence); larger n is half-res",
                           "refs": [ref(r, topic, j, t, native=native) for j in idx]})


def cmd_sync(a) -> None:
    r = rec_(a.rec)
    t = r.t_abs(a.t)
    rb = robot(a.rec)
    refs, tiles, warnings = [], [], []
    ftopic = emb.front_camera_topic(rb)
    tiles.append(cam_tile(r, ftopic, t, a.mode, (640, 360), [], warnings, native=False))
    fi = pick(r, ftopic, t, a.mode)
    front_path = None
    if fi is not None:  # the full-resolution front frame, saved alongside: its ref is native
        _, data = compressed_image(r.read(ftopic, fi).data)
        front_path = out_dir() / f"frame_{a.rec}_front_{r.topics[ftopic].ordinal[fi]}.jpg"
        front_path.write_bytes(data)
        refs.append(ref(r, ftopic, fi, t, native=True))
    for topic in emb.body_camera_topics(rb):
        tiles.append(cam_tile(r, topic, t, a.mode, (640, 360), refs, warnings, native=True))
    b, bref = bev(r, t, px=360)
    tiles.append(b.resize((640, 360)))
    if bref:
        refs.append(bref)
    path = save(grid(tiles, cols=2), f"sync_{a.rec}_{a.t:.2f}_{a.mode}.jpg")
    emit("sync", vars(a), {"path": path, "front_native_path": str(front_path) if front_path else None,
                           "mode": a.mode, "warnings": warnings,
                           "note": "front ref is native (full frame saved at front_native_path); body cams full width. 'age' = "
                                   "capture-to-receipt latency: the body image shows the scene at header_t",
                           "refs": refs})


def cmd_strip(a) -> None:
    r = rec_(a.rec)
    topic = cam_topic(r, a.cam)
    ts = np.arange(a.t0, a.t1 + 1e-9, 1.0 / a.hz)
    if len(ts) > 48:
        raise SystemExit(f"{len(ts)} tiles requested; keep strips to <= 48 (narrow the range or lower --hz)")
    refs, tiles, warnings = [], [], []
    for ts_ in ts:
        tt = r.t_abs(float(ts_))
        i = pick(r, topic, tt, "nearest")
        im = image(r, topic, i)
        rf = ref(r, topic, i, tt, native=False, warnings=warnings)
        tiles.append(label(im.resize((im.width // 4, im.height // 4)), f"{rf['t']:.1f}s", SMALL))
        refs.append(rf)
    path = save(grid(tiles, cols=6), f"strip_{a.rec}_{a.cam}_{a.t0:.1f}-{a.t1:.1f}@{a.hz}.jpg")
    emit("strip", vars(a), {"path": path, "warnings": warnings,
                            "note": "proposal only: quarter-res tiles are not native evidence", "refs": refs})


def cmd_lidar(a) -> None:
    r = rec_(a.rec)
    t = r.t_abs(a.t)
    if a.view == "scan":
        topic = emb.scan_topic(robot(a.rec))
        i = pick(r, topic, t, "last_before")
        xy, ang = dec_(a.rec).scan_xy(topic, i)
        img = signal_plot({"range_m": (np.degrees(ang), np.hypot(*xy.T))}, ["range_m"], size=(9, 5))
        path = save(img, f"scan_{a.rec}_{a.t:.2f}.jpg")
        emit("lidar", vars(a), {"path": path, "x_axis": "beam angle deg (0 = forward, + = left)",
                                "refs": [ref(r, topic, i, t, native=True)]})
        return
    img, bref = bev(r, t, rng=a.range)
    path = save(img, f"bev_{a.rec}_{a.t:.2f}_{a.range:.0f}m.jpg")
    emit("lidar", vars(a), {"path": path, "clusters": lidar_clusters(r, t, a.range),
                            "note": "clusters: 0.25 m grid connected components with >= 6 returns; range from the "
                                    "sensor; bearing + = left; in_corridor uses the ESTIMATED corridor box",
                            "refs": [bref] if bref else []})


def lidar_clusters(r: Recording, t_ns: int, rng: float) -> list[dict]:
    from scipy import ndimage
    rb = robot(r.id)
    prof = emb.prof(rb)
    xy, _, _, _ = dec_(r.id).obstacle_xy(prof, t_ns)
    xy = xy[np.hypot(xy[:, 0], xy[:, 1]) < rng]
    if not len(xy):
        return []
    cell = 0.25
    ij = np.floor((xy + rng) / cell).astype(int)
    n = int(np.ceil(2 * rng / cell)) + 1
    grid = np.zeros((n, n), dtype=np.int32)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1)
    lab, k = ndimage.label(grid > 0, structure=np.ones((3, 3)))
    point_lab = lab[ij[:, 0], ij[:, 1]]
    c = emb.corridor(rb)
    fe = emb.footprint(rb).length_m / 2
    out = []
    for L in range(1, k + 1):
        pts = xy[point_lab == L]
        if len(pts) < 6:
            continue
        cx, cy = pts.mean(0)
        near = pts[np.argmin(np.hypot(pts[:, 0], pts[:, 1]))]
        out.append({"range_m": round(float(np.hypot(*near)), 2), "bearing_deg": round(float(np.degrees(np.arctan2(cy, cx))), 1),
                    "extent_m": round(float(np.ptp(pts, 0).max()), 2), "points": int(len(pts)),
                    "in_corridor": bool(((pts[:, 0] > fe) & (pts[:, 0] < fe + c.length_m) &
                                         (np.abs(pts[:, 1]) < c.half_width_m)).any())})
    return sorted(out, key=lambda d: d["range_m"])[:25]


def cmd_signals(a) -> None:
    r = rec_(a.rec)
    s, refs = signal_series(r, r.t_abs(a.t0), r.t_abs(a.t1))
    fields = a.fields.split(",")
    bad = [f for f in fields if f not in s]
    if bad:
        raise SystemExit(f"unknown fields {bad}; available: {sorted(s)}")
    img = signal_plot(s, fields, mark_t=a.mark)
    path = save(img, f"signals_{a.rec}_{a.t0:.1f}-{a.t1:.1f}.jpg")
    summary = {}
    for f in fields:
        t, y = s[f]
        if len(y) and np.isfinite(y).any():
            k_min, k_max = int(np.nanargmin(y)), int(np.nanargmax(y))
            summary[f] = {"min": [round(float(y[k_min]), 3), round(float(t[k_min]), 2)],
                          "max": [round(float(y[k_max]), 3), round(float(t[k_max]), 2)],
                          "first": round(float(y[0]), 3), "last": round(float(y[-1]), 3)}
    summary["heading"] = heading_summary(*s["heading_change_deg"], a.t0, a.t1)
    emit("signals", vars(a), {"path": path, "label_source": "COMPUTED", "summary": summary,
                              "smoother": s.pop("_smoother"),
                              "note": "min/max as [value, t]. Use speed_smooth_mps (causal; see `smoother`) for "
                                      "slowdowns. heading = integrated yaw rate (+ = left). min_range_corridor_m = "
                                      "nearest lidar return inside the corridor box (5.0 = clear); min_range_front_m = ±30° cone. "
                                      "refs = first/last odom and scan messages used (cite for turns/speeds).",
                              "refs": refs})


def cmd_window(a) -> None:
    rec, t0, t1 = window_span_s(a.window_id)
    r = rec_(rec)
    rb = robot(rec)
    front = emb.front_camera_topic(rb)
    refs, warnings = [], []
    times = (t0, (t0 + t1) / 2, t1)
    fronts = []
    for ts in times:
        tile = cam_tile(r, front, r.t_abs(ts), "nearest", (640, 360), refs, warnings, native=False)
        fronts.append(front_corridor_overlay(tile, rb))
    bevs = []
    for ts in times:
        b, bref = bev(r, r.t_abs(ts), px=420)
        bevs.append(b)
        if bref:
            refs.append(bref)
    parts = [grid(fronts, cols=3), grid([x.resize((640, 640)) for x in bevs], cols=3)]
    if emb.body_camera_topics(rb):
        body = [cam_tile(r, topic, r.t_abs(times[1]), "nearest", (384, 288), refs, warnings, native=False)
                for topic in emb.body_camera_topics(rb)]
        parts.append(grid(body, cols=5))
    s, _ = signal_series(r, r.t_abs(max(0, t0 - 2)), r.t_abs(t1 + 2))
    _, srefs = signal_series(r, r.t_abs(t0), r.t_abs(t1))  # cite only messages inside the window
    refs += srefs
    parts.append(signal_plot(s, ["speed_mps", "yaw_rate_dps", "min_range_corridor_m"], span=(t0, t1)))
    canvas = label(stack(parts), f"window {a.window_id}  [{t0:.0f}s, {t1:.0f}s]  rows: front | lidar | "
                                 f"{'body (mid) | ' if emb.body_camera_topics(rb) else ''}signals")
    path = save(canvas, f"window_{a.window_id.replace(':', '_')}.jpg")
    emit("window", vars(a), {"path": path, "window": a.window_id, "span_s": [t0, t1], "warnings": warnings,
                             "heading": heading_summary(*s["heading_change_deg"], t0, t1),
                             "note": "composite for proposing; confirm claims with frame/step/sync/lidar. Front and "
                                     "lidar at start/mid/end; body cameras at mid-window.",
                             "refs": refs})


def cmd_windows(a) -> None:
    r = rec_(a.rec)
    dur = (r.end_ns - r.start_ns) / 1e9
    ids = segments(a.rec, dur) if a.segments else windows(a.rec, dur)
    emit("windows", vars(a), {"recording": a.rec, "count": len(ids), "ids": ids,
                              "note": f"id Rec:EEEE = [EEEE-{WINDOW_S} s, EEEE s]"})


def cmd_message(a) -> None:
    rec, topic, ordinal = parse_mid(a.mid)
    r = rec_(rec)
    i = r.index_of(topic, ordinal)
    raw = r.read(topic, i).data
    ok = hashlib.sha256(raw).digest()[:16] == r.topics[topic].sha128[i].tobytes()
    m = dec_(rec).msg(topic, i)

    def conv(x):
        if isinstance(x, np.ndarray):
            return {"array_len": int(x.size), "head": x.ravel()[:8].tolist()}
        if isinstance(x, (bytes, bytearray)):
            return {"bytes_len": len(x)}
        if isinstance(x, list):
            return [conv(v) for v in x[:16]] + ([f"... {len(x) - 16} more"] if len(x) > 16 else [])
        if hasattr(x, "__dataclass_fields__"):
            return {k: conv(getattr(x, k)) for k in x.__dataclass_fields__ if k != "__msgtype__"}
        return x

    emit("message", vars(a), {"mid": a.mid, "msgtype": dec_(rec).types[topic], "payload_hash_ok": ok,
                              "fields": conv(m), "refs": [ref(r, topic, i, None, native=True)]})


def main(argv: list[str] | None = None) -> None:
    from alloy_index.scandq import labels

    ap = argparse.ArgumentParser(prog="scandq", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("recordings", help="recordings, topics, rates, embodiment cards, recording-level tags")
    p = sp.add_parser("coverage", help="per-topic counts, rates, gaps and capture-to-receipt latency in a range")
    p.add_argument("rec"); p.add_argument("--t0", type=float); p.add_argument("--t1", type=float)
    p = sp.add_parser("windows", help="list window ids (or --segments: non-overlapping 4 s label segments)")
    p.add_argument("rec"); p.add_argument("--segments", action="store_true")
    p = sp.add_parser("frame", help="one camera frame at native resolution (native evidence)")
    p.add_argument("rec"); p.add_argument("--t", type=float, required=True); p.add_argument("--cam", default="front")
    p.add_argument("--mode", choices=["nearest", "last_before"], default="nearest")
    p.add_argument("--crop", help="x0,y0,x1,y1 in native (upright) pixels; zoomed by pixel replication")
    p = sp.add_parser("step", help="N consecutive native-rate frames (n<=4: native evidence)")
    p.add_argument("rec"); p.add_argument("--t", type=float, required=True); p.add_argument("--cam", default="front")
    p.add_argument("--dir", choices=["back", "fwd"], default="fwd"); p.add_argument("--n", type=int, default=4)
    p.add_argument("--every", type=float, help="seconds between frames (default: consecutive frames)")
    p = sp.add_parser("sync", help="every camera + lidar at one instant, each with its capture age")
    p.add_argument("rec"); p.add_argument("--t", type=float, required=True)
    p.add_argument("--mode", choices=["nearest", "last_before"], default="nearest")
    p = sp.add_parser("strip", help="timestamped contact sheet (proposal only)")
    p.add_argument("rec"); p.add_argument("--t0", type=float, required=True); p.add_argument("--t1", type=float, required=True)
    p.add_argument("--cam", default="front"); p.add_argument("--hz", type=float, default=1.0)
    p = sp.add_parser("lidar", help="lidar BEV (or --view scan: range vs angle) with nominal footprint/corridor")
    p.add_argument("rec"); p.add_argument("--t", type=float, required=True)
    p.add_argument("--view", choices=["bev", "scan"], default="bev"); p.add_argument("--range", type=float, default=10.0)
    p = sp.add_parser("signals", help="speed / yaw / heading change / ranges plot + summary (COMPUTED)")
    p.add_argument("rec"); p.add_argument("--t0", type=float, required=True); p.add_argument("--t1", type=float, required=True)
    p.add_argument("--fields", default="speed_mps,yaw_rate_dps,heading_change_deg,min_range_corridor_m")
    p.add_argument("--mark", type=float)
    p = sp.add_parser("window", help="composite labelling view of one window (proposal only)")
    p.add_argument("window_id")
    p = sp.add_parser("message", help="decoded fields of one message + payload hash check")
    p.add_argument("mid")
    labels.add_parser(sp)
    a = ap.parse_args(argv)
    handler = {"recordings": cmd_recordings, "coverage": cmd_coverage, "windows": cmd_windows, "frame": cmd_frame,
               "step": cmd_step, "sync": cmd_sync, "strip": cmd_strip, "lidar": cmd_lidar, "signals": cmd_signals,
               "window": cmd_window, "message": cmd_message, "label": labels.cmd_label}[a.cmd]
    try:
        handler(a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(json.dumps({"error": e.code}), file=sys.stderr)
            sys.exit(2)
        raise


if __name__ == "__main__":
    main()
