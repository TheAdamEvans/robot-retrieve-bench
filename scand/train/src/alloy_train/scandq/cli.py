"""scandq — anchored, audited views over the source-of-truth MCAP bags, for the labeller.

Every view is addressed by (rec, t seconds from recording start), a window_id (`Rec:EEEE`, a 4 s window ending at
second EEEE) or a MessageId string (`Rec/topic#ordinal`). Every view prints JSON listing the MessageIds it rendered,
their times, and Δt to the requested time, and appends to annotations/audit.jsonl. Images are written under
annotations/renders/<job>/ for the agent to open with Read. A ref counts as native evidence only when the view
rendered it at full resolution (`native: true`).

Set SCANDQ_JOB=<job_id> so the audit log and label shards are attributed to the job.
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

from alloy_server.catalog.windows import WINDOW_S, parse_window_id, segments, window_span_s, windows
from alloy_server.io.ros1 import compressed_image
from alloy_server.timeline.store import Recording, mid_str, parse_mid
from alloy_train import embodiment as emb
from alloy_train.decode import Decoder, odom_arrays
from alloy_train.recordings import RECORDINGS, SCAND_ROOT, robot

BUNDLE = Path(os.environ.get("SCANDQ_BUNDLE", SCAND_ROOT / "bundles" / "dev"))
ANN = Path(os.environ.get("SCANDQ_ANNOTATIONS", SCAND_ROOT / "annotations"))
JOB = os.environ.get("SCANDQ_JOB", "adhoc")
FONT = ImageFont.load_default(size=18)


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
    d = ANN / "renders" / JOB
    d.mkdir(parents=True, exist_ok=True)
    return d


def ref(r: Recording, topic: str, i: int, t_req_ns: int | None, native: bool) -> dict:
    tl = r.topics[topic]
    d = {"mid": mid_str(r.message_id(topic, i)), "t": round(r.t_rel(int(tl.log_ns[i])), 3), "native": native}
    if tl.header_ns[i] != np.iinfo(np.int64).min:
        d["header_t"] = round(r.t_rel(int(tl.header_ns[i])), 3)
    if t_req_ns is not None:
        d["dt_ms"] = round((int(tl.log_ns[i]) - t_req_ns) / 1e6, 1)
    return d


def emit(call: str, args: dict, result: dict) -> None:
    ANN.mkdir(parents=True, exist_ok=True)
    refs = result.get("refs", [])
    with open(ANN / "audit.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.time(), "job": JOB, "call": call, "args": args,
                            "refs": [{"mid": x["mid"], "native": x["native"]} for x in refs]}) + "\n")
    print(json.dumps(result, indent=1))


def pick(r: Recording, topic: str, t_ns: int, mode: str) -> int:
    tl = r.topics[topic]
    i = tl.last_before(t_ns, inclusive=True) if mode == "last_before" else tl.nearest(t_ns)
    if i is None:
        raise SystemExit(f"no {topic} message {'at or before' if mode == 'last_before' else 'near'} t")
    return i


# Spot body cameras are mounted rotated; views show them upright (lossless transpose) and say so.
DISPLAY_ROTATION = {"frontleft": -90, "frontright": -90, "right": 180}


def rotation_for(topic: str) -> int:
    parts = topic.split("/")
    return DISPLAY_ROTATION.get(parts[3], 0) if topic.startswith("/spot/camera/") else 0


def image(r: Recording, topic: str, i: int) -> Image.Image:
    _, data = compressed_image(r.read(topic, i).data)
    im = Image.open(io.BytesIO(data)).convert("RGB")
    rot = rotation_for(topic)
    return im.rotate(rot, expand=True) if rot else im


def cam_topic(r: Recording, cam: str) -> str:
    rb = robot(r.id)
    if cam == "front":
        return emb.front_camera_topic(rb)
    topic = f"/spot/camera/{cam.removeprefix('body_')}/image/compressed"
    if topic not in r.topics:
        raise SystemExit(f"camera {cam!r} not on {rb}; cameras: front" +
                         ("".join(f", body_{c}" for c in emb.SPOT_BODY_CAMS) if rb == "spot" else ""))
    return topic


def label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 10 * len(text), 26], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 0), font=FONT)
    return img


def grid(tiles: list[Image.Image], cols: int) -> Image.Image:
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    g = Image.new("RGB", (w * cols, h * rows), (30, 30, 30))
    for k, t in enumerate(tiles):
        g.paste(t.resize((w, h)), ((k % cols) * w, (k // cols) * h))
    return g


def save(img: Image.Image, name: str) -> str:
    p = out_dir() / name
    img.save(p, quality=90)
    return str(p)


# ---------- renders ----------

def bev(r: Recording, t_ns: int, rng: float = 10.0, px: int = 600) -> tuple[Image.Image, dict]:
    """Top-down lidar render in the sensor frame (x forward, up in the image) with nominal footprint + corridor."""
    rb = robot(r.id)
    dec = dec_(r.id)
    topic = "/velodyne_points" if rb == "spot" else "/velodyne_2dscan"
    i = pick(r, topic, t_ns, "last_before")
    if rb == "spot":
        p = dec.points_xyz(topic, i)
        p = p[np.isfinite(p).all(1) & (p[:, 2] > -0.6) & (p[:, 2] < 1.5)]
        xy = p[:, :2]
    else:
        xy, _ = dec.scan_xy(topic, i)
    s = px / (2 * rng)
    img = Image.new("RGB", (px, px), (12, 12, 20))
    d = ImageDraw.Draw(img)
    for ring in range(2, int(rng) + 1, 2):
        d.ellipse([px / 2 - ring * s, px / 2 - ring * s, px / 2 + ring * s, px / 2 + ring * s], outline=(45, 45, 60))
    u = (px / 2 - xy[:, 1] * s).astype(int)
    v = (px / 2 - xy[:, 0] * s).astype(int)
    ok = (u >= 0) & (u < px) & (v >= 0) & (v < px)
    arr = np.asarray(img).copy()
    arr[v[ok], u[ok]] = (120, 220, 255)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    fp = emb.CARDS[rb]["footprint_m"]
    d.rectangle([px / 2 - fp["width"] / 2 * s, px / 2 - fp["length"] / 2 * s,
                 px / 2 + fp["width"] / 2 * s, px / 2 + fp["length"] / 2 * s], outline=(255, 180, 60), width=2)
    c = emb.CORRIDOR[rb]
    d.rectangle([px / 2 - c["half_width_m"] * s, px / 2 - c["length_m"] * s,
                 px / 2 + c["half_width_m"] * s, px / 2], outline=(120, 255, 120))
    d.text((6, px - 44), f"{r.id} {topic} t={r.t_rel(int(r.topics[topic].log_ns[i])):.2f}s  rings=2m",
           fill=(220, 220, 220), font=FONT)
    d.text((6, px - 22), "footprint NOMINAL / corridor ESTIMATED (green)", fill=(160, 160, 160), font=FONT)
    return img, ref(r, topic, i, t_ns, native=True)


def signal_series(r: Recording, t0_ns: int, t1_ns: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """COMPUTED directly from raw odom and scan messages (independent of the indexing providers)."""
    rb = robot(r.id)
    dec = dec_(r.id)
    otopic = emb.odom_topic(rb)
    sl = r.topics[otopic].range(t0_ns, t1_ns)
    tl = r.topics[otopic]
    speed, yaw = [], []
    for i in range(sl.start, sl.stop):
        m = dec.msg(otopic, i)
        v = m.twist.twist.linear
        speed.append(np.hypot(v.x, v.y) if rb == "spot" else abs(v.x))
        yaw.append(np.degrees(m.twist.twist.angular.z))
    t_od = np.array([r.t_rel(int(x)) for x in tl.log_ns[sl]])
    stopic = emb.scan_topic(rb)
    ssl = r.topics[stopic].range(t0_ns, t1_ns)
    front, t_sc = [], []
    for i in range(ssl.start, ssl.stop):
        xy, a = dec.scan_xy(stopic, i, rmin=0.5)
        sel = np.abs(a) < np.radians(30)
        front.append(float(np.hypot(*xy[sel].T).min()) if sel.any() else np.nan)
        t_sc.append(r.t_rel(int(r.topics[stopic].log_ns[i])))
    speed = np.array(speed)
    smooth = np.array([np.median(speed[(t_od > tt - 1.0) & (t_od <= tt)]) for tt in t_od]) if len(speed) else speed
    return {"speed_mps": (t_od, speed), "speed_smooth_mps": (t_od, smooth), "yaw_rate_dps": (t_od, np.array(yaw)),
            "min_range_front_m": (np.array(t_sc), np.array(front))}


def signal_plot(series: dict, fields: list[str], mark_t: float | None = None, size=(9, 3.2)) -> Image.Image:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(fields), 1, figsize=(size[0], size[1] * len(fields) / 2), sharex=True, dpi=90)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, fields):
        t, y = series[f]
        if f == "speed_mps" and "speed_smooth_mps" in series:
            ax.plot(t, y, lw=0.8, alpha=0.5, label="raw (Spot gait oscillates)")
            ax.plot(*series["speed_smooth_mps"], lw=1.8, label="trailing 1 s median")
            ax.legend(fontsize=7, loc="lower left")
        else:
            ax.plot(t, y, lw=1.4)
        ax.set_ylabel(f, fontsize=8)
        ax.grid(alpha=0.3)
        if mark_t is not None:
            ax.axvline(mark_t, color="r", lw=0.8)
    axes[-1].set_xlabel("t (s from recording start)  —  COMPUTED from raw odom/scan")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return Image.open(buf).convert("RGB")


# ---------- commands ----------

def cmd_recordings(a) -> None:
    tags = {}
    with open(SCAND_ROOT.parent / "SCAND_index.csv", newline="", encoding="utf-8-sig") as f:
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
    emit("recordings", {}, {"recordings": out, "embodiment_cards": emb.CARDS})


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
        out[t] = {"n": int(len(ts)), "hz": round(1 / period, 2), "max_gap_s": round(float(gaps.max()), 3),
                  "gaps_gt_2x_period": [[round(r.t_rel(int(ts[k])), 2), round(float(gaps[k]), 3)] for k in big[:20]]}
    emit("coverage", vars(a), {"recording": a.rec, "t0": a.t0, "t1": a.t1, "topics": out})


def cmd_frame(a) -> None:
    r = rec_(a.rec)
    topic = cam_topic(r, a.cam)
    t = r.t_abs(a.t)
    i = pick(r, topic, t, a.mode)
    _, data = compressed_image(r.read(topic, i).data)
    rot = rotation_for(topic)
    if rot:  # upright for viewing; PNG keeps the decoded pixels exactly
        p = out_dir() / f"frame_{a.rec}_{a.cam}_{r.topics[topic].ordinal[i]}.png"
        image(r, topic, i).save(p)
    else:  # original encoded bytes: native resolution, no re-encode
        p = out_dir() / f"frame_{a.rec}_{a.cam}_{r.topics[topic].ordinal[i]}.jpg"
        p.write_bytes(data)
    emit("frame", vars(a), {"path": str(p), "size": Image.open(p).size, "display_rotation_deg": rot,
                            "refs": [ref(r, topic, i, t, native=True)]})


def cmd_step(a) -> None:
    r = rec_(a.rec)
    topic = cam_topic(r, a.cam)
    t = r.t_abs(a.t)
    i = pick(r, topic, t, "nearest")
    idx = [i + k * (1 if a.dir == "fwd" else -1) for k in range(a.n)]
    idx = sorted(j for j in idx if 0 <= j < len(r.topics[topic]))
    native = a.n <= 4
    tiles = []
    for j in idx:
        im = image(r, topic, j)
        if not native:
            im = im.resize((im.width // 2, im.height // 2))
        tiles.append(label(im, f"{r.t_rel(int(r.topics[topic].log_ns[j])):.3f}s #{r.topics[topic].ordinal[j]}"))
    path = save(grid(tiles, cols=2 if native else 4), f"step_{a.rec}_{a.cam}_{a.t:.2f}_{a.dir}{a.n}.jpg")
    emit("step", vars(a), {"path": path, "native": native,
                           "note": "n<=4 renders full-resolution tiles (native evidence); larger n is half-res",
                           "refs": [ref(r, topic, j, t, native=native) for j in idx]})


def cmd_sync(a) -> None:
    r = rec_(a.rec)
    t = r.t_abs(a.t)
    rb = robot(a.rec)
    refs, tiles = [], []
    front = emb.front_camera_topic(rb)
    i = pick(r, front, t, a.mode)
    tiles.append(label(image(r, front, i).resize((640, 360)), f"front {r.t_rel(int(r.topics[front].log_ns[i])):.2f}s"))
    refs.append(ref(r, front, i, t, native=False))
    for topic in emb.body_camera_topics(rb):
        j = pick(r, topic, t, a.mode)
        tiles.append(label(image(r, topic, j).resize((640, 360)),
                           f"{topic.split('/')[3]} {r.t_rel(int(r.topics[topic].log_ns[j])):.2f}s"))
        refs.append(ref(r, topic, j, t, native=True))  # 640x480 → shown at full width
    b, bref = bev(r, t, px=360)
    tiles.append(b.resize((640, 360)))
    refs.append(bref)
    path = save(grid(tiles, cols=2), f"sync_{a.rec}_{a.t:.2f}.jpg")
    emit("sync", vars(a), {"path": path, "mode": a.mode,
                           "note": "front tile is downscaled (use `frame` for native); body cams are full width",
                           "refs": refs})


def cmd_strip(a) -> None:
    r = rec_(a.rec)
    topic = cam_topic(r, a.cam)
    ts = np.arange(a.t0, a.t1 + 1e-9, 1.0 / a.hz)
    if len(ts) > 48:
        raise SystemExit(f"{len(ts)} tiles requested; keep strips to <= 48 (narrow the range or lower --hz)")
    refs, tiles = [], []
    for ts_ in ts:
        tt = r.t_abs(float(ts_))
        i = pick(r, topic, tt, "nearest")
        im = image(r, topic, i)
        tiles.append(label(im.resize((im.width // 4, im.height // 4)), f"{ts_:.1f}s"))
        refs.append(ref(r, topic, i, tt, native=False))
    path = save(grid(tiles, cols=6), f"strip_{a.rec}_{a.cam}_{a.t0:.1f}-{a.t1:.1f}@{a.hz}.jpg")
    emit("strip", vars(a), {"path": path, "note": "proposal only: quarter-res tiles are not native evidence",
                            "refs": refs})


def cmd_lidar(a) -> None:
    r = rec_(a.rec)
    t = r.t_abs(a.t)
    if a.view == "scan":
        rb = robot(a.rec)
        topic = emb.scan_topic(rb)
        i = pick(r, topic, t, "last_before")
        xy, ang = dec_(a.rec).scan_xy(topic, i)
        rr = np.hypot(*xy.T)
        img = signal_plot({"range_m": (np.degrees(ang), rr)}, ["range_m"], size=(9, 5))
        path = save(img, f"scan_{a.rec}_{a.t:.2f}.jpg")
        emit("lidar", vars(a), {"path": path, "x_axis": "beam angle deg (0 = forward)",
                                "refs": [ref(r, topic, i, t, native=True)]})
        return
    img, bref = bev(r, t, rng=a.range)
    path = save(img, f"bev_{a.rec}_{a.t:.2f}_{a.range:.0f}m.jpg")
    emit("lidar", vars(a), {"path": path, "refs": [bref]})


def cmd_signals(a) -> None:
    r = rec_(a.rec)
    s = signal_series(r, r.t_abs(a.t0), r.t_abs(a.t1))
    fields = a.fields.split(",")
    bad = [f for f in fields if f not in s]
    if bad:
        raise SystemExit(f"unknown fields {bad}; available: {sorted(s)}")
    img = signal_plot(s, fields, mark_t=a.mark)
    path = save(img, f"signals_{a.rec}_{a.t0:.1f}-{a.t1:.1f}.jpg")
    summary = {}
    for f in fields:
        t, y = s[f]
        if len(y):
            k_min, k_max = int(np.nanargmin(y)), int(np.nanargmax(y))
            summary[f] = {"min": [round(float(y[k_min]), 3), round(float(t[k_min]), 2)],
                          "max": [round(float(y[k_max]), 3), round(float(t[k_max]), 2)],
                          "first": round(float(y[0]), 3), "last": round(float(y[-1]), 3)}
    emit("signals", vars(a), {"path": path, "label_source": "COMPUTED", "summary": summary,
                              "note": "min/max given as [value, t]. speed = planar odom speed (raw oscillates with Spot's gait: use "
                                      "speed_smooth_mps, a causal trailing 1 s median, for slowdowns); "
                                      "min_range_front = min lidar range within ±30° of forward (sensor origin, "
                                      "not footprint edge)"})


def cmd_window(a) -> None:
    rec, t0, t1 = window_span_s(a.window_id)
    r = rec_(rec)
    rb = robot(rec)
    front = emb.front_camera_topic(rb)
    refs, tiles = [], []
    for ts in (t0, (t0 + t1) / 2, t1):
        tt = r.t_abs(ts)
        i = pick(r, front, tt, "nearest")
        tiles.append(label(image(r, front, i).resize((640, 360)), f"front {ts:.1f}s"))
        refs.append(ref(r, front, i, tt, native=False))
    tt = r.t_abs(t1)
    b, bref = bev(r, tt, px=360)
    tiles.append(b.resize((640, 360)))
    refs.append(bref)
    top = grid(tiles, cols=2)
    parts = [top]
    if rb == "spot":
        body = []
        for topic in emb.body_camera_topics(rb):
            j = pick(r, topic, tt, "nearest")
            body.append(label(image(r, topic, j).resize((256, 192)), topic.split("/")[3]))
            refs.append(ref(r, topic, j, tt, native=False))
        parts.append(grid(body, cols=5))
    sig = signal_plot(signal_series(r, r.t_abs(max(0, t0 - 2)), r.t_abs(t1 + 2)),
                      ["speed_mps", "yaw_rate_dps", "min_range_front_m"], mark_t=t1)
    parts.append(sig.resize((top.width, int(sig.height * top.width / sig.width))))
    W = max(p.width for p in parts)
    canvas = Image.new("RGB", (W, sum(p.height for p in parts)), (20, 20, 20))
    y = 0
    for p in parts:
        canvas.paste(p, (0, y))
        y += p.height
    label(canvas, f"window {a.window_id}  [{t0:.0f}s, {t1:.0f}s]")
    path = save(canvas, f"window_{a.window_id.replace(':', '_')}.jpg")
    emit("window", vars(a), {"path": path, "window": a.window_id, "span_s": [t0, t1],
                             "note": "composite view for proposing; confirm claims with frame/step/sync/lidar",
                             "refs": refs})


def cmd_windows(a) -> None:
    r = rec_(a.rec)
    dur = (r.end_ns - r.start_ns) / 1e9
    ids = segments(a.rec, dur) if a.segments else windows(a.rec, dur)
    emit("windows", vars(a), {"recording": a.rec, "count": len(ids), "ids": ids,
                              "note": f"id Rec:EEEE = [{'EEEE'}-{WINDOW_S}s, EEEE s]"})


def cmd_message(a) -> None:
    rec, topic, ordinal = parse_mid(a.mid)
    r = rec_(rec)
    i = r.index_of(topic, ordinal)
    raw = r.read(topic, i).data
    ok = hashlib.sha256(raw).digest()[:16] == r.topics[topic].sha128[i].tobytes()
    m = dec_(rec).msg(topic, i)

    def conv(x, depth=0):
        if isinstance(x, np.ndarray):
            return {"array_len": int(x.size), "head": x.ravel()[:8].tolist()}
        if isinstance(x, (bytes, bytearray)):
            return {"bytes_len": len(x)}
        if isinstance(x, list):
            return [conv(v, depth + 1) for v in x[:16]] + ([f"... {len(x) - 16} more"] if len(x) > 16 else [])
        if hasattr(x, "__dataclass_fields__"):
            return {k: conv(getattr(x, k), depth + 1) for k in x.__dataclass_fields__ if k != "__msgtype__"}
        return x

    emit("message", vars(a), {"mid": a.mid, "msgtype": dec_(rec).types[topic], "payload_hash_ok": ok,
                              "fields": conv(m), "refs": [ref(r, topic, i, None, native=True)]})


def main(argv: list[str] | None = None) -> None:
    from alloy_train.scandq import labels

    ap = argparse.ArgumentParser(prog="scandq", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("recordings", help="recordings, topics, rates, embodiment cards, recording-level tags")
    p = sp.add_parser("coverage", help="per-topic counts, rates and gaps in a time range")
    p.add_argument("rec"); p.add_argument("--t0", type=float); p.add_argument("--t1", type=float)
    p = sp.add_parser("windows", help="list window ids (or --segments: non-overlapping 4 s label segments)")
    p.add_argument("rec"); p.add_argument("--segments", action="store_true")
    p = sp.add_parser("frame", help="one camera frame at native resolution (native evidence)")
    p.add_argument("rec"); p.add_argument("--t", type=float, required=True); p.add_argument("--cam", default="front")
    p.add_argument("--mode", choices=["nearest", "last_before"], default="nearest")
    p = sp.add_parser("step", help="N consecutive native-rate frames (n<=4: native evidence)")
    p.add_argument("rec"); p.add_argument("--t", type=float, required=True); p.add_argument("--cam", default="front")
    p.add_argument("--dir", choices=["back", "fwd"], default="fwd"); p.add_argument("--n", type=int, default=4)
    p = sp.add_parser("sync", help="every camera + lidar BEV at one instant, each with its age")
    p.add_argument("rec"); p.add_argument("--t", type=float, required=True)
    p.add_argument("--mode", choices=["nearest", "last_before"], default="last_before")
    p = sp.add_parser("strip", help="timestamped contact sheet (proposal only)")
    p.add_argument("rec"); p.add_argument("--t0", type=float, required=True); p.add_argument("--t1", type=float, required=True)
    p.add_argument("--cam", default="front"); p.add_argument("--hz", type=float, default=1.0)
    p = sp.add_parser("lidar", help="lidar BEV (or --view scan: range vs angle) with nominal footprint/corridor")
    p.add_argument("rec"); p.add_argument("--t", type=float, required=True)
    p.add_argument("--view", choices=["bev", "scan"], default="bev"); p.add_argument("--range", type=float, default=10.0)
    p = sp.add_parser("signals", help="speed / yaw rate / front range plot + summary (COMPUTED)")
    p.add_argument("rec"); p.add_argument("--t0", type=float, required=True); p.add_argument("--t1", type=float, required=True)
    p.add_argument("--fields", default="speed_mps,speed_smooth_mps,yaw_rate_dps,min_range_front_m"); p.add_argument("--mark", type=float)
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
