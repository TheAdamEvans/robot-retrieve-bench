"""detections@1: RT-DETRv2 on the front camera at 10 Hz → counts, corridor occupancy [lo, hi], person tracks.

Corridor occupancy projects each person's foot point (box bottom-centre) onto the ground with a NOMINAL pinhole
model and counts it if it lands inside the corridor. The camera has no recorded intrinsics/extrinsics, so the
projection is evaluated over a grid of plausible (HFOV, height, pitch): lo = inside under every setting, hi = inside
under any. Results are ESTIMATED. Tracks come from a causal greedy-IoU tracker; a track is `confirmed` at its third
observation, which is its availability time.
"""
from __future__ import annotations

import io
import itertools
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from alloy_server.io.ros1 import compressed_image
from alloy_index.providers.common import write

VERSION = "detections@2"  # @2: embodiment context, stale leading frames skipped
MODEL = "PekingU/rtdetr_v2_r18vd"
REVISION = "5650961749fa93567c0d46fc7f43ea4f9e914107"
HZ = 10.0
THRESH = 0.5
VEHICLES = {"car", "truck", "bus", "motorcycle"}


def camera_bands(rb: str) -> dict:
    """(nominal, lo, hi) for HFOV, lens height and pitch, from the embodiment profile (ESTIMATED_BAND)."""
    from alloy_server.catalog.embodiment import EmbodimentContext, profile
    return EmbodimentContext(profile(rb)).camera_band()


def load():
    global _model
    if _model is None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        proc = AutoImageProcessor.from_pretrained(MODEL, revision=REVISION)
        model = AutoModelForObjectDetection.from_pretrained(MODEL, revision=REVISION).eval()
        _model = (proc, model)
    return _model


def in_corridor(foot_uv: np.ndarray, hfov: float, h: float, pitch: float, hw: float, length: float) -> np.ndarray:
    fx = (W / 2) / np.tan(np.radians(hfov / 2))
    below = np.arctan((foot_uv[:, 1] - H / 2) / fx) + np.radians(pitch)
    ok = below > np.radians(0.5)
    z = np.where(ok, h / np.tan(np.maximum(below, 1e-6)), np.inf)
    x = (foot_uv[:, 0] - W / 2) / fx * z
    return ok & (z <= length) & (np.abs(x) <= hw)


def corridor_counts(foot_uv: np.ndarray, rb: str) -> tuple[int, int, int]:
    if not len(foot_uv):
        return 0, 0, 0
    from alloy_server.catalog.embodiment import EmbodimentContext, profile
    ctx = EmbodimentContext(profile(rb))
    b, c = ctx.camera_band(), ctx.corridor
    nominal = int(in_corridor(foot_uv, b["hfov"][0], b["h"][0], b["pitch"][0], c.half_width_m, c.length_m).sum())
    masks = [in_corridor(foot_uv, f, hh, p, c.half_width_m, c.length_m)
             for f, hh, p in itertools.product(b["hfov"], b["h"], b["pitch"])]
    m = np.stack(masks)
    return nominal, int(m.all(0).sum()), int(m.any(0).sum())


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def track(frames: list[tuple[int, np.ndarray]], max_miss: int = 3, min_iou: float = 0.3):
    """Causal greedy IoU tracker. frames: [(t_ns, boxes)] → (per-track rows, per-frame track ids)."""
    active: dict[int, dict] = {}
    done, nxt = [], 0
    for t, boxes in frames:
        ids = list(active)
        assigned = set()
        if ids and len(boxes):
            m = iou(np.stack([active[i]["box"] for i in ids]), boxes)
            for k in np.argsort(-m, axis=None):
                ti, bi = divmod(int(k), m.shape[1])
                if m[ti, bi] < min_iou:
                    break
                if ids[ti] in assigned or bi in {active[j]["_b"] for j in assigned}:
                    continue
                tr = active[ids[ti]]
                if tr.get("_t") == t:
                    continue
                tr.update(box=boxes[bi], last=t, n=tr["n"] + 1, miss=0, _t=t, _b=bi)
                if tr["n"] == 3:
                    tr["confirmed"] = t
                assigned.add(ids[ti])
        used = {active[i]["_b"] for i in assigned}
        for bi in range(len(boxes)):
            if bi not in used:
                active[nxt] = {"box": boxes[bi], "first": t, "last": t, "n": 1, "miss": 0, "_t": t, "_b": bi,
                               "confirmed": None}
                nxt += 1
        for i in list(active):
            if active[i]["_t"] != t:
                active[i]["miss"] += 1
                if active[i]["miss"] > max_miss:
                    done.append((i, active.pop(i)))
    done += list(active.items())
    return [(i, tr["first"], tr["last"], tr["n"], tr["confirmed"]) for i, tr in done]


def build(bundle: Path, rec, ctx=None) -> dict:
    from alloy_server.catalog.embodiment import EmbodimentContext
    ctx = ctx or EmbodimentContext.for_recording(bundle, rec.id)
    rb = ctx.robot
    proc, model = load()
    topic = ctx.topic("front_camera")
    tl = rec.topics[topic]
    ticks = np.arange(rec.start_ns, rec.end_ns, int(1e9 / HZ))
    stale = ctx.stale_leading(topic)  # buffer-flushed frames from another moment: never detected on
    idx = sorted({tl.nearest(int(t)) for t in ticks} - set(range(stale)))
    lab = model.config.id2label
    rows = {k: [] for k in ("t_ns", "persons_visible_front", "persons_in_corridor", "persons_in_corridor_lo",
                            "persons_in_corridor_hi", "vehicles_visible_front", "vehicle_box_frac",
                            "bicycles_visible_front")}
    boxes_out = {k: [] for k in ("t_ns", "topic_ordinal", "label", "score", "x0", "y0", "x1", "y1")}
    person_frames = []
    for k in range(0, len(idx), 8):
        batch = idx[k:k + 8]
        imgs = [Image.open(io.BytesIO(compressed_image(rec.read(topic, i).data)[1])).convert("RGB") for i in batch]
        with torch.no_grad():
            out = model(**proc(images=imgs, return_tensors="pt"))
        res = proc.post_process_object_detection(out, threshold=THRESH, target_sizes=[(im.height, im.width) for im in imgs])
        for i, rr in zip(batch, res):
            t = int(tl.log_ns[i])
            names = [lab[int(l)] for l in rr["labels"]]
            bx = rr["boxes"].numpy()
            for nme, s, b in zip(names, rr["scores"].numpy(), bx):
                if nme == "person" or nme in VEHICLES or nme == "bicycle":
                    for key, v in zip(("t_ns", "topic_ordinal", "label", "score", "x0", "y0", "x1", "y1"),
                                      (t, int(tl.ordinal[i]), nme, float(s), *map(float, b))):
                        boxes_out[key].append(v)
            pmask = np.array([n == "person" for n in names], dtype=bool)
            pb = bx[pmask] if len(bx) else np.zeros((0, 4))
            feet = np.stack([(pb[:, 0] + pb[:, 2]) / 2, pb[:, 3]], axis=1) if len(pb) else np.zeros((0, 2))
            nom, lo, hi = corridor_counts(feet, rb)
            vmask = np.array([n in VEHICLES for n in names], dtype=bool)
            vb = bx[vmask] if len(bx) else np.zeros((0, 4))
            frac = float(((vb[:, 2] - vb[:, 0]) * (vb[:, 3] - vb[:, 1])).max() / (W * H)) if len(vb) else 0.0
            for key, v in zip(rows, (t, int(pmask.sum()), nom, lo, hi, int(vmask.sum()), frac,
                                     sum(n == "bicycle" for n in names))):
                rows[key].append(v)
            person_frames.append((t, pb))
    t = np.array(rows["t_ns"], dtype=np.int64)
    cols = {k: np.array(v, dtype=float) for k, v in rows.items() if k != "t_ns"}
    write(bundle, "detections", rec.id, {"t_ns": t, "available_at_ns": t, **cols},
          {"version": VERSION, "model": MODEL, "revision": REVISION, "hz": HZ, "exhaustive": True,
           "source_topics": [topic], "causal": True, "threshold": THRESH, "camera_bands": camera_bands(rb),
           "spatial_basis": "ESTIMATED"})
    tracks = track(person_frames)
    tt = [x for x in tracks if x[3] >= 3]  # confirmed tracks only
    write(bundle, "detections_tracks", rec.id, {
        "track_id": np.array([x[0] for x in tt], dtype=np.int64),
        "first_ns": np.array([x[1] for x in tt], dtype=np.int64),
        "last_ns": np.array([x[2] for x in tt], dtype=np.int64),
        "n_obs": np.array([x[3] for x in tt], dtype=np.int64),
        "confirmed_ns": np.array([x[4] for x in tt], dtype=np.int64),
    }, {"version": VERSION, "tracker": "greedy-iou>=0.3, max_miss=3, confirmed at 3 obs", "hz": HZ})
    bp = bundle / "features" / "detections_boxes" / f"{rec.id}.parquet"
    bp.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(boxes_out), bp)
    return {"provider": VERSION, "frames": len(t), "person_tracks": len(tt),
            "max_persons": int(cols["persons_visible_front"].max()), "max_corridor_hi": int(cols["persons_in_corridor_hi"].max())}
