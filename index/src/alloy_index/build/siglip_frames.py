"""Per-frame SigLIP2 image embeddings: front camera at 10 Hz, Spot body cameras at native 4.5 Hz.

Output: bundle/features/siglip2_frames/<rec>.parquet  (topic, topic_ordinal, log_time_ns, vec[float16 x 1152]).
Per-frame vectors are kept so windows can mean- or max-pool and Stage C can train on them.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from alloy_server.io.ros1 import compressed_image
from alloy_server.models.siglip import SPACE_ID, SPEC, SiglipEncoder
from alloy_server.timeline.store import Recording
from alloy_index import embodiment as emb
from alloy_index.recordings import RECORDINGS, robot

FRONT_HZ = 10.0
BATCH = 16


def _sample(r: Recording, topic: str, hz: float | None) -> list[tuple[str, int]]:
    tl = r.topics.get(topic)
    if tl is None or not len(tl):
        return []  # sensor absent in this log (intake records it)
    if hz is None:  # native rate
        return [(topic, i) for i in range(len(tl))]
    ticks = np.arange(r.start_ns, r.end_ns, int(1e9 / hz))
    return [(topic, i) for i in sorted({tl.nearest(int(t)) for t in ticks})]


def select(r: Recording, front_hz: float | None = FRONT_HZ, body_hz: float | None = None) -> list[tuple[str, int]]:
    rb = robot(r.id)
    out = _sample(r, emb.front_camera_topic(rb), front_hz)
    for topic in emb.body_camera_topics(rb):
        out += _sample(r, topic, body_hz)
    return out


def decode(r: Recording, item: tuple[str, int]) -> Image.Image:
    topic, i = item
    _, data = compressed_image(r.read(topic, i).data)
    return Image.open(io.BytesIO(data)).convert("RGB")


def run(bundle: Path, rec: str, enc: SiglipEncoder, front_hz: float | None = FRONT_HZ,
        body_hz: float | None = None, overwrite: bool = False) -> dict:
    out = bundle / "features" / "siglip2_frames" / f"{rec}.parquet"
    if out.exists() and not overwrite:
        return {"recording": rec, "skipped": True}
    r = Recording(bundle, rec)
    items = select(r, front_hz, body_hz)
    vecs = np.zeros((len(items), SPEC["dim"]), dtype=np.float16)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(4) as pool:
        batches = [items[k:k + BATCH] for k in range(0, len(items), BATCH)]
        nxt = pool.map(lambda b: [decode(r, it) for it in b], batches)
        for k, imgs in enumerate(nxt):
            vecs[k * BATCH:k * BATCH + len(imgs)] = enc.encode_images(imgs)
    dt = time.perf_counter() - t0
    tl = r.topics
    table = pa.table({
        "topic": pa.array([t for t, _ in items]).dictionary_encode(),
        "topic_ordinal": pa.array([int(tl[t].ordinal[i]) for t, i in items], pa.uint32()),
        "log_time_ns": pa.array([int(tl[t].log_ns[i]) for t, i in items], pa.int64()),
        "vec": pa.FixedSizeListArray.from_arrays(pa.array(vecs.ravel(), pa.float16()), SPEC["dim"]),
    }).replace_schema_metadata({"space_id": SPACE_ID, "spec": json.dumps(SPEC), "front_hz": str(front_hz),
                                "body_hz": str(body_hz or "native")})
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return {"recording": rec, "frames": len(items), "seconds": round(dt, 1), "img_per_s": round(len(items) / dt, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("recs", nargs="*", default=list(RECORDINGS))
    a = ap.parse_args()
    enc = SiglipEncoder()
    for rec in a.recs:
        print(json.dumps(run(a.bundle, rec, enc)), flush=True)


if __name__ == "__main__":
    main()
