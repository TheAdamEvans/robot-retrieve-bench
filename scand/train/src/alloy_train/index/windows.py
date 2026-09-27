"""Window vectors from per-frame SigLIP2 embeddings: a window's vector is the L2-normalised mean of its front-camera
frames' (L2-normalised) vectors over [end-4 s, end]."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alloy_server.catalog.windows import window_span_s, windows
from alloy_server.models.siglip import SPACE_ID, SPEC
from alloy_server.timeline.store import Recording
from alloy_train import embodiment as emb
from alloy_train.recordings import RECORDINGS, robot


def build(bundle: Path) -> dict:
    ids, recs, vecs, counts = [], [], [], []
    for rec in RECORDINGS:
        p = bundle / "features" / "siglip2_frames" / f"{rec}.parquet"
        t = pq.read_table(p)
        assert t.schema.metadata[b"space_id"].decode() == SPACE_ID
        topic = np.array(t.column("topic").to_pylist())
        ts = t.column("log_time_ns").to_numpy()
        v = np.asarray(t.column("vec").combine_chunks().values.to_numpy(zero_copy_only=False),
                       dtype=np.float32).reshape(len(ts), -1)
        front = topic == emb.front_camera_topic(robot(rec))
        ts, v = ts[front], v[front]
        r = Recording(bundle, rec)
        for wid in windows(rec, (r.end_ns - r.start_ns) / 1e9):
            _, t0, t1 = window_span_s(wid)
            m = (ts >= r.t_abs(t0)) & (ts <= r.t_abs(t1))
            mean = v[m].mean(0) if m.any() else np.zeros(v.shape[1], np.float32)
            ids.append(wid); recs.append(rec); counts.append(int(m.sum()))
            vecs.append(mean / max(np.linalg.norm(mean), 1e-9))
    arr = np.stack(vecs).astype(np.float16)
    table = pa.table({"window_id": ids, "recording_id": recs, "n_frames": counts,
                      "vec": pa.FixedSizeListArray.from_arrays(pa.array(arr.ravel(), pa.float16()), arr.shape[1])}
                     ).replace_schema_metadata({"space_id": SPACE_ID, "spec": json.dumps(SPEC),
                                                "pooling": "mean of front frames, 4 s window"})
    out = bundle / "index" / "siglip2_windows.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return {"windows": len(ids), "min_frames": min(counts), "bytes": out.stat().st_size}
