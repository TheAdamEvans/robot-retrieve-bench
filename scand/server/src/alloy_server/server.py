"""HTTP/JSON API over the bundle. Every search goes through alloy_server.pipeline.run.

  POST /v1/search     SearchRequest (proto-JSON)  → SearchResponse
  POST /v1/score      SearchRequest with window_ids → SCORE_ALL response (eval only)
  GET  /v1/pipelines  bundle PipelineSpecs (stubs flagged)
  GET  /v1/evidence/{message_id}   image bytes for a camera MessageId (counted reads)
  GET  /v1/frame?rec=&t=&cam=      nearest camera frame (thumbnails)
  GET  /v1/health     bundle id, recordings, resident sidecar size, session bytes
  GET  /              demo page
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from google.protobuf import json_format

from .bundle import Bundle
from .catalog.embodiment import sensor as profile_sensor, profile
from .gen.alloy.v1 import answer_pb2 as a
from .gen.alloy.v1 import pipeline_pb2 as pp
from .io import cost
from .io.ros1 import compressed_image
from .pipeline.runner import run
from .timeline.store import parse_mid

BUNDLE = Path(os.environ.get("ALLOY_BUNDLE", Path(__file__).resolve().parents[3] / "bundles" / "dev"))
bundle = Bundle(BUNDLE)
if os.environ.get("ALLOY_WARM", "1") == "1":  # model load is startup cost: never paid by the first live query
    from .models.siglip import SPACE_ID
    bundle.encode_query(SPACE_ID, "warm-up")
app = FastAPI(title="alloy search")
_generator = None


def generator():
    global _generator
    if _generator is None and os.environ.get("OPENAI_API_KEY"):
        from .programs.openai_generator import OpenAIProgramGenerator
        shots = json.loads((bundle.root / "prompts" / "fewshots.json").read_text())
        _generator = OpenAIProgramGenerator(bundle, [(s["utterance"], s["program"]) for s in shots],
                                            cache_dir=bundle.root.parent.parent / "cache" / "programs")
    return _generator


def _search(body: dict, mode: int) -> dict:
    req = json_format.ParseDict(body, a.SearchRequest(), ignore_unknown_fields=False)
    req.mode = mode
    spec = bundle.specs.get(req.pipeline_id)
    if spec is None:
        raise HTTPException(404, f"unknown pipeline {req.pipeline_id!r}; see /v1/pipelines")
    resp = run(bundle, spec, req, generator=generator())
    return json_format.MessageToDict(resp)


@app.post("/v1/search")
def search(body: dict):
    return JSONResponse(_search(body, pp.SEARCH))


@app.post("/v1/score")
def score(body: dict):
    return JSONResponse(_search(body, pp.SCORE_ALL))


@app.get("/v1/pipelines")
def pipelines():
    return [json_format.MessageToDict(s) for s in bundle.specs.values()]


@app.get("/v1/health")
def health():
    return {"bundle_id": bundle.bundle_id, "recordings": {r: bundle.robots[r] for r in bundle.recordings},
            "pipelines": sorted(bundle.specs), "resident": bundle.resident_bytes(),
            "starts": {r: rec.start_ns for r, rec in bundle.recordings.items()},
            "generator": "configured" if os.environ.get("OPENAI_API_KEY") else "none"}


def _image(rec_id: str, topic: str, i: int) -> Response:
    rec = bundle.recordings[rec_id]
    with cost.scope("evidence") as s:
        fmt, data = compressed_image(rec.read(topic, i).data)
    return Response(data, media_type="image/jpeg",
                    headers={"x-bytes-read": str(s.totals()["bytes_read"]), "cache-control": "max-age=3600"})


@app.get("/v1/evidence/{mid:path}")
def evidence(mid: str):
    rec_id, topic, ordinal = parse_mid(mid)
    if rec_id not in bundle.recordings or not topic.endswith("compressed"):
        raise HTTPException(404, "only camera MessageIds render as images")
    return _image(rec_id, topic, bundle.recordings[rec_id].index_of(topic, ordinal))


@app.get("/v1/frame")
def frame(rec: str, t_ns: int, cam: str = "front_camera"):
    if rec not in bundle.recordings:
        raise HTTPException(404, "unknown recording")
    s = profile_sensor(profile(bundle.robots[rec]), cam)
    if s is None:
        raise HTTPException(404, f"{cam} not on {bundle.robots[rec]}")
    tl = bundle.recordings[rec].topics[s.topics[0]]
    return _image(rec, s.topics[0], tl.nearest(t_ns))


EXAMPLES = Path(os.environ.get("ALLOY_EXAMPLES", Path(__file__).resolve().parents[3] / "benchmark" / "queries"))


@app.get("/v1/examples")
def examples():
    """Benchmark queries (data files) so the demo can run oracle programs next to generated ones."""
    out = []
    for p in sorted(EXAMPLES.glob("*.json")):
        if p.name == "MANIFEST.json":
            continue
        out += [{"query_id": q["queryId"], "set": q["querySet"], "utterance": q["utterance"],
                 "scope": q.get("scope", {}), "program": q.get("oracleProgram")} for q in json.loads(p.read_text())]
    return out


CLIP_CACHE = Path(os.environ.get("ALLOY_CLIP_CACHE", Path(__file__).resolve().parents[3] / "cache" / "clips"))


@app.get("/v1/clip")
def clip(rec: str, t0_ns: int, t1_ns: int, cam: str = "front_camera", anchor_ns: int | None = None):
    """H.264 MP4 of [t0, t1] from one camera, rendered from the MCAP and cached on disk."""
    import hashlib
    from fastapi.responses import FileResponse
    from .catalog.embodiment import topic_display
    from .io.clips import render
    if rec not in bundle.recordings:
        raise HTTPException(404, "unknown recording")
    s = profile_sensor(profile(bundle.robots[rec]), cam)
    if s is None:
        raise HTTPException(404, f"{cam} not on {bundle.robots[rec]}")
    topic = s.topics[0]
    key = hashlib.sha256(f"{bundle.bundle_id}|{rec}|{topic}|{t0_ns}|{t1_ns}|{anchor_ns}|v1".encode()).hexdigest()[:24]
    path = CLIP_CACHE / f"{key}.mp4"
    headers = {"cache-control": "max-age=86400"}
    if not path.exists():
        CLIP_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.mp4")
        with cost.scope("clip") as sc:
            render(bundle.recordings[rec], topic, t0_ns, t1_ns, str(tmp), anchor_ns,
                   rotation=topic_display(profile(bundle.robots[rec])).get(topic, (0, ""))[0], label=rec)
        tmp.replace(path)
        headers["x-bytes-read"] = str(sc.totals()["bytes_read"])
    return FileResponse(path, media_type="video/mp4", headers=headers)


@app.get("/", response_class=HTMLResponse)
def page():
    return (Path(__file__).parent / "static" / "index.html").read_text()
