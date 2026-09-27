"""Bundle: the exported, self-contained data the server serves. Loaded once; everything query-time reads through it.

Session loads (timelines, feature tables, window index, models) are counted in a session CostScope and reported,
never charged to a query. Per-query bytes are what a request actually touches beyond that (e.g. MCAP chunks for
evidence it cites).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from google.protobuf import json_format

from .catalog import embodiment as E
from .catalog.features import FeatureStore
from .catalog.registry import FEATURES, SENSORS
from .catalog.windows import window_id, windows
from .gen.alloy.v1 import answer_pb2 as a
from .gen.alloy.v1 import pipeline_pb2 as pp
from .io import cost, parquet_reader
from .programs.validator import validate
from .timeline.store import Recording
from .verify.executor import Executor

TOKEN = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "of", "in", "on", "to", "and", "or", "is", "are", "with", "for", "at", "by", "from", "where",
        "when", "find", "show", "me", "robot", "that", "this", "it", "its", "then", "after", "before"}


def toks(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class TagIndex:
    """BM25 over recording-level text (tags, locations). The naive baseline: it knows nothing about time."""

    def __init__(self, docs: dict[str, str], k1: float = 1.2, b: float = 0.75):
        self.docs = {r: toks(t) for r, t in docs.items()}
        self.k1, self.b = k1, b
        self.avgdl = np.mean([len(d) for d in self.docs.values()]) if self.docs else 1.0
        n = len(self.docs)
        vocab = {w for d in self.docs.values() for w in d}
        self.idf = {w: math.log(1 + (n - (df := sum(w in d for d in self.docs.values())) + 0.5) / (df + 0.5)) for w in vocab}

    def scores(self, query: str) -> dict[str, float]:
        qt = toks(query)
        out = {}
        for r, d in self.docs.items():
            s = 0.0
            for w in qt:
                f = d.count(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out[r] = s
        return out


class WindowIndex:
    def __init__(self, path: Path):
        t = parquet_reader.read_table(str(path), layer="window_index")
        meta = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
        self.space_id = meta["space_id"]
        self.ids = t.column("window_id").to_pylist()
        self.recs = np.array(t.column("recording_id").to_pylist())
        v = t.column("vec").combine_chunks()
        self.vecs = np.asarray(v.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(len(self.ids), -1)
        self.row = {w: i for i, w in enumerate(self.ids)}
        self.nbytes = self.vecs.nbytes

    def search(self, qv: np.ndarray, recs: list[str], k: int) -> tuple[list[str], np.ndarray]:
        mask = np.isin(self.recs, recs)
        sims = self.vecs @ qv
        sims = np.where(mask, sims, -np.inf)
        k = min(k, int(mask.sum()))
        top = np.argpartition(-sims, k - 1)[:k] if k else np.array([], int)
        top = top[np.argsort(-sims[top], kind="stable")]
        return [self.ids[i] for i in top], sims[top]

    def vector(self, wid: str) -> np.ndarray:
        return self.vecs[self.row[wid]]


class Bundle:
    def __init__(self, root: Path):
        self.root = root
        self.bundle_id = root.name
        with cost.scope("session") as s:
            recs = sorted(p.stem for p in (root / "mcap").glob("*.json"))
            self.recordings = {r: Recording(root, r) for r in recs}
            self.robots = {}
            for r in recs:
                intake = E.load_intake(root, r)
                if intake is None:
                    raise RuntimeError(f"{r}: no intake report; run alloy-train intake")
                self.robots[r] = intake.embodiment_id
            self.windows = {r: windows(r, (rec.end_ns - rec.start_ns) / 1e9) for r, rec in self.recordings.items()}
            self.features = FeatureStore(root)
            for r in recs:  # warm every provider table now so queries never pay for it
                for f in FEATURES:
                    self.features.series(f.name, r)
                    if f.kind == "track":
                        self.features.tracks(f.name, r)
            # intake measures speed_floor_mps, but it is not applied: "never slowed" and "cannot report slow" look alike
            # (see OPEN_QUESTIONS.md); a floor is applied only once a human confirms it for a log.
            self.absent_sensors = {r: set(E.load_intake(root, r).absent_sensors) for r in recs}
            self.executor = Executor(self.features, self.recordings, self.robots, absent_sensors=self.absent_sensors)
            tags = root / "index" / "tags.json"
            self.tags = TagIndex(json.loads(tags.read_text()) if tags.exists() else {})
            self._indexes: dict[str, WindowIndex] = {}
            for p in sorted((root / "index").glob("*_windows.parquet")):
                self._indexes[p.stem.removesuffix("_windows")] = WindowIndex(p)
            self.specs = self._load_specs()
        self.session = s.totals()
        self.raw_bytes = {r: rec.bag_bytes for r, rec in self.recordings.items()}
        self._encoder = None

    def _load_specs(self) -> dict[str, pp.PipelineSpec]:
        out = {}
        for p in sorted((self.root / "pipelines").glob("*.json")):
            spec = json_format.Parse(p.read_text(), pp.PipelineSpec())
            out[spec.pipeline_id] = spec
        return out

    # ---- query-time services ----
    def validate(self, prog):
        return validate(prog, self.robots)

    def sensors(self, rec: str) -> dict[str, list[str]]:
        return SENSORS[self.robots[rec]]

    def embedding_index(self, space: str) -> WindowIndex:
        if space not in self._indexes:
            raise KeyError(f"no window index for space {space!r} in bundle {self.bundle_id}")
        return self._indexes[space]

    def encode_query(self, space_id: str, text: str) -> np.ndarray:
        from .models.siglip import SPACE_ID, SiglipEncoder
        if space_id != SPACE_ID:
            raise ValueError(f"query encoder space {SPACE_ID} does not match index space {space_id}")
        if self._encoder is None:
            with cost.scope("startup_model_load"):
                self._encoder = SiglipEncoder()
        return self._encoder.encode_texts([text])[0]

    def window_for_interval(self, cand: pp.Candidate) -> str | None:
        t = cand.anchors[0].t_ns if cand.anchors else cand.seed.start_ns
        rec = self.recordings[cand.recording_id]
        end = int(math.ceil(rec.t_rel(t)))
        ids = self.windows[cand.recording_id]
        if not ids:
            return None
        end = min(max(end, int(ids[0].split(":")[1])), int(ids[-1].split(":")[1]))
        return window_id(cand.recording_id, end)

    def verify_evidence(self, resp: a.SearchResponse, max_results: int = 5, per_result: int = 2) -> None:
        for item in resp.results[:max_results]:
            for mid in list(item.candidate.evidence)[:per_result]:
                rec = self.recordings[mid.recording_id]
                i = rec.index_of(mid.topic, mid.topic_ordinal)
                if hashlib.sha256(rec.read(mid.topic, i).data).digest()[:16] != mid.payload_sha256_128:
                    raise RuntimeError(f"payload hash mismatch for {mid}")

    def cost_report(self, scope, recs: list[str], wall_ms: float, exclude: tuple[str, ...] = ()) -> pp.CostReport:
        t = scope.totals(exclude)
        raw = sum(self.raw_bytes[r] for r in recs) or 1
        rep = pp.CostReport(bytes_read=t["bytes_read"], bytes_served_from_cache=t["bytes_served_from_cache"],
                            raw_ratio=t["bytes_read"] / raw, llm_calls=t["llm_calls"], prompt_tokens=t["prompt_tokens"],
                            cached_prompt_tokens=t["cached_prompt_tokens"], reasoning_tokens=t["reasoning_tokens"],
                            completion_tokens=t["completion_tokens"], usd_est=t["usd_est"],
                            encoder_forward_ms=t["encoder_forward_ms"], wall_ms=wall_ms)
        for k, v in t["bytes_by_layer"].items():
            rep.bytes_by_layer[k] = v
        return rep

    def resident_bytes(self) -> dict:
        idx = sum(i.nbytes for i in self._indexes.values())
        return {"session_bytes_read": self.session["bytes_read"], "window_index_bytes": idx,
                "raw_corpus_bytes": sum(self.raw_bytes.values()),
                "sidecar_ratio": (self.session["bytes_read"]) / max(1, sum(self.raw_bytes.values()))}
