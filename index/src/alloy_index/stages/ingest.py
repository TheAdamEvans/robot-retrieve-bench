"""Ingest: recognise the robot (identity topics), convert bag -> MCAP (layout param) + timeline, measure (intake)."""
from __future__ import annotations

import hashlib
import json

import pyarrow.parquet as pq
from rosbags.highlevel import AnyReader

from alloy_server.catalog import embodiment as E
from alloy_index.convert.bag_to_mcap import convert
from alloy_index.intake import measure, recognise
from alloy_index.stages.base import StageImpl, register


@register
class Ingest(StageImpl):
    IMPL, VERSION, SCOPE = "Ingest", "ingest@2", "PER_RECORDING"  # @2: intake records stale leading frames

    def _paths(self, ctx, rid: str) -> list[str] | None:
        info_p = ctx.bundle / "mcap" / f"{rid}.json"
        if not info_p.exists():
            return None
        info = json.loads(info_p.read_text())
        return info.get("files", [f"mcap/{rid}.mcap"]) + [f"mcap/{rid}.json", f"timeline/{rid}.parquet",
                                                           f"intake/{rid}.json"]

    def adopt(self, ctx, rec):
        paths = self._paths(ctx, rec.recording_id)
        if not paths:
            return None
        info = json.loads((ctx.bundle / "mcap" / f"{rec.recording_id}.json").read_text())
        if info.get("layout", "interleaved") != ctx.params.get("layout", "by_sensor"):
            return None
        return paths if all((ctx.bundle / p).exists() for p in paths) else None

    def run(self, ctx, rec):
        with AnyReader([rec.bag]) as r:
            topics = {c.topic for c in r.connections}
        prof = E.profile(recognise(topics))
        layout = ctx.params.get("layout", "by_sensor")
        old = self._paths(ctx, rec.recording_id) or []
        info = convert(rec.recording_id, ctx.bundle, layout, prof)
        rep = measure(ctx.bundle, rec.recording_id, prof, rec.bag)
        E.save_intake(ctx.bundle, rep)
        for p in old:  # files of a previous layout are derived data: remove what the new layout replaced
            if p.endswith(".mcap") and p not in info["files"] and (ctx.bundle / p).exists():
                (ctx.bundle / p).unlink()
        return self._paths(ctx, rec.recording_id), {
            "layout": layout, "embodiment": prof.embodiment_id, "messages": str(info["messages"]),
            "absent_sensors": ",".join(rep.absent_sensors), "anomalies": str(len(rep.anomalies))}

    def content_key(self, ctx, rec, paths):
        """Layout-independent identity: the ordered (topic, ordinal, payload hash) set."""
        t = pq.read_table(ctx.bundle / "timeline" / f"{rec.recording_id}.parquet",
                          columns=["topic", "topic_ordinal", "payload_sha128"])
        h = hashlib.sha256()
        for topic, n, sha in zip(t.column("topic").to_pylist(), t.column("topic_ordinal").to_pylist(),
                                 t.column("payload_sha128").to_pylist()):
            h.update(f"{topic}#{n}".encode()); h.update(sha)
        return h.hexdigest()[:32]
