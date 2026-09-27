"""Provider stages: motion, clearance, detections. Each reads ingest's content and writes features/<provider>/<rec>."""
from __future__ import annotations

import importlib
import json

import pyarrow.parquet as pq

from alloy_server.catalog import embodiment as E
from alloy_server.timeline.store import Recording
from alloy_index.stages.base import StageImpl, register


class _Provider(StageImpl):
    SCOPE, DEPENDS = "PER_RECORDING", ("Ingest",)
    MODULE, OUTPUTS, SENSOR = "", ("",), ""

    @property
    def mod(self):
        return importlib.import_module(f"alloy_index.providers.{self.MODULE}")

    @property
    def VERSION(self):  # noqa: N802 — the provider module owns its version
        return self.mod.VERSION

    def applies(self, ctx, rec):
        intake = E.load_intake(ctx.bundle, rec.recording_id)
        if self.SENSOR and intake is not None and self.SENSOR in intake.absent_sensors:
            return False, f"{self.SENSOR} absent in this log"
        return True, ""

    def _outs(self, rid):
        return [f"features/{o}/{rid}.parquet" for o in self.OUTPUTS]

    def adopt(self, ctx, rec):
        outs = self._outs(rec.recording_id)
        if not all((ctx.bundle / o).exists() for o in outs):
            return None
        meta = pq.read_schema(ctx.bundle / outs[0]).metadata or {}
        version = json.loads(meta.get(b"provider", b"{}")).get("version")
        return outs if version == self.VERSION else None

    def run(self, ctx, rec):
        info = self.mod.build(ctx.bundle, Recording(ctx.bundle, rec.recording_id))
        return self._outs(rec.recording_id), {k: str(v) for k, v in info.items()}


@register
class MotionProvider(_Provider):
    IMPL, MODULE, OUTPUTS, SENSOR = "MotionProvider", "motion", ("motion",), "odom"


@register
class ClearanceProvider(_Provider):
    IMPL, MODULE, OUTPUTS = "ClearanceProvider", "clearance", ("clearance",)


@register
class DetectionsProvider(_Provider):
    IMPL, MODULE, SENSOR = "DetectionsProvider", "detections", "front_camera"
    OUTPUTS = ("detections", "detections_tracks", "detections_boxes")

    def adopt(self, ctx, rec):
        outs = self._outs(rec.recording_id)
        if not all((ctx.bundle / o).exists() for o in outs):
            return None
        meta = pq.read_schema(ctx.bundle / outs[0]).metadata or {}
        version = json.loads(meta.get(b"provider", b"{}")).get("version")
        return outs if version == self.VERSION else None
