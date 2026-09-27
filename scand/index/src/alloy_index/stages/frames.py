"""Per-frame SigLIP2 image embeddings (front camera at front_hz, body cameras at body_hz; "native" = every frame)."""
from __future__ import annotations

import pyarrow.parquet as pq

from alloy_index.stages.base import StageImpl, register


def _hz(v: str | None) -> float | None:
    return None if v in (None, "", "native", "None") else float(v)


@register
class SiglipFrames(StageImpl):
    IMPL, VERSION, SCOPE, DEPENDS = "SiglipFrames", "siglip2_frames@1", "PER_RECORDING", ("Ingest",)
    _enc = None

    def _out(self, rid):
        return f"features/siglip2_frames/{rid}.parquet"

    def adopt(self, ctx, rec):
        p = ctx.bundle / self._out(rec.recording_id)
        if not p.exists():
            return None
        meta = pq.read_schema(p).metadata or {}
        same_front = _hz(meta.get(b"front_hz", b"").decode()) == _hz(ctx.params.get("front_hz", "10"))
        same_body = _hz(meta.get(b"body_hz", b"native").decode()) == _hz(ctx.params.get("body_hz", "native"))
        return [self._out(rec.recording_id)] if same_front and same_body else None

    def run(self, ctx, rec):
        from alloy_server.models.siglip import SiglipEncoder
        from alloy_index.build import siglip_frames
        if SiglipFrames._enc is None:
            SiglipFrames._enc = SiglipEncoder()
        info = siglip_frames.run(ctx.bundle, rec.recording_id, SiglipFrames._enc,
                                 front_hz=_hz(ctx.params.get("front_hz", "10")),
                                 body_hz=_hz(ctx.params.get("body_hz", "native")), overwrite=True)
        return [self._out(rec.recording_id)], {k: str(v) for k, v in info.items()}
