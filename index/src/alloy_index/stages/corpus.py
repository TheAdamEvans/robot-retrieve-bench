"""Corpus-level stages: rebuilt when any input recording changes (all of them take seconds)."""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alloy_index.stages.base import StageImpl, register


def ingested(ctx) -> list[str]:
    return sorted(p.stem for p in (ctx.bundle / "mcap").glob("*.json"))


@register
class SiglipWindows(StageImpl):
    IMPL, VERSION, SCOPE, DEPENDS = "SiglipWindows", "siglip2_windows@2", "CORPUS", ("SiglipFrames",)

    def run(self, ctx, recs):
        from alloy_index.build import windows
        have = [r for r in recs if (ctx.bundle / "features" / "siglip2_frames" / f"{r}.parquet").exists()]
        info = windows.build(ctx.bundle, have)
        return ["index/siglip2_windows.parquet"], {k: str(v) for k, v in info.items()}


@register
class Tags(StageImpl):
    IMPL, VERSION, SCOPE, DEPENDS = "Tags", "tags@2", "CORPUS", ("Ingest",)

    def run(self, ctx, recs):
        from alloy_index.build import tags
        docs = tags.build(ctx.bundle, recs)
        return ["index/tags.json"], {"recordings": str(len(docs))}


@register
class ApplyFused(StageImpl):
    """Embed windows that a trained FUSED index does not yet cover, with that model's full-data head."""
    IMPL, VERSION, SCOPE = "ApplyFused", "apply_fused@1", "CORPUS"
    DEPENDS = ("SiglipWindows", "MotionProvider", "ClearanceProvider", "DetectionsProvider")

    def run(self, ctx, recs):
        from alloy_server.bundle import Bundle
        from alloy_index.models import fused as F
        names = [n for n in ctx.params.get("names", "fused_v1").split(",") if n]
        bundle, outs, info = None, [], {}
        for name in names:
            mdir, idx_p = ctx.bundle / "models" / name, ctx.bundle / "index" / f"{name}_windows.parquet"
            if not (mdir / "model.json").exists():
                info[name] = "no trained model (run alloy_trainer.learn.fused)"
                continue
            bundle = bundle or Bundle(ctx.bundle)
            have = set(pq.read_table(idx_p, columns=["window_id"]).column("window_id").to_pylist()) if idx_p.exists() else set()
            todo = [w for w in bundle.embedding_index("siglip2").ids if w not in have]
            if todo:
                kind, head, mu, sd, _ = F.load(mdir)
                img = np.stack([bundle.embedding_index("siglip2").vector(w) for w in todo]).astype(np.float32)
                sig = np.stack([F.standardise(F.window_signals(bundle, w), mu, sd) for w in todo])
                vec = F.encode(kind, head, img, sig).astype(np.float16)
                new = pa.table({"window_id": todo, "recording_id": [w.rsplit(":", 1)[0] for w in todo],
                                "n_frames": [0] * len(todo), "source": ["full_model"] * len(todo),
                                "vec": pa.FixedSizeListArray.from_arrays(pa.array(vec.ravel(), pa.float16()), vec.shape[1])})
                if idx_p.exists():
                    old = pq.read_table(idx_p)
                    if "source" not in old.column_names:
                        old = old.append_column("source", pa.array(["loro_out_of_fold"] * old.num_rows))
                    meta = old.schema.metadata
                    table = pa.concat_tables([old.select(new.column_names), new]).replace_schema_metadata(meta)
                else:
                    from alloy_server.models.siglip import SPACE_ID, SPEC
                    table = new.replace_schema_metadata({"space_id": SPACE_ID, "spec": json.dumps(SPEC),
                                                         "recipe": f"{name}: full model"})
                pq.write_table(table, idx_p)
            outs.append(f"index/{name}_windows.parquet")
            info[name] = f"applied to {len(todo)} windows"
        return outs, info


@register
class PipelineSpecs(StageImpl):
    IMPL, VERSION, SCOPE = "PipelineSpecs", "pipelines@2", "CORPUS"

    def run(self, ctx, recs):
        from alloy_index.build import pipelines
        ids = pipelines.build(ctx.bundle)
        return [f"pipelines/{i}.json" for i in ids], {"specs": str(len(ids))}


@register
class PromptAssets(StageImpl):
    IMPL, VERSION, SCOPE = "PromptAssets", "prompts@2", "CORPUS"

    def run(self, ctx, recs):
        from alloy_index.build import prompts
        info = prompts.build(ctx.bundle)
        return ["prompts/fewshots.json"], {k: str(v) for k, v in info.items()}
