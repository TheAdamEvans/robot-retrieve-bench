"""Standard PipelineSpecs exported into the bundle. Every config is data over the same stage library."""
from __future__ import annotations

import json
from pathlib import Path

from google.protobuf import json_format

from alloy_server.gen.alloy.v1 import pipeline_pb2 as pp


def S(sid: str, impl: str, **params) -> pp.StageSpec:
    return pp.StageSpec(stage_id=sid, impl=impl, params={k: str(v) for k, v in params.items()})


def specs() -> list[pp.PipelineSpec]:
    embed = S("embed", "EmbeddingCandidates", space="siglip2", top_k=100)
    return [
        pp.PipelineSpec(pipeline_id="TAGS", role=pp.TOP_LEVEL, final_k=10, spec_version="1",
                        doc="BM25 over recording-level tags; every window of a matching recording ties",
                        generators=[S("tags", "TagCandidates")], rankers=[S("bm25", "BM25Ranker"), S("top", "Truncate")]),
        pp.PipelineSpec(pipeline_id="EMBED", role=pp.TOP_LEVEL, final_k=10, spec_version="1",
                        doc="SigLIP2 text query vs mean-pooled front-camera window vectors",
                        generators=[embed], rankers=[S("sim", "SimilarityRanker", space="siglip2"), S("top", "Truncate")]),
        pp.PipelineSpec(pipeline_id="PROGRAM", role=pp.TOP_LEVEL, final_k=10, spec_version="1",
                        doc="QueryProgram executed over the full scope, then verified",
                        generators=[S("program", "ProgramCandidates")],
                        rankers=[S("verify", "PredicateRanker"), S("top", "Truncate")]),
        pp.PipelineSpec(pipeline_id="HYBRID", role=pp.TOP_LEVEL, final_k=10, spec_version="1",
                        doc="semantic candidates from SigLIP2, deterministic verification, lexicographic order",
                        generators=[embed],
                        rankers=[S("verify", "PredicateRanker", slack_s=4),
                                 S("sim", "SimilarityRanker", space="siglip2", order_by="verify:desc,self:desc"),
                                 S("top", "Truncate")]),
        pp.PipelineSpec(pipeline_id="FUSED", role=pp.TOP_LEVEL, final_k=10, spec_version="1",
                        doc="Stage C: contrastive window encoder (image + signals) in the SigLIP2 text space; one lookup",
                        generators=[S("embed", "EmbeddingCandidates", space="fused_v1", top_k=100)],
                        rankers=[S("sim", "SimilarityRanker", space="fused_v1"), S("top", "Truncate")]),
        pp.PipelineSpec(pipeline_id="FUSED_V", role=pp.TOP_LEVEL, final_k=10, spec_version="1",
                        doc="FUSED candidates, then deterministic verification",
                        generators=[S("embed", "EmbeddingCandidates", space="fused_v1", top_k=100)],
                        rankers=[S("verify", "PredicateRanker", slack_s=4),
                                 S("sim", "SimilarityRanker", space="fused_v1", order_by="verify:desc,self:desc"),
                                 S("top", "Truncate")]),
        pp.PipelineSpec(pipeline_id="FUSED_LINEAR", role=pp.TOP_LEVEL, final_k=10, spec_version="1",
                        doc="Stage C control C1b: linear head",
                        generators=[S("embed", "EmbeddingCandidates", space="fused_linear", top_k=100)],
                        rankers=[S("sim", "SimilarityRanker", space="fused_linear"), S("top", "Truncate")]),
        pp.PipelineSpec(pipeline_id="FUSED_CONCAT", role=pp.TOP_LEVEL, final_k=10, spec_version="1",
                        doc="Stage C control C1a: image ⊕ signals with a zero-padded text query (no learning)",
                        generators=[S("embed", "EmbeddingCandidates", space="fused_concat", top_k=100)],
                        rankers=[S("sim", "SimilarityRanker", space="fused_concat"), S("top", "Truncate")]),
        pp.PipelineSpec(pipeline_id="HYBRID_X", role=pp.TOP_LEVEL, stub=True, spec_version="0",
                        doc="stub: [EMBED, ProgramCandidates(STRUCTURED_ONLY)] → merge BY_OVERLAP → verify"),
        pp.PipelineSpec(pipeline_id="ENSEMBLE", role=pp.TOP_LEVEL, stub=True, spec_version="0",
                        doc="stub: RRF over two embedding spaces → verify"),
        pp.PipelineSpec(pipeline_id="EMBED_WEMM_FUSED", role=pp.TOP_LEVEL, stub=True, spec_version="0",
                        doc="stub: WeMM-2B fused (cards + keyframes); GPU machine"),
        pp.PipelineSpec(pipeline_id="EMBED_WEMM_IMAGE", role=pp.TOP_LEVEL, stub=True, spec_version="0",
                        doc="stub: WeMM-2B image-only"),
        pp.PipelineSpec(pipeline_id="EMBED_WEMM_CARD", role=pp.TOP_LEVEL, stub=True, spec_version="0",
                        doc="stub: WeMM-2B signal-card only"),
    ]


def build(bundle: Path) -> list[str]:
    d = bundle / "pipelines"
    d.mkdir(parents=True, exist_ok=True)
    for s in specs():
        (d / f"{s.pipeline_id}.json").write_text(json.dumps(json_format.MessageToDict(s), indent=1))
    return [s.pipeline_id for s in specs()]
