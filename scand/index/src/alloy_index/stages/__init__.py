"""Index stage implementations, registered by `impl` name (see index/index.textproto)."""
from alloy_index.stages.base import REGISTRY, IndexContext, StageImpl  # noqa: F401
from alloy_index.stages import ingest, providers, frames, corpus, annotate  # noqa: F401  (registration)
