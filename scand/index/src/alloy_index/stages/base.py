"""Stage contract: what a stage consumes (dependencies), what it declares (NAME/VERSION/SCOPE) and what it writes.

Keys cover only what can change a stage's bytes: its code VERSION, its params, and its inputs' *content* keys
(e.g. ingest's content key is a hash of payload hashes, so changing the MCAP layout rebuilds nothing downstream).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

REGISTRY: dict[str, type["StageImpl"]] = {}


def register(cls):
    REGISTRY[cls.IMPL] = cls
    return cls


@dataclass
class IndexContext:
    bundle: Path
    scand_root: Path
    sink: object
    params: dict[str, str]
    flags: set[str] = field(default_factory=set)
    log: callable = print


class StageImpl:
    IMPL = "Stage"
    VERSION = "0"
    SCOPE = "PER_RECORDING"          # or CORPUS
    DEPENDS: tuple[str, ...] = ()    # impl names whose outputs this stage reads

    def applies(self, ctx: IndexContext, rec) -> tuple[bool, str]:
        """Whether this stage has anything to do for `rec` (e.g. no front camera -> no detections)."""
        return True, ""

    def adopt(self, ctx: IndexContext, rec) -> list[str] | None:
        """Existing outputs that were built by this VERSION with these params (registered without rebuilding)."""
        return None

    def run(self, ctx: IndexContext, rec) -> tuple[list[str], dict]:
        """Build; return (output paths relative to the bundle, or 'repo:' prefixed, info)."""
        raise NotImplementedError

    def content_key(self, ctx: IndexContext, rec, paths: list[str]) -> str:
        return ""


def key_of(stage: str, version: str, params: dict, input_keys: list[str], recording_id: str, extra: str = "") -> str:
    blob = json.dumps({"stage": stage, "version": version, "params": dict(sorted(params.items())),
                       "inputs": sorted(input_keys), "recording": recording_id, "extra": extra}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def params_sha(params: dict) -> str:
    return hashlib.sha256(json.dumps(dict(sorted(params.items()))).encode()).hexdigest()[:16]
