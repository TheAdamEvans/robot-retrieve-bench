"""LocalBundleSink: the bundle directory the server loads. Its manifest is the only record of what exists.

Several runners (CPU, GPU and API stages) may share one sink, so manifest writes are read-merge-write under a lock.
"""
from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

from google.protobuf import json_format

from alloy_server.gen.alloy.v1 import index_pb2


class LocalBundleSink:
    def __init__(self, root: Path, scand_root: Path):
        self.root = root
        self.scand_root = scand_root  # some outputs (label shards) live in the repo, not the bundle
        self.path = root / "manifest.json"
        self.manifest = index_pb2.SinkManifest()
        self._reload()

    def resolve(self, rel: str) -> Path:
        return self.scand_root / rel[len("repo:"):] if rel.startswith("repo:") else self.root / rel

    def _reload(self) -> None:
        self.manifest = index_pb2.SinkManifest()
        if self.path.exists():
            json_format.Parse(self.path.read_text(), self.manifest, ignore_unknown_fields=True)

    @contextmanager
    def _locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / "manifest.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def has(self, key: str) -> bool:
        self._reload()
        rec = self.manifest.artifacts.get(key)
        return rec is not None and all(self.resolve(p).exists() for p in rec.paths)

    def get(self, key: str) -> index_pb2.ArtifactRecord | None:
        return self.manifest.artifacts.get(key)

    def latest(self, stage: str, recording_id: str = "") -> index_pb2.ArtifactRecord | None:
        self._reload()
        cands = [r for r in self.manifest.artifacts.values() if r.stage == stage and r.recording_id == recording_id
                 and all(self.resolve(p).exists() for p in r.paths)]
        return max(cands, key=lambda r: r.created_at) if cands else None

    def put(self, record: index_pb2.ArtifactRecord) -> None:
        with self._locked():
            self._reload()  # merge with whatever other runners recorded meanwhile
            # a newer build of the same (stage, recording) supersedes older keys
            for k in [k for k, r in self.manifest.artifacts.items()
                      if r.stage == record.stage and r.recording_id == record.recording_id and k != record.key]:
                del self.manifest.artifacts[k]
            self.manifest.artifacts[record.key].CopyFrom(record)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(json_format.MessageToDict(self.manifest), indent=1, sort_keys=True))
            tmp.replace(self.path)
