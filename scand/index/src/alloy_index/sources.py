"""Sources enumerate recordings and their raw files. They know nothing about stages."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alloy_index import recordings as R


@dataclass(frozen=True)
class RawRecording:
    recording_id: str
    bag: Path
    metadata: dict
    held_out: bool

    @property
    def fingerprint(self) -> str:
        st = self.bag.stat()
        return f"{self.bag.name}:{st.st_size}"


class LocalBagSource:
    """raw/*.bag joined with SCAND_index.csv; ids from the registry (aliases, then the bag stem template)."""

    def __init__(self, spec):
        self.spec = spec

    def recordings(self) -> dict[str, RawRecording]:
        meta = R.metadata()
        held = set(self.spec.source.held_out)
        return {rid: RawRecording(rid, R.bag_path(rid), meta.get(rid, {}), rid in held)
                for rid in R.discover(self.spec)}
