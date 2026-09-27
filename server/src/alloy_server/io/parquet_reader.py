"""Parquet reads through a counting file handle, so footer and row-group bytes land in the CostScope."""
from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.parquet as pq

from . import cost


class _CountingFile(io.RawIOBase):
    def __init__(self, path: str, layer: str):
        self._f = open(path, "rb")  # noqa: SIM115 — the one sanctioned open for parquet (see io/ static check)
        self.path, self.layer = path, layer
        self._scope = cost.current()  # pyarrow reads row groups on worker threads

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._f.seek(offset, whence)

    def tell(self) -> int:
        return self._f.tell()

    def read(self, n: int = -1) -> bytes:
        off = self._f.tell()
        b = self._f.read(n)
        if b:
            cost.record_read(self.layer, self.path, off, len(b), cache_hit=False, scope=self._scope)
        return b

    def readinto(self, b) -> int:
        off = self._f.tell()
        n = self._f.readinto(b)
        if n:
            cost.record_read(self.layer, self.path, off, n, cache_hit=False, scope=self._scope)
        return n

    def close(self) -> None:
        self._f.close()
        super().close()


def read_table(path: str, layer: str, columns: list[str] | None = None, row_groups: list[int] | None = None) -> pa.Table:
    f = _CountingFile(path, layer)
    try:
        pf = pq.ParquetFile(f)
        if row_groups is None:
            return pf.read(columns=columns)
        return pf.read_row_groups(row_groups, columns=columns)
    finally:
        f.close()


def read_metadata(path: str, layer: str) -> pq.FileMetaData:
    f = _CountingFile(path, layer)
    try:
        return pq.ParquetFile(f).metadata
    finally:
        f.close()
