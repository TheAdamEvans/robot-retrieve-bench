"""Minimal MCAP reader: locate a message by (chunk_offset, chunk_len, offset_in_chunk) with a pread.

Deliberately independent of the `mcap` package so the server has one small, auditable read path whose bytes
are counted exactly. Format reference: https://mcap.dev/spec
"""
from __future__ import annotations

import os
import struct
from collections import OrderedDict
from dataclasses import dataclass

import zstandard

from . import cost

MAGIC = b"\x89MCAP0\r\n"
OP_SCHEMA, OP_CHANNEL, OP_MESSAGE, OP_CHUNK, OP_DATA_END = 0x03, 0x04, 0x05, 0x06, 0x0F


def _str(buf: bytes | memoryview, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<I", buf, off)
    return bytes(buf[off + 4 : off + 4 + n]).decode(), off + 4 + n


def parse_channel(content: bytes) -> tuple[int, int, str, str]:
    cid, sid = struct.unpack_from("<HH", content, 0)
    topic, off = _str(content, 4)
    enc, off = _str(content, off)
    return cid, sid, topic, enc


def parse_schema(content: bytes) -> tuple[int, str, str, bytes]:
    (sid,) = struct.unpack_from("<H", content, 0)
    name, off = _str(content, 2)
    enc, off = _str(content, off)
    (n,) = struct.unpack_from("<I", content, off)
    return sid, name, enc, bytes(content[off + 4 : off + 4 + n])


def decompress_chunk(content: bytes) -> bytes:
    off = 8 + 8 + 8 + 4
    compression, off = _str(content, off)
    (n,) = struct.unpack_from("<Q", content, off)
    data = content[off + 8 : off + 8 + n]
    if compression == "zstd":
        (usize,) = struct.unpack_from("<Q", content, 16)
        return zstandard.ZstdDecompressor().decompress(data, max_output_size=usize)
    if compression == "":
        return bytes(data)
    raise ValueError(f"unsupported chunk compression {compression!r}")


def iter_records(buf: bytes, start: int = 0):
    """Yield (opcode, record_offset, content) for a flat run of records."""
    off, end = start, len(buf)
    while off + 9 <= end:
        op = buf[off]
        (n,) = struct.unpack_from("<Q", buf, off + 1)
        yield op, off, buf[off + 9 : off + 9 + n]
        off += 9 + n


@dataclass(frozen=True)
class RawMessage:
    channel_id: int
    sequence: int
    log_time_ns: int
    publish_time_ns: int
    data: bytes


def parse_message(content: bytes) -> RawMessage:
    cid, seq, log_t, pub_t = struct.unpack_from("<HIQQ", content, 0)
    return RawMessage(cid, seq, log_t, pub_t, bytes(content[22:]))


class McapChunkReader:
    """pread one chunk, decompress, slice a message. An LRU of decompressed chunks sits in front."""

    layer = "mcap"

    def __init__(self, path: str, cache_chunks: int = 64):
        self.path = path
        self._fd = os.open(path, os.O_RDONLY)
        self._lru: OrderedDict[int, bytes] = OrderedDict()
        self._cap = cache_chunks

    def chunk(self, chunk_offset: int, chunk_len: int) -> bytes:
        hit = chunk_offset in self._lru
        cost.record_read(self.layer, self.path, chunk_offset, chunk_len, cache_hit=hit)
        if hit:
            self._lru.move_to_end(chunk_offset)
            return self._lru[chunk_offset]
        rec = os.pread(self._fd, chunk_len, chunk_offset)
        if rec[0] != OP_CHUNK:
            raise ValueError(f"no chunk record at {chunk_offset} in {self.path}")
        records = decompress_chunk(memoryview(rec)[9:])
        self._lru[chunk_offset] = records
        if len(self._lru) > self._cap:
            self._lru.popitem(last=False)
        return records

    def message(self, chunk_offset: int, chunk_len: int, offset_in_chunk: int) -> RawMessage:
        records = self.chunk(chunk_offset, chunk_len)
        if records[offset_in_chunk] != OP_MESSAGE:
            raise ValueError("locator does not point at a message record")
        (n,) = struct.unpack_from("<Q", records, offset_in_chunk + 1)
        return parse_message(memoryview(records)[offset_in_chunk + 9 : offset_in_chunk + 9 + n])

    def close(self) -> None:
        os.close(self._fd)
