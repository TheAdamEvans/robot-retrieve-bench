"""Tiny ROS1 decoders for the payloads the server must render (no rosbags dependency on the server)."""
from __future__ import annotations

import struct


def _skip_header(raw: bytes) -> int:
    (flen,) = struct.unpack_from("<I", raw, 12)
    return 16 + flen


def compressed_image(raw: bytes) -> tuple[str, bytes]:
    """sensor_msgs/CompressedImage → (format, jpeg/png bytes)."""
    off = _skip_header(raw)
    (n,) = struct.unpack_from("<I", raw, off)
    fmt = raw[off + 4 : off + 4 + n].decode()
    off += 4 + n
    (n,) = struct.unpack_from("<I", raw, off)
    return fmt, raw[off + 4 : off + 4 + n]
