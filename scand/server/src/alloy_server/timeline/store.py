"""In-RAM per-topic timelines and the point operators over them.

Times are int64 ns. `log_time` = availability (when the recorder received the message); `header_stamp` =
measurement time where the message type carries a std_msgs/Header.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..gen.alloy.v1 import common_pb2
from ..io import parquet_reader
from ..io.mcap_chunks import McapChunkReader, RawMessage

NO_HEADER = np.iinfo(np.int64).min


@dataclass
class TopicTimeline:
    topic: str
    log_ns: np.ndarray        # sorted ascending (ordinal order == bag order, which is log-time order)
    header_ns: np.ndarray     # NO_HEADER where absent
    ordinal: np.ndarray
    sha128: np.ndarray        # (n, 16) uint8
    chunk_offset: np.ndarray
    chunk_len: np.ndarray
    offset_in_chunk: np.ndarray
    msg_len: np.ndarray

    def __len__(self) -> int:
        return len(self.log_ns)

    def last_before(self, t_ns: int, inclusive: bool = False) -> int | None:
        i = np.searchsorted(self.log_ns, t_ns, side="right" if inclusive else "left") - 1
        return int(i) if i >= 0 else None

    def first_after(self, t_ns: int, inclusive: bool = True) -> int | None:
        i = np.searchsorted(self.log_ns, t_ns, side="left" if inclusive else "right")
        return int(i) if i < len(self.log_ns) else None

    def nearest(self, t_ns: int) -> int | None:
        if not len(self.log_ns):
            return None
        i = int(np.searchsorted(self.log_ns, t_ns))
        cands = [j for j in (i - 1, i) if 0 <= j < len(self.log_ns)]
        return min(cands, key=lambda j: abs(int(self.log_ns[j]) - t_ns))

    def range(self, t0_ns: int, t1_ns: int) -> slice:
        return slice(int(np.searchsorted(self.log_ns, t0_ns)), int(np.searchsorted(self.log_ns, t1_ns, side="right")))


class Recording:
    def __init__(self, bundle: Path, rec: str):
        self.id = rec
        self.info = json.loads((bundle / "mcap" / f"{rec}.json").read_text())
        self.start_ns, self.end_ns = self.info["log_start_ns"], self.info["log_end_ns"]
        self.bag_bytes = self.info["bag_bytes"]
        t = parquet_reader.read_table(str(bundle / "timeline" / f"{rec}.parquet"), layer="timeline")
        topic = t.column("topic").to_numpy(zero_copy_only=False).astype(str)
        cols = {c: t.column(c).to_numpy(zero_copy_only=False) for c in
                ("topic_ordinal", "log_time_ns", "chunk_offset", "chunk_len", "offset_in_chunk", "msg_len")}
        header = t.column("header_stamp_ns").fill_null(NO_HEADER).to_numpy()
        sha = np.frombuffer(b"".join(t.column("payload_sha128").to_pylist()), dtype=np.uint8).reshape(-1, 16)
        self.topics: dict[str, TopicTimeline] = {}
        for name in np.unique(topic):
            m = topic == name
            order = np.argsort(cols["topic_ordinal"][m], kind="stable")
            self.topics[name] = TopicTimeline(
                name, cols["log_time_ns"][m][order].astype(np.int64), header[m][order].astype(np.int64),
                cols["topic_ordinal"][m][order], sha[m][order], cols["chunk_offset"][m][order].astype(np.int64),
                cols["chunk_len"][m][order].astype(np.int64), cols["offset_in_chunk"][m][order].astype(np.int64),
                cols["msg_len"][m][order].astype(np.int64),
            )
        self._mcap = McapChunkReader(str(bundle / "mcap" / f"{rec}.mcap"))

    def t_rel(self, t_ns: int) -> float:
        return (t_ns - self.start_ns) / 1e9

    def t_abs(self, t_s: float) -> int:
        return self.start_ns + int(round(t_s * 1e9))

    def message_id(self, topic: str, i: int) -> common_pb2.MessageId:
        tl = self.topics[topic]
        return common_pb2.MessageId(recording_id=self.id, topic=topic, topic_ordinal=int(tl.ordinal[i]),
                                    payload_sha256_128=tl.sha128[i].tobytes())

    def read(self, topic: str, i: int) -> RawMessage:
        tl = self.topics[topic]
        return self._mcap.message(int(tl.chunk_offset[i]), int(tl.chunk_len[i]), int(tl.offset_in_chunk[i]))

    def index_of(self, topic: str, ordinal: int) -> int:
        tl = self.topics[topic]
        i = int(np.searchsorted(tl.ordinal, ordinal))
        if i >= len(tl) or tl.ordinal[i] != ordinal:
            raise KeyError(f"{self.id}{topic}#{ordinal}")
        return i


def mid_str(m: common_pb2.MessageId) -> str:
    return f"{m.recording_id}{m.topic}#{m.topic_ordinal}"


def parse_mid(s: str) -> tuple[str, str, int]:
    rec, rest = s.split("/", 1)
    topic, ordinal = rest.rsplit("#", 1)
    return rec, "/" + topic, int(ordinal)
