"""raw/<stem>.bag → bundle MCAP file(s) + bundle/timeline/<rec>.parquet.

Layouts:
  * interleaved: one file, mcap/<rec>.mcap, 1 MiB chunks holding every topic (a camera read drags lidar along).
  * by_sensor:   one file per embodiment sensor group, mcap/<rec>/<group>.mcap, chunk sizes per group, so reading
                 video never reads lidar and a telemetry lookup reads kilobytes, not a megabyte.
The timeline's `file` column locates each message; MessageId is layout-independent.

Payloads are copied verbatim (ros1msg) with log_time preserved. The timeline is rebuilt from the written MCAP, not
from writer bookkeeping, and every payload hash must equal the bag's. MessageId = (rec, topic, topic_ordinal), where
the ordinal follows source-bag iteration order within the topic (connections sharing a topic are merged).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from mcap.writer import CompressionType, Writer
from rosbags.highlevel import AnyReader

from alloy_server.io.mcap_chunks import (
    MAGIC, OP_CHANNEL, OP_CHUNK, OP_DATA_END, OP_MESSAGE, decompress_chunk, iter_records, parse_channel, parse_message,
)
from alloy_index.recordings import RECORDINGS, bag_path

CHUNK_SIZE = 1 << 20
GROUP_CHUNK = {"front_camera": 256 << 10, "body_cameras": 256 << 10, "lidar": 1 << 20, "telemetry": 64 << 10,
               "other": 1 << 20}


def topic_groups(prof, topics: set[str]) -> dict[str, str]:
    """by_sensor grouping from the embodiment profile: cameras by sensor, lidars together, the rest telemetry;
    unprofiled topics (camera_info, raw packets, ...) go to `other`."""
    from alloy_server.gen.alloy.v1 import embodiment_pb2 as e
    out = {}
    for s_ in prof.sensors:
        g = s_.name if s_.kind == e.CAMERA else "lidar" if s_.kind in (e.LIDAR_3D, e.LIDAR_2D) else "telemetry"
        for t in s_.topics:
            out[t] = g
    return {t: out.get(t, "other") for t in topics}


def sha128(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()[:16]


def header_stamp(raw: bytes) -> int:
    _, sec, nsec = struct.unpack_from("<III", raw, 0)
    return sec * 1_000_000_000 + nsec


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def write_mcaps(rec: str, bundle: Path, layout: str, prof=None) -> tuple[dict, list[str]]:
    """One pass over the bag. Returns ({(topic, ordinal): sha128} from the bag payloads, relative MCAP paths)."""
    expected: dict[tuple[str, int], bytes] = {}
    ordinals: dict[str, int] = {}
    with AnyReader([bag_path(rec)]) as reader:
        topics = {c.topic for c in reader.connections}
        if layout == "by_sensor":
            groups = topic_groups(prof, topics)
            rel = {g: f"mcap/{rec}/{g}.mcap" for g in set(groups.values())}
        else:
            groups = {t: "all" for t in topics}
            rel = {"all": f"mcap/{rec}.mcap"}
        writers, handles, channels = {}, {}, {}
        for g, r in rel.items():
            path = bundle / r
            path.parent.mkdir(parents=True, exist_ok=True)
            handles[g] = open(path, "wb")
            writers[g] = Writer(handles[g], chunk_size=GROUP_CHUNK.get(g, CHUNK_SIZE), compression=CompressionType.ZSTD)
            writers[g].start(profile="ros1", library="alloy-index")
        for c in reader.connections:
            if c.topic in channels:
                continue
            w = writers[groups[c.topic]]
            sid = w.register_schema(name=c.msgtype, encoding="ros1msg", data=c.msgdef.data.encode())
            first = reader.typestore.fielddefs[c.msgtype][1][:1]
            has_header = bool(first) and first[0][0] == "header" and first[0][1][1] == "std_msgs/msg/Header"
            channels[c.topic] = w.register_channel(
                topic=c.topic, message_encoding="ros1", schema_id=sid,
                metadata={"md5sum": c.digest, "has_header": str(has_header).lower()},
            )
        for conn, t, raw in reader.messages():
            n = ordinals.get(conn.topic, 0)
            ordinals[conn.topic] = n + 1
            expected[(conn.topic, n)] = sha128(raw)
            writers[groups[conn.topic]].add_message(channel_id=channels[conn.topic], log_time=t, publish_time=t,
                                                    data=raw, sequence=n)
        for g in writers:
            writers[g].finish()
            handles[g].close()
    return expected, sorted(rel.values())


def build_timeline(rec: str, mcap: Path, rel: str, ordinals: dict[str, int]) -> dict[str, list]:
    """Timeline rows for one MCAP file, rebuilt from the file itself (not writer bookkeeping)."""
    data = mcap.read_bytes()
    assert data[:8] == MAGIC
    topics: dict[int, str] = {}
    cols: dict[str, list] = {k: [] for k in (
        "topic", "topic_ordinal", "log_time_ns", "header_stamp_ns", "payload_sha128", "msg_len",
        "chunk_offset", "chunk_len", "offset_in_chunk", "file", "_raw_head")}
    # top-level records start after the 8-byte magic; the header record comes first
    for op, off, content in iter_records(data, 8):
        if op == OP_DATA_END:
            break
        if op == OP_CHANNEL:
            cid, _, topic, _ = parse_channel(content)
            topics[cid] = topic
        if op != OP_CHUNK:
            continue
        records = decompress_chunk(content)
        for iop, ioff, icontent in iter_records(records):
            if iop == OP_CHANNEL:
                cid, _, topic, _ = parse_channel(icontent)
                topics[cid] = topic
                continue
            if iop != OP_MESSAGE:
                continue
            m = parse_message(icontent)
            topic = topics[m.channel_id]
            n = ordinals.get(topic, 0)
            ordinals[topic] = n + 1
            cols["topic"].append(topic)
            cols["topic_ordinal"].append(n)
            cols["log_time_ns"].append(m.log_time_ns)
            cols["header_stamp_ns"].append(None)
            cols["payload_sha128"].append(sha128(m.data))
            cols["msg_len"].append(len(m.data))
            cols["chunk_offset"].append(off)
            cols["chunk_len"].append(9 + len(content))
            cols["offset_in_chunk"].append(ioff)
            cols["file"].append(rel)
            cols["_raw_head"].append(bytes(m.data[:16]))
    return cols


def header_topics(mcap: Path) -> tuple[set[str], set[str]]:
    """→ (topics whose message starts with std_msgs/Header, tf2_msgs/TFMessage topics)."""
    from mcap.reader import make_reader
    with open(mcap, "rb") as f:
        summary = make_reader(f).get_summary()
    chans = summary.channels.values()
    tf = {c.topic for c in chans if summary.schemas[c.schema_id].name == "tf2_msgs/msg/TFMessage"}
    return {c.topic for c in chans if c.metadata.get("has_header") == "true"}, tf


def stamp_of(topic: str, head: bytes, htopics: set[str], tf_topics: set[str]) -> int | None:
    if topic in htopics and len(head) >= 12:
        return header_stamp(head)
    if topic in tf_topics and len(head) >= 16 and struct.unpack_from("<I", head, 0)[0] > 0:
        return header_stamp(head[4:])  # first TransformStamped's header (TFMessage has no top-level header)
    return None


def convert(rec: str, bundle: Path, layout: str = "interleaved", prof=None) -> dict:
    (bundle / "mcap").mkdir(parents=True, exist_ok=True)
    (bundle / "timeline").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    expected, files = write_mcaps(rec, bundle, layout, prof)
    t1 = time.perf_counter()
    return finish(rec, bundle, files, expected, t0, t1, layout)


def finish(rec: str, bundle: Path, files: list[str], expected: dict | None, t0: float, t1: float,
           layout: str = "interleaved") -> dict:
    ordinals: dict[str, int] = {}
    cols: dict[str, list] = {}
    htopics, tf_topics = set(), set()
    for rel in files:
        part = build_timeline(rec, bundle / rel, rel, ordinals)
        for k, v in part.items():
            cols.setdefault(k, []).extend(v)
        h, tf = header_topics(bundle / rel)
        htopics |= h
        tf_topics |= tf
    heads = cols.pop("_raw_head")
    cols["header_stamp_ns"] = [stamp_of(t, h, htopics, tf_topics) for t, h in zip(cols["topic"], heads)]
    got = {(t, n): s for t, n, s in zip(cols["topic"], cols["topic_ordinal"], cols["payload_sha128"])}
    if expected is not None and got != expected:
        missing = set(expected) - set(got)
        raise AssertionError(f"{rec}: payload mismatch ({len(missing)} missing, "
                             f"{sum(got.get(k) != v for k, v in expected.items())} differ)")
    schema = pa.schema([
        ("topic", pa.string()), ("topic_ordinal", pa.uint32()),
        ("log_time_ns", pa.int64()), ("header_stamp_ns", pa.int64()), ("payload_sha128", pa.binary(16)),
        ("msg_len", pa.uint32()), ("chunk_offset", pa.uint64()), ("chunk_len", pa.uint32()),
        ("offset_in_chunk", pa.uint32()), ("file", pa.string()),
    ])
    table = pa.Table.from_pydict(cols, schema=schema).sort_by([("topic", "ascending"), ("topic_ordinal", "ascending")])
    table = table.set_column(0, "topic", table.column("topic").dictionary_encode())
    fi = table.schema.get_field_index("file")
    table = table.set_column(fi, "file", table.column("file").dictionary_encode())
    table = table.replace_schema_metadata({"recording_id": rec})
    pq.write_table(table, bundle / "timeline" / f"{rec}.parquet", row_group_size=16384)
    bag = bag_path(rec)
    info = {
        "recording_id": rec, "bag": bag.name, "bag_bytes": bag.stat().st_size, "bag_sha256": file_sha256(bag),
        "mcap_bytes": sum((bundle / f).stat().st_size for f in files), "layout": layout, "files": files,
        "messages": table.num_rows,
        "log_start_ns": int(np.min(cols["log_time_ns"])), "log_end_ns": int(np.max(cols["log_time_ns"])),
        "topics": sorted(set(cols["topic"])), "header_topics": sorted(htopics | tf_topics),
        "convert_s": round(t1 - t0, 1), "timeline_s": round(time.perf_counter() - t1, 1),
    }
    (bundle / "mcap" / f"{rec}.json").write_text(json.dumps(info, indent=1))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--timeline-only", action="store_true", help="rebuild timelines from existing MCAPs")
    ap.add_argument("recs", nargs="*", default=list(RECORDINGS))
    a = ap.parse_args()
    for rec in a.recs:
        if a.timeline_only:
            prev = json.loads((a.bundle / "mcap" / f"{rec}.json").read_text())
            files = prev.get("files", [f"mcap/{rec}.mcap"])
            info = finish(rec, a.bundle, files, None, time.perf_counter(), time.perf_counter(),
                          prev.get("layout", "interleaved"))
        else:
            info = convert(rec, a.bundle)
        print(json.dumps({k: info[k] for k in ("recording_id", "messages", "bag_bytes", "mcap_bytes", "convert_s",
                                               "timeline_s")}))


if __name__ == "__main__":
    main()
