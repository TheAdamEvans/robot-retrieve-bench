"""Recording registry: discovered from the source (raw/*.bag + SCAND_index.csv), configured in index/index.textproto.

Historical short ids are aliases so existing labels stay valid; new recordings get ids from their bag stem via the
source's id_template (A_Spot_Bass_Garage_Fri_Nov_26_134 -> Bass_Garage_134). The CSV supplies metadata only.
A collision fails loudly.
"""
from __future__ import annotations

import csv
import functools
import json
import re
from pathlib import Path

from google.protobuf import text_format

from alloy_server.gen.alloy.v1 import index_pb2

SCAND_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = SCAND_ROOT / "index" / "index.textproto"


@functools.lru_cache(maxsize=None)
def spec(path: str | None = None) -> index_pb2.IndexSpec:
    return text_format.Parse(Path(path or SPEC_PATH).read_text(), index_pb2.IndexSpec())


STEM = re.compile(r"^(?P<person>[A-Z])_(?P<robot>Spot|Jackal)_(?P<route>.+)_(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)_"
                  r"(?P<month>[A-Za-z]+)_(?P<day>\d+)_(?P<fr>\d+)$")


def discover(sp: index_pb2.IndexSpec | None = None) -> dict[str, str]:
    """recording id -> bag stem, for every bag present in the source directory."""
    sp = sp or spec()
    raw = SCAND_ROOT / sp.source.raw_dir
    out: dict[str, str] = {}
    for bag in sorted(raw.glob("*.bag")):
        stem = bag.stem
        if stem in sp.source.aliases:
            rid = sp.source.aliases[stem]
        elif (m := STEM.match(stem)):
            rid = sp.source.id_template.format(**m.groupdict())
        else:
            rid = stem
        if rid in out:
            raise ValueError(f"recording id collision: {rid!r} for {out[rid]} and {stem}")
        out[rid] = stem
    return out


RECORDINGS: dict[str, str] = discover()
RAW = SCAND_ROOT / spec().source.raw_dir


def held_out() -> set[str]:
    return set(spec().source.held_out)


def bundle_root() -> Path:
    return SCAND_ROOT / spec().sink.path


def robot(rec: str) -> str:
    """Embodiment from the intake report when available (topics decide, not file names)."""
    p = bundle_root() / "intake" / f"{rec}.json"
    if p.exists():
        return json.loads(p.read_text())["embodimentId"]
    return "jackal" if "_Jackal_" in RECORDINGS[rec] else "spot"


@functools.lru_cache(maxsize=None)
def metadata() -> dict[str, dict]:
    """recording id -> its SCAND_index.csv row (tags, locations, split)."""
    csv_path = (SCAND_ROOT / spec().source.index_csv).resolve()
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = {r["FileName"].strip(): r for r in csv.DictReader(f)}
    return {rid: rows.get(stem, {}) for rid, stem in RECORDINGS.items()}


def bag_path(rec: str) -> Path:
    return RAW / f"{RECORDINGS[rec]}.bag"
