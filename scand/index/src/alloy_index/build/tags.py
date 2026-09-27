"""Recording-level tag documents from SCAND_index.csv (the TAGS baseline's whole view of the corpus)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from alloy_index.recordings import RECORDINGS, SCAND_ROOT


def build(bundle: Path, recordings: list[str] | None = None) -> dict:
    rows = {}
    with open(SCAND_ROOT.parent / "SCAND_index.csv", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows[row.get("FileName", "").strip()] = row
    docs = {}
    for rec, stem in RECORDINGS.items():
        if recordings is not None and rec not in recordings:
            continue
        r = rows.get(stem, {})
        docs[rec] = " ".join(x.strip() for x in (r.get("Tags", "").replace(",", " "), r.get("Robot", ""),
                                                   r.get("Start Location", ""), r.get("End Location", "")))
    p = bundle / "index" / "tags.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(docs, indent=1))
    return docs
