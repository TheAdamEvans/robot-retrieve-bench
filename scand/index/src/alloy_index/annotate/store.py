"""Read label shards with explicit revision precedence, independent of filesystem timestamps."""
from __future__ import annotations

import json
from pathlib import Path


def key_of(kind: str, row: dict) -> str:
    if kind == "attributes":
        return row["segmentId"]
    if kind == "episode":
        return f'{row["intentGroupId"]}|{row.get("jobId", "")}|{row["episodeId"]}'
    return f'{row["intentGroupId"]}|{row["windowId"]}'


def load_labels(ann: Path, kind: str) -> dict[str, dict]:
    """Higher precedence wins; ambiguous cross-shard revisions fail instead of choosing an arbitrary grade.

    Unlisted shards have precedence 0. Within one shard, later records supersede earlier records.
    """
    path = ann / "labels" / "precedence.json"
    priorities = json.loads(path.read_text()).get(kind, {}) if path.exists() else {}
    latest, origins = {}, {}
    for shard in sorted((ann / "labels" / kind).glob("*.jsonl"), key=lambda p: (priorities.get(p.name, 0), p.name)):
        priority = priorities.get(shard.name, 0)
        for line in shard.read_text().splitlines():
            row = json.loads(line)
            key = key_of(kind, row)
            if key in origins:
                prev_priority, prev_shard = origins[key]
                if priority == prev_priority and shard.name != prev_shard and row != latest[key]:
                    raise ValueError(f"Conflicting {kind} label {key} in {prev_shard} and {shard.name}; "
                                     "set explicit shard priorities in labels/precedence.json")
            latest[key], origins[key] = row, (priority, shard.name)
    return latest
