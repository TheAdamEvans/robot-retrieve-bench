"""The label store: labels/{train,eval,agreement}/<kind>/<campaign>.<job>.jsonl plus labels/metadata/<campaign>/.

The folder a label lives in is what it may be used for:
  train/      may be used for training and tuning (evaluating on it is a dev result, not a test result)
  eval/       never trained or tuned on: benchmark labels and held-out test questions
  agreement/  second opinions, used only for agreement statistics

Every record carries its `campaign`; labels/metadata/<campaign>/ holds campaign.json (purpose, judge, date, cost,
priority), jobs/<job>.json (what each job was given, its measured cost and report) and audit.jsonl (its tool calls).
Precedence between campaigns comes from campaign.json `priority`; ties that disagree fail instead of choosing.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

from alloy_index.recordings import SCAND_ROOT

LABELS = SCAND_ROOT / "labels"
USES = ("train", "eval", "agreement")
KIND_DIR = {"attributes": "attributes", "judgment": "judgments", "episode": "episodes"}


def key_of(kind: str, row: dict) -> str:
    if kind == "attributes":
        return row["segmentId"]
    if kind == "episode":
        return f'{row["intentGroupId"]}|{row.get("jobId", "")}|{row["episodeId"]}'
    return f'{row["intentGroupId"]}|{row["windowId"]}'


@functools.lru_cache(maxsize=None)
def _campaign(root: Path, slug: str) -> dict:
    p = root / "metadata" / slug / "campaign.json"
    return json.loads(p.read_text()) if p.exists() else {}


def campaign(slug: str, root: Path = LABELS) -> dict:
    return _campaign(Path(root), slug)


def priority(slug: str, root: Path = LABELS) -> float:
    return float(campaign(slug, root).get("priority", 0))


@functools.lru_cache(maxsize=None)
def test_intents(root: Path = SCAND_ROOT) -> frozenset[str]:
    """Held-out challenge intents: their labels may only live in eval/."""
    return frozenset(json.loads(l)["intentGroupId"] for f in (root / "benchmark" / "challenges").glob("*/queries_test.jsonl")
                     for l in f.read_text().splitlines() if l.strip())


def use_for(kind: str, row: dict, slug: str, root: Path = LABELS) -> str:
    """Where a new record belongs: a campaign may pin its use (agreement); attributes train; challenge dev intents
    train; held-out challenge intents and benchmark intents evaluate."""
    pinned = campaign(slug, root).get("use")
    if pinned:
        return pinned
    if kind == "attributes":
        return "train"
    intent = row.get("intentGroupId", "")
    if intent.startswith("l1x_") and intent not in test_intents():
        return "train"
    return "eval"


def shard_path(use: str, kind: str, slug: str, job: str, root: Path = LABELS) -> Path:
    return Path(root) / use / KIND_DIR[kind] / f"{slug}.{job}.jsonl"


def load_labels(kind: str, uses: tuple[str, ...] = ("train", "eval"), root: Path = LABELS) -> dict[str, dict]:
    """Latest record per key across the given use folders. Higher campaign priority wins; within one shard the later
    record wins; two shards at equal priority that disagree raise. A key may live in only one use folder."""
    root = Path(root)
    shards = []
    for use in uses:
        for p in sorted((root / use / KIND_DIR[kind]).glob("*.jsonl")):
            shards.append((priority(p.name.split(".", 1)[0], root), p.name, use, p))
    latest, origins = {}, {}
    for prio, name, use, p in sorted(shards):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = key_of(kind, row)
            if key in origins:
                p_prio, p_name, p_use = origins[key]
                if p_use != use:
                    raise ValueError(f"{kind} label {key} is in both {p_use}/ and {use}/; a label has one use")
                if prio == p_prio and name != p_name and row != latest[key]:
                    raise ValueError(f"Conflicting {kind} label {key} in {p_name} and {name}; "
                                     "give one campaign a higher priority in its campaign.json")
            latest[key], origins[key] = row, (prio, name, use)
    return latest


def ensure_campaign(slug: str, root: Path = LABELS, **fields) -> Path:
    """Create labels/metadata/<slug>/ with a campaign.json (existing fields are kept, new ones added)."""
    d = Path(root) / "metadata" / slug
    (d / "jobs").mkdir(parents=True, exist_ok=True)
    p = d / "campaign.json"
    doc = json.loads(p.read_text()) if p.exists() else {"campaign": slug, "priority": 0}
    for k, v in fields.items():
        doc.setdefault(k, v)
    p.write_text(json.dumps(doc, indent=1) + "\n")
    _campaign.cache_clear()
    return d


def write_job(slug: str, job: str, record: dict, root: Path = LABELS) -> None:
    """Merge `record` into labels/metadata/<slug>/jobs/<job>.json."""
    p = ensure_campaign(slug, root) / "jobs" / f"{job}.json"
    doc = json.loads(p.read_text()) if p.exists() else {"job": job, "campaign": slug}
    doc.update(record)
    p.write_text(json.dumps(doc, indent=1) + "\n")


def work_dir(job: str, root: Path = LABELS) -> Path:
    """Scratch for one job (brief, batches, CLI output): labels/.work/<job>/, never committed."""
    d = Path(root) / ".work" / job
    d.mkdir(parents=True, exist_ok=True)
    return d
