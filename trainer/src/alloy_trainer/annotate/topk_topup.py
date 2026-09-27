"""Blind, targeted L2 top-k top-up with a per-job spending limit.

This selects only missing (intent, window) judgments from one frozen retriever
run. It does not add attribute/random pool windows or disclose the proposing
retriever in the labeller brief. Dry-run is the default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import alloy_index.annotate
from alloy_index.annotate.headless import MODEL, run_claude
from alloy_index.annotate.store import LABELS, ensure_campaign, load_labels, work_dir, write_job
from alloy_index.recordings import RECORDINGS, SCAND_ROOT
from alloy_trainer.eval import querysets

BRIEF = Path(alloy_index.annotate.__file__).parent / "prompts" / "l2_judgments.md"


def assignments(rows: list[dict], queries: list, judged: set[tuple[str, str]], config: str,
                k: int) -> dict[str, dict]:
    """One blinded assignment per answerable intent; deduplicate paraphrases."""
    by_query = {q.query_id: q for q in queries}
    out: dict[str, dict] = {}
    for row in rows:
        if row["config"] != config or row["query_id"] not in by_query:
            continue
        q = by_query[row["query_id"]]
        if q.expect_abstain:
            continue
        intent = q.intent_group_id
        job = out.setdefault(intent, {"intent": q.intent, "scope": list(q.scope.recording_ids), "windows": set()})
        job["windows"].update(w for w in row["windows"][:k] if (intent, w) not in judged)
    return {intent: {**job, "windows": sorted(job["windows"])}
            for intent, job in sorted(out.items()) if job["windows"]}


def brief_for(intent: str, job_name: str, assignment: dict) -> str:
    by_rec: dict[str, list[str]] = defaultdict(list)
    for wid in assignment["windows"]:
        by_rec[wid.split(":")[0]].append(wid)
    listing = "\n".join(f"- {rec}: " + ", ".join(sorted(ws)) for rec, ws in sorted(by_rec.items()))
    scope = ", ".join(assignment["scope"]) if assignment["scope"] else "all indexed recordings"
    template = BRIEF.read_text().replace(
        "The\nwindows come from a pool: several retrieval systems' results, plus attribute-based and random windows.",
        "The\nwindows come from a blinded evaluation pool.")
    return (template.replace("{JOB}", job_name).replace("{INTENT_ID}", intent)
            .replace("{INTENT}", assignment["intent"]).replace("{SCOPE}", scope)
            .replace("{N}", str(len(assignment["windows"]))).replace("{WINDOWS}", listing))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="query_set", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--campaign", required=True, help="opaque campaign name shown to the judge")
    ap.add_argument("--budget-per-job", type=float, default=10)
    ap.add_argument("--parallel", type=int, default=5)
    ap.add_argument("--priority", type=int, default=-1, help="label precedence for the campaign")
    ap.add_argument("--intent", action="append", help="limit the top-up to this intent group; repeat as needed")
    ap.add_argument("--execute", action="store_true", help="run paid Opus jobs; otherwise print a dry run")
    a = ap.parse_args()
    if a.k < 1 or a.budget_per_job <= 0 or a.parallel < 1:
        ap.error("k, budget-per-job, and parallel must be positive")
    run_path = SCAND_ROOT / "results" / "eval" / a.run / "runs.jsonl"
    rows = [json.loads(line) for line in run_path.read_text().splitlines()]
    queries = querysets.load([a.query_set])
    judged = {(r["intentGroupId"], r["windowId"]) for r in load_labels("judgment", ("train", "eval")).values()
              if r.get("grade", 0) >= 0}
    work = assignments(rows, queries, judged, a.config, a.k)
    if a.intent:
        work = {intent: item for intent, item in work.items() if intent in a.intent}
    if not work:
        raise ValueError("no missing judgments for the selected run, config, set and depth")
    needed_recordings = {window.split(":", 1)[0] for item in work.values() for window in item["windows"]}
    missing_recordings = sorted(needed_recordings - RECORDINGS.keys())
    if missing_recordings:
        raise ValueError(f"recordings unavailable in this worktree: {', '.join(missing_recordings)}")
    manifest = {"run": a.run, "config": a.config, "query_set": a.query_set, "k": a.k,
                "campaign": a.campaign, "jobs": {intent: len(item["windows"]) for intent, item in work.items()},
                "total_windows": sum(len(item["windows"]) for item in work.values()),
                "max_cost_usd": round(len(work) * a.budget_per_job, 2)}
    manifest["assignment_sha256"] = hashlib.sha256(json.dumps(work, sort_keys=True).encode()).hexdigest()
    print(json.dumps(manifest, indent=2), flush=True)
    if not a.execute:
        return
    if (LABELS / "metadata" / a.campaign / "campaign.json").exists():
        raise ValueError(f"campaign already exists: {a.campaign}; use a new opaque campaign name")
    ensure_campaign(a.campaign, kinds=["judgment"], brief=BRIEF.name, judge=MODEL, use="eval", priority=a.priority,
                    purpose="Blinded top-rank relevance top-up on previously unjudged windows.")
    jobs = []
    for intent, item in work.items():
        job = f"l2-{intent}-{a.campaign}"
        brief = brief_for(intent, job, item)
        (work_dir(job) / "brief.md").write_text(brief)
        write_job(a.campaign, job, {"intent": intent, "windows": item["windows"],
                                    "assignment_sha256": hashlib.sha256(json.dumps(item["windows"]).encode()).hexdigest()})
        jobs.append((job, brief))
    def do_job(pair: tuple[str, str]) -> dict:
        job, brief = pair
        allowed = ["Bash(uv run scandq:*)", f"Bash(SCANDQ_JOB={job} uv run scandq:*)",
                   "Bash(mkdir -p labels/.work/*)", "Bash(ls:*)", "Read", "Write"]
        return run_claude(brief, job, SCAND_ROOT, allowed, budget_usd=str(a.budget_per_job), campaign=a.campaign)
    with ThreadPoolExecutor(max_workers=min(a.parallel, len(jobs))) as pool:
        results = list(pool.map(do_job, jobs))
    for result in results:
        print(json.dumps({key: result[key] for key in ("job", "exit", "cost_usd", "turns", "minutes")}), flush=True)
    total = round(sum(r["cost_usd"] for r in results), 2)
    campaign_path = LABELS / "metadata" / a.campaign / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    campaign.update({"jobs": len(results), "cost_usd": total, "assignment_sha256": manifest["assignment_sha256"],
                     "source_run": a.run, "source_config": a.config, "top_k": a.k})
    submitted = set()
    for shard in (LABELS / "eval" / "judgments").glob(f"{a.campaign}.*.jsonl"):
        for line in shard.read_text().splitlines():
            row = json.loads(line)
            if row.get("grade", 0) >= 0:
                submitted.add((row["intentGroupId"], row["windowId"]))
    missing = {intent: sorted(w for w in item["windows"] if (intent, w) not in submitted)
               for intent, item in work.items()}
    missing = {intent: windows for intent, windows in missing.items() if windows}
    campaign["submitted"] = sum(len(item["windows"]) for item in work.values()) - sum(map(len, missing.values()))
    campaign["missing"] = missing
    campaign_path.write_text(json.dumps(campaign, indent=1) + "\n")
    print(json.dumps({"cost_usd": total, "max_cost_usd": manifest["max_cost_usd"],
                      "submitted": campaign["submitted"], "missing": missing}), flush=True)
    if missing:
        raise RuntimeError("judging incomplete: see the campaign's missing assignments")


if __name__ == "__main__":
    main()
