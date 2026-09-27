"""Episode judging for challenge question sets (benchmark/challenges/<set>/): one headless Opus job per intent.

Leads are the question authors' candidate windows plus uncited discovery spans, capped per intent by the set's
judging plan. The judge sees neutral episode labels in a seeded shuffle, never the candidates' hypotheses, reasons
or roles. Output: `episode` labels (clause verdicts, anchors, grade) plus `judgment` labels for the windows each
episode covers, so the benchmark and training read them like any other judgment.

    uv run python -m alloy_trainer.annotate.episodes --set l1_compositions_v1 --split dev            # writes briefs
    uv run python -m alloy_trainer.annotate.episodes --set l1_compositions_v1 --split dev --run      # paid: runs jobs
"""
from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import alloy_index.annotate
from alloy_index.annotate.headless import run_claude
from alloy_index.annotate.store import ensure_campaign, work_dir, write_job
from alloy_trainer.challenge.sweep import exhaustive_campaign, sweep_path
from alloy_index.recordings import SCAND_ROOT

BRIEF = Path(alloy_index.annotate.__file__).parent / "prompts" / "l2_episodes.md"
TOOLS = ["Bash(uv run scandq:*)", "Bash(SCANDQ_JOB={job} uv run scandq:*)", "Bash(mkdir -p labels/.work/*)",
         "Bash(ls:*)", "Read", "Write"]


def _win(wid: str) -> tuple[str, int]:
    rec, end = wid.split(":")
    return rec, int(end)


def leads(q: dict) -> list[dict]:
    cap = q.get("judgingPlan", {}).get("maxInitialEpisodeIntentJudgments", 3)
    out = []
    for c in q.get("candidateContrasts", []):
        rec, end = _win(c["windowId"])
        disc = next((d for d in q["discoverySpans"] if d["recordingId"] == rec and d["startS"] <= end <= d["endS"] + 4), None)
        ctx = (max(0, disc["startS"] - 4), disc["endS"] + 4) if disc else (max(0, end - 12), end + 8)
        out.append({"recording": rec, "window": c["windowId"], "context": ctx})
    for d in q["discoverySpans"]:  # reserve: an uncited discovery span, starting at its middle window
        if len(out) >= cap:
            break
        if any(l["recording"] == d["recordingId"] and d["startS"] - 4 <= _win(l["window"])[1] <= d["endS"] + 4 for l in out):
            continue
        mid = int((d["startS"] + d["endS"]) / 2) + 2
        out.append({"recording": d["recordingId"], "window": f"{d['recordingId']}:{mid:04d}",
                    "context": (max(0, d["startS"] - 4), d["endS"] + 4)})
    out = out[:cap]
    random.Random(q["intentGroupId"]).shuffle(out)  # hide the authors' ordering (candidate before near miss)
    for i, l in enumerate(out, 1):
        l["episode_id"] = f"E{i}"
    return out


def contract(readme: str) -> str:
    i = readme.index("## Judging and measurement contract")
    return readme[i:].strip()


def render(q: dict, job: str, set_dir: Path) -> tuple[str, list[dict]]:
    ls = leads(q)
    listing = "\n".join(f"- **{l['episode_id']}**: `{l['window']}`; explore {l['context'][0]:.0f}–{l['context'][1]:.0f} s "
                        f"of {l['recording']}" for l in ls)
    clauses = "\n".join(f"{i}. {c}" for i, c in enumerate(q["requiredClauses"], 1))
    split = "test" if q["querySet"].endswith("test") else "dev"
    rep = {"{JOB}": job, "{INTENT_ID}": q["intentGroupId"], "{SPLIT}": split, "{UTTERANCE}": q["utterance"],
           "{SCOPE}": ", ".join(q["scope"]["recordingIds"]), "{CLAUSES}": clauses,
           "{RETURNS}": q.get("returnRequirements", ""), "{HARD_NEGATIVES}": q.get("hardNegativeRecipe", ""),
           "{GAPS}": q.get("capabilityGaps", ""), "{LEADS}": listing, "{N}": str(len(ls)),
           "{CONTRACT}": contract((set_dir / "README.md").read_text())}
    text = BRIEF.read_text()
    for k, v in rep.items():
        text = text.replace(k, v)
    assert not re.search(r"\{[A-Z_]+\}", text), "unfilled placeholder in the brief"
    return text, ls


EXHAUSTIVE = Path(alloy_index.annotate.__file__).parent / "prompts" / "l2_exhaustive.md"
CHUNK_S, JOB_S = 30.0, 150.0


def chunks(sweep: dict) -> list[dict]:
    """The union of the sweep's candidate spans per recording, split into <= CHUNK_S pieces (2 s overlap)."""
    out = []
    for rec in sweep["scope"]:
        spans = sorted((max(0.0, c["start_s"]), c["end_s"]) for m in sweep["members"].values()
                       for c in m["candidates"].get(rec, []))
        merged = []
        for s0, e0 in spans:
            if merged and s0 <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e0)
            else:
                merged.append([s0, e0])
        for s0, e0 in merged:
            n = max(1, int(np.ceil((e0 - s0) / CHUNK_S)))
            step = (e0 - s0) / n
            for k in range(n):
                out.append({"recording": rec, "start_s": round(max(0.0, s0 + k * step - (2 if k else 0)), 1),
                            "end_s": round(s0 + (k + 1) * step, 1)})
    for i, c in enumerate(out, 1):
        c["chunk_id"] = f"K{i}"
    return out


def render_exhaustive(q: dict, sweep: dict, job: str, cs: list[dict], set_dir: Path) -> str:
    listing = "\n".join(f"- **{c['chunk_id']}**: {c['recording']} {c['start_s']:.1f}–{c['end_s']:.1f} s" for c in cs)
    screen = "\n".join(f"- {name}: {m['loose']} (the question: {m['exact']})" for name, m in sweep["members"].items())
    clauses = "\n".join(f"{i}. {c}" for i, c in enumerate(q["requiredClauses"], 1))
    rep = {"{JOB}": job, "{INTENT_ID}": q["intentGroupId"], "{SPLIT}": sweep["split"], "{UTTERANCE}": q["utterance"],
           "{CLAUSES}": clauses, "{RETURNS}": q.get("returnRequirements", ""),
           "{HARD_NEGATIVES}": q.get("hardNegativeRecipe", ""), "{GAPS}": q.get("capabilityGaps", ""),
           "{SCREEN}": screen, "{CHUNKS}": listing, "{CONTRACT}": contract((set_dir / "README.md").read_text())}
    text = EXHAUSTIVE.read_text()
    for k, v in rep.items():
        text = text.replace(k, v)
    assert not re.search(r"\{[A-Z_]+\}", text), "unfilled placeholder in the brief"
    return text


def exhaustive_jobs(q: dict, set_name: str, set_dir: Path) -> list[tuple[str, str, list[dict]]]:
    sweep = json.loads(sweep_path(set_name, q["intentGroupId"]).read_text())
    assert sweep["sweep_recall_ok"], "the sweep is not a superset of judged-relevant episodes"
    cs, jobs, cur, dur = chunks(sweep), [], [], 0.0
    for c in cs:
        if cur and dur + c["end_s"] - c["start_s"] > JOB_S:
            jobs.append(cur)
            cur, dur = [], 0.0
        cur.append(c)
        dur += c["end_s"] - c["start_s"]
    if cur:
        jobs.append(cur)
    out = []
    for k, part in enumerate(jobs, 1):
        job = f"ex-{q['intentGroupId']}-j{k}"
        out.append((job, render_exhaustive(q, sweep, job, part, set_dir), part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--intents", help="comma-separated intent ids (default: all in the split)")
    ap.add_argument("--run", action="store_true", help="run the headless Opus jobs (spends money)")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--budget-usd", default="15", help="per-job cap")
    ap.add_argument("--exhaustive", action="store_true", help="judge every sweep chunk (complete ground truth)")
    a = ap.parse_args()
    set_dir = SCAND_ROOT / "benchmark" / "challenges" / a.set
    qs = [json.loads(l) for l in (set_dir / f"queries_{a.split}.jsonl").read_text().splitlines() if l.strip()]
    if a.intents:
        qs = [q for q in qs if q["intentGroupId"] in a.intents.split(",")]
    slug = exhaustive_campaign(a.set) if a.exhaustive else f"episodes-{a.set}-leads"
    jobs = []  # (job, brief, assignment recorded in labels/metadata/<campaign>/jobs/<job>.json)
    for q in qs:
        if a.exhaustive:
            if not sweep_path(a.set, q["intentGroupId"]).exists():
                print(json.dumps({"intent": q["intentGroupId"], "skipped": "no sweep"}))
                continue
            for job, text, part in exhaustive_jobs(q, a.set, set_dir):
                (work_dir(job) / "brief.md").write_text(text)
                jobs.append((job, text, {"intent": q["intentGroupId"], "split": a.split, "chunks": part}))
                print(json.dumps({"job": job, "chunks": len(part), "seconds": round(sum(c["end_s"] - c["start_s"] for c in part))}))
            continue
        job = f"ep-{q['intentGroupId']}"
        text, ls = render(q, job, set_dir)
        (work_dir(job) / "brief.md").write_text(text)
        jobs.append((job, text, {"intent": q["intentGroupId"], "split": a.split, "leads": ls}))  # never shown to the judge
        print(json.dumps({"job": job, "leads": [(l["episode_id"], l["window"]) for l in ls]}))
    if not a.run:
        return
    ensure_campaign(slug, kinds=["episode", "judgment"], challenge=a.set, judge="claude-opus-5-5",
                    brief=(EXHAUSTIVE if a.exhaustive else BRIEF).name, priority=2 if a.exhaustive else 0,
                    **({"exhaustive": True} if a.exhaustive else {}),
                    purpose=("Complete ground truth: every chunk of the numeric sweep (sweep/) judged." if a.exhaustive
                             else "Episode judgments of the question authors' leads; dev intents to train/, test to eval/."))
    for job, _, assignment in jobs:
        write_job(slug, job, assignment)
    tools = lambda job: [t.replace("{job}", job) for t in TOOLS]
    with ThreadPoolExecutor(a.parallel) as pool:
        results = list(pool.map(lambda j: run_claude(j[1], j[0], SCAND_ROOT, tools(j[0]), budget_usd=a.budget_usd,
                                                     campaign=slug), jobs))
    total = sum(r["cost_usd"] for r in results)
    for r in results:
        print(json.dumps({k: r[k] for k in ("job", "exit", "cost_usd", "turns", "minutes")}))
    print(f"total ${total:.2f} over {len(results)} jobs (campaign {slug})")


if __name__ == "__main__":
    main()
