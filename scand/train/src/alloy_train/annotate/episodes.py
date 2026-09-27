"""Episode judging for challenge question sets (benchmark/challenges/<set>/): one headless Opus job per intent.

Leads are the question authors' candidate windows plus uncited discovery spans, capped per intent by the set's
judging plan. The judge sees neutral episode labels in a seeded shuffle, never the candidates' hypotheses, reasons
or roles. Output: `episode` labels (clause verdicts, anchors, grade) plus `judgment` labels for the windows each
episode covers, so the benchmark and training read them like any other judgment.

    uv run python -m alloy_train.annotate.episodes --set l1_compositions_v1 --split dev            # writes briefs
    uv run python -m alloy_train.annotate.episodes --set l1_compositions_v1 --split dev --run      # paid: runs jobs
"""
from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import alloy_index.annotate
from alloy_index.annotate.headless import run_claude
from alloy_index.recordings import SCAND_ROOT

BRIEF = Path(alloy_index.annotate.__file__).parent / "prompts" / "l2_episodes.md"
TOOLS = ["Bash(uv run scandq:*)", "Bash(SCANDQ_JOB={job} uv run scandq:*)", "Bash(mkdir -p annotations/tmp/*)",
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--intents", help="comma-separated intent ids (default: all in the split)")
    ap.add_argument("--run", action="store_true", help="run the headless Opus jobs (spends money)")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--budget-usd", default="15", help="per-job cap")
    a = ap.parse_args()
    set_dir = SCAND_ROOT / "benchmark" / "challenges" / a.set
    qs = [json.loads(l) for l in (set_dir / f"queries_{a.split}.jsonl").read_text().splitlines() if l.strip()]
    if a.intents:
        qs = [q for q in qs if q["intentGroupId"] in a.intents.split(",")]
    jobs = []
    for q in qs:
        job = f"ep-{q['intentGroupId']}"
        text, ls = render(q, job, set_dir)
        d = SCAND_ROOT / "annotations" / "tmp" / job
        d.mkdir(parents=True, exist_ok=True)
        (d / "brief.md").write_text(text)
        (d / "leads.json").write_text(json.dumps(ls, indent=1))  # for the record; never shown to the judge
        jobs.append((job, text))
        print(json.dumps({"job": job, "leads": [(l["episode_id"], l["window"]) for l in ls]}))
    if not a.run:
        return
    tools = lambda job: [t.replace("{job}", job) for t in TOOLS]
    with ThreadPoolExecutor(a.parallel) as pool:
        results = list(pool.map(lambda j: run_claude(j[1], j[0], SCAND_ROOT, tools(j[0]), budget_usd=a.budget_usd), jobs))
    total = sum(r["cost_usd"] for r in results)
    for r in results:
        print(json.dumps({k: r[k] for k in ("job", "exit", "cost_usd", "turns", "minutes")}))
    print(f"total ${total:.2f} over {len(results)} jobs")
    (SCAND_ROOT / "annotations" / "tmp" / f"episodes-{a.set}-{a.split}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
