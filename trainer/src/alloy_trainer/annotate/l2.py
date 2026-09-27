"""Build judgment pools from an eval run and render one L2 brief per intent group (disjoint jobs)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alloy_server.bundle import Bundle
from alloy_trainer.eval import pooling, querysets
from alloy_index.recordings import SCAND_ROOT

import alloy_index.annotate

BRIEF = Path(alloy_index.annotate.__file__).parent / "prompts" / "l2_judgments.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--skip-judged", action="store_true", help="only windows without a judgment yet")
    ap.add_argument("--max-windows", type=int, default=60, help="split an intent into jobs of at most this many")
    ap.add_argument("--execute", action="store_true", help="run the headless Opus jobs (spends money)")
    ap.add_argument("--parallel", type=int, default=6)
    a_ = ap.parse_args()
    run_dir = SCAND_ROOT / "results" / "eval" / a_.run
    ann = SCAND_ROOT / "annotations"
    bundle = Bundle(SCAND_ROOT / "bundles" / "dev")
    pools = pooling.build(run_dir / "runs.jsonl", ann, bundle.windows)
    pooling.write(pools, run_dir / "pools.json")
    judged = set()
    jd = ann / "labels" / "judgment"
    if a_.skip_judged and jd.exists():
        for p in jd.glob("*.jsonl"):
            judged |= {(r["intentGroupId"], r["windowId"]) for r in map(json.loads, p.read_text().splitlines())}
    tpl = BRIEF.read_text()
    jobs: list[tuple[str, str]] = []
    a_.run_tag = a_.run
    challenge = {q.intent_group_id for name in querysets.challenge_sets() for q in querysets.load_challenge(name)}
    for intent, pool in pools.items():
        if intent in challenge:
            continue  # judged exhaustively (alloy_trainer.annotate.episodes --exhaustive), not by pooling
        todo = [w for w in pool["windows"] if (intent, w) not in judged]
        if not todo:
            continue
        by_rec: dict[str, list[str]] = {}
        for w in todo:
            by_rec.setdefault(w.split(":")[0], []).append(w)
        parts, cur = [], {}
        for rec, ws in sorted(by_rec.items()):  # whole recordings per job, so episodes are found once
            if cur and sum(map(len, cur.values())) + len(ws) > a_.max_windows:
                parts.append(cur)
                cur = {}
            cur[rec] = ws
        parts.append(cur)
        for k, part in enumerate(parts, 1):
            job = f"l2-{intent}" + (f"-{a_.run_tag}" if a_.skip_judged else "") + (f"-p{k}" if len(parts) > 1 else "")
            n = sum(map(len, part.values()))
            listing = "\n".join(f"- {rec}: " + ", ".join(ws) for rec, ws in sorted(part.items()))
            brief = (tpl.replace("{JOB}", job).replace("{INTENT_ID}", intent).replace("{INTENT}", pool["intent"])
                     .replace("{SCOPE}", ", ".join(pool["scope"])).replace("{N}", str(n)).replace("{WINDOWS}", listing))
            out = ann / "tmp" / job
            out.mkdir(parents=True, exist_ok=True)
            (out / "brief.md").write_text(brief)
            jobs.append((job, brief))
            print(json.dumps({"job": job, "windows": n, "recordings": len(part)}))
    if a_.execute:
        from concurrent.futures import ThreadPoolExecutor
        from alloy_index.annotate.headless import run_claude
        tools = lambda job: ["Bash(uv run scandq:*)", f"Bash(SCANDQ_JOB={job} uv run scandq:*)",
                             "Bash(mkdir -p annotations/tmp/*)", "Bash(ls:*)", "Read", "Write"]
        with ThreadPoolExecutor(a_.parallel) as pool_:
            results = list(pool_.map(lambda j: run_claude(j[1], j[0], SCAND_ROOT, tools(j[0])), jobs))
        for r in results:
            print(json.dumps({k: r[k] for k in ("job", "exit", "cost_usd", "turns", "minutes")}))
        print(f"total ${sum(r['cost_usd'] for r in results):.2f} over {len(results)} jobs")
        (ann / "tmp" / f"l2-{a_.run_tag}-topup.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
