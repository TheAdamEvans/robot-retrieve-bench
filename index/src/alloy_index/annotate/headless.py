"""Run one headless Claude Code labeller job in a campaign and record its measured cost."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from alloy_index.annotate.store import work_dir, write_job

MODEL = "claude-opus-5-5"


def run_claude(prompt: str, job: str, cwd: Path, tools: list[str], model: str = MODEL, budget_usd: str = "15",
               campaign: str = "adhoc") -> dict:
    """Runs the job with SCANDQ_CAMPAIGN/SCANDQ_JOB set; raw CLI output stays in labels/.work/<job>/, and the job's
    cost, turns and report are merged into labels/metadata/<campaign>/jobs/<job>.json."""
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json", "--permission-mode", "acceptEdits",
           "--allowedTools", *tools, "--max-budget-usd", str(budget_usd)]
    labels = Path(cwd) / "labels"
    log = work_dir(job, labels)
    (log / "brief.md").write_text(prompt)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env={**os.environ, "SCANDQ_CAMPAIGN": campaign, "SCANDQ_JOB": job},
                       capture_output=True, text=True)
    (log / "claude_result.json").write_text(p.stdout or "")
    (log / "claude_stderr.txt").write_text(p.stderr or "")
    res = {}
    for line in reversed((p.stdout or "").splitlines()):  # the CLI may print warnings before its JSON result
        if line.startswith("{"):
            try:
                res = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    out = {"job": job, "exit": p.returncode, "cost_usd": float(res.get("total_cost_usd") or 0.0),
           "turns": res.get("num_turns"), "minutes": round((time.time() - t0) / 60, 1),
           "result": (res.get("result") or "")[:3000]}
    write_job(campaign, job, {"model": model, "exit": out["exit"], "cost_usd": out["cost_usd"], "turns": out["turns"],
                              "minutes": out["minutes"], "is_error": res.get("is_error"), "report": res.get("result")},
              labels)
    return out
