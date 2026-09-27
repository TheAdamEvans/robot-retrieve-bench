"""Run one headless Claude Code labeller job and record its measured cost."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

MODEL = "claude-opus-5-5"


def run_claude(prompt: str, job: str, cwd: Path, tools: list[str], model: str = MODEL, budget_usd: str = "15") -> dict:
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json", "--permission-mode", "acceptEdits",
           "--allowedTools", *tools, "--max-budget-usd", str(budget_usd)]
    log = Path(cwd) / "annotations" / "tmp" / job
    log.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env={**os.environ, "SCANDQ_JOB": job}, capture_output=True, text=True)
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
    return {"job": job, "exit": p.returncode, "cost_usd": float(res.get("total_cost_usd") or 0.0),
            "turns": res.get("num_turns"), "minutes": round((time.time() - t0) / 60, 1),
            "result": (res.get("result") or "")[:3000]}
