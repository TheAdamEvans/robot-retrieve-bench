"""annotate.l1: the Opus labeller as an optional index stage (runs only with --annotate: it spends money).

Unlabelled 4 s segments are chunked into disjoint jobs (<= max_segments_per_job) and run headless with the Claude
CLI, several in parallel. Each job sees only `scandq` (audited, anchored views over the MCAP) and writes validated
label shards; its measured cost (from the CLI's JSON output) is recorded. Label shards live in the repo
(annotations/labels/attributes/), recorded in the sink manifest with a `repo:` prefix.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from alloy_server.catalog.windows import segments
from alloy_index.annotate.store import load_labels
from alloy_index.stages.base import StageImpl, register

BRIEF = Path(__file__).resolve().parents[1] / "annotate" / "prompts" / "l1_attributes.md"


@register
class L1Labeller(StageImpl):
    IMPL, VERSION, SCOPE, DEPENDS = "L1Labeller", "l1_labeller@1", "PER_RECORDING", ("Ingest",)

    def _ann(self, ctx) -> Path:
        return ctx.scand_root / "annotations"

    def _todo(self, ctx, rid: str) -> list[str]:
        info = json.loads((ctx.bundle / "mcap" / f"{rid}.json").read_text())
        have = load_labels(self._ann(ctx), "attributes")
        return [s for s in segments(rid, (info["log_end_ns"] - info["log_start_ns"]) / 1e9) if s not in have]

    def _shards(self, ctx, rid: str) -> list[str]:
        d = self._ann(ctx) / "labels" / "attributes"
        return sorted(f"repo:annotations/labels/attributes/{p.name}" for p in d.glob("*.jsonl")
                      if any(json.loads(l)["recordingId"] == rid for l in p.read_text().splitlines()))

    def adopt(self, ctx, rec):
        return self._shards(ctx, rec.recording_id) if not self._todo(ctx, rec.recording_id) else None

    def run(self, ctx, rec):
        return self.run_many(ctx, [rec])[rec.recording_id]

    def run_many(self, ctx, recs) -> dict:
        """All recordings' jobs in one pool (parallel across recordings, disjoint segments)."""
        per = int(ctx.params.get("max_segments_per_job", "30"))
        jobs = []
        for rec in recs:
            todo = self._todo(ctx, rec.recording_id)
            for k in range(0, len(todo), per):
                jobs.append((rec.recording_id, f"l1-{rec.recording_id}-c{k // per + 1}", todo[k:k + per]))
        ctx.log(f"annotate.l1: {sum(len(j[2]) for j in jobs)} segments in {len(jobs)} jobs")
        with ThreadPoolExecutor(int(ctx.params.get("parallel", "8"))) as pool:
            results = list(pool.map(lambda j: self._job(ctx, *j), jobs))
        out = {}
        for rec in recs:
            mine = [r for r in results if r["recording"] == rec.recording_id]
            left = self._todo(ctx, rec.recording_id)
            out[rec.recording_id] = (self._shards(ctx, rec.recording_id), {
                "jobs": str(len(mine)), "cost_usd": f"{sum(r['cost_usd'] for r in mine):.2f}",
                "unlabelled_after": str(len(left)), "model": ctx.params.get("model", "claude-opus-5-5")})
            if left:
                ctx.log(f"annotate.l1: {rec.recording_id} still has {len(left)} unlabelled segments: {left[:5]}")
        return out

    def _job(self, ctx, rid: str, job: str, segs: list[str]) -> dict:
        prompt = (BRIEF.read_text().replace("{JOB}", job).replace("{SEGMENTS}", ", ".join(segs)) +
                  f"\n\nOther labeller jobs run in parallel on other segments: touch only job `{job}` and "
                  f"annotations/tmp/{job}/. Label all {len(segs)} listed segments, submit with `label put`, verify "
                  f"with `label get attributes --rec {rid}`, then give the short report.")
        allowed = ["Bash(uv run scandq:*)", f"Bash(SCANDQ_JOB={job} uv run scandq:*)", "Bash(mkdir -p annotations/tmp/*)",
                   "Bash(ls:*)", "Read", "Write"]
        cmd = ["claude", "-p", prompt, "--model", ctx.params.get("model", "claude-opus-5-5"), "--output-format", "json",
               "--permission-mode", "acceptEdits", "--allowedTools", *allowed,
               "--max-budget-usd", ctx.params.get("max_budget_usd_per_job", "15")]
        log_dir = self._ann(ctx) / "tmp" / job
        log_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        env = {**os.environ, "SCANDQ_JOB": job}
        p = subprocess.run(cmd, cwd=ctx.scand_root, env=env, capture_output=True, text=True)
        (log_dir / "claude_result.json").write_text(p.stdout or "")
        (log_dir / "claude_stderr.txt").write_text(p.stderr or "")
        res = {}
        for line in reversed((p.stdout or "").splitlines()):  # the CLI may print warnings before its JSON result
            if line.startswith("{"):
                try:
                    res = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        cost = float(res.get("total_cost_usd") or 0.0)
        ctx.log(f"annotate.l1: {job} ({len(segs)} segments) exit {p.returncode} in {time.time() - t0:.0f}s "
                f"cost ${cost:.2f}")
        return {"recording": rid, "job": job, "cost_usd": cost, "exit": p.returncode,
                "turns": res.get("num_turns"), "result": (res.get("result") or "")[:2000]}
