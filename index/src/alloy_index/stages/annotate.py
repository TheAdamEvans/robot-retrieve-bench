"""annotate.l1: the Opus labeller as an optional index stage (runs only with --annotate: it spends money).

Unlabelled 4 s segments are chunked into disjoint jobs (<= max_segments_per_job) and run headless with the Claude
CLI, several in parallel, as one labelling campaign (param `campaign`, default `l1-<date>`). Each job sees only
`scandq` (audited, anchored views over the MCAP) and writes validated label shards to labels/train/attributes/; its
assignment, measured cost and report go to labels/metadata/<campaign>/jobs/. Shards are recorded in the sink
manifest with a `repo:` prefix.
"""
from __future__ import annotations

import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from alloy_server.catalog.windows import segments
from alloy_index.annotate.headless import run_claude
from alloy_index.annotate.store import KIND_DIR, ensure_campaign, load_labels, write_job
from alloy_index.stages.base import StageImpl, register

BRIEF = Path(__file__).resolve().parents[1] / "annotate" / "prompts" / "l1_attributes.md"


@register
class L1Labeller(StageImpl):
    IMPL, VERSION, SCOPE, DEPENDS = "L1Labeller", "l1_labeller@1", "PER_RECORDING", ("Ingest",)

    def _labels(self, ctx) -> Path:
        return ctx.scand_root / "labels"

    def _todo(self, ctx, rid: str) -> list[str]:
        info = json.loads((ctx.bundle / "mcap" / f"{rid}.json").read_text())
        have = load_labels("attributes", ("train",), self._labels(ctx))
        return [s for s in segments(rid, (info["log_end_ns"] - info["log_start_ns"]) / 1e9) if s not in have]

    def _shards(self, ctx, rid: str) -> list[str]:
        d = self._labels(ctx) / "train" / KIND_DIR["attributes"]
        return sorted(f"repo:labels/train/attributes/{p.name}" for p in d.glob("*.jsonl")
                      if any(json.loads(l)["recordingId"] == rid for l in p.read_text().splitlines() if l.strip()))

    def adopt(self, ctx, rec):
        return self._shards(ctx, rec.recording_id) if not self._todo(ctx, rec.recording_id) else None

    def run(self, ctx, rec):
        return self.run_many(ctx, [rec])[rec.recording_id]

    def run_many(self, ctx, recs) -> dict:
        """All recordings' jobs in one pool (parallel across recordings, disjoint segments)."""
        per = int(ctx.params.get("max_segments_per_job", "30"))
        slug = ctx.params.get("campaign") or f"l1-{datetime.date.today():%Y-%m-%d}"
        ensure_campaign(slug, self._labels(ctx), kinds=["attributes"], brief=BRIEF.name,
                        judge=ctx.params.get("model", "claude-opus-5-5"), date=f"{datetime.date.today():%Y-%m-%d}",
                        purpose="L1 per-segment attributes and captions for newly indexed recordings (annotate.l1).")
        jobs = []
        for rec in recs:
            todo = self._todo(ctx, rec.recording_id)
            for k in range(0, len(todo), per):
                jobs.append((rec.recording_id, f"l1-{rec.recording_id}-c{k // per + 1}", todo[k:k + per]))
        ctx.log(f"annotate.l1: {sum(len(j[2]) for j in jobs)} segments in {len(jobs)} jobs (campaign {slug})")
        with ThreadPoolExecutor(int(ctx.params.get("parallel", "8"))) as pool:
            results = list(pool.map(lambda j: self._job(ctx, slug, *j), jobs))
        out = {}
        for rec in recs:
            mine = [r for r in results if r["recording"] == rec.recording_id]
            left = self._todo(ctx, rec.recording_id)
            out[rec.recording_id] = (self._shards(ctx, rec.recording_id), {
                "campaign": slug, "jobs": str(len(mine)), "cost_usd": f"{sum(r['cost_usd'] for r in mine):.2f}",
                "unlabelled_after": str(len(left)), "model": ctx.params.get("model", "claude-opus-5-5")})
            if left:
                ctx.log(f"annotate.l1: {rec.recording_id} still has {len(left)} unlabelled segments: {left[:5]}")
        return out

    def _job(self, ctx, slug: str, rid: str, job: str, segs: list[str]) -> dict:
        prompt = (BRIEF.read_text().replace("{JOB}", job).replace("{SEGMENTS}", ", ".join(segs)) +
                  f"\n\nOther labeller jobs run in parallel on other segments: touch only job `{job}` and "
                  f"labels/.work/{job}/. Label all {len(segs)} listed segments, submit with `label put`, verify "
                  f"with `label get attributes --rec {rid}`, then give the short report.")
        allowed = ["Bash(uv run scandq:*)", f"Bash(SCANDQ_JOB={job} uv run scandq:*)", "Bash(mkdir -p labels/.work/*)",
                   "Bash(ls:*)", "Read", "Write"]
        write_job(slug, job, {"recording": rid, "segments": segs}, self._labels(ctx))
        r = run_claude(prompt, job, ctx.scand_root, allowed, model=ctx.params.get("model", "claude-opus-5-5"),
                       budget_usd=ctx.params.get("max_budget_usd_per_job", "15"), campaign=slug)
        ctx.log(f"annotate.l1: {job} ({len(segs)} segments) exit {r['exit']} in {r['minutes']:.0f} min "
                f"cost ${r['cost_usd']:.2f}")
        return {"recording": rid, **r}
