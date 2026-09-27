"""Re-label specific L1 segments in a correction campaign that outranks the original labels.

The original labels are never edited: the correction campaign gets a higher priority, so the store returns the new
record for the same segment and the old one stays in history. The judge is told that a cross-check disputes the
segments and what it measured, never what the answer should be, and re-labels every attribute from the data.

    uv run python -m alloy_index.annotate.correct --campaign l1-corrections-2026-09-27 \\
        --segments Brackenridge:0084,Brackenridge:0088,Brackenridge:0100 --reason "..." --run
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from alloy_index.annotate.headless import run_claude
from alloy_index.annotate.store import ensure_campaign, load_labels, write_job
from alloy_index.recordings import SCAND_ROOT

BRIEF = Path(__file__).parent / "prompts" / "l1_attributes.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--segments", required=True, help="comma-separated segment ids (Rec:EEEE)")
    ap.add_argument("--reason", required=True, help="what the cross-check measured (shown to the judge)")
    ap.add_argument("--priority", type=int, default=1)
    ap.add_argument("--run", action="store_true", help="run the headless Opus job (spends money)")
    a = ap.parse_args()
    segs = a.segments.split(",")
    have = load_labels("attributes", ("train",))
    missing = [s for s in segs if s not in have]
    if missing:
        raise SystemExit(f"not labelled yet (use annotate.l1, not a correction): {missing}")
    supersedes = sorted({have[s]["campaign"] for s in segs})
    job = f"fix-{segs[0].split(':')[0]}-{len(segs)}seg"
    preamble = (f"# Correction job\n\nThese segments already have L1 labels, but a sensor cross-check disputes them:\n\n"
                f"> {a.reason}\n\nRe-label every listed segment from scratch, from the data: all attributes and the "
                f"caption, not only the disputed part. Read speeds from `signals` (COMPUTED from odometry) and confirm "
                f"what you see at native resolution. Do not assume the earlier labels or this note are right.\n\n")
    prompt = preamble + BRIEF.read_text().replace("{JOB}", job).replace("{SEGMENTS}", ", ".join(segs))
    prompt += (f"\n\nTouch only job `{job}` and labels/.work/{job}/. Label all {len(segs)} listed segments, submit with "
               f"`label put`, verify with `label get attributes --rec {segs[0].split(':')[0]}`, then give the short report.")
    print(json.dumps({"campaign": a.campaign, "job": job, "segments": segs, "supersedes": supersedes}))
    if not a.run:
        return
    ensure_campaign(a.campaign, kinds=["attributes"], priority=a.priority, brief=BRIEF.name, judge="claude-opus-5-5",
                    date=f"{datetime.date.today():%Y-%m-%d}", supersedes=supersedes, reason=a.reason,
                    purpose="Correction of L1 labels disputed by a sensor cross-check; outranks the original labels.")
    write_job(a.campaign, job, {"segments": segs, "reason": a.reason,
                                "previous": {s: {"campaign": have[s]["campaign"], "caption": have[s].get("caption")}
                                             for s in segs}})
    tools = ["Bash(uv run scandq:*)", f"Bash(SCANDQ_JOB={job} uv run scandq:*)", "Bash(mkdir -p labels/.work/*)",
             "Bash(ls:*)", "Read", "Write"]
    r = run_claude(prompt, job, SCAND_ROOT, tools, campaign=a.campaign, budget_usd="5")
    print(json.dumps({k: r[k] for k in ("job", "exit", "cost_usd", "turns", "minutes")}))
    after = load_labels("attributes", ("train",))
    for s in segs:
        print(f"{s}: [{after[s]['campaign']}] {after[s].get('caption')}")


if __name__ == "__main__":
    main()
