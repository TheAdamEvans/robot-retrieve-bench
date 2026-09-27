"""alloy-index plan | run | status — drop bags into raw/, then plan and run.

  plan     what is built / adoptable / to do / gated, without running anything
  run      build what's missing (only stages whose key the sink doesn't have)
  status   recordings x per-recording stages grid from the sink manifest
Flags: --recordings A,B  --stages ingest,frames.siglip2  --annotate (allow the paid labeller stage)  --adopt
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from alloy_index import recordings as R
from alloy_index.runner import Runner

SYMBOL = {"built": "✓", "adopted": "✓", "ran": "✓", "todo": "·", "blocked": "…", "n/a": "–", "gated": "$"}


def print_report(rep) -> None:
    grid = defaultdict(dict)
    for s in rep.steps:
        grid[s.recording][s.stage] = s
    stages = list(dict.fromkeys(s.stage for s in rep.steps))
    per_rec = [st for st in stages if any(r != "*" and st in grid[r] for r in grid)]
    corpus = [st for st in stages if st in grid.get("*", {})]
    w = max([len(r) for r in grid] + [10])
    print(" " * (w + 2) + "  ".join(st.split(".")[-1][:10].ljust(10) for st in per_rec))
    for rid in sorted(r for r in grid if r != "*"):
        held = " (held out)" if rid in R.held_out() else ""
        print(rid.ljust(w + 2) + "  ".join(SYMBOL.get(grid[rid].get(st, None).status if st in grid[rid] else "", " ").ljust(10)
                                           for st in per_rec) + held)
    for st in corpus:
        s = grid["*"][st]
        print(f"[corpus] {st:22s} {s.status:8s} {s.detail}")
    todo = [s for s in rep.steps if s.status in ("todo", "gated", "blocked")]
    print(f"\n{sum(s.status in ('built', 'adopted', 'ran') for s in rep.steps)} built, "
          f"{sum(s.status == 'todo' for s in rep.steps)} to do, {sum(s.status == 'blocked' for s in rep.steps)} blocked, "
          f"{sum(s.status == 'gated' for s in rep.steps)} gated ($ = needs its flag), "
          f"{sum(s.status == 'n/a' for s in rep.steps)} not applicable (–)")
    for s in rep.steps:
        if s.status in ("ran", "adopted") or (s.status == "n/a" and s.detail):
            print(f"  {s.status:8s} {s.stage:22s} {s.recording:24s} {s.detail}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="alloy-index", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run", "status"])
    ap.add_argument("--recordings", help="comma-separated recording ids (default: every bag in the source)")
    ap.add_argument("--stages", help="comma-separated stage names to run (others are only reported)")
    ap.add_argument("--annotate", action="store_true", help="allow annotate.* stages (paid labeller jobs)")
    ap.add_argument("--adopt", action="store_true", help="register existing outputs built by the same version/params")
    a = ap.parse_args()
    flags = {"annotate"} if a.annotate else set()
    runner = Runner(flags=flags, recordings=a.recordings.split(",") if a.recordings else None,
                    stages=a.stages.split(",") if a.stages else None, adopt=a.adopt, dry=a.cmd != "run")
    print_report(runner.run())


if __name__ == "__main__":
    main()
