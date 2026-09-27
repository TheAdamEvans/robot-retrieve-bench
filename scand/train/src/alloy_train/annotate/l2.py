"""Build judgment pools from an eval run and render one L2 brief per intent group (disjoint jobs)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alloy_server.bundle import Bundle
from alloy_train.eval import pooling
from alloy_index.recordings import SCAND_ROOT

import alloy_index.annotate

BRIEF = Path(alloy_index.annotate.__file__).parent / "prompts" / "l2_judgments.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--skip-judged", action="store_true", help="only windows without a judgment yet")
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
    for intent, pool in pools.items():
        todo = [w for w in pool["windows"] if (intent, w) not in judged]
        if not todo:
            continue
        job = f"l2-{intent}" + ("-extra" if a_.skip_judged else "")
        by_rec: dict[str, list[str]] = {}
        for w in todo:
            by_rec.setdefault(w.split(":")[0], []).append(w)
        listing = "\n".join(f"- {rec}: " + ", ".join(ws) for rec, ws in sorted(by_rec.items()))
        brief = (tpl.replace("{JOB}", job).replace("{INTENT_ID}", intent).replace("{INTENT}", pool["intent"])
                 .replace("{SCOPE}", ", ".join(pool["scope"])).replace("{N}", str(len(todo))).replace("{WINDOWS}", listing))
        out = ann / "tmp" / job
        out.mkdir(parents=True, exist_ok=True)
        (out / "brief.md").write_text(brief)
        print(json.dumps({"job": job, "windows": len(todo), "recordings": len(by_rec), "brief": str(out / "brief.md")}))


if __name__ == "__main__":
    main()
