"""alloy-evals: run, extend and hill-climb the module eval suites (evals/*/confeval.py in every package).

    alloy-evals list
    alloy-evals run [SUITE_FILTER ...] [--split dev|test|all] [--param prompt=v2]
    alloy-evals add SUITE --id ID --input '{...}' --expect '{...}' [--split test] [--note ...]
    alloy-evals add programs/generator --from-query doorway_crossing:para1 [--split test]
    alloy-evals hillclimb programs/generator --grid prompt=v1,v2 --grid reasoning=low,medium
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import alloy_index
import alloy_server
import alloy_trainer
from alloy_server.evalkit import REPO, EvalContext, discover, load_suite, run_suite, suite_name

ROOTS = [Path(m.__file__).parent for m in (alloy_server, alloy_index, alloy_trainer)]


def suites() -> dict[str, Path]:
    return {suite_name(p): p for p in discover(*ROOTS)}


def _params(kvs: list[str]) -> dict:
    return dict(kv.split("=", 1) for kv in kvs)


def cmd_run(a) -> int:
    ctx = EvalContext(params=_params(a.param))
    splits = ("dev", "test") if a.split == "all" else (a.split,)
    failed = 0
    for name, p in suites().items():
        if a.filter and not any(f in name for f in a.filter):
            continue
        card = run_suite(p, ctx, splits)
        print(card.text())
        failed += not card.passed
    return 1 if failed else 0


def _query(qid: str) -> dict:
    for f in sorted((REPO / "benchmark" / "queries").glob("*.json")):
        if f.name == "MANIFEST.json":
            continue
        for q in json.loads(f.read_text()):
            if q["queryId"] == qid:
                return q
    raise SystemExit(f"no benchmark query {qid!r}")


def expect_from_oracle(q: dict) -> dict:
    """What a generated program should share with the benchmark's hand-written one."""
    prog = q["oracleProgram"]
    exp = {"valid": True, "leak_free": True, "partial": bool(prog.get("unexpressible"))}
    if not (prog.get("unexpressible") and prog.get("abstainIfInsufficient")):  # abstain oracles carry placeholders
        exp["features"] = sorted({ev["feature"] for ev in prog.get("events", [])})
    return exp


def cmd_add(a) -> int:
    path = suites().get(a.suite) or sys.exit(f"unknown suite {a.suite!r}; see `alloy-evals list`")
    mod = load_suite(path)
    ds = next((d for d in mod.DATASETS if d.split == a.split and (not a.dataset or d.name == a.dataset)), None)
    if ds is None:
        sys.exit(f"{a.suite} has no {a.split} dataset")
    if a.from_query:
        q = _query(a.from_query)
        case = {"id": a.id or a.from_query, "input": {"utterance": q["utterance"]}, "expect": expect_from_oracle(q),
                "source": f"benchmark:{a.from_query}"}
    else:
        case = {"id": a.id, "input": json.loads(a.input), "expect": json.loads(a.expect)}
    if a.expect and a.from_query:
        case["expect"].update(json.loads(a.expect))
    if a.note:
        case["note"] = a.note
    f = path.parent / ds.path
    ids = {json.loads(line)["id"] for line in f.read_text().splitlines() if line.strip()} if f.exists() else set()
    if case["id"] in ids:
        sys.exit(f"{case['id']!r} already in {f}")
    with f.open("a") as fh:
        fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"added {case['id']} to {f.relative_to(REPO)}")
    return 0


def cmd_hillclimb(a) -> int:
    path = suites()[a.suite]
    axes = [(k, vs.split(",")) for k, vs in (g.split("=", 1) for g in a.grid)]
    rows = []
    for combo in itertools.product(*[vs for _, vs in axes]):
        params = dict(zip([k for k, _ in axes], combo))
        row = {"params": params}
        for split in ("dev", "test"):
            card = run_suite(path, EvalContext(params=params), (split,))
            row[split] = {"metrics": card.metrics, "passed": card.passed, "n": len(card.rows),
                          "misses": {r["id"]: r["misses"] for r in card.rows if r["misses"]}, "skipped": card.skipped}
        rows.append(row)
    keys = a.metrics.split(",")
    head = " | ".join(f"{k}" for k, _ in axes)
    print(f"| {head} | split | n | " + " | ".join(keys) + " |")
    print("|" + "---|" * (len(axes) + 2 + len(keys)))
    for r in rows:
        for split in ("dev", "test"):
            m = r[split]["metrics"]
            print(f"| {' | '.join(r['params'].values())} | {split} | {r[split]['n']} | "
                  + " | ".join(f"{m[k]:.3g}" if k in m else "–" for k in keys) + " |")
    out = REPO / "results" / "evals" / "hillclimb" / f"{a.suite.replace('/', '_')}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="alloy-evals")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("filter", nargs="*")
    r.add_argument("--split", default="all", choices=["dev", "test", "all"])
    r.add_argument("--param", action="append", default=[])
    ad = sub.add_parser("add")
    ad.add_argument("suite")
    ad.add_argument("--id")
    ad.add_argument("--input")
    ad.add_argument("--expect")
    ad.add_argument("--from-query")
    ad.add_argument("--split", default="dev", choices=["dev", "test"])
    ad.add_argument("--dataset")
    ad.add_argument("--note")
    h = sub.add_parser("hillclimb")
    h.add_argument("suite")
    h.add_argument("--grid", action="append", required=True, help="knob=v1,v2 (repeatable)")
    h.add_argument("--metrics", default="valid.accuracy,features.recall,features_none.clean,partial.accuracy,"
                                        "leak_free.accuracy,attempts.value.mean,tokens.value.mean,tokens.value.p90")
    a = ap.parse_args()
    if a.cmd == "list":
        for name, p in suites().items():
            mod = load_suite(p)
            print(f"{name:32} " + ", ".join(f"{d.name}[{d.split}]{'+' + '+'.join(d.requires) if d.requires else ''}"
                                             for d in mod.DATASETS) + (f"  requires {mod.REQUIRES}" if getattr(mod, "REQUIRES", None) else ""))
        return
    sys.exit({"run": cmd_run, "add": cmd_add, "hillclimb": cmd_hillclimb}[a.cmd](a))


if __name__ == "__main__":
    main()
