"""Self-contained module evals: a folder `evals/<suite>/` holds `confeval.py` plus one JSONL line per case.

confeval.py declares:
    DATASETS = [Dataset("onsets", "cases.jsonl", requires=["bundle"])]
    SCORERS  = [abs_error("onset_s"), exact("status")]
    GATES    = [Gate("onset_s.abs_error.max", "<=", 0.10), Gate("status.accuracy", ">=", 1.0)]
    def run_case(case, ctx) -> dict   # calls the module; returns observed fields

A case is {"id", "input": {...}, "expect": {...}, "note"?, "source"?}. Scorers compare observed fields with `expect`;
a case without the scorer's field is not scored by it. Requirements ("bundle", "llm") skip a dataset where unmet.
"""
from __future__ import annotations

import importlib.util
import json
import operator
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[4]  # scand/


@dataclass
class Dataset:
    name: str
    path: str
    requires: list[str] = field(default_factory=list)
    split: str = "dev"  # dev: tune on it; test: report only


@dataclass
class Scorer:
    field: str
    kind: str
    fn: Callable[[object, object], float]
    aggs: tuple[str, ...]


def exact(f: str) -> Scorer:
    return Scorer(f, "accuracy", lambda obs, exp: float(obs == exp), ("mean",))


def abs_error(f: str) -> Scorer:
    return Scorer(f, "abs_error", lambda obs, exp: abs(float(obs) - float(exp)), ("mean", "p90", "max"))


def at_least(f: str) -> Scorer:
    return Scorer(f, "at_least", lambda obs, exp: float(obs >= exp), ("mean",))


def at_most(f: str) -> Scorer:
    return Scorer(f, "at_most", lambda obs, exp: float(obs <= exp), ("mean",))


def includes(f: str) -> Scorer:
    """Recall of the expected items among the observed ones."""
    return Scorer(f, "recall", lambda obs, exp: len(set(exp) & set(obs)) / len(exp) if exp else 1.0, ("mean",))


def excludes(f: str) -> Scorer:
    """1 when none of the forbidden items was observed. Expected under `<field>_none`."""
    return Scorer(f + "_none", "clean", lambda obs, exp: float(not set(exp) & set(obs)), ("mean",))


def measured(f: str) -> Scorer:
    """Aggregates an observed number (tokens, latency) with no expectation."""
    return Scorer(f, "value", lambda obs, exp: float(obs), ("mean", "p90", "max"))


OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt, "==": operator.eq}


@dataclass
class Gate:
    metric: str
    op: str
    value: float

    def check(self, metrics: dict[str, float]) -> bool | None:
        v = metrics.get(self.metric)
        return None if v is None else OPS[self.op](v, self.value)


class EvalContext:
    """What a case may use. `params` carries hill-climb knobs (prompt version, model)."""

    def __init__(self, bundle_root: Path | None = None, params: dict | None = None):
        self.bundle_root = Path(bundle_root or os.environ.get("ALLOY_BUNDLE", REPO / "bundles" / "dev"))
        self.params = params or {}
        self._bundle = None

    @property
    def bundle(self):
        if self._bundle is None:
            from ..bundle import Bundle
            self._bundle = Bundle(self.bundle_root)
        return self._bundle

    def unmet(self, requires: list[str]) -> list[str]:
        out = []
        if "bundle" in requires and not (self.bundle_root / "intake").exists():
            out.append("bundle (not built)")
        if "llm" in requires and not (os.environ.get("OPENAI_API_KEY") and os.environ.get("ALLOY_EVALS_LLM") == "1"):
            out.append("llm (set ALLOY_EVALS_LLM=1; spends API tokens)")
        return out


@dataclass
class Scorecard:
    suite: str
    rows: list[dict]
    metrics: dict[str, float]
    gates: list[tuple[Gate, bool | None]]
    skipped: dict[str, list[str]]

    @property
    def passed(self) -> bool:
        return all(ok is not False for _, ok in self.gates)

    def failures(self) -> list[str]:
        return [f"{g.metric} = {self.metrics.get(g.metric)} (want {g.op} {g.value})" for g, ok in self.gates if ok is False]

    def text(self) -> str:
        lines = [f"== {self.suite}: {len(self.rows)} cases" + (f"; skipped {self.skipped}" if self.skipped else "")]
        lines += [f"  {k:40} {v:.4g}" for k, v in sorted(self.metrics.items())]
        lines += [f"  gate {g.metric} {g.op} {g.value}: {'PASS' if ok else 'n/a' if ok is None else 'FAIL'}"
                  for g, ok in self.gates]
        lines += [f"  ✗ {r['id']}: {r['misses']}" for r in self.rows if r.get("misses")]
        return "\n".join(lines)


def load_suite(path: Path):
    path = Path(path)
    name = f"confeval_{path.parent.parent.parent.name}_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def suite_name(path: Path) -> str:
    """e.g. alloy_server/verify/evals/onsets/confeval.py -> verify/onsets"""
    parts = Path(path).parent.parts
    i = parts.index("evals")
    pkg = next(k for k in range(i, -1, -1) if parts[k].startswith("alloy_"))
    return "/".join(parts[pkg + 1:i] + parts[i + 1:])


def discover(*roots: Path) -> list[Path]:
    return sorted(p for r in roots for p in Path(r).rglob("evals/*/confeval.py"))


def read_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("//")]


def _agg(vals: list[float], how: str) -> float:
    return {"mean": float(np.mean(vals)), "p90": float(np.percentile(vals, 90)), "max": float(np.max(vals))}[how]


def run_suite(path: Path, ctx: EvalContext | None = None, splits: tuple[str, ...] = ("dev", "test")) -> Scorecard:
    ctx = ctx or EvalContext()
    mod = load_suite(path)
    rows, per, skipped = [], {}, {}
    for ds in mod.DATASETS:
        if ds.split not in splits:
            continue
        unmet = ctx.unmet(list(ds.requires) + list(getattr(mod, "REQUIRES", [])))
        if unmet:
            skipped[ds.name] = unmet
            continue
        for case in read_cases(Path(path).parent / ds.path):
            t = time.perf_counter()
            obs = mod.run_case(case, ctx)
            row = {"id": case["id"], "dataset": ds.name, "obs": obs, "ms": (time.perf_counter() - t) * 1e3, "misses": []}
            exp = case.get("expect", {})
            for sc in mod.SCORERS:
                src = sc.field[:-5] if sc.field.endswith("_none") else sc.field
                if sc.kind != "value" and sc.field not in exp or src not in obs:
                    continue
                v = sc.fn(obs[src], exp.get(sc.field))
                per.setdefault((sc.field, sc.kind, sc.aggs), []).append(v)
                if sc.kind in ("accuracy", "recall", "clean", "at_least", "at_most") and v < 1:
                    row["misses"].append(f"{sc.field}: got {obs[src]!r}, want {exp.get(sc.field)!r}")
            rows.append(row)
    metrics = {f"{f}.{k}.{a}" if a != "mean" or k in ("abs_error", "value") else f"{f}.{k}": _agg(v, a)
               for (f, k, aggs), v in per.items() for a in aggs}
    return Scorecard(suite_name(path), rows, metrics, [(g, g.check(metrics)) for g in mod.GATES], skipped)
