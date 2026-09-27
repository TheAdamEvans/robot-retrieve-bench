"""Collect every `evals/*/confeval.py` under a package as one pytest case per suite.

    # <package>/tests/test_module_evals.py
    from alloy_server.evalkit.pytest_support import module_eval_test
    test_module_evals = module_eval_test("alloy_index")
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from . import EvalContext, discover, load_suite, run_suite, suite_name


def module_eval_test(package: str):
    root = Path(importlib.import_module(package).__file__).parent
    params = []
    for p in discover(root):
        mod = load_suite(p)
        reqs = set(getattr(mod, "REQUIRES", [])) | {r for ds in mod.DATASETS for r in ds.requires}
        marks = [pytest.mark.moduleeval] + [getattr(pytest.mark, f"requires_{r}") for r in sorted(reqs)]
        params.append(pytest.param(p, id=suite_name(p), marks=marks))

    @pytest.mark.parametrize("confeval", params)
    def test(confeval):
        card = run_suite(confeval, EvalContext(), splits=("dev", "test"))
        if not card.rows:
            pytest.skip(f"{card.suite}: {card.skipped}")
        assert card.passed, card.text()

    return test
