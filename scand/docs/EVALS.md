# Evals

There are two levels:

- **The benchmark** answers "which config should serve this query?" end to end, with pooled graded relevance
  judgments. See [`benchmark/CORRECTNESS.md`](../benchmark/CORRECTNESS.md) and [FUSED.md](FUSED.md#evaluation).
- **Module evals** answer "does this one module still do its job?" They are cheap, run in pytest, and adding a
  case takes one line.

## Module evals

Every module can keep suites next to its code:

```
alloy_server/verify/evals/onsets/
  confeval.py              # datasets, scorers, gates, run_case
  cases_synthetic.jsonl    # one JSON case per line
  cases_recorded.jsonl
```

```python
# confeval.py
DATASETS = [Dataset("synthetic", "cases_synthetic.jsonl"),
            Dataset("recorded", "cases_recorded.jsonl", requires=["bundle"])]
SCORERS  = [abs_error("onset_s"), exact("n_onsets")]
GATES    = [Gate("onset_s.abs_error.max", "<=", 0.10), Gate("n_onsets.accuracy", ">=", 1.0)]

def run_case(case, ctx) -> dict:          # call the module; return the observed fields
    ...
```

A case is `{"id", "input": {...}, "expect": {...}, "note"?, "source"?}`. A scorer compares one observed field with
`expect[field]`, and a case without that field is not scored by it.

| Scorer | Per case | Metric names |
|---|---|---|
| `exact(f)` | observed == expected | `f.accuracy` |
| `abs_error(f)` | \|observed − expected\| | `f.abs_error.{mean,p90,max}` |
| `at_least(f)`, `at_most(f)` | threshold met | `f.at_least`, `f.at_most` |
| `includes(f)` | recall of expected items | `f.recall` |
| `excludes(f)` | none of `expect[f_none]` observed | `f_none.clean` |
| `measured(f)` | an observed number with no expectation (tokens, latency) | `f.value.{mean,p90,max}` |

`requires` gates where a dataset runs: `bundle` needs the built dev bundle, and `llm` needs `OPENAI_API_KEY` plus
`ALLOY_EVALS_LLM=1`, because it spends API tokens. A `dev` dataset may be tuned on; a `test` dataset is only reported.

### Suites today

| Suite | Checks | Seeded by |
|---|---|---|
| `catalog/embodiment` | body-side room, front margin, robot-relative speed | the labeller's follower-behind finding |
| `verify/onsets` | ONSET timing on synthetic gait oscillation and on JCL | the labeller's finding that onsets were ~0.6 s late; JCL stop→go at 24.15 s |
| `pipeline/merge_adjacent` | which results merge, spans shown, weak results never widen a span | W6 |
| `providers/detections` (index) | the pinned RT-DETR loads and re-detects known frames | the refactor that deleted the model cache |
| `programs/generator` | valid programs, oracle features recalled, forbidden features absent, partial when it should be, no system words leaked, tokens | demo5 and regressions (dev); compose_test and demo5_para (test) |

### Running

```bash
uv run pytest -q                                   # every suite runs as a test; LLM suites skip without ALLOY_EVALS_LLM=1
uv run pytest -q -m "moduleeval and not requires_llm"
uv run alloy-evals list
uv run alloy-evals run onsets merge                # scorecards for suites matching a filter
ALLOY_EVALS_LLM=1 uv run alloy-evals run programs/generator --split dev
```

### Adding a case

Append one line to the suite's JSONL, or use the CLI:

```bash
uv run alloy-evals add catalog/embodiment --id spot_wall_left \
  --input '{"robot": "spot", "op": "body_side_room", "xy": [[0.0, 0.75]]}' --expect '{"left_m": 0.5, "right_m": 5.0}'

# promote a benchmark query into a generator case (expectations taken from its oracle program)
uv run alloy-evals add programs/generator --from-query test_turn_then_brake:canonical --split test
```

The server-side suites import only server code, and their cases are plain data. Cases derived from benchmark labels
are *promoted* into them by the train-side CLI, which keeps the import rule intact.

## Hill-climbing the program generator

The generator's grammar and rules live in versioned files, `server/src/alloy_server/programs/prompts/<v>.md`
(select one with `ALLOY_PROGRAM_PROMPT`). The prompt hash is part of the program cache key, so every version keeps
its own cache and re-running a configuration costs nothing.

```bash
cp server/src/alloy_server/programs/prompts/v1.md server/src/alloy_server/programs/prompts/v2.md   # edit v2
ALLOY_EVALS_LLM=1 uv run alloy-evals hillclimb programs/generator --grid prompt=v1,v2 --grid reasoning=low
```

This prints a dev/test table (validity, feature recall, forbidden features, partial accuracy, leak-free, attempts,
tokens) and writes `results/evals/hillclimb/<suite>-<time>.json`. **Choose on dev only.** Test is for reporting.
Then confirm on the benchmark.

**Baseline (prompt v1, gpt-6-luna, 16 recordings in the prompt):** see the latest file in `results/evals/hillclimb/`.
