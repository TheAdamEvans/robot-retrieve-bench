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
(select one with `ALLOY_PROGRAM_PROMPT`; default `v3`). The prompt hash is part of the program cache key, so every version keeps
its own cache and re-running a configuration costs nothing.

```bash
cp server/src/alloy_server/programs/prompts/v1.md server/src/alloy_server/programs/prompts/v2.md   # edit v2
ALLOY_EVALS_LLM=1 uv run alloy-evals hillclimb programs/generator --grid prompt=v1,v2 --grid reasoning=low
```

This prints a dev/test table (validity, feature recall, forbidden features, partial accuracy, leak-free, attempts,
tokens) and writes `results/evals/hillclimb/<suite>-<time>.json`. **Choose on dev only.** Test is for reporting.
Then confirm on the benchmark.

**Baseline, 27 Sep (prompt v1, gpt-6-luna, 16 recordings in the prompt):**

| reasoning | split | n | valid | feature recall | forbidden absent | partial correct | leak-free | attempts | tokens (mean / p90) |
|---|---|---|---|---|---|---|---|---|---|
| low | dev | 10 | 1.00 | 0.91 | 1.00 | 0.63 | 1.00 | 1.00 | 5.3k / 5.8k |
| low | test | 21 | 0.95 | 0.87 | – | 0.70 | 1.00 | 1.05 | 5.4k / 6.4k |
| medium | dev | 10 | 0.90 | 0.93 | 1.00 | 0.86 | 0.89 | 1.00 | 5.4k / 6.8k |
| medium | test | 21 | 0.86 | 0.91 | – | 0.67 | 1.00 | 1.14 | 5.9k / 12.3k |

Medium reasoning is not a clear win: it recalls slightly more features, but its validity drops. On test,
`chained_turn_person` paraphrases fail validation 3 times out of 3. **Low stays the default.**

The dev failures that recur at both settings are the ones v2 targets. Test failures are reported, never tuned on:
- **Over-partial answers.** Reportable details ("full-body-clear times", "supporting frames") are put in
  `unexpressible`, which marks answers partial. Rule 2 already says these are not requirements.
- **Vehicle interaction without a motion response.** `speed_mps` is omitted.

**First hill-climb, 27 Sep (gpt-6-luna, low reasoning).** v2 and v3 were written from **dev** failures only.

| prompt | change | dev partial ✓ | dev leak-free | test valid | test feature recall | test partial ✓ | tokens (mean) |
|---|---|---|---|---|---|---|---|
| v1 | baseline | 0.63 | 1.00 | 0.95 | 0.87 | 0.70 | 5.4k |
| v2 | "what to return is output, not a requirement"; interaction = presence + speed change | 0.88 | 0.90 | 0.95 | 0.92 | 0.75 | 6.6k |
| v3 | v2 + "state defaults in plain words, never feature names" (v2 leaked `speed_mps` into `doc`) | **1.00** | **1.00** | **1.00** | **0.93** | **0.81** | 6.9k |

v3 is the default. It costs about 1.5k more tokens per generation (more repair turns: 1.24 attempts on test,
against 1.05). A failed generation was replayed as 0 tokens until 27 Sep; the table's token means are lower bounds.

**Confirmed on the benchmark (eval v9 corpus, same judgments; `benchmark/results/v9-prompt-v1` against `benchmark/results/v9`).**

| Set | v1 → v3 (PROGRAM / HYBRID / FUSED_V with gpt-6-luna) |
|---|---|
| compose_test | no change: 5 of 6 programs have identical clauses |
| demo5_para | status accuracy 0.73 → 0.93; nDCG@10 +0.05 / −0.11 / +0.04; fewer `unexpressible` entries |
| l1_compositions_dev (never tuned on) | ROC-AUC 0.46–0.50 → **0.87–0.90**; `unexpressible` entries 22 → 15; tokens 9.5k → 8.2k; p50 16 s → 12 s |

The rule "what the question asks you to return is output, not a requirement" transferred to questions the
hill-climb never saw. HYBRID's lower demo5_para nDCG is the one regression to watch.

## Exhaustive ground truth (challenge sets)

Pooled judgments cannot support recall claims: a window nobody judged is unknown, not irrelevant. For the
`benchmark/challenges/` question sets, ground truth is instead built to be **complete** over each question's scope:

1. **Numeric sweep** (`uv run python -m alloy_train.challenge.sweep --set l1_compositions_v1 --split dev`). Every
   span that could pass the question's numeric clauses:
   - thresholds are loosened by a stated tolerance;
   - maneuver windows are searched on a grid of 1–10 s, with onsets every 0.25 s;
   - signals are computed as `scandq signals` computes them, plus raw odometry pose.

   The sweep must contain every episode already judged relevant (the sweep-recall check), or it fails.
2. **Exhaustive judging** (`uv run python -m alloy_train.annotate.episodes --set … --split dev --exhaustive --run`).
   The candidates are split into chunks of at most 30 s. Opus judges every episode in every chunk, and each chunk
   needs at least one record, so coverage can be checked by machine.
3. **Scoring.** Once every chunk of an intent is covered, the intent is *complete*:
   - an in-scope window outside every relevant episode is a real 0, so ROC-AUC and nDCG have no pooling bias;
   - the report adds episode recall@10 and @50 (a ground-truth episode counts as found when a top-k window overlaps it);
   - an intent with no numeric candidates is complete with zero episodes.

**l1_compositions_v1 dev** cost $51.25 over 12 jobs. The ground truth:
- multiple_attempts: none in scope (numeric proof);
- opportunity_to_go: none in scope (every candidate judged);
- steer_or_brake: one qualifying pair;
- recover_person_present: 3 full matches and 10 partial;
- turn_creates_exposure and person_adjusts: 2 partial each. person_adjusts recall is relative to the people seen
  by L1 labels or the detector.

Where the lead-based pass and the exhaustive pass overlap, they agree on relevant versus not for 16 of 18
episodes. Exhaustive shards take precedence (`annotations/labels/precedence.json`).

**What the ground truth exposed.** On the two questions whose true answer is "none", every config still returns
results, marked unverified or partial. None says `none_found_exhaustive` or `insufficient_evidence`. Pooled
evaluation could not see this failure.
