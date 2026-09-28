# Results

Every benchmark checkpoint, what it changed, its headline numbers and how to replay it. **The current checkpoint
is `v9`.** How results are scored is in [`benchmark/CORRECTNESS.md`](../benchmark/CORRECTNESS.md) and
[EVALS.md](EVALS.md); what the configs are is in [ARCHITECTURE.md](ARCHITECTURE.md#the-pipeline-runner).

## Checkpoints

A checkpoint is a frozen copy of one eval run under `benchmark/results/<run>/`:
- `manifest.json`: the code commit and a note on what changed;
- `runs.jsonl` and `score_all.jsonl`: every search, and every judged window scored;
- `judgments.json` and `pools.json`: the labels the run was scored against;
- `program_cache/`: the generated programs, so the run replays with no API calls;
- `report.json` and `report.html`: the tables.

| Run | Corpus | What it is |
|---|---|---|
| `v7` | 7 recordings | The first full evaluation. Requests for 20 or 50 candidates were silently capped at 10 |
| `v8-correctness` | 7 | v7 replayed with no API calls, after the correctness fixes: request limits, label precedence, API failure handling. AUC unchanged, R@50 corrected. See [`CORRECTNESS.md`](../benchmark/CORRECTNESS.md) |
| **`v9`** | 16 (2 Val held out) | By-sensor MCAPs; providers motion@3, clearance@4, detections@2; FUSED retrained (LORO over the 14 Train recordings); program prompt v3. Adds `l1_compositions_dev` with complete ground truth, and the `compose_test@val` / `@train` slices |
| `v9-prompt-v1` | 16 | Prompt A/B: the three gpt-6-luna configs with prompt v1, against v9's v3 |
| `v9-abstain` | 16 | The abstention control `abstain_v1`, with imu@2 |
| `v9-abstain-r2` | 16 | `abstain_v1` replayed under the coverage rule |

**v9 is not directly comparable with v7.** The corpus, providers, prompt and judged pools all changed. The one
like-for-like check: on the original 7 recordings, v9's hand-written programs score 0.78 / 0.74 on compose_test,
against v7's 0.77 / 0.74.

## Query sets

| Set | Queries | What it tests |
|---|---|---|
| `demo5` | 5 intents | The five demo questions, as written |
| `demo5_para` | 15 | Three paraphrases of each demo intent: robustness to wording |
| `compose_test` | 6 | Held-out compositions (turn→brake, person→speed-up, tight right-side room, close car while fast, …). Never tuned on; FUSED's pseudo-labels exclude these pairings |
| `compose_test@val`, `@train` | 6 | compose_test scored only on the 2 held-out Val recordings, or only on the Train recordings |
| `l1_compositions_dev` | 6 | Behavior questions from the challenge set, with **complete** ground truth: every in-scope window is judged, so a 0 is real |
| `abstain_controls` | 1 | A question whose true answer is "nowhere" (see [EVALS.md](EVALS.md#abstention-controls)) |

`compose_dev` is the program generator's few-shot pool, never scored. Intent groups are small (5–6 per set), so
the bootstrap CIs in `report.html` are wide: read differences as directional.

## v9 headline numbers

**Metrics:**
- **ROC** is macro ROC-AUC, penalised for abstention.
- **nDCG@10** and **R@50** use condensed lists, with unjudged windows removed.
- **judged@10** is the share of the top 10 that has a judgment.
- **status** is the share of answers whose status matches the expected one.
- **p50** is the median uncached search. For `*_LUNA` configs that includes generating the program; cached, they
  answer in tens of milliseconds.

**demo5_para (15 paraphrases)**

| Config | ROC | nDCG@10 | R@50 | judged@10 | status | tokens | p50 |
|---|---|---|---|---|---|---|---|
| TAGS | 0.52 | 0.06 | 0.33 | 0.60 | – | 0 | 2 ms |
| EMBED | 0.46 | 0.25 | 0.36 | 1.00 | – | 0 | 28 ms |
| FUSED | **0.73** | **0.47** | **0.61** | 1.00 | – | 0 | 33 ms |
| PROGRAM_ORACLE | 0.56 | 0.35 | 0.43 | 1.00 | 1.00 | 0 | 61 ms |
| PROGRAM_LUNA | 0.55 | 0.24 | 0.34 | 0.93 | 0.93 | 6.6k | 7.0 s |
| HYBRID_LUNA | 0.52 | 0.13 | 0.18 | 1.00 | 0.87 | 6.6k | 5.8 s |
| FUSED_V_LUNA | 0.58 | 0.32 | 0.24 | 1.00 | 0.93 | 6.6k | 6.1 s |

**compose_test (6 held-out compositions)**

| Config | ROC | nDCG@10 | R@50 | judged@10 | status | tokens | p50 |
|---|---|---|---|---|---|---|---|
| TAGS | 0.55 | 0.00 | 0.00 | 0.20 | – | 0 | < 1 ms |
| EMBED | 0.47 | 0.22 | 0.19 | 1.00 | – | 0 | 26 ms |
| FUSED | 0.59 | 0.46 | 0.17 | 1.00 | – | 0 | 30 ms |
| PROGRAM_ORACLE | **0.66** | 0.45 | **0.41** | 1.00 | 1.00 | 0 | 38 ms |
| PROGRAM_LUNA | 0.59 | 0.38 | 0.23 | 0.80 | 0.83 | 5.3k | 4.6 s |
| HYBRID_LUNA | 0.56 | 0.05 | 0.04 | 1.00 | 0.80 | 5.3k | 4.5 s |
| FUSED_V_LUNA | 0.61 | **0.47** | 0.18 | 1.00 | 0.80 | 5.3k | 4.6 s |

**compose_test by recording group (ROC / nDCG@10)**

| Config | original 7 | 9 new | Val (2 held out) |
|---|---|---|---|
| EMBED | 0.39 / 0.02 | 0.40 / 0.25 | 0.25 / 0.24 |
| FUSED | 0.64 / 0.59 | 0.57 / 0.45 | **0.75 / 0.54** |
| FUSED_V_LUNA | 0.77 / 0.65 | 0.60 / 0.46 | 0.61 / 0.37 |
| PROGRAM_LUNA | 0.69 / 0.52 | **0.68** / 0.38 | 0.58 / 0.39 |
| PROGRAM_ORACLE | **0.78 / 0.74** | 0.65 / 0.38 | 0.58 / 0.39 |

**l1_compositions_dev (6 behavior questions, complete ground truth)**

| Config | ROC | nDCG@10 | R@50 | tokens | p50 |
|---|---|---|---|---|---|
| EMBED | 0.41 | 0.05 | 0.05 | 0 | 26 ms |
| FUSED | 0.72 | 0.15 | 0.06 | 0 | 31 ms |
| PROGRAM_LUNA | 0.89 | 0.14 | 0.19 | 8.2k | 12.5 s |
| HYBRID_LUNA | 0.87 | 0.14 | 0.05 | 8.2k | 11.5 s |
| FUSED_V_LUNA | **0.90** | **0.23** | **0.20** | 8.2k | 11.4 s |

These questions have no hand-written programs, so there are no ORACLE rows. Positives are rare: two questions
have none in scope. That is why ROC is high while nDCG is low.

### What v9 says
- **FUSED is the best zero-cost config.** It leads on paraphrases and on the held-out Val recordings in about
  30 ms, with no LLM call.
- **Programs win on compositions where the features transfer.** Hand-written programs lead on the original 7
  recordings. On the 9 new ones every config drops, programs included: the structured signals transfer imperfectly
  to new scenes.
- **Generated programs trail the hand-written ceiling by about 0.07 ROC and nDCG on compose_test.** On behavior
  questions nobody tuned for, they reach ROC 0.87–0.90 after prompt v3.
- **HYBRID's nDCG is the weak spot.** It verifies only EMBED's candidates, and EMBED rarely surfaces the right
  windows for compositions.
- **No config says "none" when the true answer is none.** On the two l1_compositions_dev questions with no
  qualifying episode, every config still returns partial or unverified results. Only complete ground truth could
  show this.

## Side runs on v9

- **Prompt A/B (`v9-prompt-v1`).** v1 → v3 for the gpt-6-luna configs:
  - demo5_para status accuracy rose from 0.73 to 0.93;
  - l1_compositions_dev ROC rose from 0.46–0.50 to 0.87–0.90;
  - HYBRID's demo5_para nDCG fell by 0.11, the one regression.

  Details in [EVALS.md](EVALS.md#hill-climbing-the-program-generator).
- **Abstention (`v9-abstain`, `v9-abstain-r2`).**
  - Only PROGRAM_ORACLE and PROGRAM_LUNA prove "nowhere" (`none_found_exhaustive`).
  - The verify configs abstain without proof (`insufficient_evidence`).
  - TAGS, EMBED and FUSED return 50 unverified windows each.
  - r2 replays the first run under the coverage rule.

  Details in [EVALS.md](EVALS.md#abstention-controls).

## Earlier checkpoints

**v7 (7 recordings; v8-correctness leaves AUC unchanged).**

| Config | demo5_para ROC | demo5_para nDCG@10 | compose_test ROC | compose_test nDCG@10 | p50 latency | tokens |
|---|---|---|---|---|---|---|
| EMBED | 0.47 | 0.25 | 0.38 | 0.07 | 24 ms | 0 |
| FUSED_CONCAT | 0.47 | 0.21 | 0.41 | 0.20 | 24 ms | 0 |
| FUSED_LINEAR | 0.62 | 0.33 | 0.47 | 0.39 | 26 ms | 0 |
| FUSED | **0.64** | **0.44** | 0.54 | 0.45 | 28 ms | 0 |
| FUSED_V_LUNA | 0.61 | 0.31 | 0.74 | 0.54 | ~5–7 s cold, ~40 ms cached | ~5k |
| PROGRAM_LUNA | 0.60 | 0.27 | 0.69 | 0.45 | ~5–7 s cold | ~5k |
| PROGRAM_ORACLE | 0.58 | 0.40 | **0.77** | **0.74** | 17–58 ms | 0 |

**v8-correctness** fixed the result limit, which changes pooled R@50 substantially. The corrected compose_test
R@50 values are FUSED 0.29, PROGRAM_LUNA 0.55 and PROGRAM_ORACLE 0.83; the full table is in
[`CORRECTNESS.md`](../benchmark/CORRECTNESS.md#effect-on-the-measured-results).

## Replaying and freezing

Replays run from the repository root against the built dev bundle. Reusing a checkpoint's program cache makes no
API calls. A missing cache entry fails explicitly, and generation costs are carried over from the original run.
Use a fresh run name, because the runner resumes completed rows.

```bash
uv run python -m alloy_trainer.eval.run --run v9-replay --reuse-programs-from benchmark/results/v9
uv run python -m alloy_trainer.eval.run --run v9-replay --score-all --reuse-programs-from benchmark/results/v9
uv run python -m alloy_trainer.eval.report --run v9-replay      # writes results/eval/v9-replay/report.{json,html}
```

`--configs` and `--sets` narrow a run (for example `--sets abstain_controls`). Working runs land in the ignored
`results/eval/<run>/`. To freeze one, copy it to `benchmark/results/<run>/` with a `manifest.json` holding the code
commit and a note on what changed, and add a row to the checkpoint table above.
