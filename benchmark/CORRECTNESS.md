# Correctness checkpoint

The original v7 artifacts are preserved in [`results/v7/`](results/v7/). The corrected run is
[`results/v8-correctness/report.html`](results/v8-correctness/report.html); `report.html` in this directory is a copy
of the current report. Both checkpoints include hashes of their artifacts.

## Changes

- A nonzero `SearchRequest.k` overrides the pipeline's default `final_k` consistently. An omitted limit uses
  `final_k`, falling back to 10. Previously, requests for 20 or 50 candidates were capped at 10 after the truncation
  stage. `SCORE_ALL` continues to score every supplied window.
- Program API errors return failed diagnostics and an explanatory note. PROGRAM abstains; HYBRID/FUSED_V keep
  unverified retrieval results. Requests use a 20-second timeout and no automatic API retries. Failed API requests
  count as calls, and transient failures are never cached. The existing one-repair policy for invalid programs remains.
- Label readers share explicit shard precedence in `annotations/labels/precedence.json`. Follow-up judgments
  supersede the original shards; complete Butler labels supersede the smoke sample. Equal-priority cross-shard
  conflicts fail. The resolved 173 L1 labels and 1,061 L2 judgments match the original checkpoint.
- The report distinguishes supplied ORACLE programs from generated LUNA programs and includes R@50 and judged@50.

## Replaying the benchmark

Run from the repository root, with the existing dev bundle. Use a fresh run name: the runner resumes completed query/config rows.

```bash
uv run pytest -q
uv run python -m alloy_trainer.eval.run --run v8-correctness --reuse-programs-from benchmark/results/v7
uv run python -m alloy_trainer.eval.run --run v8-correctness --score-all --reuse-programs-from benchmark/results/v7
uv run python -m alloy_trainer.eval.report --run v8-correctness
```

The replay reuses v7's generated programs and makes no API calls. A missing cache entry fails explicitly. Generation
token usage comes from the original run. Uncached latency is reconstructed from the measured replay execution time
plus v7's measured generation time; it is labelled as an estimate. Cached-program latency measures the replay.

The new result limit can expose unjudged windows beyond the old top ten. Recall is against known positives in the
frozen judgment pool, not all relevant windows in the corpus. The main nDCG and recall metrics use condensed lists
(unjudged windows removed); the `unj=0` sensitivity column retains unjudged windows as zero. Judged@10 and judged@50
refer to the actual returned prefixes. Window metrics expand PROGRAM intervals around their primary anchors.

The v7 composition numbers 0.54, 0.69 and 0.77 are penalised ROC-AUC for FUSED, PROGRAM_LUNA and PROGRAM_ORACLE,
respectively. They are not nDCG scores. The composition AUC covers four eligible intent groups. Judgments are
model-generated; the blind re-check measures consistency within the same model family.

## Effect on the measured results

All 250 `SCORE_ALL` rows are identical to v7, so the AUC comparisons are unchanged. Correcting the search limit
changes pooled R@50 substantially. On `compose_test`:

| Config | v7 R@50 | Corrected R@50 | Corrected judged@50 |
|---|---:|---:|---:|
| EMBED | 0.007 | 0.068 | 45.2% |
| FUSED | 0.090 | 0.293 | 56.8% |
| PROGRAM_LUNA | 0.372 | 0.554 | 54.0% |
| PROGRAM_ORACLE | 0.684 | 0.832 | 61.3% |
| FUSED_V_LUNA | 0.112 | 0.361 | 55.4% |

The corrected run contains 260 search rows and 250 scoring rows, with zero new model API calls. The original label
pool is preserved; the coverage columns make the remaining unjudged depth visible. All 34 regression/contract tests
pass, including request limits, timeout/rate-limit/connection failures, recovery after a failure, label precedence,
and generation-cost preservation during replay.
