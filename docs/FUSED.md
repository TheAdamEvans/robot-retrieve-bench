# FUSED — single-vector retrieval over robot windows (Stage C)

This is the brief for anyone improving FUSED. Read all of it before changing anything; the **guardrails** are what
keep the numbers honest.

## What FUSED is

**The job.** FUSED answers a natural-language question about robot recordings with **one vector lookup**. There is
no LLM call, no program and no per-query tokens.

**The target to beat is PROGRAM.** PROGRAM uses an LLM (`gpt-6-luna`) to write a `QueryProgram` that is executed
over structured features, and it is more precise on compositional questions. It also costs 5–8 s and about 5–6k
tokens per query when the program isn't cached. FUSED runs in about 28 ms.

The goal is to close the quality gap to PROGRAM without giving up that cost profile.

```
query side (unchanged, frozen):  utterance ──► SigLIP2 text tower ──► q ∈ R^1152 (L2)
document side (learned):         window w ──► [ SigLIP2 image mean (frozen, 1152) ; s(w) signals+masks (46) ]
                                            ──► head (residual MLP, zero-init) ──► f(w) ∈ R^1152 (L2)
score(q, w) = q · f(w)          (exact dot product over the window index; same pipeline runner as every config)
```

- **Windows.** A window is a 4 s trailing span at a 1 s stride: `Rec:EEEE` covers `[EEEE-4 s, EEEE]`, about 680
  windows across 7 recordings.
- **Image features.** The mean of L2-normalised SigLIP2 (`google/siglip2-so400m-patch14-384`) vectors from the front
  camera's frames at 10 Hz inside the window. Per-frame vectors are kept in `bundle/features/siglip2_frames/`.
- **`s(w)`.** 23 window statistics from the provider features: speed (mean, min, max, Δ, max drop), yaw-rate max,
  heading Δ, acceleration (min, max), front clearance, any-direction clearance, lateral room on each side, gap width,
  a doorway flag, persons (mean, max), persons in the corridor, vehicles, vehicle box fraction, bicycles, new person
  tracks, and a robot flag. Each is standardised using **training-fold statistics only** and paired with a presence
  mask, giving 46 dimensions. See `SIGNALS` in `index/src/alloy_index/models/fused.py`.
- **Head.**
  - Architecture: `Dropout(0.2) → Linear(1198→512) → GELU → Dropout(0.2) → Linear(512→1152)`.
  - Output: added to the image vector (a residual), then L2-normalised.
  - **The last layer is zero-initialised**, so FUSED starts exactly at EMBED and learns only a correction from
    signals. This stopped an early version collapsing into hubs.
- **Loss.** A SigLIP-style pairwise sigmoid loss over a multi-positive window × text matrix, with a learned
  temperature (initialised at log 10) and bias (−10).
- **Training.** Full batch, AdamW (learning rate 5e-4, weight decay 1e-2), 150 epochs, per fold. It takes seconds
  on MPS.

## Training data (no human grades are ever used)

Text-window pairs come from two sources, built per fold from the **training recordings only**:

1. **Program pseudo-labels.** These are the "usage becomes a dataset" loop, and they distil PROGRAM into the
   embedding.
   - 20 grammar-sampled `QueryProgram`s (`PSEUDO` in `fused.py`), each rendered as 2–3 texts.
   - Each program is **executed** with the same executor PROGRAM uses.
   - Windows containing a *definite* match's primary anchor are positives.
   - Examples: slowdowns, turns (left and right), stop, go, a person appearing, a crowd, a close obstacle, a doorway,
     a car, a bike, moving fast, a narrow passage, and compositions such as "slows as a person appears" or "turns in
     a narrow passage".
2. **L1 labeller text.** For each 4 s segment, the Opus labeller's one-sentence caption and templated attribute
   sentences, such as "a group of people standing together near the robot's path".

**Excluded on purpose:** the `compose_test` feature × operator pairings (turn→brake, person→speed-up, tight
right-side room, close car while fast). Adding them would leak the held-out test.

## Evaluation

- **Leave-one-recording-out (LORO), always.**
  - Each of 7 folds trains on 6 recordings, then embeds the held-out recording.
  - The out-of-fold vectors from all folds form one corpus index, `bundle/index/fused_v1_windows.parquet`. Every
    window is embedded by a model that never saw its recording.
  - The benchmark queries that index through the standard pipeline specs `FUSED` (lookup) and `FUSED_V` (lookup,
    then program verification).
- **The benchmark is the only arbiter.**
  - Judgments: pooled relevance, graded 0/1/2 by an Opus labeller with audited tools. Unjudged windows are never
    counted as 0.
  - Metrics: macro ROC-AUC (penalised for abstention), nDCG@10, and R@10/R@50 on condensed lists.
  - Query sets: `demo5_para` (15 paraphrases) and `compose_test` (6 held-out compositions).
  - Uncertainty: bootstrap CIs over intent groups, with n = 5–6 groups per set, so read differences as directional.
- **Diagnostics** (cheap, and *not* ground truth), written to `bundle/index/<name>.json`:
  - out-of-fold pseudo-label AUC (EMBED vs FUSED) per held-out recording, which measures distillation fidelity;
  - hubness: the top-5 concentration over probe texts.

### Results (eval v7; v8-correctness leaves AUC unchanged)

| Config | demo5_para ROC | demo5_para nDCG@10 | compose_test ROC | compose_test nDCG@10 | p50 latency | tokens |
|---|---|---|---|---|---|---|
| EMBED (image only) | 0.47 | 0.25 | 0.38 | 0.07 | 24 ms | 0 |
| FUSED_CONCAT (control: no learning) | 0.47 | 0.21 | 0.41 | 0.20 | 24 ms | 0 |
| FUSED_LINEAR (control: linear head) | 0.62 | 0.33 | 0.47 | 0.39 | 26 ms | 0 |
| **FUSED (MLP head)** | **0.64** | **0.44** | 0.54 | 0.45 | 28 ms | 0 |
| FUSED_V (FUSED + LLM program verification) | 0.61 | 0.31 | 0.74 | 0.54 | ~5–7 s cold, ~40 ms cached | ~5k |
| PROGRAM_LUNA (LLM program) | 0.60 | 0.27 | 0.69 | 0.45 | ~5–7 s cold | ~5k |
| PROGRAM_ORACLE (hand-written program: the ceiling) | 0.58 | 0.40 | **0.77** | **0.74** | 17–58 ms | 0 |

- **Out-of-fold pseudo-label AUC:** EMBED 0.51, FUSED 0.69 (clearance@3 signals).
- **Hubness:** the busiest window appears in the top 5 twenty times for FUSED, against fifteen for EMBED.
- **v8-correctness, compose_test R@50:** FUSED 0.29, PROGRAM_LUNA 0.55, PROGRAM_ORACLE 0.83.

### Results, eval v9 (16 recordings; LORO over the 14 Train recordings; Val embedded by the full-data model)

| Config | demo5_para ROC / nDCG@10 | compose_test ROC / nDCG@10 | compose_test, original 7 recordings | compose_test, 9 new recordings | compose_test, Val (2 held out) |
|---|---|---|---|---|---|
| EMBED | 0.46 / 0.25 | 0.47 / 0.22 | 0.39 / 0.02 | 0.40 / 0.25 | 0.25 / 0.24 |
| **FUSED** | **0.73 / 0.47** | 0.59 / 0.46 | 0.64 / 0.59 | 0.57 / 0.45 | **0.75 / 0.54** |
| FUSED_V_LUNA | 0.58 / 0.32 | 0.61 / 0.47 | 0.77 / 0.65 | 0.60 / 0.46 | 0.61 / 0.37 |
| PROGRAM_LUNA | 0.55 / 0.24 | 0.59 / 0.38 | 0.69 / 0.52 | 0.68 / 0.38 | 0.58 / 0.39 |
| PROGRAM_ORACLE | 0.56 / 0.35 | 0.66 / 0.45 | 0.78 / 0.74 | 0.65 / 0.38 | 0.58 / 0.39 |

- **On the original 7 recordings, v9 reproduces v7.** PROGRAM_ORACLE scores 0.78 / 0.74, against 0.77 / 0.74.
  FUSED improves from 0.54 / 0.45 to 0.64 / 0.59 with twice the training data.
- **The 9 new recordings are harder for everything,** including hand-written programs. That is a transfer gap in the
  structured signals.
- **FUSED holds up best on the two held-out Val recordings.**
- **Out-of-fold pseudo-label AUC:** EMBED 0.51, FUSED 0.70. **Hubness** (the busiest window's top-5 count): 15,
  down from 20.

## Known weaknesses (the improvement surface)

1. **Mean pooling throws away temporal order.** "Turn *then* brake" and "brake then turn" look the same, and the
   signal statistics are order-free too. This is the biggest lever on `compose_test`.
2. **The pseudo-labels come from our own providers.** FUSED can at best reproduce what the executor sees, including
   its blind spots: the doorway detector has precision 0.15, and the detector-based person and vehicle counts miss
   things.
3. **Too little text diversity.** About 45 templated pseudo-label texts, plus about 150 captions per fold. Queries
   phrased unlike either generalise poorly.
4. **Hubness.** A few windows win many unrelated queries.
5. **A shared model family.** Captions (L1) and judgments (L2) both come from the same Opus labeller, so FUSED
   could learn the judge's phrasing. LORO guards against leaking recordings, not against this.
6. **Small data.** 16 recordings (5 Jackal) since eval v9. The SCAND Val recordings `Rec_Tent_129` and
   `Bass_Garage_134` are **held out**: never trained on, and embedded by the full-data model.

## Ranked ideas

1. **A temporal encoder over the per-frame vectors plus a 20 Hz telemetry sequence.** A small 1D conv or a 2-layer
   transformer, with order probes such as segment-swapped windows.
2. **All-but-one-clause hard negatives.** For each pseudo-label program, render texts that violate exactly one
   clause ("…at constant speed", "…with nobody ahead") as explicit negatives.
3. **Paraphrase diversity.** An offline `gpt-6-luna` pass paraphrases each pseudo-label template, costed once.
4. **Per-frame max-sim (late interaction).** Score `max_t q·f(frame_t)`, or attention pooling, instead of mean
   pooling.
5. **FUSED as the HYBRID generator**, followed by `PredicateRanker`. It already exists as `FUSED_V`; a cheap local
   program generator would remove its LLM cost.
6. **Tune the loss**: temperature and bias, balancing the loss between multi-positive templates and single-positive
   captions, and a hubness penalty (CSLS or centring).
7. **More data** from the new recordings, once indexed, run through LORO over all Train recordings.

## Guardrails (non-negotiable)

- **Never train the retriever's relevance targets on L2 judgments.** The v2 salience hint uses positive Train episodes only, as described below; Eval labels stay out of training.
- **LORO always.** Standardisation statistics come from training folds only.
- **Don't add the `compose_test` pairings** to the pseudo-label programs or templates.
- **Keep the Val recordings out** of anything used for training or tuning.
- **Report changes on the benchmark** with CIs, next to the controls. Pseudo-label AUC is a diagnostic only.
- **Keep the zero-init residual,** or show on the benchmark why removing it helps.

## FUSED v2: interval salience and verified negatives

The trainer and server remain separate. `alloy_trainer.learn.fused_v2` executes
Train programs, creates window/text labels, fits the document-side head and writes
two indexes. `fused_v2_<sampler>_oof_windows.parquet` is the LORO benchmark
artifact. `fused_v2_<sampler>_windows.parquet` is made by the full Train model
for serving and for newly indexed windows. The server loads either index and
uses its existing SigLIP2 text encoder and dot-product lookup. The query path
has no program synthesis.

`alloy_trainer.learn.importance` fits a shallow CPU XGBoost Poisson regressor
directly on the **1152-dimensional frame vectors**, without PCA. The weak target
is the number of distinct positive `labels/train/episodes` answer intents
whose interval covers each frame. No known interval means *unlabeled*, not
confirmed irrelevant. Background frames are subsampled with inverse sampling
weights. A frame-level prediction is averaged into each four-second window.
The public `benchmark/importance/fused_v2_scores.jsonl` gives one prediction
per window: Train scores are leave-one-recording-out; Val scores use Train only.
The manifest records the recipe and provenance.
Nested fold scores are cached inside the ignored bundle by a separate CPU
process; this avoids an OpenMP runtime conflict between XGBoost and PyTorch on
some machines. On macOS, XGBoost may also require `brew install libomp`.

The answer intervals influence
only the **sampling probability**, not window/text truth labels or the online
ranker. It still carries a query-set bias; compare against the uniform sampler
on the same frozen benchmark, especially on held-out Val and composition
questions. The importance sampler uses 50% uniform and 50% score-proportional
selection within each recording, with a four-times-uniform cap.

The v2 retriever expands the 20 fixed programs into 140 program/text examples,
including numeric threshold variants and safe compositions. It excludes the
held-out `compose_test` pairings.
The executor supplies temporal and co-occurrence supervision; these
observations alone do not establish that one event caused another.
Each required clause is executed on the Train recording. A definite match is
positive; a fully covered one-clause failure is a hard negative; an unknown
never becomes a negative. L1 captions and attributes add visual supervision.
Each step samples one positive, one near miss when available and one other
verified negative, with equal total positive and negative loss weight.
`--mine-every 25` optionally refreshes a pool of top-scoring false windows
across Train recordings and draws half of the second negatives from it.
Only executor-verified negatives enter this pool. The default keeps online
mining off because it reduced global top-10 quality in the development
ablation. Both sampling arms use the same program pool, architecture, seed,
steps and negative-selection procedure.
The loss also gives a small weight to a positive-over-negative ranking term and
to an in-batch caption contrastive term, which preserve text specificity.
The one-hidden-layer head defaults to width 512 and is configurable with
`--hidden-dim` and `--dropout`.

### Train-only model selection

`alloy_trainer.eval.fused_development` scores 26 held-out Train queries.
Their exact wording and numeric program settings are absent from training.
Each Train recording is embedded by a model that did not train on it; the
command refuses an in-sample serving index. It reports global and
within-recording hit@10, recall@50, judged precision and nDCG@10, judged
coverage@10, and per-recording results. Unknown executor outcomes stay
unjudged: they earn no nDCG credit and their frequency is exposed by coverage,
without being asserted false. The main selection target is **global judged
nDCG@10**, alongside global recall@50 and judged coverage@10. A higher pairwise
AUC alone is not
grounds to ship a new retriever. Keep the frozen Val and `compose_test` sets
for final confirmation after choosing a recipe on Train-only development.

```bash
uv run python -m alloy_trainer.eval.fused_development --bundle bundles/dev
```

Before expanding the programs or mining negatives, FUSED v1 achieved global
nDCG@10 .640 and recall@50 .183 on these queries; v2 uniform achieved .589
and .141, and v2 importance .565 and .140. Within-recording nDCG was .568
for v1 and .644 for v2 uniform. The gap shows that local separation is not
yet transferring to global retrieval.

The expanded program set raises executor-verified positive window/text pairs
from 47,900 to 61,124 and one-clause hard-negative pairs from 16,862 to
21,947. These are additional query/window pairings on the same recordings,
not 13,224 independent sensor scenes. At 150 steps with uniform sampling:

| Train-only development model | Global nDCG@10 | Global recall@50 | Judged@10 | Within-recording nDCG@10 |
|---|---:|---:|---:|---:|
| FUSED v1 | .640 | .183 | .981 | .568 |
| Prior v2 uniform, 105 examples | .589 | .141 | .954 | .644 |
| Expanded v2 uniform, online mining off | .597 | .136 | .946 | .628 |
| Expanded v2 uniform, mining every 25 steps | .548 | .150 | .965 | .632 |

The extra verified examples slightly improved v2's global top-10 score when
online mining was off, but v1 is still stronger on the selection target and
recall@50. Online mining improved recall but caused a larger top-10 regression.
Keep v1 as the serving default. This comparison is directional because the
26 development queries share ten program families.

```bash
uv run python -m alloy_trainer.learn.importance --bundle bundles/dev --labels labels
uv run python -m alloy_trainer.learn.fused_v2 --bundle bundles/dev --labels labels --sampler both
uv run python -m alloy_trainer.learn.fused_v2 --bundle bundles/dev --labels labels --sampler both --diagnose-only
uv run python -c "from pathlib import Path; from alloy_index.build.pipelines import build; build(Path('bundles/dev'))"
uv run python -m alloy_trainer.eval.run --run fused-v2 --configs EMBED,FUSED,FUSED_V2_UNIFORM_OOF,FUSED_V2_IMPORTANCE_OOF --sets demo5_para,compose_test
uv run python -m alloy_trainer.eval.run --run fused-v2 --score-all --configs EMBED,FUSED,FUSED_V2_UNIFORM_OOF,FUSED_V2_IMPORTANCE_OOF --sets demo5_para,compose_test
uv run python -m alloy_trainer.eval.report --run fused-v2
```

The diagnostic command writes per-family out-of-fold AUC and hard-negative
pair win rates to `bundle/index/fused_v2_<sampler>_diagnostics.json`. The
benchmark report then shows query-level retrieval metrics and judged coverage.

### Measured result (16 recordings, 14 Train LORO folds; 150 steps, seed 0)

| Retriever | demo5_para ROC / nDCG@10 | compose_test ROC / nDCG@10 | compose_test Val ROC / nDCG@10 | compose_test judged@10 |
|---|---:|---:|---:|---:|
| EMBED | .463 / .246 | .466 / .225 | .246 / .236 | 1.00 |
| FUSED v1 | .726 / .467 | .591 / .456 | .754 / .539 | 1.00 |
| v2 uniform | .768 / .366 | .698 / .450 | .784 / .000 | .40 |
| v2 importance | .778 / .358 | .704 / .437 | .784 / .000 | .44 |

The v2 head discriminates judged positive and negative windows better, but
its global top results miss known Val positives and have low judged coverage.
**Keep FUSED v1 as the default serving retriever** until top-rank recall and
judgment coverage improve. The importance sampler adds only a small ROC change
against the matched uniform arm and does not improve nDCG@10 here.

The interval-salience regressor has only 30 positive Train intervals across
five recordings. Its global LORO top-decile interval lift is .097, so its
predictions are a weak sampling hint, not evidence that importance transfer
has been solved. Within-program diagnostics are stronger: the v2 importance
index has macro family AUC .788 and a .740 positive-over-hard-negative win
rate. Fast-speed variants and crowd/slow compositions are among the weak
families to target with additional verified data. These metrics use overlapping
windows and few query groups, so differences are directional.

## How to run

From the repository root:

```bash
# train LORO folds + write the out-of-fold index (also writes diagnostics to bundles/dev/index/fused_v1.json)
uv run python -m alloy_trainer.learn.fused --kind mlp --name fused_v1
# controls
uv run python -m alloy_trainer.learn.fused --kind linear --name fused_linear
uv run python -m alloy_trainer.learn.fused --kind concat --name fused_concat

# benchmark: a fresh run name; reuse frozen programs so LLM configs cost nothing
uv run python -m alloy_trainer.eval.run --run myexp --configs FUSED,FUSED_LINEAR,FUSED_CONCAT,EMBED --reuse-programs-from benchmark/results/v7
uv run python -m alloy_trainer.eval.run --run myexp --score-all --configs FUSED,FUSED_LINEAR,FUSED_CONCAT,EMBED --reuse-programs-from benchmark/results/v7
uv run python -m alloy_trainer.annotate.l2 --run myexp --skip-judged   # lists windows your config surfaced that nobody has judged yet
uv run python -m alloy_trainer.eval.report --run myexp
```

If `annotate.l2` lists unjudged windows, your top results reach outside the judged pool. Either get them judged,
which is a labeller job, or report `judged@10` alongside the metrics. Condensed-list metrics ignore unjudged windows.

## Files

| File | What it holds |
|---|---|
| `trainer/src/alloy_trainer/learn/fused.py` | features, pseudo-label programs, head, loss, LORO, diagnostics |
| `server/src/alloy_server/models/siglip.py` | the frozen encoder recipe shared by the query and document sides |
| `index/src/alloy_index/build/pipelines.py` | the `FUSED`, `FUSED_V`, `FUSED_LINEAR` and `FUSED_CONCAT` specs |
| `trainer/src/alloy_trainer/eval/*` | the benchmark runner, pooling, metrics and report |
| `benchmark/results/v7/`, `benchmark/results/v8-correctness/` | frozen reference results |
