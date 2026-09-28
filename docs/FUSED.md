# FUSED — single-vector retrieval over robot windows (Stage C)

This is the brief for anyone improving FUSED. Read all of it before changing anything; the **guardrails** are what
keep the numbers honest.

## What FUSED is

**The job.** FUSED answers a natural-language question about robot recordings with **one vector lookup**. There is
no LLM call, no program and no per-query tokens.

**The target to beat is PROGRAM.** PROGRAM uses an LLM (`gpt-6-luna`) to write a `QueryProgram` that is executed
over structured features, and it is more precise on compositional questions. It also costs 5–12 s and about 5–8k
tokens per query when the program isn't cached. FUSED runs in about 30 ms.

The goal is to close the quality gap to PROGRAM without giving up that cost profile.

```
query side (unchanged, frozen):  utterance ──► SigLIP2 text tower ──► q ∈ R^1152 (L2)
document side (learned):         window w ──► [ SigLIP2 image mean (frozen, 1152) ; s(w) signals+masks (46) ]
                                            ──► head (residual MLP, zero-init) ──► f(w) ∈ R^1152 (L2)
score(q, w) = q · f(w)          (exact dot product over the window index; same pipeline runner as every config)
```

- **Windows.** A window is a 4 s trailing span at a 1 s stride: `Rec:EEEE` covers `[EEEE-4 s, EEEE]`, 2,253
  windows across 16 recordings.
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
  - Each fold holds out one of the 14 Train recordings, trains on the other 13, then embeds the held-out one.
  - The 2 Val recordings are never in a fold: the full-data model (trained on all 14) embeds them.
  - The out-of-fold vectors from all folds form one corpus index, `bundle/index/fused_v1_windows.parquet`. Every
    window is embedded by a model that never saw its recording.
  - The benchmark queries that index through the standard pipeline specs `FUSED` (lookup) and `FUSED_V` (lookup,
    then program verification).
- **The benchmark is the only arbiter.**
  - Judgments: pooled relevance, graded 0/1/2 by an Opus labeller with audited tools. Unjudged windows are never
    counted as 0.
  - Metrics: macro ROC-AUC (penalised for abstention), nDCG@10, and R@10/R@50 on condensed lists.
  - Query sets: `demo5_para` (15 paraphrases), `compose_test` (6 held-out compositions, also sliced to the Val
    recordings as `compose_test@val`) and `l1_compositions_dev` (6 behavior questions with complete ground truth).
  - Uncertainty: bootstrap CIs over intent groups, with n = 5–6 groups per set, so read differences as directional.
- **Diagnostics** (cheap, and *not* ground truth), written to `bundle/index/<name>.json`:
  - out-of-fold pseudo-label AUC (EMBED vs FUSED) per held-out recording, which measures distillation fidelity;
  - hubness: the top-5 concentration over probe texts.

### Where FUSED stands (eval v9)

| Config | demo5_para ROC / nDCG@10 | compose_test ROC / nDCG@10 | compose_test@val ROC / nDCG@10 | p50 latency | tokens |
|---|---|---|---|---|---|
| EMBED (image only) | 0.46 / 0.25 | 0.47 / 0.22 | 0.25 / 0.24 | 28 ms | 0 |
| FUSED_CONCAT (control: no learning) | 0.43 / 0.19 | 0.44 / 0.11 | 0.23 / 0.19 | 30 ms | 0 |
| FUSED_LINEAR (control: linear head) | 0.65 / 0.42 | 0.58 / 0.35 | 0.71 / 0.25 | 31 ms | 0 |
| **FUSED (MLP head)** | **0.73 / 0.47** | 0.59 / 0.46 | **0.75 / 0.54** | 33 ms | 0 |
| FUSED_V_LUNA | 0.58 / 0.32 | 0.61 / 0.47 | 0.61 / 0.37 | ~5–6 s cold | ~5–7k |
| PROGRAM_LUNA | 0.55 / 0.24 | 0.59 / 0.38 | 0.58 / 0.39 | ~5–7 s cold | ~5–7k |
| PROGRAM_ORACLE (the ceiling) | 0.56 / 0.35 | **0.66** / 0.45 | 0.58 / 0.39 | 40–60 ms | 0 |

- **The head is doing the work.** Concatenation alone is no better than EMBED; a linear head gets most of the
  gain; the MLP adds the rest.
- **FUSED leads on paraphrases and on the two held-out Val recordings,** at no per-query cost. Hand-written
  programs still lead on compositions over the original 7 recordings (0.78 / 0.74 against FUSED's 0.64 / 0.59).
- **Diagnostics.** Out-of-fold pseudo-label AUC: EMBED 0.51, FUSED 0.70. Hubness (the busiest window's top-5
  count): 15, down from 20 in v7.

Every set, the v7 → v9 history and the per-slice breakdown are in [RESULTS.md](RESULTS.md).

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
7. **More data.** Doubling the training recordings in v9 lifted compose_test on the original 7 from 0.54 / 0.45 to
   0.64 / 0.59. New bags go through the same LORO, so this lever stays open.

## Guardrails (non-negotiable)

- **Never train on L2 judgments.** They are the test labels.
- **LORO always.** Standardisation statistics come from training folds only.
- **Don't add the `compose_test` pairings** to the pseudo-label programs or templates.
- **Keep the Val recordings out** of anything used for training or tuning.
- **Report changes on the benchmark** with CIs, next to the controls. Pseudo-label AUC is a diagnostic only.
- **Keep the zero-init residual,** or show on the benchmark why removing it helps.

## How to run

From the repository root:

```bash
# train LORO folds + write the out-of-fold index (also writes diagnostics to bundles/dev/index/fused_v1.json)
uv run python -m alloy_trainer.learn.fused --kind mlp --name fused_v1
# controls
uv run python -m alloy_trainer.learn.fused --kind linear --name fused_linear
uv run python -m alloy_trainer.learn.fused --kind concat --name fused_concat

# benchmark: a fresh run name; reuse frozen programs so LLM configs cost nothing
uv run python -m alloy_trainer.eval.run --run myexp --configs FUSED,FUSED_LINEAR,FUSED_CONCAT,EMBED --reuse-programs-from benchmark/results/v9
uv run python -m alloy_trainer.eval.run --run myexp --score-all --configs FUSED,FUSED_LINEAR,FUSED_CONCAT,EMBED --reuse-programs-from benchmark/results/v9
uv run python -m alloy_trainer.annotate.l2 --run myexp --skip-judged   # lists windows your config surfaced that nobody has judged yet
uv run python -m alloy_trainer.eval.report --run myexp
```

If `annotate.l2` lists unjudged windows, your top results reach outside the judged pool. Either get them judged,
which is a labeller job, or report `judged@10` alongside the metrics. Condensed-list metrics ignore unjudged windows.

## Files

| File | What it holds |
|---|---|
| `index/src/alloy_index/models/fused.py` | `SIGNALS`, window features and the head (shared by training and `apply.fused`) |
| `trainer/src/alloy_trainer/learn/fused.py` | pseudo-label programs, loss, LORO training, diagnostics |
| `server/src/alloy_server/models/siglip.py` | the frozen encoder recipe shared by the query and document sides |
| `index/src/alloy_index/build/pipelines.py` | the `FUSED`, `FUSED_V`, `FUSED_LINEAR` and `FUSED_CONCAT` specs |
| `trainer/src/alloy_trainer/eval/*` | the benchmark runner, pooling, metrics and report |
| `benchmark/results/v9/` | the current frozen reference (earlier checkpoints: [RESULTS.md](RESULTS.md)) |
