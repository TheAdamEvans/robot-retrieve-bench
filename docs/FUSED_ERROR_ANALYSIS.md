# FUSED retrieval error analysis (2026-09-28; updated after top-50 judging)

This compares the FUSED v1 leave-one-recording-out index, the expanded v2
uniform index trained for 150 steps without online mining, and the frozen v9
PROGRAM results. The v2 index was selected using Train-only development
queries before this analysis. The benchmark contains 14 Train and two held-out
Val recordings. PROGRAM_LUNA rows are the already recorded v9 run; no program
was regenerated. All retrieval rows use the same 16-recording bundle.

## Measure the first cut

For a reranker, **literal top-50** means the first 50 returned windows. The
published benchmark `R@50` instead removes unjudged windows before counting
50. A blinded Opus top-up judged all 159 previously unjudged intent/window
pairs in v2's literal first 50 on the five answerable `compose_test` intents.
Three temporary `-1` marks caused by an unavailable recording were resolved
with native evidence. The 159 effective new grades are 32 grade 2, 27 grade 1,
and 100 grade 0. V2's first 50 is now 250/250 judged. Battery abstention has
no relevant-window target and was excluded.

| Compose-test model | Known grade-2 fraction in literal first 50, before → after | Judged first 10, after | Judged first 50, after | nDCG@10, after |
|---|---:|---:|---:|---:|
| FUSED v1 | .173 → .161 | 1.00 | .748 | .456 |
| Expanded v2 uniform | .091 → .175 | 1.00 | 1.00 | .301 |
| PROGRAM_LUNA | .126 → .111 | .80 | .532 | .378 |
| PROGRAM_ORACLE | .276 → .264 | 1.00 | .564 | .448 |

These fractions use the same enlarged pool of known grade-2 windows for every
model, averaged by intent. They are **not exhaustive recall**: the pool is
partly built from retrieved results, many positive windows overlap one event,
and v1 and PROGRAM still have unjudged first-50 results. The new pool especially
favors coverage of v2's proposals, so the .175 versus .161 difference does not
establish a v2 win over v1. It does remove the main reason to believe v2 had
poor first-50 candidate coverage. V2 finds 68 known positive windows versus
v1's 74 across these intents; the macro fractions weight each intent equally.

Before the top-up, only 42% of v2's first ten were judged. Its condensed
nDCG@10 of .440 was therefore optimistic about the literal first ten; treating
unjudged ranks as zero gave a .111 lower bound. With full coverage the literal
and condensed values agree at **.301**, below v1's .456. V2's current issue is
more clearly *ordering within the first cut*, especially tight right-side
clearance, even though its first-50 candidates are more competitive than the
original pooled labels suggested.

The paid judging jobs cost **$16.76** in total, including failed setup attempts
and the three-window resolution; a separate CLI compatibility check cost about
$0.15. The labels are single-judge, agent-provisional judgments with audited
sensor and frame references. The successful shards are in campaigns
`l2-rank-audit-20260928-r2`, `-r3`, and `-r5`.

On the five `demo5_para` intent groups, literal first-50 known-hit fractions
are .610 for v1 and .590 for v2. The aggregate conceals a large trade: v2
improves vehicle interaction (.83 vs .67) while losing last-safe-evidence
retrieval (.07 vs .31). Both find all known crowd and doorway windows by rank
50, but v1 places the correct doorway much higher.

## Where the models succeed and fail

| Intent | v1 / v2 / PROGRAM_LUNA known grade-2 hits in first 50 | Interpretation |
|---|---|---|
| Person then speed-up | 30 / 19 / 22 of 119 | V2 improves after judging but still trails v1 on this broad event. |
| Tight right-side clearance | 21 / 8 / 3 of 59 | V2 gains six positives, yet still puts only one known positive in its first ten; v1 puts eight. |
| Turn then brake | 23 / 38 / 37 of 116 | V2 puts ten positives in its first ten and now has more known first-50 hits than v1 or PROGRAM_LUNA. |
| Close car while fast | 0 / 3 / 0 of 12 | V2 alone retrieves three known positives, though none is in its first ten. |
| Body camera sees person, front does not | 0 / 0 / 0 of 20 | The hand-written program finds 17 by rank 50. FUSED has front-camera image vectors and no side/rear-camera input. |

The vehicle-interaction query is v2's clearest qualitative gain. Across its
three paraphrases, v2 puts about 2.3 grade-2 windows in the first ten versus
zero for v1. One is a utility-cart encounter where the robot veers while the
cart passes; both PROGRAM_LUNA and the hand-written program miss all known
positive windows for that intent. Likewise, the close-car hit is a fast Jeep
pass that the structured program does not retrieve. These examples show that
image-plus-sensor retrieval can recover provider blind spots. They do not
demonstrate causal understanding: the labels describe observed sequences and
co-occurrences.

The doorway query shows a ranking failure. For the “tightest doorway actually
crossed” paraphrase, v1 has seven grade-2 windows in its first ten, while v2
has none. V2 favors another open doorway and approach windows. Yet both have
all known positives by rank 50, so a reranker could repair this case. The
right-clearance and last-safe-evidence failures are more serious because v2
often fails to put the right windows in the candidate set at all.

On the two held-out Val recordings, v1 retrieves nine known grade-2 windows in
the *global* first 50 across the three compose-test intents with Val positives;
v2 retrieves five after the top-up (four turn-then-brake and one right-side
clearance). This is a small set of overlapping windows, but the earlier
zero-versus-nine statement was a judgment-coverage artifact.

## What v1 did differently

The heads are the same one-hidden-layer residual MLP. The main changes are
optimization and supervision:

1. **Exposure.** V1 applies a sigmoid loss to every training window and text
   each step. A typical fold has about 1,683 windows and 475 texts: roughly
   800,000 pairwise terms per step, repeated 150 times. V2 uses one batch of
   256 positive triplets per step: 768 labelled pairwise terms, repeated 150
   times, plus a 64-caption contrastive batch. That is about 115,000 triplet
   terms for the whole fold. There are only 38,400 positive draws, fewer than
   the roughly 61,000 positive window/text pairings in the full expanded set;
   repeated draws reduce unique coverage further. The `epochs` argument in v2
   currently counts optimizer steps, not dataset passes.
2. **Global competition.** V1 compares each text with windows from every
   training recording. V2 usually selects its verified negatives from the
   positive's recording. This can improve local discrimination without
   teaching a score that ranks recordings against one another. On the 26
   Train-only held-out queries, v2 beats v1 within a recording (.628 vs .568
   nDCG@10) but trails globally (.597 vs .640). Simple per-record z-score and
   percentile normalization made global development recall worse, so a
   post-hoc score fix is not established.
3. **Visual text retention.** V1 includes all L1 captions in the full matrix
   on every step. V2 samples at most 64 caption texts per step and gives the
   caption contrastive term weight .05. Doorway and scene-specific semantics
   may therefore receive much less pressure. This is a hypothesis to test;
   the v2 loss also improves some visual cases.

V1's all-pairs loss wrongly treats unlabelled windows as negative. Its stronger
global retrieval does **not** justify importing that label error into v2.
Instead, train longer on verified pairs and compare each query with verified
false windows from multiple recordings. Keep the complete Train-only
development suite for checkpoint selection, then use frozen Val and
`compose_test` once per chosen recipe.

## Next experiment

Make `optimizer_steps` explicit and run a matched 150/600/1,500-step ablation.
At each step, score a batch of program texts against a larger cross-record
pool of executor-verified positives and negatives, with a ranking loss aimed
at recall in the first 50. Retain meaningful L1 caption exposure. Select on
literal Train-only recall@50, nDCG@10, judged coverage and the worst
recordings; report raw candidate@100 as a separate ceiling. Avoid deciding
from pairwise AUC or the final training loss.

The v2 first-50 pool is now complete. For a fair cross-model ranking comparison,
top up the remaining v1 and PROGRAM first-50 windows as well, then judge a new
pool chosen without reference to any one retriever. Body-camera-only and
sensor-receipt questions require new input
channels. A wider MLP or transformer over the current 23 order-free signal
statistics cannot recover information that those statistics omit.
