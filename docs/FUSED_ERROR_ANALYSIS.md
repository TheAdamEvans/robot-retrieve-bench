# FUSED retrieval error analysis (2026-09-28)

This compares the FUSED v1 leave-one-recording-out index, the expanded v2
uniform index trained for 150 steps without online mining, and the frozen v9
PROGRAM results. The v2 index was selected using Train-only development
queries before this analysis. The benchmark contains 14 Train and two held-out
Val recordings. PROGRAM_LUNA rows are the already recorded v9 run; no program
was regenerated. All retrieval rows use the same 16-recording bundle.

## Measure the first cut

For a reranker, **literal top-50** means the first 50 returned windows. The
published benchmark `R@50` instead removes unjudged windows before counting
50. The figures below are the fraction of *known grade-2 windows* present in
the literal first 50 or raw generator candidates. They are lower bounds on
relevance, not exhaustive recall: the v9 judgment pool included FUSED v1 and
PROGRAM, but not v2. Compose-test judgments are not exhaustive, and several
grade-2 windows overlap the same episode. The five answerable compose-test
intents are the averaging units; battery abstention has no positives.

| Compose-test model | Known hits in first 50 | Known hits in generator | Judged first 10 |
|---|---:|---:|---:|
| FUSED v1 | .173 | .217 | 1.00 |
| Expanded v2 uniform | .091 | .190 | .42 |
| PROGRAM_LUNA | .126 | .511 | .80 |
| PROGRAM_ORACLE | .276 | .683 | 1.00 |

PROGRAM's executor finds a much broader candidate set on these compositions,
but its ordering often leaves relevant windows below rank 50. V2 has fewer
known hits in the first 50 than v1, and its much lower judgment coverage makes
the size of that gap uncertain. The official condensed compose-test nDCG@10 is
.456 for v1 and .440 for expanded v2. Scoring only the literal first ten and
giving unjudged windows no credit yields .456 and .111 respectively; this is a
coverage-sensitive lower bound, not a claim that every unjudged v2 result is
false.

On the five `demo5_para` intent groups, literal first-50 known-hit fractions
are .610 for v1 and .590 for v2. The aggregate conceals a large trade: v2
improves vehicle interaction (.83 vs .67) while losing last-safe-evidence
retrieval (.07 vs .31). Both find all known crowd and doorway windows by rank
50, but v1 places the correct doorway much higher.

## Where the models succeed and fail

| Intent | v1 / v2 / PROGRAM_LUNA known grade-2 hits in first 50 | Interpretation |
|---|---|---|
| Person then speed-up | 30 / 14 / 22 of 114 | V2 loses broad recall; PROGRAM generates 99 known hits before ranking. |
| Tight right-side clearance | 20 / 2 / 3 of 53 | V2's top results favor visually narrow entrances or steps without a verified right-side gap. PROGRAM generates 49 hits but ranks most deep. |
| Turn then brake | 22 / 19 / 37 of 97 | V2 has some signal but not enough first-cut coverage; PROGRAM generates 74. |
| Close car while fast | 0 / 1 / 0 of 10 | V2 alone retrieves a judged Jeep encounter at rank 21; its raw top 100 contains three graded windows from that scene. |
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
v2 retrieves zero. This is a small, overlapping-window sample, but it agrees
with the top-rank generalization problem. Raising raw retrieval depth from 50
to 200 on all five answerable compose-test intents changes mean known-hit
fraction from .173 to .280 for v1 and from .091 to .427 for v2. Thus some v2
evidence is present but ranked deep. A top-200 first cut is 9% of this
2,253-window bundle and is not yet evidence of a scalable production fix.

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

Before making strong claims about v2's held-out quality, add its unique top
results to a blinded judgment pool. Only 42% of its compose-test first ten
are judged, against 100% for v1, because the existing pool predates v2.
Finally, body-camera-only and sensor-receipt questions require new input
channels. A wider MLP or transformer over the current 23 order-free signal
statistics cannot recover information that those statistics omit.
