<p class="eyebrow">FLEET SEARCH / EXPERIMENT 01 / SEPTEMBER 2026</p>

# When search needs evidence.

<p class="dek">Embeddings, executable queries, and the price of knowing why a result belongs.</p>

A robot log makes a familiar search problem unusually concrete. A frame can look right while the event is wrong. A valid program can check the wrong interpretation. And sometimes the data cannot answer the question at all.

This experiment puts those possibilities through **one execution path and one benchmark**. The result is a way to compare retrieval quality, latency, model usage, and the strength of the evidence behind an answer.

<!-- component: hero -->

<!-- component: stats -->

<p class="reading-guide">A 15–18 minute walkthrough. The charts and clips work offline. Expand the evidence and method notes when you want to inspect a claim.</p>

## 01 / Start with a question that crosses modalities

> “Find where the robot turns and then brakes: a turn of at least 45 degrees followed within five seconds by a drop of 30% or more in speed.”

There are several requirements hiding in one sentence: a change in heading, a change in speed, a relative magnitude, and a temporal relationship. A visually similar frame addresses only part of that request.

The recordings contain front-camera images, lidar, odometry, transforms, and—in Spot’s case—additional body cameras. The challenge is to connect language to those signals without silently weakening the question.

<!-- component: corpus -->

The local corpus is a sixteen-recording subset of [SCAND, the Socially CompliAnt Navigation Dataset](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html), collected with Spot and Jackal: 2,309 seconds and 21 GB of raw logs. Two recordings, SCAND’s own validation split, are **held out**: nothing is trained or tuned on them. The source dataset consists of human-teleoperated navigation demonstrations. These are archived recordings; this experiment searches them retrospectively.

**The question we are testing:** how much can a fast vector lookup recover, and when does explicit execution justify its extra cost?

## 02 / Six approaches, one execution path

The serving code and the evaluation harness call the same pipeline runner. Each approach is a composition of candidate generation and ranking stages. This keeps differences in retrieval strategy visible while sharing execution, instrumentation, and answer semantics.

~~~mermaid
flowchart LR
  Q["Natural-language question"] --> E["EMBED\nFrozen image/text embeddings"]
  Q --> F["FUSED\nImage + measured signals"]
  Q --> P["PROGRAM\nGenerate an executable query"]
  E --> R["Rank windows"]
  F --> R
  E --> V["Verify clauses\nOptional for embedding candidates"]
  F --> V
  P --> X["Execute across the full scope"]
  X --> V
  R --> A["Results + status + cost"]
  V --> A
  classDef visual fill:#e6eff6,stroke:#3475a5,color:#243b4d;
  classDef learned fill:#e2f1eb,stroke:#087f72,color:#17473d;
  classDef program fill:#fbeadf,stroke:#c45b36,color:#613823;
  class E visual;
  class F learned;
  class P,X,V program;
~~~

| Approach | What it does | What the answer can establish |
|---|---|---|
| **TAGS** | BM25 over recording-level metadata | A matching recording, with no knowledge of when an event happens |
| **EMBED** | A text vector against mean-pooled camera window vectors | Visual relevance; results remain unverified |
| **FUSED** | A learned vector combining images and signal summaries | Relevance informed by motion and scene features; still unverified |
| **PROGRAM** | Language → typed query → execution over the full scope | Whether the expressed clauses hold under the indexed features |
| **HYBRID** | EMBED candidates followed by verification | Verified or partial matches within the retrieved candidates |
| **FUSED_V** | FUSED candidates followed by verification | The same verification, with a different candidate source |

The generated program has a strict schema, semantic validation, and one repair opportunity. The deterministic executor evaluates clauses using **TRUE, FALSE, and UNKNOWN**. Only a definitively false required clause filters a candidate.

<details><summary>Why separate candidate generation from verification?</summary>

A verifier can reject a false match. It cannot recover an event that the candidate generator never retrieved. The benchmark therefore measures candidate recall as well as final ranking quality. PROGRAM can execute across the full recording scope; the two hybrids begin with a limited candidate set.

A stage report records candidate counts, filtering reasons, latency, bytes read, and model usage. The runner checks stage invariants. Stable message identifiers allow a result to point back to the original sensor payload.

</details>

## 03 / Teach a fast retriever some of the structure

FUSED keeps the query side simple: the frozen SigLIP2 text encoder followed by a dot product. The work happens offline, where a small residual head learns to combine the image vector with 23 signal summaries and their availability masks.

~~~mermaid
flowchart LR
  I["Frozen image vector"] --> H["Residual head\nStarts at the image baseline"]
  S["23 signal summaries\n+ availability masks"] --> H
  H --> W["One vector per window"]
  T["Frozen text encoder"] --> D["Dot product"]
  W --> D
  G["Executed grammar programs\n+ L1 captions and attributes"] -. training supervision .-> H
  classDef learned fill:#e2f1eb,stroke:#087f72,color:#17473d;
  class H,W learned;
~~~

The training supervision comes from executed grammar programs and separately collected L1 captions and attributes. **L2 relevance grades are reserved for evaluation.** Every window of a training recording is embedded by a leave-one-recording-out model that never saw that recording; the two held-out recordings are embedded by a model trained on all fourteen others. The pseudo-program generator excludes the composition pairings designated for `compose_test`.

This gives a concrete mechanism for transferring some structured knowledge into a cheap retrieval path. It also preserves clear limitations: signal summaries and mean pooling lose temporal detail, provider mistakes affect pseudo-labels, and vectors do not supply clause-level verification.

<details><summary>Training controls and what the holdout means</summary>

The model uses a multi-positive sigmoid loss. A linear head and an image-plus-signal concatenation baseline are included in the full results, alongside the residual model. Out-of-fold pseudo-label AUC rises from 0.51 for plain embeddings to 0.70 for FUSED; the busiest “hub” window appears in 15 top-five lists, down from 20 before the new data.

Leave-one-recording-out prevents each window’s model from seeing that recording during training. The two SCAND validation recordings go further: they were never used for training, prompt tuning, or development cases. The benchmark has informed other development decisions, so the remaining results should be read as a development benchmark.

</details>

## 04 / Make the benchmark explain the tradeoff

A single quality score would hide several distinct failure modes. We measure ranking, coverage, answer status, and cost separately, and keep the assumptions visible.

~~~mermaid
flowchart LR
  C["Every configuration\nCanonical + paraphrased questions"] --> P["Pool candidate windows"]
  L["L1 attribute candidates\n+ sampled negatives"] --> P
  P --> J["L2 relevance judgments\nAudited sensor views"]
  S["Numeric sweep of the whole scope"] --> X["Exhaustive episode judging\nBehavior questions"]
  J --> R["Ranking + coverage\nStatus + cost"]
  X --> R
  B["Blind re-checks"] --> R
  classDef measured fill:#f1eddf,stroke:#ad8640,color:#534728;
  class J,X,R measured;
~~~

<!-- component: measures -->

**Reading the scores.** nDCG@10 measures how well graded matches concentrate near the top of the ranking; 1 is an ideal ordering. ROC-AUC measures ordering of relevant versus non-relevant windows in the judged set; 0.5 is chance. R@50 measures the fraction of known relevant windows recovered. The explorer switches between nDCG and AUC; the full results also include recall.

There are **32 evaluation utterances** under **10 configurations**: five original demo questions, fifteen paraphrases, six composition/control questions, and six behavior questions written by reviewing the new recordings. The labels include **2,344 window-level relevance judgments** by the Opus judge, every window any configuration surfaced for these questions included, plus complete episode-level ground truth for the behavior questions. The frozen run contains 308 search runs.

The unit of relevance is a four-second window. PROGRAM’s interval results expand into nearby anchor windows for comparison. This is a window-level benchmark; overlapping windows are not independent events.

<details><summary>Judging protocol, uncertainty, and incomplete coverage</summary>

The Opus labeller uses audited tools to open frames and other sensor views. Positive judgments must cite evidence opened at native resolution. Grades are 0 (not relevant), 1 (partial), and 2 (relevant). A fresh labeller instance rechecked 80 windows without seeing the previous grades: 83.8% exact agreement, 97.5% within one grade, and 90% agreement on relevant versus not relevant; graded Cohen’s κ is 0.65. For the behavior questions, two independent judging passes agree on relevant versus not for 16 of 18 episodes. Both passes use the same model family, so this measures consistency, with shared blind spots still possible.

The main nDCG and recall metrics use condensed lists: unjudged windows are removed. A separate nDCG sensitivity column treats them as zero. Recall is against known positives in the pool. Judged@10 and judged@50 report how much of the actual returned prefix has been graded. The behavior questions are different: their ground truth is complete, so an unjudged window in scope is a true negative.

Confidence intervals bootstrap intent groups, so paraphrases do not count as independent samples. Read small differences as directional evidence.

</details>

## 05 / Change the question, change the best point on the curve

**On the composition slice**, FUSED improves nDCG@10 from 0.22 for plain embeddings to 0.46 at roughly 30 ms median query latency. Generated PROGRAM reaches 0.38; FUSED plus verification reaches 0.47, with an estimated fresh-program median latency around 4.6 seconds.

**On the paraphrase slice**, FUSED reaches 0.47 nDCG@10 (ROC-AUC 0.73), compared with 0.25 for plain embeddings and 0.24 for generated PROGRAM. Adding verification to FUSED lowers that slice’s nDCG to 0.32. Verification is a capability whose benefit depends on the question, candidate set, and program quality.

<!-- component: explorer -->

The oracle row uses a supplied reference program. It isolates part of the translation problem; it is not a result obtained by generating a program from the question. On the original seven recordings its composition nDCG is 0.74; across all sixteen it is 0.45, for reasons the next section shows.

<details><summary>How to read latency, tokens, and bytes</summary>

The “fresh program” latency is measured generation time plus execution. The cached-program mode shows execution latency alone; a validated cached program costs no tokens. The generator prompt went through a measured hill-climb: its third version raised status accuracy on the paraphrases from 0.73 to 0.93 and, on behavior questions it was never tuned on, ROC-AUC from about 0.48 to 0.89.

Embedding approaches still run a text encoder. Their zero LLM-token figure does not mean zero computation. Model loading and sidecar loading are startup work; provider extraction, image encoding, training, and labelling are offline work. Clip rendering and playback are separate from search latency.

The runtime loads roughly 42 MB of window indexes for a 21 GB raw corpus. That is counted startup I/O, not a measurement of total resident memory; model weights are additional. Per-query raw-byte accounting measures incremental reads, such as evidence checks, and the per-sensor storage layout keeps those reads small.

</details>

## 06 / New recordings are harder, for everything

The corpus grew from seven recordings to sixteen. Splitting the composition questions by where each judged window comes from shows what that did:

<!-- component: generalization -->

**On the original seven recordings, the results reproduce the earlier checkpoint** (oracle program 0.74 nDCG@10), and FUSED improves from 0.45 to 0.59 with twice the training data. **On the nine new recordings, every approach drops**, including the supplied oracle program (0.38). The new scenes—stadium crowds, tailgates, tents—expose where the structured signals and detectors transfer imperfectly.

**On the two held-out recordings, FUSED holds up best** (0.54 nDCG@10, ROC-AUC 0.75), embedded by a model that never saw them. Two recordings are a small sample; this is a direction, not a generalization guarantee.

## 07 / Ask the questions a roboticist would ask

Six behavior questions were written by reviewing the new recordings: *When does it steer instead of brake? Where does it stay slow after the route clears? Can it recover while the person is still nearby?* Each has numeric clauses and semantic ones.

For these questions the ground truth is **complete** over each question’s scope. Code sweeps every sample for spans that could pass the numeric clauses (with deliberately loosened thresholds), and the judge then settles the semantic clauses on every candidate. That supports statements pooled judging cannot make: how many qualifying episodes exist, and how many each system finds.

<!-- component: dream -->

Two of the six have **no qualifying episode at all** in their scope—one proven by the numbers alone, one after judging every candidate. Every approach still returns results for them, marked unverified or partial; none says “none found”. That failure is invisible without complete ground truth. Where episodes do exist, the executable approaches rank relevant windows well (ROC-AUC 0.87–0.90 against 0.72 for FUSED and 0.41 for plain embeddings), but full matches are rarely at the top.

## 08 / Inspect the results

> “Tightest doorway traversal on the library route — give the crossing times and the left and right gaps.”

These are the actual first windows returned by EMBED and FUSED for `doorway_crossing:para3`. Both remain **unverified**. The grade beside each clip is the separate L2 relevance judgment.

<!-- component: doorway -->

FUSED’s signal-aware representation retrieves a graded doorway event here, and the generated PROGRAM’s first window is also graded relevant, with its clauses checked. Locating a useful event is valuable; exact crossing times and physical clearances still require appropriate evidence and interpretation.

<details><summary>Same question, two readings of “then”</summary>

For the turn-then-brake question from the opening, the generated program checks braking after the turn’s **END**; the reference program checks after its **START**. Both pass structural validation, and both now return a relevant first window—but at different moments of different turns, because the event boundary differs. Schema validity alone cannot say which reading the user meant.

<!-- component: program-diff -->

</details>

## 09 / An answer can include an evidence receipt

> “Immediately before the robot accelerates from a stop, return the latest available front image, side/rear images, lidar scan, pose, and transform that could have informed the action. Report each message’s age.”

A similarity score cannot express this temporal availability requirement. The generated program finds an onset, and the receipt selects sensor messages under a strict-before boundary. Each message retains its identity and a payload hash. The evaluated program is replayed here on the Library-to-MLK recording.

<!-- component: receipt -->

The front image is about **35 ms** old; the body-camera images are **490–630 ms** old. The transform is stamped a few milliseconds after the cutoff and is marked **UNKNOWN / FUTURE_MEASUREMENT**. The system exposes that inconsistency instead of silently treating every sensor reading as usable.

The onset is found retrospectively. The receipt establishes which recorded messages satisfy the cutoff rules; it does not establish which messages the robot’s controller actually consumed, or that they caused the action. This is an auditable temporal boundary over archived evidence.

<details><summary>Inspect the exact receipt</summary>

<!-- component: receipt-json -->

The program’s generated maximum-age threshold is 1,000 seconds, which is visible in this receipt and deserves scrutiny. All the valid messages shown here happen to be much fresher. An explicit program makes this assumption inspectable; it does not make the assumption correct by construction.

</details>

## 10 / Make uncertainty part of the result

The battery question asks for a signal that is not available in this indexed corpus. Generated PROGRAM returns **insufficient evidence**. FUSED still returns ranked windows, with an **unverified** status. Those outcomes need different treatment in an interface and in an evaluation.

The body-camera question reveals a different limit: the recordings contain those cameras, but the decisive body-camera person feature is not indexed. FUSED plus verification returns **partial** results and identifies the missing clause; generated PROGRAM abstains. “Recorded,” “indexed,” and “supported by the current query language” are separate conditions. One held-out Spot recording has no front camera at all; clauses that need it come back **UNKNOWN (sensor absent in this log)**, never false.

<!-- component: uncertainty -->

Three-valued logic protects a useful distinction: a false clause can rule out a candidate; an unknown clause cannot. Likewise, silence from a learned detector is not sufficient evidence of absence. Exhaustive no-match claims require appropriate coverage and features that support that conclusion.

**ANSWERED means the expressed clauses passed the executor’s checks.** It is not a guarantee that the natural-language intent was perfectly translated. One paraphrase of the receipt question shows it: the generated program adds a standstill requirement nothing satisfies and reports none found—exhaustive for the program, wrong for the question.

## 11 / What should a system choose?

The results suggest a policy to test, rather than one universal winner.

| Need | Plausible starting point | What to check next |
|---|---|---|
| Fast exploratory search | FUSED | Coverage on new scenes, recurring irrelevant “hub” windows, and presentation of unverified results |
| Numeric or temporal constraints | PROGRAM, or FUSED followed by verification | Translation quality, event boundaries, provider accuracy, and candidate recall |
| A defensible evidence trail | An executable query with explicit receipts | Time semantics, missing inputs, measurement uncertainty, and unsupported clauses |
| Repeated questions | Reuse a validated cached program | Invalidate when the prompt, schema, registry, or relevant data assumptions change |

A router or verify-on-demand interaction would make these choices explicit. That is proposed future work. Today, the benchmark measures the constituent approaches and the tradeoffs a policy would need to manage.

The transferable idea is to give **relevance, constraint satisfaction, uncertainty, and cost their own measurements**. In another domain, the signals might be timestamps, product attributes, permissions, or numeric facts. Whether the same architecture helps there is a hypothesis to test with that domain’s own evidence and labels.

## 12 / What is established, and what earns the next claim?

The working system searches sixteen real recordings through one runner, indexes new bags with one command, returns playable source clips, stage-level costs, and inspectable evidence, and passes 42 contract, regression, and module-evaluation tests. Frozen checkpoints and their label sets are preserved for comparison.

The immediate research priorities follow from the observed gaps:

1. **Say “none” when nothing qualifies.** Complete ground truth shows every approach returning results for questions with no answer in scope.
2. **Close the transfer gap on new scenes.** Detectors and signal providers lose accuracy on crowds and new environments; measure them per scene type.
3. **Improve translation and event semantics.** Track clause fidelity and concrete regression cases, including START-versus-END interpretations.
4. **Improve candidate coverage.** Verification cannot rescue events it never sees. Evaluate temporal encoders and difficult clause-level negatives against the same baseline.
5. **Test a decision policy.** Measure whether selective verification gives useful quality and evidence improvements within an explicit latency budget.

Before stronger claims, we need more independent evidence: the corpus is still small, judgments come from a model, the same family supplies some captions and judging, feature quality is uneven, and the benchmark has guided iteration. These are measurable next steps, with the current checkpoint as a stable comparison.

<p class="closing">The useful artifact is the comparison: a fast answer, a checked answer, and enough evidence to decide which one the question deserves.</p>

<details><summary>Artifact provenance, reproducibility, and full results</summary>

This document is generated from `benchmark/results/v9/`. Media comes from the local SCAND recordings; the receipt replays v9’s stored generated program on the Library-to-MLK recording. The behavior-question ground truth is in `annotations/exhaustive/` and `annotations/labels/episode/`. Frozen artifacts have SHA-256 manifests.

The HTML contains its diagrams, plots, photographs, three short clips, and benchmark data. Nothing needs a running server or a network connection to present it. The Markdown source and build scripts are included alongside it for editing and reproduction.

Dataset attribution: Haresh Karnan and colleagues, *Socially CompliAnt Navigation Dataset (SCAND)*, 2022. [Project and paper](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html) · [Dataset DOI](https://doi.org/10.18738/T8/0PRYRH). Clips and frames shown here are excerpts of that dataset.

<!-- component: downloads -->

<!-- component: full-results -->

</details>
