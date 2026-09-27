# A 15–18 minute walkthrough

Open `fleet-search.html` from disk. A browser at 100% zoom with a wide window works well. All three clips are muted;
play one at a time. The top navigation jumps to the main stops. The expanded method details are available for questions.

1. **Question, 1 minute.** Start with “turns and then brakes.” Ask the audience to notice the numeric constraints and
   temporal relation. This is the search problem the rest of the document measures.
2. **Approaches and FUSED, 2 minutes.** Trace the shared runner. Explain how FUSED transfers some structured signal
   into a fast vector lookup, and why its results remain unverified.
3. **Benchmark, 2 minutes.** Explain the separate measurements. Say explicitly that the judgments are model-produced,
   paraphrases are grouped, and the corpus is small. Do not equate label consistency with human validation.
4. **Results, 3 minutes.** Begin on compositions / nDCG / fresh program. Set the median budget to 100 ms, then 10 s.
   Switch to paraphrases: verification does not improve every slice. Switch to cached generation to separate query
   translation cost from execution. Point out that the oracle is excluded from the budget recommendation.
5. **New recordings, 1 minute.** The chart splits compositions by recording. On the original seven, v9 reproduces the
   earlier checkpoint; on the nine new ones every approach drops, the oracle program included; on the two held-out
   recordings FUSED holds up best. Two held-out recordings are a direction, not a guarantee.
6. **Behavior questions, 2 minutes.** Explain complete ground truth: code sweeps every sample for the numeric
   clauses, the judge settles the rest on every candidate. Point at the two "none in scope" rows: every system still
   answers. Then at recovery: the executable approaches rank well, but full matches are rarely at the top.
7. **Doorway, 1 minute.** Play the two first-result clips. FUSED retrieves a grade-2 window here; that still does not
   independently establish every requested crossing time and physical clearance.
8. **Receipt, 2 minutes.** Play the onset clip and trace the sensor ages. Call attention to the future-stamped transform.
   The receipt constrains recorded availability, not what the robot actually used or what caused its motion.
9. **Limits and next decisions, 2 minutes.** Use the battery and body-camera examples. If time allows, expand the
   turn/brake example: END versus START changes the program and the moment returned, even though both first windows
   are now relevant. The receipt paraphrase that reports none found is the sharper example of ANSWERED ≠ understood.
   Close with the next experiments needed to choose a policy.

## Keep these distinctions explicit

- AUC versus nDCG: different metrics answering different questions.
- PROGRAM_LUNA versus PROGRAM_ORACLE: generated versus supplied programs.
- Verified clauses versus correct interpretation of natural language.
- Fresh-program estimates versus measured cached execution.
- Query-time costs versus startup, indexing, labelling and training.
- Window-level relevance versus distinct events; overlapping windows are correlated.
- Pooled judgments (unjudged is unknown) versus complete ground truth (unjudged in scope is a true negative).
- A latency-budget comparison versus an implemented router or tail-latency guarantee.

The narrative, charts, clips, receipt, full report and downloadable source are embedded in the HTML. No live service
is needed for the walkthrough. The live search UI remains a separate optional demonstration.
