# A focused benchmark for robot behavior

**Eight questions, reduced from 56: six development intents and two held-out test intents.**
The active JSONL files contain only these eight. Each asks about a behavior decision or an observable interaction
that could inform review and selection of demonstrations. Narrow scene lookups, redundant variants and standalone
sensor diagnostics have been removed from the additions.

| Priority | Question | Why spend a judgment on it? |
|---|---|---|
| 1 · dev | [When does it steer instead of brake?](catalog_dev.md#l1x_dream_steer_or_brake) | Compare response choices under similar initial conditions |
| 2 · dev | [Where does it stay slow after the route clears?](catalog_dev.md#l1x_dream_opportunity_to_go) | Measure recovery delay and inspect remaining constraints |
| 3 · dev | [Can it recover while the person is still nearby?](catalog_dev.md#l1x_dream_recover_person_present) | Distinguish route clearance from disappearance of all people |
| 4 · dev | [Did its turn bring someone into the route?](catalog_dev.md#l1x_dream_turn_creates_exposure) | Separate exposure from ego-motion and a person's crossing |
| 5 · dev | [Where do repeated corrections produce little progress?](catalog_dev.md#l1x_dream_multiple_attempts) | Find sustained navigation difficulty rather than one awkward frame |
| 6 · dev | [Do people change course while it holds its motion?](catalog_dev.md#l1x_dream_person_adjusts) | Observe which participant adjusts during an encounter |
| 7 · test | [Can it preserve progress through a close pass?](catalog_test.md#l1x_dream_test_small_turn_close_pass) | Check steady passing versus slowdown on held-out crowd evidence |
| 8 · test | [Does it respond differently to a crowd and a staircase?](catalog_test.md#l1x_dream_test_stairs_vs_people) | Interpret similar close-range signals in different physical contexts |

## Keep the first judging pass small

Cap the initial pass at **three distinct episodes per intent: 18 dev + 6 test = at most 24 episode–intent
judgments**. This is a work-unit cap, not an estimate of API calls or dollars. A retry, recheck or extra candidate
would consume additional work; this plan does not authorize automatic expansion or invoke any paid judging.

- For a single-episode question, inspect one strong candidate and one specific near miss, retaining one reserve.
- For a comparison question, inspect the two intended comparison members, retaining one reserve or hard negative.
- Select from deduplicated retrieval candidates and cited discovery leads; do not judge every window in the leads.
- Merge overlapping windows into complete encounters. Reuse decoded clips, signals and sensor references across
  intents; each intent still gets its own clause verdict. Prefer the shared crossing/recovery episodes for 2 and 3.
- Stop at the cap even if there is no confirmed positive. Record uncertainty or a failed clause. Do not relax
  thresholds or expand the search for a judgeable positive automatically.

The catalogs give concrete ungraded anchors and what could make each fail. Some of the dream questions have only
plausible discovery leads: neither the required numeric conditions nor the existence of a matched pair is established.
This first pass measures discrimination on a small judged pool. It cannot establish exhaustive recall or absence of
matches across a recording. Run the six dev intents first; use the two test intents only after freezing the system.

## Artifacts

| File | Purpose |
|---|---|
| [catalog_dev.md](catalog_dev.md), [catalog_test.md](catalog_test.md) | Full questions, behavior value, clauses, initial anchors and near misses |
| `requests_dev.jsonl`, `requests_test.jsonl` | Retrieval inputs: IDs, utterance and explicit recording scope only |
| `queries_dev.jsonl`, `queries_test.jsonl` | Author/judge specifications, priority and per-intent judging cap |
| `sources_dev.jsonl`, `sources_test.jsonl` | Provenance for the retained questions: label lines, captions, sensor refs, reports, audit lines and inspected views |
| `attention_dev.json`, `attention_test.json` | Request counts filtered to source segments for the retained questions |
| [regression_notes.md](regression_notes.md) | The user's legs-versus-shadows finding, preserved outside the active queries and judging budget |
| `manifest.json`, `validate.py` | Checksums, protected original query fingerprints, split and budget checks |

Validate from the SCAND root:

```sh
python3 benchmark/challenges/l1_compositions_v1/validate.py
```

Request exports are adapter inputs, not executable oracle programs or a drop-in extension of the fixed `EVAL_SETS`.
Pass only `utterance` and `scope` to retrieval/generation and retain `queryId` for attribution. Do not feed author
captions, discovery spans, proposed roles or evidence refs to the retriever. Unsupported semantics remain explicit;
no gold relevance labels or expected positive statuses have been invented. All eight need independent judging.

The evidence review behind these questions covered 17 headless L1 reports and 397 labeled segments, audited tool
requests and selected saved views. Provenance records identify the relevant subset for each retained query.
The current reduction made no new model calls or judgments and leaves the frozen original benchmark untouched.

## Hold-out and interpretation

Anything derived from `Rec_Tent_129` or `Bass_Garage_134`, including a negative or comparison member, stays in test.
Keep test queries, evidence, thresholds and judgments out of training, generator dev cases and prompt tuning.
All retrieval scopes are explicit. Test questions were authored from held-out scenes; they are a test of a frozen
system on those recordings, not an independently sampled distribution of user questions.

These are human-teleoperated demonstrations. A behavior contrast can inform review and example selection, but
motion alone cannot establish controller intent, social comfort, causation, safety or the best autonomous action.

## Judging and measurement contract

Retrieval uses the full explicit recording scope. Discovery spans guide candidate selection; they do not restrict
retrieval or define gold positives. Hide system identity, rank and proposed candidate role from the judge. Freeze
these operational choices before scoring:


1. **Time and motion.** Seconds are recording-relative. Use capture timestamps for physical timing when reliable,
   receipt timestamps for evidence availability, and preserve both. Measure numeric clauses from original samples,
   with a causal 0.9 s trailing mean for Spot speed and 0.5 s trailing median for Jackal speed, matching the viewed
   plots approximately. Record actual sample support; never bridge a known gap to certify a clause. If filtering
   choices or timestamp uncertainty change a boundary verdict, mark that clause UNKNOWN. Raw samples must also
   support a claimed stop; a filtered dip alone is insufficient.
2. **Baselines and anchors.** Unless specified otherwise, baseline speed is the median filtered speed in the two
   seconds immediately before the visually established encounter/approach onset. A baseline below 0.2 m/s or
   incomplete coverage makes relative-drop/recovery clauses UNKNOWN. Freeze onset independently of whether a
   desired threshold passes. For sustained slowdown onset use a >=0.1 m/s loss from baseline lasting >=0.3 s;
   this locates onset, while each query's stated magnitude still determines qualification. For before/after
   speed comparisons, use two-second neighborhoods outside the encounter; flag censoring.
3. **Heading.** Unwrap heading; robot-left is positive. Excursion is max minus min unwrapped heading over the named
   maneuver; a directed turn uses signed net change over that maneuver. Do not span unrelated turns to meet a
   threshold or treat +/-180-degree wrap as a reversal. Steering reversals need the stated magnitude, not yaw noise.
4. **Geometry.** The nominal corridor is 1.5 m wide and 5 m forward. Camera projection and body clearance are
   approximate. `scandq`'s `min_range_corridor_m`, indexed `min_clearance_front_m`, image box fraction, physical
   pedestrian distance and body clearance are different measurements. Never substitute one silently. Use count
   ranges for edge people, merged clusters and above-scan-plane people. “Clear” needs adequate visual/scan coverage.
5. **Identity and semantics.** Follow the same object across ordered phases. Cross-camera continuity may be UNKNOWN.
   A door, arch, queue, group, parked vehicle or staircase needs the requested visual evidence. A label caption
   alone cannot certify it. Clauses concern observable behavior, not controller intent, comfort, causation or safety.
6. **Results and near misses.** For every candidate, record each clause TRUE/FALSE/UNKNOWN, anchor intervals,
   source-message refs and coverage. A full match requires all clauses; a known failed clause is a near miss,
   and unresolved evidence stays distinct. A graded retrieval score can use 2 for a complete matching result,
   1 for relevant but incomplete episode evidence, and 0 for a contradicted/off-topic result. Keep uncertainty
   separate from that relevance grade.
7. **Episodes and cohorts.** Deduplicate overlapping windows belonging to one episode. Evaluate full ordered chains
   and identity continuity separately from anchor-window recall. A comparison requires all requested members and
   matching tolerances; missing members must be reported. Do not claim an episode complete because one frame is
   relevant. For this bounded first pass, report clause satisfaction, contrast discrimination, anchor error and judged
   coverage. Do not report exhaustive episode recall or corpus-wide no-match conclusions from this small pool.
   Group results by intent/episode rather than counting overlaps independently.

The desired outcome is a collection of inspectable behavior contrasts with honest unknowns. A query that finds no
qualifying pair can still be useful; do not relax its thresholds after inspecting results to manufacture a match.
