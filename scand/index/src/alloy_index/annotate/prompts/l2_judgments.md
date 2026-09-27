# L2 labelling brief — graded relevance judgments for one intent

You are the benchmark labeller. You grade how relevant each listed window is to one search **intent**. The
windows come from a pool: several retrieval systems' results, plus attribute-based and random windows. You are not
told which system proposed which window, and you must not guess. Grade the data, not the pool.

Job id: `{JOB}`. Intent group: `{INTENT_ID}`.

**Intent:** {INTENT}

**Scope:** {SCOPE}

**Windows to grade** ({N}); `Rec:EEEE` is the 4 s window `[EEEE-4 s, EEEE s]`:

{WINDOWS}

## Tools
These are the same `scandq` tools as L1, run from `/Users/adam.e/robots/scand` with `SCANDQ_JOB={JOB}`:
`signals`, `strip`, `window`, `frame` (and `--crop` to zoom), `step --every`, `sync`, `lidar` (with its cluster list),
`coverage`, `message`, `label put judgment`, `label get judgment`.

Numbers come from code, not from you: speeds, speed drops, turns (heading change), clearances and message ages.
Read them from `signals` (COMPUTED) and cite the odom and lidar refs it returns. You judge what the numbers cannot
say: people, vehicles, doorways, whether something is in the robot's path, and whether an event happened at all.

## Grades
| Grade | Meaning |
|---|---|
| **2** | The window contains, or is within 1 s of, an **anchor** of an episode that satisfies the intent, and every required part of the intent holds over the episode. |
| **1** | Partial. Either the window lies inside a qualifying episode's context without containing its anchor, or everything checkable holds but a required part cannot be established from the data. Say which in the rationale. |
| **0** | Not relevant: some required part is false (including near-misses that satisfy all but one part), or the window is unrelated. |
| **-1** | UNJUDGED: you genuinely cannot tell. This is never counted as 0. Use it sparingly and say why. |

## Method (efficient and consistent)
1. **Find the episodes first.** For each recording in the list, run `signals` over the full span that covers its
   windows, plus `strip` at 1 Hz where useful. Locate candidate episodes and their anchors in time (for example,
   deceleration start, minimum speed, doorway crossing, first visibility, acceleration onset). Confirm the semantic
   parts at native resolution with `frame`, `step`, `sync` or `lidar`.
2. **Then grade every listed window** by its relation to the episodes you found. Adjacent windows overlap (1 s
   stride), so grades should change smoothly around an anchor.
3. **Required evidence.** A grade of 1 or 2 needs at least one native ref. A grade of 0 needs at least one opened ref
   (a `window` or `signals` ref is fine). Each rationale is one sentence naming the episode, e.g. `"slowdown
   1.4→0.5 m/s at 113.9 s; no crowd in the corridor"`.
4. **Submit.** Submit in batches with `label put judgment --json @annotations/tmp/{JOB}/batch_N.json`, each item
   `{"intent_group_id": "{INTENT_ID}", "window_id": ..., "grade": ..., "rationale": ..., "refs": [...]}`. Fix
   and resubmit any rejects, then finish with `label get judgment --intent {INTENT_ID}`.

## Report back (short)
- Episodes found, with times and the evidence behind each.
- Grade counts, and any `-1` with the reason.
- Anything in the intent that the data cannot establish (this feeds `answered_partial` analysis).
