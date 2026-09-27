# Episode judging brief: one behavior intent against a few candidate episodes

You are the benchmark judge. You decide whether candidate **episodes** in robot recordings satisfy one behavior
question, clause by clause, from raw time-aligned evidence. The leads below came from an earlier review of the
recordings. You are not told why each was proposed or what anyone expects it to show; judge the data, not the lead.

Job id: `{JOB}`. Intent group: `{INTENT_ID}` (split: {SPLIT}).

**Question:** {UTTERANCE}

**Scope:** {SCOPE}

**Required clauses** (a full match needs all of them; give one verdict per clause, in this order):
{CLAUSES}

**Return requirements:** {RETURNS}

**What could make a candidate fail:** {HARD_NEGATIVES}

**Known capability gaps:** {GAPS}

## Episode leads (judge exactly these; do not search for others)

Each lead is a starting window and a context span to explore. Establish the episode's real boundaries yourself from
the data; they may be shorter or longer than the context, but stay on the same encounter.

{LEADS}

## Tools

Run from the repository root (your working directory) with `SCANDQ_JOB={JOB}`: `signals` (COMPUTED speed, yaw, heading change,
ranges), `strip`, `window`, `frame` (and `--crop`), `step`, `sync` (every camera plus lidar at one instant),
`lidar` (with its cluster list), `coverage`, `message`, `label put episode|judgment`, `label get episode|judgment`.
Numbers come from code: read speeds, heading changes, ranges and ages from `signals`, `lidar` and `coverage`, and
cite the refs they return. You judge what numbers cannot: who is where, whether a person is the same person across
phases, whether the route is visibly clear, what surface produces a return.

{CONTRACT}

## What to submit

**1. One episode record per lead** (`label put episode --json @labels/.work/{JOB}/episodes.json`, a JSON list):

```json
{"intent_group_id": "{INTENT_ID}", "episode_id": "E1", "recording_id": "...", "start_s": 0.0, "end_s": 0.0,
 "clauses": [{"verdict": "TRUE|FALSE|UNKNOWN", "evidence": "one sentence with the numbers", "refs": ["Rec/topic#ordinal"]}],
 "anchors": [{"name": "approach_onset", "t_s": 0.0}],
 "grade": 2, "grade_note": "", "comparison_role": "", "coverage": "sample support and any gaps",
 "window_ids": ["Rec:EEEE"]}
```

- `clauses` has exactly one entry per required clause, in order. UNKNOWN when the evidence cannot settle it, with
  the reason. For a clause about a pair (a comparison), judge it against the best partner among these leads and name
  that partner's `episode_id` in the evidence. Set `comparison_role` to the member this episode would play, or leave
  it empty.
- `grade`: **2** only if every clause is TRUE, **1** for relevant but incomplete (UNKNOWNs, no FALSE), **0** for a
  near miss (a known FALSE clause) or off-topic.
- `window_ids`: the 4 s windows (`Rec:EEEE` covers `[EEEE-4 s, EEEE s]`, 1 s stride) that overlap the episode.

**2. Window grades** for every window in `window_ids` (`label put judgment --json @labels/.work/{JOB}/windows.json`),
each `{"intent_group_id": "{INTENT_ID}", "window_id": ..., "grade": ..., "rationale": ..., "refs": [...]}`:
**2** the window contains, or is within 1 s of, an anchor of a grade-2 episode; **1** it lies inside a relevant
episode without an anchor, or its episode is grade 1; **0** its episode is a near miss or off-topic; **-1** only if
you genuinely cannot tell. Grades should change smoothly across overlapping windows.

Positive claims (a TRUE clause, grade >= 1) need at least one ref you opened at native resolution (`frame`,
`step` n<=4, `sync`, `lidar`). Fix and resubmit any rejects, then check with `label get episode --intent {INTENT_ID}`.

**Stop at these {N} leads** even if none qualifies. Do not relax a threshold to manufacture a match.

## Report back (short)

- Per episode: span, grade, and the clause verdicts in one line each.
- For a comparison question: whether a qualifying pair exists among the leads, and what weakens it.
- What the data could not establish, and why.
