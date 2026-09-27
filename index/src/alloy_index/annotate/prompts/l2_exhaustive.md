# Exhaustive episode judging: every episode in these spans, for one behavior question

You are the benchmark judge building **complete ground truth** for one behavior question. Code has already swept
the whole scope and removed every stretch that cannot satisfy the question's numeric clauses. What remains is split
into the chunks below. Your job: find **every** episode inside each chunk that could satisfy the question, judge it
clause by clause, and leave nothing out. Recall will be measured against what you record, so a missed qualifying
episode is the worst error; an extra near miss costs little.

Job id: `{JOB}`. Intent group: `{INTENT_ID}` (split: {SPLIT}).

**Question:** {UTTERANCE}

**Required clauses** (a full match needs all of them; give one verdict per clause, in this order):
{CLAUSES}

**Return requirements:** {RETURNS}

**What could make a candidate fail:** {HARD_NEGATIVES}

**Known capability gaps:** {GAPS}

**Numeric screen already applied** (anything outside the chunks provably fails at least one of these; they are
deliberately looser than the clauses):
{SCREEN}

## Chunks to cover completely

{CHUNKS}

## Method

1. For each chunk, run `signals` over the whole chunk first, then `strip` at 1–2 Hz across it. List every moment
   where the numeric clauses could hold: every slowdown, turn or steady stretch that matters for this question.
2. For each such moment, check the semantic clauses at native resolution (`frame`, `step`, `sync`, `lidar`) until
   one clause is clearly FALSE (a near miss: stop there) or all are settled.
3. Record **one episode per distinct candidate moment** that passes the numeric screen. If a chunk contains no
   such moment after all, record its single closest stretch as an episode with the failing clause FALSE, so every
   chunk has at least one record.

{CONTRACT}

## Tools

Run from the repository root (your working directory) with `SCANDQ_JOB={JOB}`: `signals`, `strip`, `window`, `frame` (`--crop`),
`step`, `sync`, `lidar`, `coverage`, `message`, `label put episode|judgment`, `label get episode|judgment`. Read the
numbers from `signals` and cite its refs; you judge who is where, identity across phases, and visual clearance.

## What to submit

**Episodes** (`label put episode --json @annotations/tmp/{JOB}/episodes.json`, a JSON list), one per candidate moment:

```json
{"intent_group_id": "{INTENT_ID}", "chunk_id": "K1", "episode_id": "K1-a", "recording_id": "...",
 "start_s": 0.0, "end_s": 0.0,
 "clauses": [{"verdict": "TRUE|FALSE|UNKNOWN", "evidence": "one sentence with the numbers", "refs": ["Rec/topic#ordinal"]}],
 "anchors": [{"name": "...", "t_s": 0.0}], "grade": 0, "grade_note": "", "comparison_role": "",
 "coverage": "sample support and any gaps", "window_ids": ["Rec:EEEE"]}
```

- Give exactly one clause entry per required clause, in order. For a near miss, the failing clause is FALSE with
  evidence. Clauses you did not need to assess are UNKNOWN with the evidence `"not assessed: clause N is FALSE"`.
- For a comparison question, judge each episode as a candidate member. Set `comparison_role` to the member it could
  play. For a pair-level clause, name the best partner episode among all chunks in the evidence, or say none exists.
- `grade`: **2** only if every clause is TRUE; **1** relevant but incomplete (UNKNOWNs, no FALSE); **0** near miss.
- `window_ids`: the 4 s windows (`Rec:EEEE` covers `[EEEE-4 s, EEEE s]`) overlapping the episode.

**Window grades**, only for windows of grade 1 and 2 episodes (`label put judgment --json @annotations/tmp/{JOB}/windows.json`):
each `{"intent_group_id": "{INTENT_ID}", "window_id": ..., "grade": ..., "rationale": ..., "refs": [...]}`, where
2 = contains or is within 1 s of an anchor of a grade-2 episode, and 1 = inside a relevant episode.

Positive claims (a TRUE clause, grade >= 1) need a ref you opened at native resolution. Fix and resubmit rejects,
then check with `label get episode --intent {INTENT_ID}`. Never relax a threshold to manufacture a match.

## Report back (short)

- Per chunk: the candidate moments found and their grades, one line each.
- The complete list of grade 2 and grade 1 episodes.
- For a comparison question: every qualifying pair, or why none exists.
