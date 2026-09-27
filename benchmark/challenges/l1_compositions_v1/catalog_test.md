# Behavior questions — test

2 active intents. At most three distinct episode–intent judgments each in the initial pass. Leads and contrasts are ungraded hypotheses.

**Test only.** Evaluate after freezing the system. Keep all scene-derived contents out of training and prompt tuning.

<a id="l1x_dream_test_small_turn_close_pass"></a>
## 7. Can it preserve progress through a close pass?

Test whether retrieval distinguishes steady close passing from a larger slowdown in the held-out crowd recording.

`l1x_dream_test_small_turn_close_pass:canonical`

> Find a close oncoming pass where the robot preserves at least 90% of baseline speed using less than 15 degrees heading excursion. Return before, closest-pass and after evidence, including whether the people changed course. Compare with another oncoming encounter in the same recording with at least 30% speed loss; report if no qualifying comparator is confirmed.

**Judge these clauses**

- Close oncoming pass
- Speed >=90% baseline and heading excursion <15 degrees
- Person trajectory change assessed
- Comparator is same-recording oncoming encounter with >=30% speed loss

**Discovery leads:** Rec_Tent_129 292–308 s; Rec_Tent_129 356–368 s.

**Decisive near miss:** Nearest lidar range from food truck; compare following episode with oncoming encounter without flagging mismatch.

**Evidence limit:** Close physical separation and human trajectory may remain uncertain; no claim of safety from speed preservation.

| Initial candidate anchor | Ungraded hypothesis | What to check |
|---|---|---|
| `Rec_Tent_129:0364` | uncertain | Close oncoming pass with steady plotted mean speed; check all numeric and trajectory clauses. |
| `Rec_Tent_129:0304` | near_miss | Large slowdown is reported while following someone; this may fail the oncoming comparator requirement. Do not silently substitute it. |

Expand anchors to the whole encounter, including the baseline and recovery. Deduplicate overlapping windows. For a comparison, seek both required members; an unsuitable suggested comparator is a near miss, not permission to relax the question.

**Budget:** up to three distinct episodes for this intent; stop even if none qualifies. Absence of a confirmed match in this pool is not corpus-wide absence.

**Provenance:** exact label lines, sensor refs, report/session IDs and audit lines are in the corresponding `sources_test.jsonl` record.

**Previously inspected views:** [window_Rec_Tent_129_0364.jpg](../../../labels/.work/renders/l1-Rec_Tent_129-c4/window_Rec_Tent_129_0364.jpg)

<a id="l1x_dream_test_stairs_vs_people"></a>
## 8. Does it respond differently to a crowd and a staircase?

Test whether similar close-range signals can be interpreted in their physical context, with missing-camera limits preserved.

`l1x_dream_test_stairs_vs_people:canonical`

> Compare one crowd pass and one stair climb in Bass_Garage with corridor range below 0.3 m. Show which physical surfaces generate the close returns and how robot speed differs. State what the missing main front camera prevents you from resolving.

**Judge these clauses**

- Crowd episode and stair episode
- Corridor range <0.3 m in each
- Associate returns with people/structure or mark unknown
- Compare speed over equal 4 s spans
- Main-front absence disclosed

**Discovery leads:** Bass_Garage_134 24–48 s; Bass_Garage_134 72–100 s.

**Decisive near miss:** Treat equal range as equal risk; substitute body-camera count for absent main-front detections.

**Evidence limit:** Raw range is not calibrated clearance or safety; body cameras may not disambiguate all reflectors.

| Initial candidate anchor | Ungraded hypothesis | What to check |
|---|---|---|
| `Bass_Garage_134:0036` | uncertain | Crowd pass: verify close-range threshold and which object produces the return. |
| `Bass_Garage_134:0084` | uncertain | Stair slowdown: compare supported range and speed with the crowd episode; two intended comparison members, not a positive/negative pair. |
| `Bass_Garage_134:0100` | near_miss | People above the stairs can make reflector identity ambiguous; reject an overconfident stairs-only attribution. |

Expand anchors to the whole encounter, including the baseline and recovery. Deduplicate overlapping windows. For a comparison, seek both required members; an unsuitable suggested comparator is a near miss, not permission to relax the question.

**Budget:** up to three distinct episodes for this intent; stop even if none qualifies. Absence of a confirmed match in this pool is not corpus-wide absence.

**Provenance:** exact label lines, sensor refs, report/session IDs and audit lines are in the corresponding `sources_test.jsonl` record.

**Previously inspected views:** [window_Bass_Garage_134_0084.jpg](../../../labels/.work/renders/l1-Bass_Garage_134-c1/window_Bass_Garage_134_0084.jpg)
