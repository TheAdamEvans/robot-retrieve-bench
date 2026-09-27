# Behavior questions — dev

6 active intents. At most three distinct episode–intent judgments each in the initial pass. Leads and contrasts are ungraded hypotheses.

<a id="l1x_dream_steer_or_brake"></a>
## 1. When does it steer instead of brake?

Compare response choices under similar encounter conditions; identify examples worth reviewing as behavior targets.

`l1x_dream_steer_or_brake:canonical`

> Find two oncoming-pedestrian encounters with baseline speeds within 0.3 m/s and the same corridor-count band: one with at least 30 degrees heading excursion and less than 20% speed loss, the other with at least 30% speed loss and less than 15 degrees heading excursion. Show approach and separation, report closest supported separation, and flag differences in terrain or other actors that weaken the comparison.

**Judge these clauses**

- Oncoming pedestrian in both episodes
- Baseline speeds within 0.3 m/s and same corridor-count band
- Steering case: >=30-degree excursion and <20% speed loss
- Slowing case: >=30% speed loss and <15-degree excursion
- Physical separation only if object association supports it

**Discovery leads:** Library_Jester_75 8–24 s; Stadium_Sanjac_90 52–80 s; Thompson_Thompson_100 244–260 s.

**Decisive near miss:** Crowd context differs materially; a turn was required by walkway geometry; nearest wall range used as pedestrian distance.

**Evidence limit:** May find no adequately matched pair; observations do not identify a better policy.

| Initial candidate anchor | Ungraded hypothesis | What to check |
|---|---|---|
| `Stadium_Sanjac_90:0056` | uncertain | Oncoming crowd with weaving: candidate steering response; speed and heading thresholds need checking. |
| `Library_Jester_75:0016` | uncertain | Oncoming pedestrian plus walked bicycle: candidate slowdown response; mixed actors may invalidate comparability. |
| `Thompson_Thompson_100:0256` | near_miss | Turn beside a bending person looks similar but may fail the oncoming-encounter requirement. |

Expand anchors to the whole encounter, including the baseline and recovery. Deduplicate overlapping windows. For a comparison, seek both required members; an unsuitable suggested comparator is a near miss, not permission to relax the question.

**Budget:** up to three distinct episodes for this intent; stop even if none qualifies. Absence of a confirmed match in this pool is not corpus-wide absence.

**Provenance:** exact label lines, sensor refs, report/session IDs and audit lines are in the corresponding `sources_dev.jsonl` record.

**Previously inspected views:** [window_Thompson_Thompson_100_0256.jpg](../../../labels/.work/renders/l1-Thompson_Thompson_100-c3/window_Thompson_Thompson_100_0256.jpg)

<a id="l1x_dream_opportunity_to_go"></a>
## 2. Where does it stay slow after the route clears?

Locate clearance-to-recovery delays and visible remaining constraints before deciding whether recovery behavior deserves changes.

`l1x_dream_opportunity_to_go:canonical`

> Find a pedestrian crossing after which the route is visibly clear for at least two seconds but robot speed remains below half its pre-encounter baseline. Return the clearance-to-recovery delay and any remaining visible constraint that could explain it.

**Judge these clauses**

- Pedestrian crossing localized
- Corridor clear >=2 s afterward with sufficient coverage
- Speed remains <50% baseline during those 2 s
- Search side views, terrain and next maneuver for remaining constraints

**Discovery leads:** Library_Jester_75 12–24 s; Stadium_Sanjac_90 44–56 s; GDC_Library_42 292–308 s.

**Decisive near miss:** Front clear but turn/steps ahead; camera dropout called open route; low baseline makes ratio unstable.

**Evidence limit:** A delay is an observation, not proof of needless hesitation or controller failure.

| Initial candidate anchor | Ungraded hypothesis | What to check |
|---|---|---|
| `Stadium_Sanjac_90:0052` | uncertain | Crossing and restart: candidate delay episode; a matching two-second delay is not established. |
| `GDC_Library_42:0304` | near_miss | Reported recovery after a crossing: useful contrast if speed recovers too promptly to satisfy the delay clause. |

Expand anchors to the whole encounter, including the baseline and recovery. Deduplicate overlapping windows. For a comparison, seek both required members; an unsuitable suggested comparator is a near miss, not permission to relax the question.

**Budget:** up to three distinct episodes for this intent; stop even if none qualifies. Absence of a confirmed match in this pool is not corpus-wide absence.

**Provenance:** exact label lines, sensor refs, report/session IDs and audit lines are in the corresponding `sources_dev.jsonl` record.

**Previously inspected views:** [window_Stadium_Sanjac_90_0048.jpg](../../../labels/.work/renders/l1-Stadium_Sanjac_90-c1/window_Stadium_Sanjac_90_0048.jpg)

<a id="l1x_dream_recover_person_present"></a>
## 3. Can it recover while the person is still nearby?

Distinguish appropriate route-based recovery from waiting until all people disappear from view.

`l1x_dream_recover_person_present:canonical`

> Find a speed loss of at least 30% followed by recovery to 90% of baseline, where the same pedestrian has left the travel corridor but remains visible beside the robot at recovery. Return corridor exit, recovery and last supported visibility times separately.

**Judge these clauses**

- Same person associated across episode
- Slowdown >=30%
- Corridor exit before recovery >=90%
- Person still visible beside robot at recovery

**Discovery leads:** Library_Jester_75 12–24 s; GDC_Library_42 292–308 s; Stadium_Sanjac_90 72–88 s.

**Decisive near miss:** Unrelated bystander supplies persistence; front disappearance treated as every-camera disappearance.

**Evidence limit:** Person identity and all-camera visibility require raw review; do not infer disappearance outside coverage.

| Initial candidate anchor | Ungraded hypothesis | What to check |
|---|---|---|
| `Library_Jester_75:0020` | uncertain | Recovery after a bicycle clears; confirm the same pedestrian remains visible and the required speed recovery. |
| `Stadium_Sanjac_90:0084` | near_miss | Nearby standing people could falsely provide persistence; require the same person from the slowdown. |

Expand anchors to the whole encounter, including the baseline and recovery. Deduplicate overlapping windows. For a comparison, seek both required members; an unsuitable suggested comparator is a near miss, not permission to relax the question.

**Budget:** up to three distinct episodes for this intent; stop even if none qualifies. Absence of a confirmed match in this pool is not corpus-wide absence.

**Provenance:** exact label lines, sensor refs, report/session IDs and audit lines are in the corresponding `sources_dev.jsonl` record.

<a id="l1x_dream_turn_creates_exposure"></a>
## 4. Did its turn bring someone into the route?

Inspect whether a maneuver changes exposure to a person already present, and whether slowing precedes or follows that change.

`l1x_dream_turn_creates_exposure:canonical`

> Find a turn of at least 45 degrees that brings a previously side-visible pedestrian into the forward travel corridor without that person crossing the robot path. Measure whether slowing begins before or after this change of relative geometry.

**Judge these clauses**

- Turn >=45 degrees
- Same pedestrian visible at side before turn
- Corridor occupancy changes because robot reorients, not pedestrian crossing
- Compare slowdown onset with geometric transition

**Discovery leads:** Parlin_Parlin_51 36–48 s; Library_Jester_75 52–64 s; GDC_Library_42 24–36 s.

**Decisive near miss:** Treat every newly front-visible person as a new arrival; track ID reset mistaken for appearance.

**Evidence limit:** Requires ego-motion-compensated person geometry and cross-camera association.

| Initial candidate anchor | Ungraded hypothesis | What to check |
|---|---|---|
| `Parlin_Parlin_51:0044` | uncertain | Robot turns toward people beside the landing; verify identity, corridor geometry and signed turn. |
| `GDC_Library_42:0032` | near_miss | A pedestrian actually crosses from the stairs; new front visibility alone does not establish exposure caused by rotation. |

Expand anchors to the whole encounter, including the baseline and recovery. Deduplicate overlapping windows. For a comparison, seek both required members; an unsuitable suggested comparator is a near miss, not permission to relax the question.

**Budget:** up to three distinct episodes for this intent; stop even if none qualifies. Absence of a confirmed match in this pool is not corpus-wide absence.

**Provenance:** exact label lines, sensor refs, report/session IDs and audit lines are in the corresponding `sources_dev.jsonl` record.

**Previously inspected views:** [window_GDC_Library_42_0032.jpg](../../../labels/.work/renders/l1-GDC_Library_42-c1/window_GDC_Library_42_0032.jpg), [window_Parlin_Parlin_51_0040.jpg](../../../labels/.work/renders/l1-Parlin_Parlin_51-c1/window_Parlin_Parlin_51_0040.jpg)

<a id="l1x_dream_multiple_attempts"></a>
## 5. Where do repeated corrections produce little progress?

Find episodes worth reviewing for persistent navigation difficulty rather than isolated turns or gait-related speed dips.

`l1x_dream_multiple_attempts:canonical`

> Find an eight-second crowded passage with at least two left-right steering reversals of 15 degrees or more, two speed dips of at least 0.2 m/s, and less than two metres net displacement. Return the sequence and show whether a stable opening remained available or the crowd kept changing.

**Judge these clauses**

- 8 s crowded passage
- At least two signed steering reversals of >=15 degrees each
- At least two speed dips >=0.2 m/s
- Net planar displacement <2 m from raw pose
- Available opening tracked over time

**Discovery leads:** Stadium_Sanjac_90 60–80 s; Thompson_Thompson_100 220–260 s; Library_Jester_75 48–64 s.

**Decisive near miss:** Gait/yaw noise counted as indecision; path length substituted for net progress; assume a visible opening is traversable.

**Evidence limit:** Net displacement and opening persistence require raw pose/visual review beyond current feature registry.

| Initial candidate anchor | Ungraded hypothesis | What to check |
|---|---|---|
| `Stadium_Sanjac_90:0068` | uncertain | Slow left-right motion in a crowd; verify both reversals, speed dips and net displacement over eight seconds. |
| `Library_Jester_75:0056` | near_miss | One substantial lobby turn and slowdown can look difficult without meeting repeated-correction and low-progress clauses. |

Expand anchors to the whole encounter, including the baseline and recovery. Deduplicate overlapping windows. For a comparison, seek both required members; an unsuitable suggested comparator is a near miss, not permission to relax the question.

**Budget:** up to three distinct episodes for this intent; stop even if none qualifies. Absence of a confirmed match in this pool is not corpus-wide absence.

**Provenance:** exact label lines, sensor refs, report/session IDs and audit lines are in the corresponding `sources_dev.jsonl` record.

**Previously inspected views:** [window_Thompson_Thompson_100_0256.jpg](../../../labels/.work/renders/l1-Thompson_Thompson_100-c3/window_Thompson_Thompson_100_0256.jpg)

<a id="l1x_dream_person_adjusts"></a>
## 6. Do people change course while it holds its motion?

Observe how much of the encounter adjustment comes from people, without inferring their intent or comfort.

`l1x_dream_person_adjusts:canonical`

> Find an oncoming encounter where the pedestrian visibly changes course before passing while robot speed changes by less than 0.2 m/s and heading by less than 15 degrees. Return the pedestrian trajectory change and robot signals; do not infer discomfort or intent from the maneuver.

**Judge these clauses**

- Oncoming person changes world-relative course
- Robot speed range <0.2 m/s
- Robot heading excursion <15 degrees
- Person adjustment precedes passing

**Discovery leads:** Library_Fountain_5 0–40 s; GDC_Library_42 284–308 s; Sanjac_Rec_91 0–40 s.

**Decisive near miss:** Apparent lateral motion from camera rotation; person simply follows curved walkway; label discomfort from posture.

**Evidence limit:** Human trajectory needs ego-motion compensation; social intent not observable.

| Initial candidate anchor | Ungraded hypothesis | What to check |
|---|---|---|
| `Sanjac_Rec_91:0020` | uncertain | Close oncoming passes: inspect world-relative pedestrian course and robot motion before claiming an adjustment. |
| `GDC_Library_42:0300` | near_miss | A pedestrian crosses while the robot slows; visible human motion is insufficient for the steady-robot condition. |

Expand anchors to the whole encounter, including the baseline and recovery. Deduplicate overlapping windows. For a comparison, seek both required members; an unsuitable suggested comparator is a near miss, not permission to relax the question.

**Budget:** up to three distinct episodes for this intent; stop even if none qualifies. Absence of a confirmed match in this pool is not corpus-wide absence.

**Provenance:** exact label lines, sensor refs, report/session IDs and audit lines are in the corresponding `sources_dev.jsonl` record.
