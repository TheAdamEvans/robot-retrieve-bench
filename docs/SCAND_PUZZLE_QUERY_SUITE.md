# SCAND Puzzle Query Suite

Status: design draft grounded in the seven complete local SCAND bags and the dashboard screenshots captured on 2026-09-26.

This document turns the retrieval benchmark into a set of deliberately difficult, compositional questions. The purpose is not to make a language model sound clever. A successful system must retrieve the right interval, align the right sensor messages, perform the requested temporal and spatial operations, and return inspectable evidence.

## 1. The benchmark object

The benchmark unit is a **query program over one or more views of a canonical recording**.

The MCAP or ROS bag remains the source of truth. A frame table, pose table, event table, embedding index, lidar summary, and Rerun visualization are materialized views over the same messages. Every derived row must retain enough provenance to recover its source:

- recording ID and source object version
- topic and message schema
- log timestamp and header timestamp, when both exist
- message index or byte location
- transform chain and calibration version
- derivation name, version, parameters, and code revision

This is the central product idea. Users should be able to ask across convenient analytical views while the system preserves the evidentiary link to raw robot data.

## 2. A query is more than text

Each benchmark item should contain these fields:

```yaml
query_id: scand.puzzle.001
utterance: >-
  Find the first crowd-induced hesitation after Spot enters the event area.
scope:
  recordings: [A_Spot_Butler_LBJ_Sat_Nov_13_104]
anchor:
  event: enters_dense_crowd
  selector: first
window:
  start: anchor - 4s
  end: anchor + 12s
predicates:
  - visible_person_count >= 6
  - observed_speed drops_by >= 0.7 m/s within 3s
  - lidar_clearance_p05 decreases
operators:
  - first_after
  - within
answer:
  type: interval
  tolerance_ms: 500
evidence:
  required: [front_rgb, odom, lidar]
  optional: [spot_surround]
hard_negatives:
  - visual_crowd_without_slowdown
  - slowdown_without_visual_crowd
```

The natural-language utterance tests query understanding. The structured program defines what the annotator meant and makes evaluation reproducible. It also gives us a target intermediate representation for text-to-query generation.

## 3. Canonical anchors and windows

An anchor is an identity-bearing point or interval. It is not merely a timestamp.

### Scene anchors

- first person enters a safety corridor
- closest approach to a person, bicycle, wall, vehicle, or doorway
- robot crosses a doorway plane or curb boundary
- traffic stream begins or clears
- a previously occluded region becomes visible
- conversational group first blocks the intended path

### Motion anchors

- local minimum or maximum of observed speed
- beginning of sustained stop, with a minimum dwell duration
- maximum absolute yaw rate in a bounded interval
- largest divergence between commanded and observed motion
- maximum IMU impulse or Spot joint-effort excursion

### Data anchors

- first message after a topic gap
- camera frame nearest a lidar event under a maximum-age rule
- last known pose strictly before an event
- transform valid at the source message's header timestamp

An answer window is computed from an anchor. We should support `before`, `after`, `during`, `overlaps`, `contains`, `until`, `first_after`, `last_before`, `nearest`, `argmin`, and `argmax`. `Nearest` must declare its clock, direction, and maximum age. A causal query should usually use `last_before`; a symmetric nearest-neighbor join can leak future state.

## 4. Views of the same recording

The benchmark should expose these logical views even if the first implementation computes some of them on demand:

| View | Grain | Examples |
|---|---|---|
| `messages` | raw message | topic, timestamps, schema, source location |
| `frames` | camera frame | camera, image, intrinsics, pose, SigLIP2 embedding |
| `motion` | resampled instant | pose, twist, acceleration, joint summaries |
| `geometry` | lidar instant | clearance sectors, clusters, free-space corridor |
| `tracks` | object observation | class, image box, bearing, range, track ID |
| `events` | semantic interval | doorway traversal, yielding, crowd encounter |
| `episodes` | bounded sequence | approach, interaction, response, recovery |

The views must not invent one universal sampling rate. A query planner should align only the columns it needs and report the age or skew of each joined observation.

## 5. What the screenshots reveal

The local corpus is small but contains useful contrasts:

- `A_Jackal_Brackenridge_Greg_Sat_Nov_13_94` shows a sidewalk, bicycles and street furniture with highly variable speed, yaw, clearance, IMU acceleration, and angular rate.
- `A_Spot_Butler_LBJ_Sat_Nov_13_104` moves through an outdoor event crowd. The front view and five body cameras offer different evidence around people, while motion and clearance include several interruptions.
- `A_Spot_JCL_JCL_Wed_Nov_10_65` is an indoor crowd and conversational-group route with early stopped/slow intervals, a looping trajectory, tight clearance, and strong changes in joint motion and effort.
- `A_Spot_Library_MLK_Thu_Nov_18_122` contains indoor corridors, traffic, and narrow doorways. Its nearly hairpin trajectory and repeated speed collapses make temporal ordering essential.
- `B_Jackal_Sanjac_Stadium_Sat_Nov_13_89` combines crowds, a street crossing, and conversational groups. Its IMU and yaw spikes let us test whether visual semantics explain physical response.
- `B_Spot_GDC_AHG_Mon_Nov_15_116` appears visually open in sparse keyframes, despite vehicle-interaction and traffic labels. This is a good hard-negative source: recording-level labels cannot localize an event.
- `B_Spot_RLM_RLM_Wed_Nov_10_45` moves between indoor and outdoor-looking spaces, with multiple turns, clearance troughs, and a long late slowdown. It is useful for doorway and transition questions.

These are hypotheses from sparse screenshots. Exact gold intervals must be annotated against timestamped bag frames and supporting sensor streams.

## 6. Puzzle deck

### P01 — Crowd-induced hesitation

**Query:** In the Butler-to-LBJ run, find the first interval in which at least six people occupy the forward travel corridor and Spot loses at least 0.7 m/s within three seconds. Return the frame just before deceleration, the minimum-speed time, and the first recovery to 90% of pre-event speed.

**Requires:** person instances, projected travel corridor, odometry, lidar, `first_after`, three related anchors.

**Hard negatives:** a dense crowd passed at constant speed; a slowdown with no crowd in the corridor.

### P02 — Rear camera resolves the cause

**Query:** Find an outdoor crowd interval where front RGB alone cannot show whether a pedestrian has cleared the robot, but the rear camera can. Return the earliest causally safe resume time using only messages available at that time.

**Requires:** synchronized front and rear views, body-frame geometry, observed speed, causal joins.

**Hard negative:** resuming based on a future rear frame or a symmetric nearest-frame join.

### P03 — Conversational group versus traffic flow

**Query:** In the indoor JCL run, identify a stationary conversational group that constrains the route. Distinguish it from people walking in the same direction, then measure the robot's lateral detour and speed loss.

**Requires:** multi-frame human motion, grouping, pose, intended-path approximation, interval classification.

**Hard negatives:** a dense moving crowd; people visible outside the travel corridor.

### P04 — Blind-corner reveal

**Query:** Find the blind-corner event in which a person becomes visible only after the robot begins turning. How far and how long had the robot committed to the turn before first visibility?

**Requires:** yaw onset, track first-observation time, pose integration, visibility reasoning.

**Hard negative:** a person visible before turn onset but temporarily missed by a detector.

### P05 — Doorway plane crossing

**Query:** In the Library-to-MLK run, find the narrowest doorway actually traversed. Return approach, threshold crossing, and full-body-clear times, plus minimum left and right clearance in the doorway frame.

**Requires:** doorway-plane estimation, robot footprint, lidar geometry, pose transforms, interval boundaries.

**Hard negative:** a narrow-looking opening that the robot passes beside rather than through.

### P06 — Which slowdown belongs to the doorway?

**Query:** The Library-to-MLK run contains several speed collapses. Select the one caused by a doorway rather than a person or turn, and return the visual and geometric evidence that disambiguates it.

**Requires:** competing-event attribution, temporal overlap, evidence ranking.

**Hard negatives:** all other local speed minima in the same recording.

### P07 — Crossing, then yielding

**Query:** In the San Jacinto-to-stadium run, locate a street crossing where the robot begins crossing, yields to a moving agent, and then resumes. Return the ordered phase boundaries and the agent's crossing direction in the robot frame.

**Requires:** phase segmentation, person/vehicle motion, ego pose, relative direction, IMU and odometry.

**Hard negative:** stopping before entering the crossing; turning without yielding.

### P08 — Same clearance, different risk

**Query:** Retrieve two intervals with matched lidar p05 clearance within 10 cm: one caused by a static wall or doorway and one caused by a moving person. Rank which is more hazardous using time-to-collision, then cite the observations that support the ranking.

**Requires:** cross-recording retrieval, dynamic/static classification, matched structured filter, visual reranking.

**Hard negative:** treating equal range as equal risk.

### P09 — Image looks clear, lidar disagrees

**Query:** Find a moment when front RGB appears to show a clear route but lidar reports a close obstacle in the intended corridor. Determine whether the obstacle is below the camera's useful field of view, laterally offset, or a likely sensing artifact.

**Requires:** camera-lidar projection, travel corridor, surrounding frames, calibrated uncertainty.

**Hard negative:** low global p05 clearance produced behind the robot or outside the corridor.

### P10 — Image looks crowded, geometry is safe

**Query:** Find the densest-looking front image for which the projected travel corridor remains clear and speed does not fall materially over the following three seconds.

**Requires:** SigLIP2 or detector candidate generation, spatial reranking, lidar, future motion.

**Hard negative:** any visually similar crowd that enters the corridor.

### P11 — Bicycle interaction on a sidewalk

**Query:** In the Brackenridge-to-Greg run, find the closest bicycle encounter. Decide whether the bicycle is static, crossing, approaching, or receding, then return the robot's response window.

**Requires:** temporal visual track, relative motion, lidar association, Jackal odometry.

**Hard negative:** parked bicycles visible for a long time; an A-frame sign with similar geometry.

### P12 — Vehicle label without a localized vehicle event

**Query:** In the GDC-to-AHG run, either localize the strongest vehicle interaction with supporting frames and motion response or return `insufficient evidence`. A recording-level `Vehicle Interaction` tag is not sufficient evidence.

**Requires:** abstention, label-versus-instance distinction, evidence completeness.

**Hard negative:** returning the whole recording or the visually emptiest sampled keyframe.

### P13 — Indoor-to-outdoor transition

**Query:** In the RLM run, find the first transition from an enclosed corridor to an open exterior or atrium-like space. Measure the change in lidar free-space area and speed over symmetric five-second windows.

**Requires:** semantic transition localization, symmetric aggregation, clearance beyond a single percentile.

**Hard negative:** a brighter indoor lobby that looks exterior in one frame.

### P14 — Turn caused by route or obstacle?

**Query:** Find the largest absolute yaw-rate event in each embodiment. Classify whether it is a planned route turn, obstacle avoidance, or recovery from a stop. Return the evidence and confidence.

**Requires:** argmax within each recording, visual context, trajectory curvature, obstacle dynamics, cross-embodiment normalization.

**Hard negative:** raw sensor spikes without sustained pose change.

### P15 — Commanded versus achieved motion

**Query:** For Spot, retrieve the interval with the largest sustained disagreement between commanded and observed forward speed, excluding startup and shutdown. Was the disagreement accompanied by crowd occupancy, low clearance, or elevated joint effort?

**Requires:** aligned command and odometry, exclusion windows, duration threshold, multimodal explanation.

**Hard negative:** the orange command trace in the current dashboard may be zero because the command field or frame convention needs validation; the benchmark must first verify command semantics.

### P16 — Physical contact proxy

**Query:** Find the highest joint-effort excursion that is not explained by ordinary gait phase. Determine whether it coincides with a tight passage, abrupt speed change, body-camera proximity, or no visible cause.

**Requires:** gait-normalized residuals, joint state, body cameras, lidar, motion.

**Hard negative:** periodic gait peaks and recording boundaries.

### P17 — Rough terrain versus braking

**Query:** For Jackal, find two high-IMU-acceleration events: one caused primarily by terrain vibration and one caused primarily by a commanded motion change. Explain the distinction using spectral content, odometry, and images.

**Requires:** IMU windows, frequency features, odometry derivatives, visual surface type.

**Hard negative:** a single-sample IMU outlier.

### P18 — Loop closure with changed semantics

**Query:** Find two poses in the JCL run that are spatially close and similarly oriented but occur at different times. Show that the social scene has changed, and describe how the robot's behavior differs on the two visits.

**Requires:** pose-neighbor self-join with temporal exclusion, heading tolerance, scene comparison, motion comparison.

**Hard negative:** adjacent samples from the same pass.

### P19 — Same scene, different camera view

**Query:** At an annotated anchor, rank the five Spot body cameras by how well each supports the claim that the robot has lateral clearance. Compare this ranking with front RGB and lidar.

**Requires:** view identity, camera geometry, timestamp age, spatial evidence quality.

**Hard negative:** ranking cameras from image similarity alone without considering their field of view.

### P20 — Last safe evidence before action

**Query:** Immediately before the robot accelerates from a stop, return the latest available front image, side/rear images, lidar scan, pose, and transform that could have informed the action. Report each message's age.

**Requires:** `last_before` joins across 30 Hz, 10 Hz, 4.5 Hz, and 16 Hz topics; no future leakage.

**Hard negative:** using the closest message when it occurs after acceleration onset.

### P21 — Sensor freshness trap

**Query:** Find a high-motion interval where a body-camera observation is more than 150 ms old at the front-camera anchor. Quantify how far a nearby object could move during that staleness and decide whether fusion is still defensible.

**Requires:** header timestamps, maximum-age policy, relative-motion bound, uncertainty.

**Hard negative:** comparing log timestamps from one topic with header timestamps from another.

### P22 — Presentation-video clock trap

**Query:** A user supplies an AVI time and asks for the matching lidar scan. Resolve it to bag time, return the scan, and explain the uncertainty. Reject a direct equality join because the local AVIs are approximately ten-times accelerated.

**Requires:** explicit clock-domain mapping and provenance.

**Hard negative:** interpreting `00:05` in the AVI as five seconds after bag start.

### P23 — Missing-topic counterfactual

**Query:** Answer a crowd-and-braking question once with all sensors and once after hiding lidar. Identify whether the answer changes, which evidence becomes unsupported, and whether the confidence should fall.

**Requires:** controlled sensor ablation, calibrated confidence, evidence accounting.

**Hard negative:** producing identical certainty after deleting decisive evidence.

### P24 — Cross-embodiment analogue

**Query:** Find the Jackal interval most behaviorally analogous to Spot passing a conversational group. Match on event structure and response rather than raw speed, joint state, or topic names.

**Requires:** embodiment-normalized event representation, semantic retrieval, structured constraints.

**Hard negative:** nearest raw feature vector across incompatible sensors.

### P25 — The chained final puzzle

**Query:** Find the earliest event, across all complete recordings, in which a person or group becomes relevant only after a turn; the robot then reduces speed by at least 30%; lidar clearance reaches a local minimum within two seconds of the speed minimum; and the robot returns to 90% of its pre-event speed before the person disappears from every camera view. Return the episode, ordered anchors, transformed trajectory segment, and minimal evidence set.

**Requires:** semantic candidate generation, interval algebra, pose and frame transforms, cross-topic alignment, recovery logic, multi-camera persistence, cross-recording ranking.

**Hard negatives:** each query clause should have examples that satisfy all but one condition.

## 7. Annotation protocol

Annotators should work in a synchronized viewer backed by bag timestamps. For each puzzle:

1. Mark coarse candidate intervals from video and signal summaries.
2. Refine semantic anchors on canonical timestamped frames.
3. Refine motion and geometry anchors from their native topics.
4. Save positive evidence, contradictory evidence, and all-but-one-clause negatives.
5. Record uncertainty as intervals rather than pretending every boundary is exact.
6. Have a second annotator adjudicate queries involving intent, causality, occlusion, or social grouping.

Model-assisted proposals are useful, but a vision-language model must not create the gold answer from presentation-video timestamps. It can propose descriptions and candidate intervals; a deterministic extractor resolves source messages and clocks.

## 8. Scoring

A single recall number would hide the interesting failures. Score these layers separately:

- **recording recall:** did the correct recording enter the candidate set?
- **episode recall and ranking:** did the right interval rank highly?
- **anchor error:** absolute time error for each named anchor
- **interval quality:** temporal IoU with uncertainty-aware boundaries
- **predicate accuracy:** which query clauses were actually satisfied?
- **ordering accuracy:** were phase and event relations correct?
- **spatial accuracy:** position, distance, bearing, corridor occupancy, and frame correctness
- **evidence precision/recall:** were cited messages necessary and sufficient?
- **freshness compliance:** did every join obey direction and maximum-age constraints?
- **abstention quality:** did the system decline unsupported claims?
- **counterfactual consistency:** does confidence respond sensibly when evidence is removed?

We should record execution cost as bytes read by topic, decoded images, lidar points processed, embedding candidates examined, and wall-clock latency. That connects retrieval quality directly to the Iceberg/object-store product story.

## 9. Monday-sized demonstration

Eight items make a compact first demo:

1. P01 crowd-induced hesitation
2. P05 doorway plane crossing
3. P08 same clearance, different risk
4. P12 label without localized evidence
5. P18 loop closure with changed semantics
6. P20 causal last-safe-evidence join
7. P22 presentation-video clock trap
8. P25 chained final puzzle

The demo can show one query moving through four layers: natural language, structured query plan, ranked intervals, and a synchronized evidence view. Start with metadata/SQL, add SigLIP2 candidate retrieval, then add spatial and temporal verification. This makes each improvement and failure visible.

## 10. Immediate implementation sequence

1. Convert one Spot and one Jackal bag to MCAP without changing canonical timestamps.
2. Materialize message, frame, motion, and lidar-summary views with source pointers.
3. Build the anchor and interval schema before producing embeddings.
4. Add a synchronized annotation page with frame stepping and native bag time.
5. Annotate the eight demo puzzles and their all-but-one-clause negatives.
6. Run metadata/SQL and SigLIP2 candidate baselines.
7. Add deterministic temporal joins and geometry reranking.
8. Emit an evidence receipt for every answer and measure bytes read as well as relevance.

