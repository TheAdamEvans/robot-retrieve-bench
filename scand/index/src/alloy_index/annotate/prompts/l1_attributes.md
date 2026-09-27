# L1 labelling brief — per-segment scene attributes

You are the benchmark labeller for a small corpus of robot recordings (SCAND: a Boston Dynamics Spot and a Clearpath
Jackal, teleoperated around a university campus). Your labels become **provisional reference data** for a retrieval
benchmark, so accuracy and honesty matter more than speed. Say `TRUTH_UNKNOWN` / `UNSURE` rather than guess.

## Your job
Job id: `{JOB}`. Label these segments (each is a 4 s span `[EEEE-4 s, EEEE s]` of recording `Rec`):

{SEGMENTS}

## Tools — the only way you may look at the data
Run from the `scand/` directory (your working directory), always with the job id set:

```
SCANDQ_JOB={JOB} uv run scandq <command> ...
```

| Command | Use |
|---|---|
| `window Rec:EEEE` | **Start here for every segment.** Composite: 3 front frames, lidar top-down view, body cameras (Spot), speed / yaw-rate / front-range plot. Proposal only (downscaled). |
| `frame Rec --t S [--cam front\|body_frontleft\|body_frontright\|body_left\|body_right\|body_back]` | One frame at **native resolution** (native evidence). |
| `step Rec --t S --n 4 [--dir back\|fwd] [--cam C]` | 4 consecutive native frames (native evidence); for people entering/leaving view. |
| `sync Rec --t S` | Every camera + lidar at one instant (body cameras and lidar count as native). |
| `lidar Rec --t S [--range 10]` | Top-down lidar with the robot footprint (orange, NOMINAL) and travel corridor (green, ESTIMATED). Native. |
| `signals Rec --t0 A --t1 B` | Speed (raw + smoothed), yaw rate, front range. COMPUTED by code, not by you. |
| `strip Rec --t0 A --t1 B --hz 1` | Contact sheet; proposal only. |
| `recordings`, `coverage Rec`, `message MID` | Metadata, topic coverage, raw decoded message. |
| `label put attributes --json @file.json` | Submit labels (a JSON list). Rejected items come back with reasons: fix and resubmit. |
| `label get attributes --rec Rec` | Read back what is stored. |

Reading the views:
- **Times.** `t` is when the recorder received a message, and `header_t` is when the sensor captured it. Spot's body
  cameras arrive about 0.4–0.7 s after capture (tiles show `age`), so a body frame shows the scene at `header_t`.
  `dt_ms` is `t` minus the time you asked for. `warnings` flags frames far from the requested time, such as at a
  recording edge or across a gap.
- **Camera names.** Spot's front stereo pair is cross-eyed: `frontleft` looks front-right and `frontright` looks
  front-left. Body cameras are shown upright.
- **Corridor.** The green corridor drawn on front frames uses a nominal camera model and is approximate; it tends
  to look too wide near the robot. Judge corridor membership mainly from `lidar` (the green box) and from people's
  position relative to the path.
- **Turns.** `signals` and `window` report `heading`: the net change, the largest excursion and the largest change
  in any 1 s. Use these for `turn_visible`, and cite the odom refs `signals` returns.

Every command prints JSON with a `path` to an image — open it with the Read tool — and `refs`: the MessageIds shown,
each flagged `native`. **Cite refs from what you actually opened.** Do not read other files in the repository, run
Python, or look at any other outputs; the tools above are deliberately your whole view of the data. Recording-level
tags printed by `recordings` are **not** evidence about any segment.

## Attributes (one record per segment)
Times are seconds from recording start. "Corridor" = the robot's forward travel path: roughly 1.5 m wide, out to about
5 m ahead (the green box in the lidar view; it is an estimate, so judge from the camera too).

| Field | Values | Definition |
|---|---|---|
| `persons_in_corridor` | `ZERO`, `ONE_TWO`, `THREE_FIVE`, `SIX_PLUS`, `UNSURE` | Max number of people simultaneously in the corridor at any moment in the segment. People beside the path do not count. |
| `stationary_group` | Truth | ≥ 2 people standing or talking together (not walking) within ~5 m of the robot's path. |
| `doorway_traversal` | Truth | The robot passes through a doorway, door frame or narrow gate during the segment. |
| `vehicle_present` | Truth | A motor vehicle (car, truck, bus, golf cart, motor scooter) visible in any camera within ~20 m. |
| `vehicle_interaction` | Truth | A vehicle moves across or near the robot's path within ~10 m, or the robot visibly yields to one. |
| `bicycle` | Truth | A bicycle or kick/e-scooter (ridden or parked) within ~10 m. |
| `indoor` | Truth | The robot is inside a building for most of the segment. |
| `turn_visible` | Truth | The robot's heading swings by ≥ 30° within the segment: `heading.max_abs_excursion_deg` ≥ 30. |
| `caption` | text | One factual sentence: what the robot does and what is around it. No guesses about intent. |
| `refs` | list of MessageId strings | The refs you relied on. Positive claims (any TRUE, or persons ≥ 1) need ≥ 1 **native** ref. |

Truth values are `TRUTH_TRUE`, `TRUTH_FALSE` or `TRUTH_UNKNOWN` (occluded, motion-blurred, or genuinely ambiguous).

Example item:
```json
{"segment_id": "Butler:0052", "persons_in_corridor": "ONE_TWO", "stationary_group": "TRUTH_TRUE",
 "doorway_traversal": "TRUTH_FALSE", "vehicle_present": "TRUTH_FALSE", "vehicle_interaction": "TRUTH_FALSE",
 "bicycle": "TRUTH_FALSE", "indoor": "TRUTH_FALSE", "turn_visible": "TRUTH_FALSE",
 "caption": "Spot walks a sidewalk behind a person on crutches past seated tailgaters under tents.",
 "refs": ["Butler/image_raw/compressed#1445", "Butler/velodyne_points#480"]}
```

## Method
1. For each segment open `window`. Decide which attributes are clearly negative and which need checking.
2. Confirm every positive or borderline attribute at native resolution (`frame`, `step`, `sync`, `lidar`).
   Counting people in the corridor: use a native front `frame` at the busiest moment, cross-check with `lidar`.
3. Write items to `annotations/tmp/{JOB}/batch_N.json` and submit with `label put`. Batches of ~5 are fine.
4. Finish with `label get attributes --rec <Rec>` to confirm everything is stored.

## Report back (short)
- Segments labelled and any left `UNSURE` / `TRUTH_UNKNOWN`, with why.
- Tool friction: anything slow, confusing, missing or wrong in the views. This feeds tool fixes, so be concrete.
