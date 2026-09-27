# Architecture

Natural-language search over archived robot recordings (SCAND: Spot and Jackal ROS bags). Raw bags become a
**bundle** of derived artifacts. A server answers questions over the bundle through **one pipeline runner** shared
by every search config. One evaluation harness scores quality, speed and cost for all of them.

```
            ┌──────────────────────── alloy-index ────────────────────────┐
raw/*.bag ─►│ Source ─► ingest ─► providers ─► frames ─► windows/tags ─►   │─► Sink: bundles/dev ─► alloy-server (HTTP, runner)
            │           [annotate.l1: optional, paid] ─► apply.fused      │           ▲
            └─────────────────────────────────────────────────────────────┘           │
                  scandq (audited views) ─► Opus labeller ─► L1 labels / L2 judgments ─► alloy-train (eval, learn)
                                                                                         │
                         usage and labels become datasets ─► FUSED model ─► bundle/models ┘
```

## Packages and the import rule

| Package | Owns | May import |
|---|---|---|
| `server/` `alloy-server` | proto contracts, instrumented readers, the embodiment registry, the pipeline runner, the program generator, validator and executor, receipts, the HTTP API, `evalkit` | nothing from the other two |
| `index/` `alloy-index` | Sources, Stages and the Sink; conversion, intake, providers, frame and window encoders, applying trained models, the `scandq` labeller tools and the label store | `alloy_server` |
| `trainer/` `alloy-train` | the benchmark (query sets, pooling, metrics, reports), L2 judging, FUSED training, `alloy-evals` | both |

Layering is enforced by tests (`server/tests/test_contracts.py`, `index/tests/test_layering.py`). The server also
never touches bundle files outside its `io` package, so byte accounting stays complete.

## Contracts (`server/proto/alloy/v1`)

The protos are the interfaces between packages. Regenerate them with `./tools_codegen.sh`.

| File | Holds |
|---|---|
| `common.proto` | truth values, UNKNOWN reasons (`NOT_INDEXED`, `STALE`, `SENSOR_ABSENT_IN_LOG`, …), boundaries |
| `query.proto` | `QueryProgram`: events, relations, selection, receipt and context |
| `pipeline.proto` | `PipelineSpec`, `Candidate` (`WINDOW`, `EVENT`, `MERGED`), stage scores, filter records, stage reports |
| `answer.proto` | `SearchRequest` and `SearchResponse`, answer statuses, completeness, cost reports |
| `embodiment.proto` | `EmbodimentProfile` (reviewed robot facts) and `IntakeReport` (measured per log) |
| `index.proto` | `IndexSpec` (Source, Stages, Sink), `ArtifactRecord` and `SinkManifest` |
| `eval.proto` | `EvalQuery` with its oracle program and expected status |

## The bundle

```
bundles/dev/
  manifest.json      the Sink's only state: every artifact key, its inputs, outputs and timings
  mcap/<rec>/        by_sensor MCAPs: front_camera, body_cameras, lidar, telemetry, other (+ <rec>.json info)
  timeline/<rec>     one row per message: topic, ordinal, log/header time, file, chunk offset and length
  intake/<rec>       IntakeReport: robot, sensor availability, stale leading frames, gait, speed floor
  features/          provider outputs (motion, clearance, detections, tracks, boxes) and per-frame SigLIP2 vectors
  index/             window indexes (siglip2, fused_*), tags
  models/<name>/     full-data trained heads applied at index time (safetensors + model.json)
  pipelines/         PipelineSpec JSON for every config
  prompts/           few-shot pool for the program generator (compose_dev only)
```

A `MessageId` is (recording, topic, ordinal), where the ordinal is the message's position in the bag. So it is
independent of the storage layout. Chunk locations live only in the timeline.

### Storage layout: `by_sensor`

Each profile sensor group gets its own MCAP, with chunk sizes suited to its readers: 256 KiB for cameras, 1 MiB
for lidar and 64 KiB for telemetry. Readers fetch only the chunks they need, and every byte read is counted.
`uv run python -m alloy_index.bench_io` compares this layout with the old single interleaved file:

| Read | Butler | Sanjac |
|---|---|---|
| 4 s front-camera clip | 2.1× fewer bytes (38.8 → 18.4 MB) | 1.2× |
| telemetry for a window | 93× (0.99 MB → 11 KB) | 147× |
| causal receipt (latest message per sensor) | 2.0× | 2.8× |
| one lidar frame | 1.2× | 2.1× |

Video now dominates clip bytes. The next saving would be a video-proxy stage (a low-resolution clip per window).

## Embodiment

- **Profiles** (`server/src/alloy_server/embodiments/*.textproto`) are reviewed facts per robot model: footprint,
  nominal maximum speed, and the sensors with topics and camera bands (ESTIMATED ranges where nothing was recorded).
  `identity_topics` is the minimum set that recognises the robot. Every other sensor is optional per log.
- **Intake** measures each log. It records the robot, `absent_sensors`, stale leading frames (buffer flushes from
  another moment), gait period and speed floor. A log missing a profiled sensor is **degraded, not rejected**.
  `Bass_Garage_134` is a Spot log with no front camera.
- **`EmbodimentContext`** is the one place measurements are normalised: body-side room (a follower directly behind
  is not side room), front margin from the front edge, corridor, camera band, speed as a fraction of nominal
  maximum, the speed smoother and stale frames. Providers receive a context; they never read profiles directly.
- **Robot-relative features** let one question mean the same thing on both robots. For example, "fast" is
  `speed_frac_max > 0.75`.
- **A clause on a sensor absent from the log** evaluates to `UNKNOWN(SENSOR_ABSENT_IN_LOG)`, never FALSE.

## The pipeline runner

Every config is a `PipelineSpec`: generators produce candidates, then rankers order, filter and truncate them.
Stages declare what they do, and the runner enforces it:

| Declaration | Meaning | Enforced |
|---|---|---|
| `needs_program` | reads the `QueryProgram` | reading it without declaring raises |
| `filters` | may remove candidates | every removal needs a `FilterRecord` with a reason |
| `truncates` | may cut the list | skipped under `SCORE_ALL`, so every judged window is scored |
| `display_only` | presentation only (`MergeAdjacent`) | runs only when `SearchRequest.presentation` is set: the live UI sets it, the eval never does |

The runner also checks append-only invariants: a stage may not rewrite seeds, other stages' scores, lineage or
evidence. Each stage reports candidates in and out, filter reasons and its cost.

**Configs.**

| Config | Pipeline |
|---|---|
| TAGS | BM25 over recording-level tags |
| EMBED | SigLIP2 text query against mean-pooled window vectors |
| PROGRAM | text → `QueryProgram` (gpt-6-luna, strict schema, one repair turn) → executed over the full scope → verified |
| HYBRID | EMBED candidates, then deterministic verification |
| FUSED | one lookup in a learned window index (image plus signals); see [FUSED.md](FUSED.md) |
| FUSED_V | FUSED candidates, then program verification |

**Honest answers.** Clauses are three-valued (TRUE, FALSE or UNKNOWN), and every UNKNOWN carries its reason. A
candidate is filtered only when a required clause is definitively FALSE. Answers carry `answered`,
`answered_partial`, `answered_unverified`, `insufficient_evidence` or `none_found_exhaustive`. Superlatives need
exhaustive coverage. Detector silence is never proof of absence.

**Cost accounting.** Each response has a `CostReport`: wall time, bytes read (equal to the timeline's chunk lengths,
which is tested), resident bytes, LLM calls and tokens (cached separately), and whether the program came from cache.

## Learning loop: usage becomes datasets

1. **Labels.** The Opus labeller writes L1 per-segment attributes and captions through audited `scandq` views. L2
   graded judgments over pooled results are the benchmark's test labels.
2. **Pseudo-labels.** Grammar-sampled `QueryProgram`s are executed by the same executor PROGRAM uses. Their matches
   become text-window pairs, which distils PROGRAM into an embedding.
3. **FUSED** trains on pseudo-labels plus L1 captions, never on L2. It is evaluated leave-one-recording-out and
   saves a full-data model.
4. **`apply.fused`** embeds new recordings with the saved model at index time, with no retraining.

## Evaluation

- **The benchmark** (`trainer/src/alloy_train/eval`): pooled graded relevance, macro ROC-AUC (penalised for
  abstention), nDCG@10, recall on condensed lists, bootstrap CIs over intent groups, and latency and token costs.
  See [`benchmark/CORRECTNESS.md`](../benchmark/CORRECTNESS.md).
- **Module evals**: each module keeps small, self-contained suites. See [EVALS.md](EVALS.md).
