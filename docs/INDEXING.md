# Indexing: Source → Stages → Sink

`alloy-index` turns raw bags into the bundle the server loads. It is config-driven (`index/index.textproto`),
incremental (only missing or stale artifacts are built), and repeatable (artifact keys are content hashes). The paid
labeller is an optional stage that never runs unless asked.

```bash
uv run alloy-index plan      # what's missing or stale
uv run alloy-index run       # build exactly that
uv run alloy-index status    # stages × recordings grid
```

```
Source (LocalBagSource: raw/*.bag + SCAND_index.csv)
  └─ per recording: ingest ─► providers.{motion, clearance, imu, detections} ─► frames.siglip2 ─► [annotate.l1]
  └─ corpus:        windows.siglip2 ─► tags ─► apply.fused ─► pipelines ─► prompts
Sink (LocalBundleSink: bundles/dev + manifest.json)
```

## Source

A **Source** enumerates recordings and hands each stage the raw bag and its metadata. It knows nothing about stages.

`LocalBagSource` globs `raw/*.bag` and joins `SCAND_index.csv`. Recording IDs come from:
1. `aliases`, which keep the original seven short IDs (`Butler`, `JCL`, …), so every label and judgment stays valid;
2. otherwise `id_template "{route}_{fr}"` parsed from the bag stem, e.g. `A_Spot_Bass_Garage_Fri_Nov_26_134` →
   `Bass_Garage_134`. A collision fails loudly.

The robot is never taken from the filename. Intake recognises it from the profile's `identity_topics`.

`held_out` lists SCAND's Val split (`Rec_Tent_129`, `Bass_Garage_134`). These recordings are indexed and searchable,
but never used for training, prompt tuning or module-eval dev sets. FUSED embeds them with the full-data model.

A second source (for example, downloading by `FileName` from the SCAND index links) would implement the same
`recordings()` method.

## Stages

| Stage | Scope | Version | Writes |
|---|---|---|---|
| `ingest` | recording | ingest@3 (intake@2) | `mcap/<rec>/*.mcap` (`layout: by_sensor`), `timeline/<rec>`, `intake/<rec>` |
| `providers.motion` | recording | motion@3 | speed, yaw rate, heading, acceleration, `speed_frac_max` |
| `providers.clearance` | recording | clearance@4 | front, any-direction and body-side clearance, gap, doorway, `clearance_margin_front_m` |
| `providers.imu` | recording | imu@2 | `imu_vibration_rms`, gyro yaw rate, `odom_gyro_yaw_disagreement_dps`. Jackal only; Spot has no IMU (not applicable) |
| `providers.detections` | recording | detections@2 | RT-DETR counts, corridor occupancy, tracks, boxes. Not applicable without a front camera |
| `frames.siglip2` | recording | siglip2_frames@1; params `front_hz: 10`, `body_hz: native` | per-frame SigLIP2 vectors |
| `annotate.l1` | recording | l1_labeller@1; `requires_flag` | L1 per-segment labels (paid; see below) |
| `windows.siglip2` | corpus | siglip2_windows@2 | 4 s windows at a 1 s stride; body-camera fallback when the front camera is absent (`source` column) |
| `tags` | corpus | tags@2 | recording-level tags |
| `apply.fused` | corpus | apply_fused@1 | FUSED vectors for windows the saved full-data models haven't embedded |
| `pipelines`, `prompts` | corpus | pipelines@2, prompts@2 | every `PipelineSpec`; the generator's few-shot pool |

A stage declares `IMPL`, `VERSION`, `SCOPE` and `DEPENDS`, and implements `run`. Optionally it also implements
`applies` (for example, "front camera absent in this log"), `adopt` (register outputs built earlier by the same
version and params) and `content_key`.

**Keys.** An artifact's key is `sha256(stage, VERSION, params, input content keys, recording)`. Ingest's content key
hashes (topic, ordinal, payload sha256) over the whole bag. So:
- bumping a provider's `VERSION` rebuilds that provider and whatever reads it, and nothing else;
- changing the MCAP layout re-ingests but rebuilds nothing downstream, because the payloads are unchanged;
- `plan` on an unchanged tree prints nothing to do.

### Adding a stage

1. Put a class in `index/src/alloy_index/stages/` with `@register`, `IMPL`, `VERSION`, `SCOPE`, `DEPENDS` and
   `run(ctx, rec) -> (paths, info)`. Take robot facts from `EmbodimentContext.for_recording(...)`.
2. Add a `stages { name: … impl: … }` line to `index/index.textproto`.
3. Add a module eval suite next to the code (see [EVALS.md](EVALS.md)).

## Sink

`LocalBundleSink` owns `bundles/dev/manifest.json`, which records every artifact with its key, stage, version,
params hash, input keys, output paths, content key, wall time and info. That manifest is the only index state.
Writers take a file lock and read, merge and write, so several `alloy-index run` processes (for example, detections,
frames and the labeller) can build different stages at the same time.

A warehouse or object-store sink would implement the same `has`, `get`, `latest` and `put` methods. A separate state
store (history, multi-writer coordination) is deferred until a second sink exists.

## The optional labeller stage: `annotate.l1`

```bash
uv run alloy-index run --stages annotate.l1 --annotate
```

- **Jobs.** Unlabelled 4 s segments are chunked into disjoint jobs of at most 30 segments. Up to 8 run in parallel
  across recordings.
- **How a job runs.** Each job runs headless, with only the audited labeller tools:

  ```
  claude -p <brief> --model claude-opus-5-5 --output-format json --permission-mode acceptEdits \
         --allowedTools "Bash(uv run scandq:*)" "Bash(ls:*)" Read Write --max-budget-usd 15   (env SCANDQ_JOB=<job>)
  ```

  The brief is `annotate/prompts/l1_attributes.md`, plus the job's segment list.
- **Output.** Validated label shards, `labels/train/attributes/<campaign>.<job>.jsonl`. Each job's assignment,
  measured cost and report go to `labels/metadata/<campaign>/jobs/`. Campaign priority resolves overlaps (see
  [`labels/README.md`](../labels/README.md)).
- **Cost.** Each job's measured `total_cost_usd` is recorded in the artifact info.

**Data wave 1, measured.** Nine recordings, 1,608 s, **397 segments in 17 jobs, $84.05 total** with Opus 5.5
(measured `total_cost_usd`). That is **$0.21 per segment**, about **$5.20 per 100 s of recording**. Jobs took 2–21
min each, 7.7–12 min for a typical 30-segment job, with 8 running in parallel. Every segment was labelled
(`unlabelled_after=0`). The corpus now has 570 L1 segments; three Brackenridge captions were later corrected
(`l1-corrections-2026-09-27`, see [`benchmark/OPEN_QUESTIONS.md`](../benchmark/OPEN_QUESTIONS.md)).

The plan estimated about $0.24 per segment, about $95 for this wave. Cheaper future options: a
smaller model for L1 (with Opus reserved for L2 judging), or caption-only L1.

## Measured wall times (M-series Mac, MPS)

| Stage | Typical | Notes |
|---|---|---|
| ingest | 3–26 s per bag | scales with bag size (15 GB wave: ~3 min total) |
| motion, clearance | < 12 s per bag | |
| detections | ~200–400 s per recording | measured while frames were also running |
| frames.siglip2 | ~6–7 min per 100 s of Spot | five cameras; Jackal is front-only |
| annotate.l1 | 8–12 min per 30-segment job | 8 in parallel; $0.21 per segment |
