# Labels

Every label the benchmark and the models use, organized by **what it may be used for**, with its provenance
kept separately by **campaign**: one labelling run, with one brief, one judge and one purpose.

```
labels/
  train/        may be used for training and tuning; evaluating on these labels is a dev result, not a test result
  eval/         never trained or tuned on: the benchmark's labels and the held-out test questions
  agreement/    second opinions (blind re-judgments), used only for agreement statistics
    └─ attributes/   L1 per-segment attributes and captions   key: segmentId
       judgments/    graded window relevance per intent       key: intentGroupId | windowId
       episodes/     one intent judged against one episode    key: intentGroupId | jobId | episodeId
       shard files:  <campaign>.<job>.jsonl, one per parallel job; every record carries "campaign"
  metadata/<campaign>/
    campaign.json   purpose, judge, brief, date, cost_usd, priority (and "use" when pinned, "exhaustive")
    jobs/<job>.json what the job was given (segments, windows, leads or sweep chunks), its measured cost, turns and report
    audit.jsonl     every labeller tool call in the campaign; `seq` is the row's line in the original combined log
    sweep/          (exhaustive campaigns) the numeric sweep that defined the candidates
  .work/            ignored: briefs, CLI output, submitted batches, rendered views
```

## Rules

- **The folder is the use.** The FUSED trainer reads only `train/`, so it cannot train on a benchmark or test
  label by accident. Evaluation reads `train/` and `eval/`. Only agreement statistics read `agreement/`.
- **A label has exactly one use.** A key found in two use folders is an error.
- **Where new labels go** (`alloy_index.annotate.store.use_for`):
  - L1 attributes go to `train/`;
  - judgments and episodes for benchmark intents, and for held-out challenge questions, go to `eval/`;
  - dev challenge questions go to `train/`;
  - a campaign can pin its use in `campaign.json` (the blind re-check pins `agreement`).
- **Precedence is a campaign property.** A higher `priority` wins for the same key. Unlisted means 0. Two campaigns
  at equal priority that disagree on a key fail loudly instead of choosing. Within one shard, the later record wins.

| Campaign | Kinds | Priority | Notes |
|---|---|---|---|
| `l1-smoke` | attributes | −1 | first smoke run; superseded where l1-original labels the same segment |
| `l1-original`, `l1-wave1` | attributes | 0 | original 7 recordings; data wave 1 (9 recordings, $84.06) |
| `l1-corrections-2026-09-27` | attributes | 1 | Brackenridge 0084, 0088, 0100 re-labelled after the joystick/IMU/odometry cross-check ($1.05) |
| `l2-original` | judgments | 0 | pooled judgments for eval v7 |
| `l2-followups` | judgments | 1 | follow-up rounds; supersede l2-original where they re-grade |
| `l2-recheck` | judgments | 0 | blind 80-window re-check → `agreement/` |
| `l2-v9-topup` | judgments | −1 | fills gaps from eval v9, never overrides ($53.06) |
| `episodes-l1_compositions_v1-leads` | episodes, judgments | 0 | author leads; dev → `train/`, test → `eval/` ($17.93) |
| `episodes-l1_compositions_v1-exhaustive` | episodes, judgments | 2 | complete ground truth for the dev behavior questions ($51.25) |
| `adhoc` | – | – | tool smoke tests and ad hoc reviews; audit only |

## Reading and writing

```python
from alloy_index.annotate.store import load_labels
load_labels("judgment", ("train", "eval"))     # {key: record}, precedence applied
load_labels("attributes", ("train",))
```

The labeller writes through `scandq label put`, with `SCANDQ_CAMPAIGN` and `SCANDQ_JOB` set. The orchestrators
(`alloy-index run --stages annotate.l1 --annotate`, `alloy_trainer.annotate.l2 --execute`,
`alloy_trainer.annotate.episodes --run`) create the campaign, record each job's assignment and cost, and set both
variables.
