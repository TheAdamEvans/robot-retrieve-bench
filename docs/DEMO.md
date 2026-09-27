# Demo runbook

The operational side of the demo: what to start, what to check, what to click, and what to do when something
fails. The narrative walkthrough is the offline presentation (`presentation/`, with presenter notes in
`presentation/PRESENTER.md`). The live UI is the optional second half.

## One hour before

```bash
uv sync                                    # from the repository root
uv run alloy-index status                  # every recording ✓ except gated ($) or n/a (–) cells
uv run pytest -q                           # contracts, layering, module evals
uv run uvicorn alloy_server.server:app --port 8787 &
curl -s localhost:8787/v1/health | python3 -m json.tool   # recordings, pipelines, resident bytes, generator: configured
```

**Warm the program cache** so live PROGRAM, HYBRID and FUSED_V searches answer in tens of milliseconds, not 5–8 s.
The cache key includes the prompt and the recordings list, so re-warm after any reindex.

```bash
python3 - <<'EOF'
import json, urllib.request
for f in ["demo5", "demo5_para", "compose_test"]:
    for q in json.load(open(f"benchmark/queries/{f}.json")):
        body = json.dumps({"utterance": q["utterance"], "pipelineId": "PROGRAM", "k": 10}).encode()
        req = urllib.request.Request("http://localhost:8787/v1/search", body, {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=60))
        print(f'{q["queryId"]:40} {r.get("status")}')
EOF
```

Open <http://localhost:8787>, run one query per config and play one clip. The first clip for a span is encoded on
demand and cached under `cache/clips/`.

## What to show in the live UI

1. **One question, every config.** Try "Spot slows down as a group of people appears ahead". Switch TAGS → EMBED
   → FUSED → PROGRAM → HYBRID. Point at the status chip (`answered` versus `answered_unverified`), the per-stage
   report and the cost line (bytes read, tokens, cache hit).
2. **Merged results.** The live UI merges overlapping windows from one recording into one result ("N windows
   merged · 13 s"), and the clip plays the whole span. The benchmark scores the unmerged list.
3. **Honest degradation.** Ask a front-camera question scoped to `Bass_Garage_134`, which has no front camera. Its
   clauses come back `UNKNOWN (sensor absent in this log)`, never "none found".
4. **Unexpressible input.** "legs close up": PROGRAM abstains, and HYBRID ranks by similarity and says the results
   are unverified.
5. **A receipt.** The last-safe-evidence query shows each sensor's latest message before the cutoff, with its age.

## Adding a bag live

The data wave showed every step is one command. Put a bag in `raw/`, then:

```bash
uv run alloy-index plan                          # shows only the new recording's artifacts as missing
uv run alloy-index run --recordings <ID> --stages ingest,providers.motion,providers.clearance
uv run alloy-index status
```

| Step (90 s Spot bag, M-series Mac) | Wall time |
|---|---|
| ingest: recognise, `by_sensor` MCAPs, timeline, intake | ~5 s |
| motion and clearance providers | ~3 s |
| RT-DETR detections, 10 Hz front camera | ~3–4 min |
| SigLIP2 frames, front 10 Hz plus body cameras at native 4.5 Hz | ~5–7 min |
| windows, tags, apply.fused, pipelines | < 1 min |
| L1 labels (optional, `--annotate`, Opus 5.5) | ~8–12 min (one job per ≤30 segments, parallel); ~$0.21 per 4 s segment |

The detection and frame figures were measured while both ran at once, so treat them as upper bounds. Live, run
ingest and the cheap providers (about 10 s), show `plan` and `status`, and let the encoders finish in the background
or use a pre-indexed bag. Restart the server afterwards to load the new recording.

## If something breaks

| Symptom | Do this |
|---|---|
| OpenAI unreachable or slow | Cached queries still answer. Uncached ones return `insufficient_evidence` or `unverified` with "retry the search". Stay on TAGS, EMBED and FUSED, which make no LLM call. |
| Server won't start | Check `ALLOY_BUNDLE` points at `bundles/dev` and `alloy-index status` is green. `ALLOY_WARM=0` skips the encoder warm-up. |
| Clip is slow the first time | It is encoding once; later plays are cached. |
| Anything else | Switch to the offline presentation. It needs no service. |

## Known live behaviours (eval v9, prompt v3)

- **Receipt demo:** use `last_safe_evidence:canonical` or `:para3`. For `:para1` ("right before the robot starts
  moving again after standing still…") the generated program adds a ≥1 s standstill before the onset. Nothing
  satisfies that composition, so the answer is `none_found_exhaustive`: exhaustive for the *program*, wrong for the
  *question*. That is the verified-clauses versus correct-interpretation distinction, if you want an example of it.
- **The six dream questions** (`benchmark/challenges/l1_compositions_v1`) take 11–38 s to generate cold; they are
  cached now. Two of them have no qualifying episode in scope by exhaustive ground truth ("repeated corrections",
  "staying slow after the route clears"), but the system still returns partial answers. Say so if you show them.
- The demo server keeps its own program cache (`cache/programs/`). Its keys differ from the eval's, because the
  few-shot pool is serialised differently, so a live program can differ from the evaluated one. Warm it after any
  reindex or prompt change.

## Honest caveats to say out loud

- The judgments are model-produced (Opus labeller, audited tools), not human.
- The corpus is small: 16 recordings, 2 held out. Read intent-group CIs as directional.
- The 9 newest recordings are harder for every config, hand-written programs included (compose_test oracle nDCG
  0.74 on the original 7, 0.38 on the new 9). Structured signals transfer imperfectly to new scenes.
- FUSED results are unverified by design: a vector lookup has nothing to check.
- ESTIMATED spatial claims come from nominal camera models, because SCAND has no calibration.
