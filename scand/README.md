# Fleet search over robot logs: SCAND retrieval benchmark

This repo runs natural-language search over archived robot recordings. It uses 7 complete bags from the public
[SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html) dataset: Boston Dynamics Spot and Clearpath Jackal, 701 s, 6.05 GB.

Every search config runs through one pipeline runner, and one evaluation harness scores quality, speed and cost for all of them.

```
bag ─► intake (recognise robot, convert to MCAP, measure) ─► providers ─► window index ─► bundle ─► server
                                                                                          ▲
   scandq (audited views) ─► Opus labeller ─► L1 attributes / L2 pooled judgments ─► eval ─┘
```

## Layout
- **`server/` (`alloy-server`).** Proto contracts, instrumented readers, the pipeline runner, the program executor,
  causal receipts and the HTTP API. It never imports `train/`, a rule enforced by a test.
- **`train/` (`alloy-train`).** Intake, providers, indexing, the `scandq` labeller tools, the eval harness and Stage C learning.
- **`server/src/alloy_server/embodiments/*.textproto`.** Robot facts as reviewed data, one profile per robot model.
- **`benchmark/queries/`.** The frozen query sets and the oracle `QueryProgram`s, with a sha256 manifest.
- **`benchmark/OPEN_QUESTIONS.md`.** Data questions that change what the system may claim.

## Search configs (all through `alloy_server.pipeline.run`)
| Config | Pipeline |
|---|---|
| TAGS | BM25 over recording-level tags. Every window of a matching recording ties |
| EMBED | SigLIP2 text query against mean-pooled front-camera window vectors |
| PROGRAM | text → `QueryProgram` (strict JSON Schema from the proto) → executed over the full scope → verified |
| HYBRID | EMBED candidates → deterministic verification → lexicographic order (truth, then similarity) |
| FUSED | Stage C: a contrastive window encoder (image + signals) in the SigLIP2 text space. One lookup, no LLM |
| stubs | HYBRID_X, ENSEMBLE, the WeMM variants. The specs load and report `UNAVAILABLE` |

## Honest answers
Clauses are three-valued (Kleene): TRUE, FALSE or UNKNOWN, and every UNKNOWN carries a reason, such as not
indexed, calibration uncertain, stale, future-stamped or outside coverage.
- **Filtering.** A candidate is filtered only when a required clause is definitively FALSE.
- **Statuses.** Answers carry `answered`, `answered_partial`, `answered_unverified`, `insufficient_evidence` or
  `none_found_exhaustive`.
- **Superlatives.** They are asserted only when the answer is exhaustive.
- **Spatial basis.** Claims based on nominal camera models are labelled ESTIMATED.

## Quick start
```bash
uv sync
uv run python -m alloy_train.intake --bundle bundles/dev          # recognise, convert, measure
uv run python -m alloy_train.providers.run --bundle bundles/dev --providers motion,clearance,detections
uv run python -m alloy_train.index.siglip_frames --bundle bundles/dev
uv run uvicorn alloy_server.server:app --port 8787               # demo page at http://localhost:8787
uv run pytest -q server/tests
```
Raw bags, bundles, renders and results are not in version control.
