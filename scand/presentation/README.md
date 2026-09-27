# When search needs evidence

Open [`fleet-search.html`](fleet-search.html) directly in a browser. It is the portable presentation: a single file,
with three playable clips, real camera frames, three Mermaid diagrams rendered to SVG, sixteen chart variants, and
the frozen eval v9 data embedded. No server or network is required. Optional source links open the SCAND project.

[`story.md`](story.md) is the editable narrative. `<!-- component: ... -->` markers insert data-backed media,
charts and tables at build time. Mermaid fences render during the build; the viewer does not need Mermaid installed.
[`PRESENTER.md`](PRESENTER.md) gives a short run-of-show.

## Edit and rebuild

Requirements: Node.js, npm, and Google Chrome. Build dependencies are pinned in `package-lock.json`.

```bash
cd scand/presentation
npm ci --ignore-scripts
npm run build
npm run check
```

The build reads bundled assets, so editing the narrative does not need the raw logs, Python environment, or a
running search server. `check` opens the file in offline Chrome, exercises every results view and the latency
budget, checks image loading, plays every clip, checks navigation and mobile overflow, and captures screenshots
in the ignored `.cache/` directory.

To regenerate the data, clips and Matplotlib charts from the frozen benchmark and existing local bundle:

```bash
cd scand
uv run python presentation/prepare.py
cd presentation
npm run build
npm run check
```

`prepare.py` reuses existing rendered MP4 files; remove the particular clip under `assets/` if changing its span.
It makes no model API calls. The benchmark data comes from `benchmark/results/v9/` (the blind label re-check from
`benchmark/results/v8-correctness/`); the receipt replays v9's stored generated program on the Library-to-MLK
recording in `bundles/dev`. The behavior-question table reads the exhaustive ground truth under `annotations/`.

## Files

- `fleet-search.html`: share this file by itself.
- `story.md`: narrative and Mermaid source.
- `style.css`, `client.js`, `build.mjs`: presentation styling, controls and assembly.
- `prepare.py`: source-media and plot export.
- `assets/`: source JPEGs, MP4s, SVGs and JSON. SVG plots are also standalone exportable figures.
- `check.mjs`: offline browser verification.

Frames and clips are excerpts from SCAND (Karnan and colleagues, 2022), attributed and linked in the presentation.
The text distinguishes pooled model judgments, oracle programs, measured execution latency and reconstructed
fresh-program latency. The interactive budget is a way to inspect point estimates, not a significance claim or
a deployed routing policy.
