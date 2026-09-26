# SCAND retrieval benchmark

The Python environment is managed exclusively with `uv`. Raw recordings and generated outputs are intentionally excluded from version control.

```bash
uv sync --project scand --frozen
uv run --project scand --frozen python --version
uv run --project scand --frozen python scand/build_dashboard.py
```

Add or remove dependencies through `uv add --project scand` and `uv remove --project scand` so `pyproject.toml` and `uv.lock` remain synchronized. Do not install packages directly into `scand/.venv`.

Directory roles:

- `raw/`: untouched source downloads.
- `derived/`: reproducible conversions and extracted media.
- `annotations/`: benchmark annotation sources.
- `results/`: generated experiment results.
