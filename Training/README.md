# Training Repository

This repository contains a reproducible workflow for:
1. processing experimental data,
2. training BINN models on the data,
3. analyzing and visualizing results in a notebook.

It is organized as a pipeline, where each stage has its own README and command-line entry point.

## Repository map

- `data/python_v2/`: data-processing pipeline (creates `dataObj` artifacts).
- `dataObj_v2/`: generated data artifacts used for model training.
- `binn_v2/`: BINN training pipeline and model code.
- `binn_v2_models/`: trained model artifacts.
- `JN/`: analysis notebook and helper modules.

## Quick start

Run these commands from the `Training` directory:

```bash
# 1) Generate data artifacts
python -m data.python_v2.exec.data__sim_v2

# 2) Train BINN models
python -m binn_v2.python.pipeline.exec.binn_exec

# 3) Explore outputs in the notebook
# Open JN/experim.ipynb and run top-to-bottom
```

## Where to begin

- New users: start with `data/python_v2/README.md`, then `binn_v2/README.md`, then `JN/README.md`.
- Returning users: inspect run summaries under each pipeline's `runs/` folder to compare experiments.

## Reproducibility notes

Both pipelines write run metadata as:
- `run_header.json` (environment and run setup)
- `records.jsonl` (per-run records)
- `summary.json` (aggregate run summary)

This makes each run auditable and easier to compare over time.