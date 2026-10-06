# binn_v2

`binn_v2` trains BINN models from generated `dataObj` artifacts.

This package is designed so a user can:
- run the training pipeline from a single command,
- inspect reproducible run metadata,
- and extend experiment settings in one configuration module.

## What this folder contains

- `python/pipeline/exec/binn_exec.py`: CLI entry point for BINN training runs.
- `python/pipeline/config/experiment_config.py`: default parameter grid and paths.
- `python/Modules/`: BINN model and utility code used by the pipeline.
- `runs/`: timestamped run logs and summaries.

## Inputs and outputs

- Expected inputs: `main/dataObj_v2/.../data_obj.npy`
- Trained model outputs: `main/binn_v2_models/...`
- Run metadata directory: `main/binn_v2/runs/run_YYYYMMDD_HHMMSS/`
- Run metadata files in each run directory: `run_header.json`, `records.jsonl`, `summary.json`
- Global run index: `main/binn_v2/runs/run_index.jsonl`

## Quick start

Run from the `Training` directory:

```bash
python -m binn_v2.python.pipeline.exec.binn_exec
```

Run a short validation job (first N scheduled runs only):

```bash
python -m binn_v2.python.pipeline.exec.binn_exec --max-runs 1
```

## How to customize experiments

Edit `python/pipeline/config/experiment_config.py` to change:
- parameter grids,
- training behavior,
- data/model output roots.

This keeps experiment logic centralized and makes runs easier to reproduce.
