# data/python_v2

`data/python_v2` generates simulation-ready `dataObj` artifacts used by the BINN training pipeline.

The goal of this package is to make data generation reproducible, inspectable, and simple to rerun.

## What this folder contains

- `exec/data__sim_v2.py`: CLI entry point to generate data artifacts.
- `config/experiment_config.py`: default parameter grid, schedules, and paths.
- `components/`: orchestration utilities (pathing, run metadata, simulation loop).
- `modules/`: scientific/data-construction modules used by each simulation.
- `runs/`: timestamped run logs and summaries.

## Inputs and outputs

- Main input CSVs: `Training/data/split_csvs/`
- Main simulation settings: `config/experiment_config.py`
- Generated artifacts: `Training/dataObj_v2/.../data_obj.npy`
- Run metadata directory: `Training/data/python_v2/runs/run_YYYYMMDD_HHMMSS/`
- Run metadata files in each run directory: `run_header.json`, `records.jsonl`, `summary.json`
- Global run index: `Training/data/python_v2/runs/run_index.jsonl`

## Quick start

Run from the `Training` directory:

```bash
python -m data.python_v2.exec.data__sim_v2
```

Useful flags:

```bash
# Always regenerate artifacts, even if they already exist.
python -m data.python_v2.exec.data__sim_v2 --overwrite

# Never overwrite existing artifacts and enable plotting.
python -m data.python_v2.exec.data__sim_v2 --no-overwrite --plot
```

## How to customize experiments

Edit `config/experiment_config.py` to change:
- parameter ranges,
- hour schedules,
- overwrite/plot defaults,
- data and artifact root directories.

A good workflow is:
1. Run once with defaults.
2. Inspect `summary.json` and `records.jsonl`.
3. Adjust config and rerun.
