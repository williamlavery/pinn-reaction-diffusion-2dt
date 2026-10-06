# data/python_v2

`data/python_v2` generates ready `dataObj` artifacts used to store the experimental data in human friendly way and one in which that is used by the BINN training pipeline.

The goal of this component is to make data fully inspectable and simple to rerun.

## What this folder contains

- `exec/data__sim_v2.py`: CLI entry point to generate data artifacts.
- `config/experiment_config.py`: default parameter grid, schedules, and paths.
- `components/`: orchestration utilities (pathing, run metadata, simulation loop).
- `modules/`: scientific/data-construction modules used by each simulation.
- `runs/`: timestamped run logs and summaries.

## Inputs and outputs

- Main input CSVs: `main/data/split_csvs/`
- Main simulation settings: `config/experiment_config.py`
- Generated artifacts: `main/dataObj_v2/.../data_obj.npy`
- Run metadata directory: `main/data/python_v2/runs/run_YYYYMMDD_HHMMSS/`
- Run metadata files in each run directory: `run_header.json`, `records.jsonl`, `summary.json`
- Global run index: `main/data/python_v2/runs/run_index.jsonl`

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
- hour ranges of experimental data,
- overwrite/plot defaults,
- data and artifact root directories.

