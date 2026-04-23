# main

This repository contains a reproducible workflow for:
1. processing experimental data,
2. training BINN models on the data,
3. analyzing and visualizing results in a notebook.

It is organized as a pipeline, where each stage has its own README and command-line entry point.

## `main` map

- `data/python_v2/`: data-processing pipeline (creates `dataObj`  objects).
- `dataObj_v2/`: generated data objects used for model training.
- `binn_v2/`: BINN training pipeline and model code.
- `binn_v2_models/`: trained models used in paper
- `binn_v2_models_updated/`: trained models on latest module versions
- `JN/`: analysis notebook and helper modules.

<!--
## Quick start

Run these commands from the `main` directory:

```bash
# 1) Generate data  objects
python -m data.python_v2.exec.data__sim_v2

# 2) Train BINN models
python -m binn_v2.python.pipeline.exec.binn_exec

# 3) Explore outputs in the notebook
# Open JN/paper_notebook.ipynb and run top-to-bottom
```
-->

## How to generate data stores and train models

Note: Configuration files are pre-set to replicate the paper's full pipeline by default. This includes all data preprocessing and BINN training across all replicates and training-validation splits.

Run all commands from [main](.):

```bash
cd main
```

1. Generate data objects from CSV splits:

```bash
python -m data.python_v2.exec.data__sim_v2
```

This writes `.npy` objects storing the raw experimental data under [main/dataObj_v2](dataObj_v2) and run metadata under [main/data/python_v2/runs](data/python_v2/runs).

2. Train BINN models:

```bash
python -m binn_v2.python.pipeline.exec.binn_exec
```

This writes models under [main/binn_v2_models](binn_v2_models) and run metadata under [main/binn_v2/runs](binn_v2/runs).

3. Recreate figures/tables used in the paper workflow:

Open [main/JN/paper_notebook.ipynb](JN/paper_notebook.ipynb) and run cells top-to-bottom. Outputs are written under [main/JN/outputs](JN/outputs).

## Optional checks

Data generation test:

```bash
python -m data.python_v2.exec.data__sim_v2 --no-overwrite
```

Short BINN training test:

```bash
python -m binn_v2.python.pipeline.exec.binn_exec --max-runs 1
```

This is to test the code before performing training for all replicates and training-validation splits.

## Where to begin

- New users: start with `data/python_v2/README.md`, then `binn_v2/README.md`, then `JN/README.md`.
- Returning users: inspect run summaries under each pipeline's `runs/` folder to compare experiments.

## Reproducibility notes

Both pipelines write run metadata as:
- `run_header.json` (environment and run setup)
- `records.jsonl` (per-run records)
- `summary.json` (aggregate run summary)

This makes each run auditable and easier to compare over time.