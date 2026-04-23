# paper_helpers

Helper modules used by `Training/JN/paper_notebook.ipynb` for paper-ready analysis, plotting, forward simulation, and symbolic-regression workflows.

## Visualization helpers (figure generation and diagnostics)

These modules are primarily for plotting and reporting outputs from data, trained BINNs, forward simulations, and SR results.

### `plotting/`

- `snapshots.py`
  - 2D density heatmaps and cell-position scatter snapshots (single/multi-panel paper figures).
- `density.py`
  - Midline slice comparisons for data vs BINN predictions (and optional forward simulations), including uncertainty bands across splits.
- `training_dynamics.py`
  - Tracks learned diffusion/growth curve evolution across saved epochs and highlights best-epoch behavior.
- `loss_trajectories.py`
  - Loss trajectory visualization across early-stopping settings and repeated runs.
- `training_summary.py`
  - Aggregates validation/training losses and runtime statistics into summary plots/tables.
- `forward_eval.py`
  - Visualization and diagnostics for forward simulation trajectories and errors (data vs BINN vs forward, optional SR-forward).
- `sr_eval.py`
  - Final SR-vs-ensemble plotting utilities for diffusion/growth evaluations and histogram overlays.

### `styles/`

- `matplotlib_rc.py`
  - Applies shared Matplotlib defaults for consistent paper figure styling.

## Pipeline helpers (modeling and computation)

These modules are part of the computational pipeline used to run forward simulation and symbolic regression, rather than pure plotting.

### Forward simulation pipeline (`simulation/`)

- `pde_solver.py`
  - Implements numerical PDE components for forward simulation:
    - sparse diffusion operator assembly (`Du_2d`),
    - PDE RHS wrapper (`PDE_RHS_2D`),
    - time integration/orchestration (`PDE_sim`).
- `forward_batch.py`
  - Batch orchestration for forward simulation experiments.
  - Runs learned-parameter forward simulations and symbolic-regression-based forward simulations, with deterministic save tagging and caching-friendly outputs.

### Symbolic regression pipeline (`sr/`)

- `runner.py`
  - Single-run symbolic regression workflow:
    - data scaling,
    - ensemble target assembly,
    - multi-seed SR fitting,
    - expression post-processing and simplification.
- `batch.py`
  - Multi-case symbolic-regression runner with disk caching, timing, and optional plotting hooks.
  - Collects per-key symbolic expressions/lambdas/predictions for downstream analysis.
- `analysis.py`
  - SR post-analysis workflow:
    - per-seed ranking and filtering,
    - weighted/unweighted error summaries,
    - expression-form counting,
    - diagnostic evaluation outputs.

## Shared support helpers

These modules support both the visualization and pipeline code paths.

### `io/`

- `discover.py`
  - Finds `data_obj.npy` files recursively and converts discovered paths into a metadata table (`paths_to_df`).
  - Supports simple key/value filtering (`condense_df`) used to select subsets of datasets.
- `paths.py`
  - Shared utility functions for save-path construction, overwrite-safe saving, tensor conversion, histogram diagnostics, and model prediction reshaping.
  - Centralizes common helper logic used by plotting, SR, and simulation modules.

## Notes

- `__init__.py` files are intentionally lightweight package markers.
- The helpers are designed to be imported directly from notebook cells, with most configuration passed at call time rather than hard-coded globally.
