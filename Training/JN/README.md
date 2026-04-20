# JN

This folder contains the main analysis notebook and helper modules for exploring generated data and trained BINN models.

## What this folder contains

- `experim.ipynb`: end-to-end analysis notebook (loading artifacts, plotting, and result inspection).
- `paper_helpers/`: reusable utilities imported by the notebook.
- `outputs/`: saved figures and notebook-generated artifacts.

## Before you run the notebook

Make sure these artifacts exist first:
- data objects in `Training/dataObj_v2/...`
- trained BINN models in `Training/binn_v2_models/...`

If they are missing, generate them by running:
1. `python -m data.python_v2.exec.data__sim_v2`
2. `python -m binn_v2.python.pipeline.exec.binn_exec`

## Recommended usage

1. Open `experim.ipynb` in VS Code or Jupyter.
2. Select the Python environment used for this repository.
3. Run cells from top to bottom.
4. Check `outputs/` for exported figures and artifacts.

