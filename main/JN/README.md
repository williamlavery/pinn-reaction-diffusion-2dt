# JN

This folder contains the main analysis notebook and helper modules for exploring generated data and trained BINN models.

## What this folder contains

- `paper_notebook.ipynb`: end-to-end analysis notebook (loading artifacts, plotting, and result inspection).
- `paper_helpers/`: reusable utilities imported by the notebook.
- `outputs/`: saved figures and notebook-generated artifacts.

## Before you run the notebook

Make sure these artifacts exist first:
- data objects in `main/dataObj_v2/...`
- trained BINN models in `main/binn_v2_models/...`

If they are missing, generate them by running:
1. `python -m data.python_v2.exec.data__sim_v2`
2. `python -m binn_v2.python.pipeline.exec.binn_exec`

## Recommended usage

1. Open `paper_notebook.ipynb` in VS Code or Jupyter.
2. Select the Python environment used for this repository.
3. Run cells from top to bottom.
4. Check `outputs/` for exported figures and artifacts.

## Notebook structure (paper-aligned)

Execution of JN loads all models used and generates all plots used in results part of paper.

The notebook is [main/JN/paper_notebook.ipynb](paper_notebook.ipynb). It follows the same high-level logic as the paper:

1. Load and configure experimental replicates (keys `2_5`, `2_3`, `3_1`) with their time ranges and metadata. Build data dictionaries storing the loaded data objects `data_obj.npy`.
2. Produce the initial data plots that the paper uses to ground interpretation:
	- density heatmaps
	- scatter plots of observed green-cell coordinates (positions)
3. Load trained BINN models and run prediction/evaluation sections.
4. Generate training dynamics, losses, and prediction-vs-data plots.
5. Run symbolic-diffusion/equation-analysis blocks for interpretable post-processing.

In other words, the notebook starts with observational scatter/coordinate structure and then moves through model-based and symbolic analyses, mirroring the paper's results narrative.


## Statement on trained models

Execution of `main/JN/paper_notebook.ipynb` loads the models and SR results used in the paper. Reproducing these models and SR expressions from scratch may introduce minor numerical differences due to variations in torch and PySR versions. Therefore, for full reproducibility, the models and SR outputs generated using the `main/environment.yml` configuration are provided in main/binn_v2_models_updated and main/JN/outputs/sr_updated. These differences do not affect the results reported in the paper, although they may lead to slight variations in the candidate expressions seen across the ten SR runs. Importantly, the dominant expression form each replicate remain the same. 

