# Physics-Informed Neural Networks for Biological 2D+t Reaction-Diffusion Systems



Physics-informed neural networks (PINNs) provide a powerful framework for learning governing equations from data [[1]](#raissi-et-al-2019). Biologically-informed neural networks (BINNs) extend that idea by preserving known differential-operator structure while learning constitutive terms via trainable subnetworks [[2]](#lagergren-et-al-2020).

We introduce a framework that combines data preprocessing, BINN-based equation learning, and symbolic regression for explicit, closed-form equation discovery directly from experimental data—demonstrated here by recovering $2\mathrm{D}{+}t$ reaction–diffusion models of lung cancer cell population dynamics from time-lapse microscopy.

Readily applicable to other spatio-temporal systems, this framework provides a practical and interpretable tool for fast analytic equation discovery from data.

For more detail see the manuscript (reference withheld for anonymous review).


## Pipeline

The schematic of the pipeline used to apply the PINN framework to the experimental data is shown in [pipeline_schematic.png](pipeline_schematic.png).

## Repository structure

```text
pinn-reaction-diffusion-2dt/
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── pipeline_schematic.png
└── main/
	├── README.md
	├── data/
	│   ├── split_csvs/              # experimental CSV splits
	│   └── python_v2/               # data-processing pipeline
	│       ├── README.md
	│       ├── config/
	│       ├── components/
	│       ├── exec/
	│       ├── modules/
	│       └── runs/                # generated
	├── dataObj_v2/                  # binned data objects
	├── binn_v2/
	│   ├── README.md
	│   ├── python/
	│   │   ├── pipeline/
	│   │   │   ├── config/
	│   │   │   ├── components/
	│   │   │   └── exec/
	│   │   └── Modules/
	│   │       ├── README.md
	│   │       ├── dataClass.py
	│   │       ├── Models/
	│   │       │   ├── BuildBINNs_2D.py
	│   │       │   └── BuildMLP.py
	│   │       └── Utils/
	│   │           ├── Gradient.py
	│   │           ├── ModelWrapper.py
	│   │           └── PDESolver_2D.py
	│   └── runs/                    # generated
	├── binn_v2_models_updated/      # trained models
	└── JN/
		├── README.md
		├── paper_notebook.ipynb
		├── paper_helpers/
		└── outputs/
		    ├── sr_updated/             # cached SR fits
		    └── figures/, tables/, ...  # generated
```

Directories marked `# generated` are not tracked in git. They are created when you run
the pipeline or the notebook; everything else ships with the repository, so
`paper_notebook.ipynb` can be run without first retraining anything.

## Environment setup

Create the project environment first:

```bash
conda env create -f environment.yml
conda activate pinn-rd-2dt
```

Then open the notebook and select this same environment as kernel.

### Platform

Tested on macOS with Apple Silicon (`arm64`). Check the architecture conda will
resolve for:

```bash
python -c "import platform; print(platform.machine())"
```

On an Apple Silicon Mac this should report `arm64`. If it reports `x86_64`,
conda is an Intel build running under Rosetta and will install `osx-64`
packages, which may not reproduce the results here. Install an `arm64` conda,
or create the environment with
`CONDA_SUBDIR=osx-arm64 conda env create -f environment.yml`.

## Documentation guide

Use the README files below for focused guidance on each stage of the pipeline:

- [README.md](README.md): high-level overview of the paper context, repository layout, and top-level setup.
- [main/README.md](main/README.md): full training pipeline guide, including how to generate data stores and train BINN models.
- [main/binn_v2/README.md](main/binn_v2/README.md): BINN training pipeline usage and experiment configuration points.
- [main/binn_v2/python/Modules/README.md](main/binn_v2/python/Modules/README.md): code-level model/loss documentation, including constraints, loss weights, and PDE customization points.
- [main/data/python_v2/README.md](main/data/python_v2/README.md): data-preprocessing/data-object generation details and run controls.
- [main/data/split_csvs/README.md](main/data/split_csvs/README.md): details on CSV dataset contents and filename metadata semantics.
- [main/JN/README.md](main/JN/README.md): notebook workflow guide, including the paper-aligned notebook structure used to reproduce result figures.
- [main/JN/paper_helpers/README.md](main/JN/paper_helpers/README.md): detailed overview of paper-ready analysis, plotting, SR pipeline, and forward simulation helper modules.

## Optional checks

Quick validation commands are documented in [main/README.md](main/README.md), since they are run from the training pipeline context.

## Citation

Citation withheld for double-blind review.

## References

1. Raissi M, Perdikaris P, Karniadakis GE. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*. 2019;378:686–707. [https://doi.org/10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

2. Lagergren JH, Nardini JT, Baker RE, Simpson MJ, Flores KB. Biologically-informed neural networks guide mechanistic modeling from sparse experimental data. *PLoS Computational Biology*. 2020;16(12):e1008462. [https://doi.org/10.1371/journal.pcbi.1008462](https://doi.org/10.1371/journal.pcbi.1008462)