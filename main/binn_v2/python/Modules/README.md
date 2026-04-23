# Modules (BINN core code)

This folder contains the core BINN model code used by the training pipeline and paper analyses.

## What is in this module

- `Models/BuildBINNs_2D.py`
  - Core 2D+t BINN building blocks: state network (`u_MLP`), diffusion network (`D_MLP`), growth network (`G_MLP`), PDE residual losses, and constraint penalties. Mirrors [[1]](#lagergren-et-al-2020) for consistency in BINN literature.
  - Main model class: `BINN_2d`.
- `Models/BuildMLP.py`
  - Generic MLP constructor used by the BINN heads. Mirrors [[1]](#lagergren-et-al-2020) for consistency in BINN literature.
- `Utils/Gradient.py`
  - Autograd helper for first/second derivatives used in PDE terms and monotonicity penalties. Mirrors [[1]](#lagergren-et-al-2020) for consistency in BINN literature.
- `Utils/ModelWrapper.py`
  - Training loop, early stopping, checkpointing, and logging helpers. Mirrors [1] for consistency in BINN literature.
- `Utils/PDESolver_2D.py`
  - PDE solving/simulation utilities used in evaluation and helper workflows. Mirrors [[1]](#lagergren-et-al-2020) for consistency in BINN literature.
- `dataClass.py`
  - Data container classes passed through training/evaluation.


## Intentional use of work from [1]
We intentially match script styles from the original BINN paper by [1]
- Source code : https://github.com/jlager/BINNs/tree/master/Modules/





## Where key parameters are set

There are three levels: defaults, pipeline wiring, and model consumption.

1. Defaults (experiment-level)

- File: `Training/binn_v2/python/pipeline/config/experiment_config.py`
- Class: `BinnParams`
- Key defaults include:
  - `all_constraints`
  - `bc_bool`
  - `num_pde_samples`
  - `data_loss_labels`
  - `d_constraint_bools`
  - `d_bounds`
  - architecture sizes (`binn_usizes`, `binn_dsizes`, `binn_gsizes`)
  - optimization/training controls (`lr`, `batch_size`, `es_values`, `epochs`)

2. Pipeline wiring (parameter mapping)

- File: `Training/binn_v2/python/pipeline/exec/binn_exec.py`
- Function: `build_base_payload`
- The defaults above are mapped into runtime dictionaries:
  - `model_params["binn_model_params"]["binn_construction_params"]`
  - `model_params["binn_model_params"]["pde_loss_params"]`
  - `model_params["binn_model_params"]["BNdata_loss_params"]`

3. Model consumption (actual behavior)

- File: `Training/binn_v2/python/Modules/Models/BuildBINNs_2D.py`
- In `BINN_2d.__init__`, these values are read and used to build the model and loss terms.

## Loss weights used in the model

In `BINN_2d.__init__` (`BuildBINNs_2D.py`) the key weights are set as:

- `self.surface_weight = 1e0`
- `self.pde_weight = 1e0`
- `self.D_weight = 1e3 / self.D_max`
- `self.dDdu_weight = self.D_weight * self.K`
- `self.G_weight = 1e3 / self.G_max` (when growth is enabled)
- `self.dGdu_weight = self.G_weight * self.K` (when growth is enabled)
- `self.bc_weight = 1e0` (BC-enabled path)

How this relates to the paper:

- The paper models a 2D+t reaction-diffusion system where observed dynamics are fit by balancing data mismatch and PDE residual terms.
- `surface_weight` and `pde_weight` are the direct balancing factors between supervised fit and physics residual.
- `D_weight`/`G_weight` and derivative penalties (`dDdu_weight`, `dGdu_weight`) scale biological plausibility constraints so learned constitutive terms stay in realistic ranges and monotonicity trends.

## Key default parameters (explained)

This section explains the defaults currently used by the BINN pipeline and what changes when each option is enabled.

### 1) `all_constraints`

- Default value:
  - `0` (from `BinnParams.all_constraints` in `Training/binn_v2/python/pipeline/config/experiment_config.py`)
- Runtime mapping:
  - `binn_construction_params["allConstraints"]` in `Training/binn_v2/python/pipeline/exec/binn_exec.py`
- Model usage:
  - read in `BINN_2d.__init__` (`BuildBINNs_2D.py`) as `self.allConstraints`
  - gate in `apply_constraints`: `if not self.allConstraints: return`

Consequence at default (`0`):

- No diffusion/growth bound or monotonicity penalties are added.
- The PDE residual and supervised data terms still train normally.

When enabled (`1`):

- The following penalties are activated in `apply_constraints`:
  - diffusion range penalty (`alpha_D_min` / `alpha_D_max`)
  - optional diffusion monotonicity penalty via `dD/du` (only if `D_constraint_bool` is also enabled)
  - growth range penalty (`alpha_G_min` / `alpha_G_max`)
  - growth monotonicity penalty via `dG/du`

### 2) `bc_bool`

- Default value:
  - `0` (from `BinnParams.bc_bool` in `experiment_config.py`)
- Runtime mapping:
  - `pde_loss_params["BCbool"]` in `binn_exec.py`
- Dispatch location:
  - `bn_model_pde_loss_func` in `Training/binn_v2/python/pipeline/components/simulate.py`

Consequence at default (`0`):

- The training uses `pde_loss_without_bc_2d`.
- No boundary-condition term is added.

When enabled (`1`):

- The training route switches to `pde_loss_with_bc_2d`.
- A boundary-condition penalty term is included (scaled by `self.bc_weight`) using boundary samples.

### 3) `d_constraint_bools` and dependence on `all_constraints`

- Default value:
  - `[0]` (from `BinnParams.d_constraint_bools` in `experiment_config.py`)
- Runtime mapping:
  - `binn_construction_params["D_constraint_bool"]` in `binn_exec.py`
- Model usage:
  - stored as `self.D_constraint_bool` in `BINN_2d.__init__`
  - checked inside `apply_constraints`

Important dependency:

- `D_constraint_bool` has practical effect only when `all_constraints=1`.
- If `all_constraints=0`, `apply_constraints` exits early, so `dD/du` penalties are not evaluated even if `D_constraint_bool=1`.

### 4) `d_bounds`

- Default value:
  - `[1]` (from `BinnParams.d_bounds` in `experiment_config.py`)
- Runtime mapping:
  - `binn_construction_params["D_bound"]` in `binn_exec.py`
- Model usage:
  - passed to `D_MLP(..., D_bound=D_bound)`
  - sets diffusivity upper scale (`self.diffusion.max` -> `self.D_max`)

Consequence even when bounds are not enforced (`all_constraints=0`):

- `D_bound` still affects model scaling because `self.D_max` is used in core computations:
  - PDE residual term scale (`RHS = self.D_max * div + ...`)
  - loss scaling (`self.D_weight = 1e3 / self.D_max`)
- So `d_bounds` changes training scale and optimization dynamics even when explicit constraint penalties are disabled.

### 5) `num_pde_samples`

- Default value:
  - `[100]` (from `BinnParams.num_pde_samples` in `experiment_config.py`)
- Runtime mapping:
  - `pde_loss_params["numPDEsamples"]` in `binn_exec.py`
- Model usage:
  - assigned to `self.num_samples` in `BINN_2d.__init__`
  - consumed by PDE input sampling (random/LHS/Sobol samplers)

Effect of increasing `num_pde_samples`:

- More collocation points per loss evaluation.
- Typically improves PDE residual coverage/stability (lower stochastic variance in physics term).
- Increases per-step compute and memory cost.
- In practice: larger values can give smoother physics guidance but may slow each epoch.

## Loss-function customization points

Data loss (MSE vs GLS):

- Select label in config: `BNdataLossFuncLabel`
- Wiring and dispatch:
  - `Training/binn_v2/python/pipeline/components/simulate.py`
  - `bn_model_data_loss_func`
- Implementations in:
  - `BuildBINNs_2D.py`: `data_loss_MSE`, `data_loss_GLS`

PDE loss with/without BC:

- Toggle in config: `BCbool`
- Wiring and dispatch:
  - `simulate.py`: `bn_model_pde_loss_func`
- Implementations in:
  - `BuildBINNs_2D.py`: `pde_loss_without_bc_2d`, `pde_loss_with_bc_2d`

## If a different PDE is assumed

For a different mechanistic PDE, the main edits should be in `BuildBINNs_2D.py`:

1. Update residual definition

- Edit `pde_loss_without_bc_2d` (and `pde_loss_with_bc_2d` if BC is used).
- Replace current residual terms (`ut`, diffusion divergence, growth reaction) with your new operator terms.

2. Update constitutive heads if needed

- If PDE requires different learned fields, add/modify heads analogous to `D_MLP` and `G_MLP`.
- Wire them into `BINN_2d.__init__` and residual computation.

3. Update constraints for new terms

- Extend `apply_constraints` with bounds/monotonicity rules for any new constitutive outputs.

4. Update boundary/initial condition treatment

- BC sampling and no-flux logic are in:
  - `generate_bc_inputs_2d`
  - `bc_no_flux_loss_2d`
  - `apply_BC_2d`
- Modify these if your PDE uses different boundary operators.

5. Ensure pipeline uses your loss path

- Keep/extend dispatch in `simulate.py` (`bn_model_pde_loss_func`) so training calls the intended residual function.

## Practical guidance

For paper-consistent reproduction, keep:

- current residual form in `pde_loss_without_bc_2d`
- current constraint toggles and weights


Then change one switch at a time (loss type, constraints, PDE form) to isolate the effect of each modeling decision.

## References

1. Lagergren JH, Nardini JT, Baker RE, Simpson MJ, Flores KB. Biologically-informed neural networks guide mechanistic modeling from sparse experimental data. *PLoS Computational Biology*. 2020;16(12):e1008462. [https://doi.org/10.1371/journal.pcbi.1008462](https://doi.org/10.1371/journal.pcbi.1008462)