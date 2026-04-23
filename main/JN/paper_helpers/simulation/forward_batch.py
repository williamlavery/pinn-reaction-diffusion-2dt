from __future__ import annotations

# =========================
# Imports
# =========================
import os
import sys
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

# sys.path.append("../")
# sys.path.append("../../")

from .pde_solver import PDE_RHS_2D, PDE_sim
from ..io.paths import build_save_base, predict_u_pred


# =========================
# Internal helpers
# =========================
def _prepare_head_runner(
    model: Any,
    device: str,
) -> Callable[[Callable[[torch.Tensor], torch.Tensor], np.ndarray], np.ndarray]:
    """
    Create a helper that evaluates a model head on a numpy array of u values.

    Parameters
    ----------
    model : Any
        Model containing parameters whose dtype determines input casting.
    device : str
        Torch device string, e.g. "cpu" or "cuda".

    Returns
    -------
    Callable
        Function with signature run_head(head, u) -> np.ndarray.
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    on_cuda = device.startswith("cuda")

    @torch.inference_mode()
    def run_head(head: Callable[[torch.Tensor], torch.Tensor], u: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(u, dtype=dtype, device=device).reshape(-1, 1).contiguous()
        y = head(x).reshape(-1)
        return y.to("cpu", non_blocking=on_cuda).numpy()

    return run_head


def _build_diffusion_and_growth(
    model: Any,
    run_head: Callable[[Callable[[torch.Tensor], torch.Tensor], np.ndarray], np.ndarray],
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Build callable diffusion and growth functions from a trained model.

    Parameters
    ----------
    model : Any
        Model with diffusion and growth heads and scaling factors D_scale, G_scale.
    run_head : Callable
        Helper for evaluating a model head on numpy inputs.

    Returns
    -------
    tuple
        (diffusion_func, growth_func)
    """
    Dmax = model.D_scale
    Gmax = model.G_scale

    def diffusion_func(u: np.ndarray) -> np.ndarray:
        return Dmax * run_head(model.diffusion, u)

    def growth_func(u: np.ndarray) -> np.ndarray:
        return Gmax * run_head(model.growth, u)

    return diffusion_func, growth_func


def _select_initial_condition(
    *,
    init_type: str,
    species_label: str,
    u_pred: Optional[np.ndarray],
    data_obj: Any,
) -> np.ndarray:
    """
    Select the initial condition for forward simulation.

    Parameters
    ----------
    init_type : str
        Either "pred" for model-predicted initial condition or "data" for observed data.
    species_label : str
        Species identifier, expected to be "green" or "red".
    u_pred : np.ndarray or None
        Predicted state tensor. Required when init_type == "pred".
    data_obj : Any
        Dataset object containing observed arrays.

    Returns
    -------
    np.ndarray
        Initial condition array for the requested species.

    Raises
    ------
    ValueError
        If init_type or species_label is invalid.
    """
    if init_type not in {"pred", "data"}:
        raise ValueError(f"Unknown init_type: {init_type}. Expected 'pred' or 'data'.")

    if species_label not in {"green", "red"}:
        raise ValueError(f"Unknown species_label: {species_label}. Expected 'green' or 'red'.")

    if init_type == "pred":
        if u_pred is None:
            raise ValueError("u_pred must be provided when init_type='pred'.")
        return u_pred[..., 0]

    if species_label == "green":
        return data_obj.u_green[..., 0]
    return data_obj.u_red[..., 0]


def _require_mapping_entry(mapping: Mapping[Any, Any], key: Any, mapping_name: str) -> Any:
    """
    Fetch a value from a mapping with a clear error message if the key is absent.
    """
    if key not in mapping:
        available_keys = list(mapping.keys())
        raise KeyError(
            f"Missing key {key!r} in {mapping_name}. "
            f"Available keys: {available_keys}"
        )
    return mapping[key]


# =========================
# Public API: learned D/G
# =========================
def forward_simulate_learned_DG(
    keys: Sequence[Any],
    dataObj_dict: Mapping[Any, Any],
    binn_ES_list: Sequence[Any],
    binnSplitSeeds: Sequence[Any],
    binn_exts: Mapping[Any, Mapping[Any, Mapping[Any, Any]]],
    binn_models_dics_ES: Mapping[Any, Mapping[Any, Mapping[Any, Any]]],
    speciesLabel: str,
    *,
    key_idx: int = 2,
    device: str = "cpu",
    forward_sim_bool: bool = True,
    overwrite_forward: bool = False,
    init_type: str = "pred",
    base_dir: str = "plots",
    numtsim: int = 500,
    numxsim1: int = 100,
    numxsim2: int = 100,
    clear: bool = True,
) -> Dict[Any, Dict[Any, str]]:
    """
    Forward-simulate a PDE using learned neural-network diffusion D(u)
    and growth G(u).

    Parameters
    ----------
    keys : Sequence[Any]
        Dataset keys.
    dataObj_dict : Mapping[Any, Any]
        Maps keys to data objects.
    binn_ES_list : Sequence[Any]
        Experimental settings to iterate over.
    binnSplitSeeds : Sequence[Any]
        Split seeds to iterate over.
    binn_exts : Mapping
        Nested metadata used for file naming.
    binn_models_dics_ES : Mapping
        Nested dictionary of trained model wrappers indexed by ES/key/seed.
    speciesLabel : str
        Species label, expected to be "green" or "red".
    key_idx : int, default=2
        Index into `keys` selecting which dataset key to simulate.
    device : str, default="cpu"
        Torch device string.
    forward_sim_bool : bool, default=True
        Global switch controlling whether simulation is run.
    overwrite_forward : bool, default=False
        If True, overwrite existing forward-simulation files.
    init_type : str, default="pred"
        Initial-condition source: "pred" or "data".
    base_dir : str, default="plots"
        Base directory for saved outputs.
    numtsim : int, default=500
        Number of temporal points used by the PDE solver.
    numxsim1 : int, default=100
        Number of spatial grid points in x1.
    numxsim2 : int, default=100
        Number of spatial grid points in x2.
    clear : bool, default=True
        Passed through to the PDE solver.

    Returns
    -------
    dict
        Nested mapping:
            results[binn_ES][binnSplitSeed] = saved_filename
    """
    key = keys[key_idx]
    data_obj = _require_mapping_entry(dataObj_dict, key, "dataObj_dict")

    print(f"[INFO] Using key: {key} (index={key_idx}), device={device}")

    results: Dict[Any, Dict[Any, str]] = {}

    for binn_ES in binn_ES_list:
        print(f"\n##### Forward learned DG for ES = {binn_ES} #####")
        results[binn_ES] = {}

        binn_exts_for_es = _require_mapping_entry(binn_exts, binn_ES, "binn_exts")
        models_for_es = _require_mapping_entry(binn_models_dics_ES, binn_ES, "binn_models_dics_ES")

        binn_exts_for_key = _require_mapping_entry(binn_exts_for_es, key, f"binn_exts[{binn_ES!r}]")
        models_for_key = _require_mapping_entry(models_for_es, key, f"binn_models_dics_ES[{binn_ES!r}]")

        for binnSplitSeed in binnSplitSeeds:
            print(f"\n--- Simulation: key={key}, ES={binn_ES}, seed={binnSplitSeed} ---")

            binn_ext = _require_mapping_entry(
                binn_exts_for_key,
                binnSplitSeed,
                f"binn_exts[{binn_ES!r}][{key!r}]",
            )
            model_wrapper = _require_mapping_entry(
                models_for_key,
                binnSplitSeed,
                f"binn_models_dics_ES[{binn_ES!r}][{key!r}]",
            )
            model = model_wrapper.model

            filename = build_save_base(
                binn_ext,
                f"ufwd_init{init_type}.npy",
                base_dir=base_dir,
            )
            results[binn_ES][binnSplitSeed] = filename

            if not forward_sim_bool:
                print("[SKIP] forward_sim_bool=False")
                continue

            if os.path.exists(filename) and not overwrite_forward:
                print(f"[SKIP] File exists: {filename}")
                continue

            u_pred = predict_u_pred(
                model_wrapper,
                dataobj=data_obj,
                device=device,
            )

            run_head = _prepare_head_runner(model, device)
            diffusion_func, growth_func = _build_diffusion_and_growth(model, run_head)

            u_init = _select_initial_condition(
                init_type=init_type,
                species_label=speciesLabel,
                u_pred=u_pred,
                data_obj=data_obj,
            )

            print(f"[INFO] Forward simulating species: {speciesLabel}")

            u_fwd = PDE_sim(
                PDE_RHS_2D,
                u_init,
                data_obj.x1,
                data_obj.x2,
                data_obj.t,
                diffusion_func,
                growth_func,
                numtsim=numtsim,
                numxsim1=numxsim1,
                numxsim2=numxsim2,
                clear=clear,
            )

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.save(filename, u_fwd)
            print(f"[SAVE] {filename}")

    return results


# =========================
# Public API: SR tag builder
# =========================
def build_forward_sr_settings_tag(
    *,
    population_size: int,
    niter: int,
    label: str = "DG",
    suffix: str = "updated_mean",
    **kwargs,
) -> str:
    """
    Build a minimal descriptive save tag for SR-based forward simulations
    focusing only on population and iterations.
    """
    parts = [
        f"{label}fwdSR",
        f"pop{population_size}",
        f"niter{niter}",
    ]

    if suffix:
        parts.append(str(suffix))

    return "_".join(str(part) for part in parts)


# =========================
# Public API: SR forward batch
# =========================
def run_forward_sr_batch(
    *,
    keys: Sequence[Any],
    binnSplitSeeds: Sequence[Any],
    dataObj_dict: Mapping[Any, Any],
    binn_models_dics_ES: Mapping[Any, Mapping[Any, Mapping[Any, Any]]],
    binn_exts: Mapping[Any, Mapping[Any, Mapping[Any, Any]]],
    d_lambdas: Mapping[Any, Callable[[np.ndarray], np.ndarray]],
    g_lambdas: Mapping[Any, Callable[[np.ndarray], np.ndarray]],
    u_scale_SRs: Mapping[Any, float],
    ES: Any,
    speciesLabel: str,
    population_size: int,
    niter: int,
    percentiles: Sequence[float],
    density_scaling: Any,
    y_scaling: Optional[Any] = None,
    use_sf_for_predictions: Optional[bool] = None,
    seeds: Optional[Iterable[int]] = None,
    loss: Optional[str] = None,
    unary_operators: Optional[Sequence[str]] = None,
    binary_operators: Optional[Sequence[str]] = None,
    constraints: Optional[Mapping[str, Any]] = None,
    maxsize: Optional[int] = None,
    parsimony: Optional[float] = None,
    model_selection: Optional[str] = None,
    device: str = "cpu",
    forward_SR_bool: bool = True,
    overwrite_SR_forward: bool = False,
    init_type: str = "pred",
    u_scale_type: Optional[Any] = None,
    base_dir: str = "plots",
    numtsim: int = 500,
    numxsim1: int = 100,
    numxsim2: int = 100,
    clear: bool = True,
    save_suffix: str = "updated_mean",
) -> Dict[Any, Dict[Any, np.ndarray]]:
    """
    Run or load forward PDE simulations using symbolic-regression diffusion
    and growth functions for each dataset key and split seed.

    Parameters
    ----------
    keys : Sequence[Any]
        Dataset keys.
    binnSplitSeeds : Sequence[Any]
        Split seeds to iterate over.
    dataObj_dict : Mapping[Any, Any]
        Maps keys to data objects.
    binn_models_dics_ES : Mapping
        Nested trained model wrappers used when `init_type="pred"`.
    binn_exts : Mapping
        Nested metadata used for file naming.
    d_lambdas : Mapping[Any, Callable]
        SR diffusion functions keyed by dataset key.
    g_lambdas : Mapping[Any, Callable]
        SR growth functions keyed by dataset key.
    u_scale_SRs : Mapping[Any, float]
        Scaling factors applied to u during SR forward simulation, keyed by dataset key.
    ES : Any
        Experimental setting identifier.
    speciesLabel : str
        Species label, expected to be "green" or "red".
    population_size, niter, percentiles, density_scaling : required
        Metadata used both in simulation bookkeeping and save-tag generation.
    y_scaling, use_sf_for_predictions, seeds, loss, unary_operators,
    binary_operators, constraints, maxsize, parsimony, model_selection : optional
        Additional SR metadata included in the save tag.
    device : str, default="cpu"
        Torch device string used when generating predicted initial conditions.
    forward_SR_bool : bool, default=True
        Global switch controlling whether SR forward results are run/loaded.
    overwrite_SR_forward : bool, default=False
        If True, overwrite existing saved SR forward simulations.
    init_type : str, default="pred"
        Initial-condition source: "pred" or "data".
    u_scale_type : optional
        Descriptor used in the save tag. Defaults to `density_scaling` if None.
    base_dir : str, default="plots"
        Base directory for saved outputs.
    numtsim, numxsim1, numxsim2 : int
        PDE solver resolution settings.
    clear : bool, default=True
        Passed through to the PDE solver.
    save_suffix : str, default="updated_mean"
        Final suffix added to the SR save tag.

    Returns
    -------
    dict
        Nested mapping:
            forward_results[key][binnSplitSeed] = u_fwd_SR_normalized
    """
    if u_scale_type is None:
        u_scale_type = density_scaling

    forward_results: Dict[Any, Dict[Any, np.ndarray]] = {}

    if not forward_SR_bool:
        print("[SKIP] forward_SR_bool=False")
        return forward_results

    save_tag = build_forward_sr_settings_tag(
        population_size=population_size,
        niter=niter,
        label="DG",
        suffix=save_suffix,
    )
    print(f"\n##### Forward SR simulations for ES = {ES} #####")

    binn_exts_for_es = _require_mapping_entry(binn_exts, ES, "binn_exts")
    models_for_es = _require_mapping_entry(binn_models_dics_ES, ES, "binn_models_dics_ES")

    for key in keys:
        D_SR_func = _require_mapping_entry(d_lambdas, key, "d_lambdas")
        G_SR_func = _require_mapping_entry(g_lambdas, key, "g_lambdas")
        u_sf = _require_mapping_entry(u_scale_SRs, key, "u_scale_SRs")

        data_obj = _require_mapping_entry(dataObj_dict, key, "dataObj_dict")
        binn_exts_for_key = _require_mapping_entry(binn_exts_for_es, key, f"binn_exts[{ES!r}]")
        models_for_key = _require_mapping_entry(models_for_es, key, f"binn_models_dics_ES[{ES!r}]")

        forward_results[key] = {}

        for binnSplitSeed in binnSplitSeeds:
            print(f"\n--- Forward SR: key={key}, ES={ES}, seed={binnSplitSeed} ---")

            binn_ext = _require_mapping_entry(
                binn_exts_for_key,
                binnSplitSeed,
                f"binn_exts[{ES!r}][{key!r}]",
            )

            filename = build_save_base(
                binn_ext,
                f"{save_tag}.npy",
                base_dir=base_dir,
            )

            file_exists = os.path.exists(filename)

            if file_exists and not overwrite_SR_forward:
                print(f"[LOAD] Existing SR forward simulation: {filename}")
                u_fwd_SR_normalized = np.load(filename)
                forward_results[key][binnSplitSeed] = u_fwd_SR_normalized
                continue

            if file_exists and overwrite_SR_forward:
                print(f"[OVERWRITE] Re-simulating existing file: {filename}")
            else:
                print(f"[SIMULATE] File not found: {filename}")

            u_pred: Optional[np.ndarray] = None
            if init_type == "pred":
                model_wrapper = _require_mapping_entry(
                    models_for_key,
                    binnSplitSeed,
                    f"binn_models_dics_ES[{ES!r}][{key!r}]",
                )
                u_pred = predict_u_pred(
                    model_wrapper,
                    dataobj=data_obj,
                    device=device,
                )

            u_init = _select_initial_condition(
                init_type=init_type,
                species_label=speciesLabel,
                u_pred=u_pred,
                data_obj=data_obj,
            )

            print(f"[INFO] Forward simulating species: {speciesLabel} (SR)")

            u_fwd_SR = PDE_sim(
                PDE_RHS_2D,
                u_init * u_sf,
                data_obj.x1,
                data_obj.x2,
                data_obj.t,
                D_SR_func,
                G_SR_func,
                numtsim=numtsim,
                numxsim1=numxsim1,
                numxsim2=numxsim2,
                clear=clear,
            )

            u_fwd_SR_normalized = u_fwd_SR / u_sf

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.save(filename, u_fwd_SR_normalized)
            print(f"[SAVE] {filename}")

            forward_results[key][binnSplitSeed] = u_fwd_SR_normalized

    return forward_results