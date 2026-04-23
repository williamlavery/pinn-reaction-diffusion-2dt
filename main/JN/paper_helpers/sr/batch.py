from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import dill
import matplotlib.pyplot as plt
import numpy as np
from sympy import lambdify, symbols

from pysr import PySRRegressor
from .runner import SymbolicRegressionRunner


dill.settings["recurse"] = True


class SymbolicRegressionBatchRunner:
    """
    Run symbolic regression over multiple cases with optional caching and plotting.

    Parameters
    ----------
    keys
        Iterable of case keys to process.
    dataObj_dict
        Mapping key -> data object.
    binn_models_dics_ES
        Nested dictionary indexed like binn_models_dics_ES[ES][key].
    binn_exts
        Nested dictionary indexed like binn_exts[ES][key][binnSplitSeeds[-1]].
    ES
        Early-stopping key/index used to access binn_models_dics_ES and binn_exts.
    binnSplitSeeds
        Sequence used to choose the final binn_ext entry.
    build_save_base
        Callable with signature:
        build_save_base(binn_ext, filename, base_dir="plots") -> str
    """

    def __init__(
        self,
        *,
        keys: Iterable[Any],
        dataObj_dict: Dict[Any, Any],
        binn_models_dics_ES: Dict[Any, Dict[Any, Any]],
        binn_exts: Dict[Any, Dict[Any, Any]],
        ES: Any,
        binnSplitSeeds: Sequence[Any],
        build_save_base: Any,
        device: str = "cpu",
        density_scaling: Any = "dimensionless",
        y_scaling: Any = "dimensionless",
        use_sf_for_predictions: bool = True,
        population_size: int = 20,
        percentiles: Sequence[float] = (5, 95),
        niter: int = 10,
        seeds: Iterable[int] = range(10),
        loss: str = "(x - y)^2",
        unary_operators: Optional[Sequence[str]] = None,
        binary_operators: Optional[Sequence[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        maxsize: int = 10,
        parsimony: float = 0,
        model_selection: str = "score",
        sig_figs: int = 6,
        label: str = "diff",
        speciesLabel: str = "green",
        base_dir: str = "plots",
        persist_hof_files: bool = False,
        verbose: bool = True,
    ) -> None:
        self.keys = list(keys)
        self.dataObj_dict = dataObj_dict
        self.binn_models_dics_ES = binn_models_dics_ES
        self.binn_exts = binn_exts
        self.ES = ES
        self.binnSplitSeeds = list(binnSplitSeeds)
        self.build_save_base = build_save_base

        self.device = device
        self.density_scaling = density_scaling
        self.y_scaling = y_scaling
        self.use_sf_for_predictions = use_sf_for_predictions
        self.population_size = population_size
        self.percentiles = list(percentiles)
        self.niter = niter
        self.seeds = list(seeds)
        self.loss = loss
        self.unary_operators = list(unary_operators) if unary_operators is not None else ["log", "exp", "sqrt"]
        self.binary_operators = list(binary_operators) if binary_operators is not None else ["+", "-", "*", "/"]
        self.constraints = constraints if constraints is not None else {}
        self.maxsize = maxsize
        self.parsimony = parsimony
        self.model_selection = model_selection
        self.sig_figs = sig_figs
        self.label = label
        self.speciesLabel = speciesLabel
        self.base_dir = base_dir
        self.persist_hof_files = bool(persist_hof_files)
        self.verbose = verbose

        self._validate_label()
        self._validate_scaling_modes()

        self.response_sym_simp_rounded_by_key: Dict[Any, Any] = {}
        self.response_lambdas_by_key: Dict[Any, Any] = {}
        self.response_vals_by_key: Dict[Any, Any] = {}
        self.u_vals_np_by_key: Dict[Any, Any] = {}
        self.response_sym_vals_by_key: Dict[Any, Any] = {}
        self.u_scale_SRs: Dict[Any, Any] = {}
        self.y_scale_SRs: Dict[Any, Any] = {}
        self.response_SR_runs_by_key: Dict[Any, Any] = {}

        # Backward-compatible ordered views
        self.response_sym_simp_rounded_list: List[Any] = []
        self.response_lambdas: List[Any] = []
        self.response_vals_list: List[Any] = []
        self.u_vals_np_list: List[Any] = []
        self.response_sym_vals_list: List[Any] = []
        self.response_SR_runs: List[Any] = []

        # Timing stats
        self.load_times: List[float] = []
        self.compute_times: List[float] = []
        self.n_seen_loads = 0
        self.n_seen_computes = 0

    # ------------------------------------------------------------
    # Validation / naming helpers
    # ------------------------------------------------------------

    def _validate_label(self) -> None:
        valid_labels = {"diff", "grow"}
        if self.label not in valid_labels:
            raise ValueError(
                f"Unknown label: {self.label}. "
                f"Expected one of {sorted(valid_labels)}."
            )

    def _validate_scaling_modes(self) -> None:
        valid_density_scalings = {"dimensionless", "dimensionfull"}
        valid_y_scalings = {"dimensionless", "dimensionfull"}

        if self.density_scaling not in valid_density_scalings:
            raise ValueError(
                f"Unknown density_scaling: {self.density_scaling}. "
                f"Expected one of {sorted(valid_density_scalings)}."
            )

        if self.y_scaling not in valid_y_scalings:
            raise ValueError(
                f"Unknown y_scaling: {self.y_scaling}. "
                f"Expected one of {sorted(valid_y_scalings)}."
            )

    @property
    def response_symbol(self) -> str:
        return "D" if self.label == "diff" else "G"

    @property
    def response_function_label(self) -> str:
        return f"{self.response_symbol}(u)"

    @property
    def response_run_key(self) -> str:
        return f"{self.response_symbol}_SR_runs_by_key"

    @property
    def response_sym_expr_key(self) -> str:
        return f"{self.response_symbol}_sym_simp_rounded_by_key"

    @property
    def response_lambda_key(self) -> str:
        return f"{self.response_symbol}_lambdas_by_key"

    @property
    def response_vals_key(self) -> str:
        return f"{self.response_symbol}_vals_by_key"

    @property
    def response_sym_vals_key(self) -> str:
        return f"{self.response_symbol}_sym_vals_by_key"

    @staticmethod
    def format_time(seconds: float) -> str:
        seconds = int(seconds)
        h, r = divmod(seconds, 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def resolve_expected_u_scale(density_scaling: Any, dataobj: Any) -> float:
        if density_scaling == "dimensionfull":
            return 1e6 * dataobj.u_green_max
        if density_scaling == "dimensionless":
            return 1.0
        raise ValueError(
            f"Unknown density_scaling: {density_scaling}. "
            "Expected 'dimensionless' or 'dimensionfull'."
        )

    @staticmethod
    def resolve_expected_y_scale(y_scaling: Any, y_values: Any) -> float:
        y_values = np.asarray(y_values, dtype=float).ravel()

        if y_scaling == "dimensionless":
            sf = float(np.max(np.abs(y_values)))
        elif y_scaling == "dimensionfull":
            sf = 1.0
        else:
            raise ValueError(
                f"Unknown y_scaling: {y_scaling}. "
                "Expected 'dimensionless' or 'dimensionfull'."
            )

        if not np.isfinite(sf) or sf == 0.0:
            raise ValueError(f"Resolved y scaling factor must be finite and nonzero, got {sf}.")
        return sf

    @staticmethod
    def _require_mapping_entry(mapping: Mapping[Any, Any], key: Any, mapping_name: str) -> Any:
        if key not in mapping:
            raise KeyError(
                f"Missing key {key!r} in {mapping_name}. "
                f"Available keys: {list(mapping.keys())}"
            )
        return mapping[key]

    def _refresh_ordered_views(self) -> None:
        """
        Maintain backward-compatible list outputs in the order of self.keys.
        """
        self.response_sym_simp_rounded_list = [
            self.response_sym_simp_rounded_by_key[key]
            for key in self.keys
            if key in self.response_sym_simp_rounded_by_key
        ]
        self.response_lambdas = [
            self.response_lambdas_by_key[key]
            for key in self.keys
            if key in self.response_lambdas_by_key
        ]
        self.response_vals_list = [
            self.response_vals_by_key[key]
            for key in self.keys
            if key in self.response_vals_by_key
        ]
        self.u_vals_np_list = [
            self.u_vals_np_by_key[key]
            for key in self.keys
            if key in self.u_vals_np_by_key
        ]
        self.response_sym_vals_list = [
            self.response_sym_vals_by_key[key]
            for key in self.keys
            if key in self.response_sym_vals_by_key
        ]
        self.response_SR_runs = [
            self.response_SR_runs_by_key[key]
            for key in self.keys
            if key in self.response_SR_runs_by_key
        ]

    def build_case_filename(self, key: Any) -> Tuple[Any, str]:
        binn_ext = self.binn_exts[self.ES][key][self.binnSplitSeeds[-1]]
        filename = self.build_save_base(
            binn_ext,
            (
                f"{self.label}_SR_densityScaling{self.density_scaling}"
                f"_yScaling{self.y_scaling}"
                f"_predSF{self.use_sf_for_predictions}"
                f"_pop{self.population_size}"
                f"_con{self.constraints}"
                f"_maxsize{self.maxsize}"
                f"_parsimony{self.parsimony}"
                f"_model_selection{self.model_selection}"
                f"_niter{self.niter}"
                f"_percentiles{self.percentiles}"
                f"_seeds{list(self.seeds)}"
                f"_loss{self.loss}"
                f"_ops{self.unary_operators}.pkl"
            ),
            base_dir=self.base_dir,
        )
        return binn_ext, filename

    def _make_regressor(self) -> PySRRegressor:
        return PySRRegressor(
            model_selection=self.model_selection,
            niterations=self.niter,
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            elementwise_loss=f"loss(x, y, w) = {self.loss}",
            population_size=self.population_size,
            maxsize=self.maxsize,
            verbosity=self.verbose,
            random_state=42,
            deterministic=True,
            parallelism="serial",
            parsimony=self.parsimony,
        )

    def _fit_single_case(self, key: Any) -> Any:
        SRmodel_response = self._make_regressor()
        return SymbolicRegressionRunner(
            SRmodel=SRmodel_response,
            dataobj=self.dataObj_dict[key],
            orig_dic=self.binn_models_dics_ES[self.ES][key],
            device=self.device,
            density_scaling=self.density_scaling,
            y_scaling=self.y_scaling,
            use_sf_for_predictions=self.use_sf_for_predictions,
            percentiles=self.percentiles,
            sig_figs=self.sig_figs,
            seeds=self.seeds,
            persist_hof_files=self.persist_hof_files,
            label=self.label,
            speciesLabel=self.speciesLabel,
        ).fit(verbose=False)

    def _extract_scaling_info(
        self,
        response_SR: Any,
        expected_u_scale_SR: float,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        if isinstance(response_SR, dict):
            resolved_u_sf = response_SR.get("u_sf", expected_u_scale_SR)

            if "y_sf" in response_SR and response_SR["y_sf"] is not None:
                resolved_y_sf = response_SR["y_sf"]
            else:
                y_for_fallback = response_SR.get("f_vals_np_raw", response_SR.get("f_vals", None))
                resolved_y_sf = (
                    self.resolve_expected_y_scale(self.y_scaling, y_for_fallback)
                    if y_for_fallback is not None else None
                )

            runner_density_scaling = response_SR.get("density_scaling", None)
            runner_y_scaling = response_SR.get("y_scaling", None)
            runner_use_sf_for_predictions = response_SR.get("use_sf_for_predictions", None)

        else:
            resolved_u_sf = getattr(response_SR, "u_sf", expected_u_scale_SR)

            if hasattr(response_SR, "y_sf") and response_SR.y_sf is not None:
                resolved_y_sf = response_SR.y_sf
            else:
                y_for_fallback = getattr(response_SR, "f_vals_np_raw", getattr(response_SR, "f_vals_np", None))
                resolved_y_sf = (
                    self.resolve_expected_y_scale(self.y_scaling, y_for_fallback)
                    if y_for_fallback is not None else None
                )

            runner_density_scaling = getattr(response_SR, "density_scaling", None)
            runner_y_scaling = getattr(response_SR, "y_scaling", None)
            runner_use_sf_for_predictions = getattr(response_SR, "use_sf_for_predictions", None)

        return (
            resolved_u_sf,
            resolved_y_sf,
            runner_density_scaling,
            runner_y_scaling,
            runner_use_sf_for_predictions,
        )

    def _store_outputs(self, key: Any, response_SR: Any) -> Dict[str, Any]:
        if isinstance(response_SR, dict):
            self.response_sym_simp_rounded_by_key[key] = response_SR["sym_expr_simplified_rounded"]
            self.response_lambdas_by_key[key] = response_SR["f_lambda"]
            self.response_vals_by_key[key] = response_SR.get("f_vals_np_raw", response_SR.get("f_vals"))
            self.u_vals_np_by_key[key] = response_SR["u_vals_np"]
            self.response_sym_vals_by_key[key] = response_SR["f_lambda_vals"]

            per_seed = response_SR.get("per_seed", None)
            final_pysr_complexity = response_SR.get("final_pysr_complexity", None)
            final_sympy_complexity = response_SR.get("final_rounded_sympy_node_complexity", None)
        else:
            self.response_sym_simp_rounded_by_key[key] = response_SR.sym_expr_simplified_rounded
            self.response_lambdas_by_key[key] = response_SR.f_lambda
            self.response_vals_by_key[key] = getattr(response_SR, "f_vals_np_raw", response_SR.f_vals_np)
            self.u_vals_np_by_key[key] = response_SR.u_vals_np
            self.response_sym_vals_by_key[key] = response_SR.f_lambda_vals

            per_seed = getattr(response_SR, "per_seed", None)
            final_pysr_complexity = getattr(response_SR, "final_pysr_complexity", None)
            final_sympy_complexity = getattr(response_SR, "final_rounded_sympy_node_complexity", None)

        self.response_SR_runs_by_key[key] = response_SR
        self._refresh_ordered_views()

        return {
            "per_seed": per_seed,
            "final_pysr_complexity": final_pysr_complexity,
            "final_sympy_complexity": final_sympy_complexity,
        }

    def _print_case_summary(
        self,
        *,
        i: int,
        n_total: int,
        key: Any,
        action: str,
        iter_start: float,
        total_start_time: float,
        resolved_u_sf: Any,
        resolved_y_sf: Any,
        runner_density_scaling: Any,
        runner_y_scaling: Any,
        runner_use_sf_for_predictions: Any,
        final_pysr_complexity: Any,
        final_sympy_complexity: Any,
        per_seed: Any,
    ) -> None:
        avg_load = np.mean(self.load_times) if self.load_times else 0.0
        avg_compute = np.mean(self.compute_times) if self.compute_times else 0.0

        n_done = i + 1
        n_remaining = n_total - n_done
        n_seen = self.n_seen_loads + self.n_seen_computes

        if n_seen > 0:
            frac_load = self.n_seen_loads / n_seen
            est_remaining_loads = n_remaining * frac_load
            est_remaining_computes = n_remaining * (1.0 - frac_load)
        else:
            est_remaining_loads = 0.0
            est_remaining_computes = float(n_remaining)

        eta = est_remaining_loads * avg_load + est_remaining_computes * avg_compute
        total_elapsed = time.time() - total_start_time
        iter_time = time.time() - iter_start

        print(f"\n[{i+1}/{n_total}] key={key}")
        print(
            f"{action.upper()} took {self.format_time(iter_time)} | "
            f"Elapsed: {self.format_time(total_elapsed)} | "
            f"ETA: {self.format_time(eta)} "
            f"(estimated remaining: {est_remaining_loads:.1f} load, "
            f"{est_remaining_computes:.1f} compute)"
        )
        print(
            f"Scaling: density_scaling={runner_density_scaling}, "
            f"y_scaling={runner_y_scaling}, "
            f"use_sf_for_predictions={runner_use_sf_for_predictions}, "
            f"resolved_u_sf={resolved_u_sf}, "
            f"resolved_y_sf={resolved_y_sf}"
        )
        print(
            f"Final/reference complexity: PySR={final_pysr_complexity}, "
            f"rounded_sympy_nodes={final_sympy_complexity}"
        )

        if per_seed is not None:
            for run in per_seed:
                print(
                    f"  Seed {run['seed']}: "
                    f"PySR complexity={run.get('pysr_complexity', None)} | "
                    f"SymPy nodes={run.get('sympy_node_complexity', None)} | "
                    f"Expr={run.get('sym_expr_post', None)}"
                )

    def run(
        self,
        *,
        plot: bool = False,
        force_recompute: bool = False,
    ) -> Dict[Any, Any]:
        """
        Run all cases, loading cached results when available unless force_recompute=True.

        Parameters
        ----------
        plot
            If True, call plot_runs() after processing.
        force_recompute
            If True, ignore existing cache files and recompute everything.

        Returns
        -------
        dict
            Mapping:
                key -> symbolic-regression run object/dict
        """
        total_start_time = time.time()
        n_total = len(self.keys)

        # Reset stored outputs for a fresh run
        self.response_sym_simp_rounded_by_key = {}
        self.response_lambdas_by_key = {}
        self.response_vals_by_key = {}
        self.u_vals_np_by_key = {}
        self.response_sym_vals_by_key = {}
        self.u_scale_SRs = {}
        self.y_scale_SRs = {}
        self.response_SR_runs_by_key = {}

        self.response_sym_simp_rounded_list = []
        self.response_lambdas = []
        self.response_vals_list = []
        self.u_vals_np_list = []
        self.response_sym_vals_list = []
        self.response_SR_runs = []

        self.load_times = []
        self.compute_times = []
        self.n_seen_loads = 0
        self.n_seen_computes = 0

        for i, key in enumerate(self.keys):
            iter_start = time.time()
            _, filename = self.build_case_filename(key)

            if self.verbose:
                print(f"\n[{i+1}/{n_total}] key={key}")
                print("Checking file existence for this case...")

            expected_u_scale_SR = self.resolve_expected_u_scale(
                density_scaling=self.density_scaling,
                dataobj=self.dataObj_dict[key],
            )

            file_exists = os.path.isfile(filename) and not force_recompute

            if file_exists:
                if self.verbose:
                    print(f"Loading from {filename}...")
                t0 = time.time()
                with open(filename, "rb") as f:
                    response_SR = dill.load(f)
                dt = time.time() - t0
                self.load_times.append(dt)
                self.n_seen_loads += 1
                action = "load"
            else:
                if self.verbose:
                    print(f"Computing and saving to {filename}...")
                t0 = time.time()
                response_SR = self._fit_single_case(key)
                dt = time.time() - t0
                self.compute_times.append(dt)
                self.n_seen_computes += 1
                action = "compute"

                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "wb") as f:
                    dill.dump(response_SR, f, protocol=dill.HIGHEST_PROTOCOL)

                if self.verbose:
                    print(f"Saved → {filename}")

            (
                resolved_u_sf,
                resolved_y_sf,
                runner_density_scaling,
                runner_y_scaling,
                runner_use_sf_for_predictions,
            ) = self._extract_scaling_info(response_SR, expected_u_scale_SR)

            self.u_scale_SRs[key] = resolved_u_sf
            self.y_scale_SRs[key] = resolved_y_sf

            stored = self._store_outputs(key, response_SR)

            if self.verbose:
                self._print_case_summary(
                    i=i,
                    n_total=n_total,
                    key=key,
                    action=action,
                    iter_start=iter_start,
                    total_start_time=total_start_time,
                    resolved_u_sf=resolved_u_sf,
                    resolved_y_sf=resolved_y_sf,
                    runner_density_scaling=runner_density_scaling,
                    runner_y_scaling=runner_y_scaling,
                    runner_use_sf_for_predictions=runner_use_sf_for_predictions,
                    final_pysr_complexity=stored["final_pysr_complexity"],
                    final_sympy_complexity=stored["final_sympy_complexity"],
                    per_seed=stored["per_seed"],
                )

        if plot:
            self.plot_runs()

        return self.response_SR_runs_by_key

    def plot_runs(
        self,
        *,
        show_per_seed: bool = True,
        show_final: bool = True,
        show_nn_ensemble: bool = True,
        figsize: Tuple[float, float] = (7, 5),
        legend: bool = True,
    ) -> None:
        """
        Plot all stored symbolic regression runs.
        """
        for key in self.keys:
            if key not in self.response_SR_runs_by_key:
                continue

            response_SR = self.response_SR_runs_by_key[key]
            plt.figure(figsize=figsize)

            if isinstance(response_SR, dict):
                u_vals = response_SR["u_vals_np"]
                f_vals_np = response_SR.get("f_vals_np_raw", response_SR.get("f_vals"))
                per_seed = response_SR.get("per_seed", None)
                final_vals = response_SR["f_lambda_vals"]
                final_comp = response_SR.get("final_pysr_complexity", None)
                use_sf_pred = response_SR.get("use_sf_for_predictions", False)
                runner_y_scaling = response_SR.get("y_scaling", None)
                x0 = symbols("x0")
            else:
                u_vals = response_SR.u_vals_np
                f_vals_np = getattr(response_SR, "f_vals_np_raw", response_SR.f_vals_np)
                per_seed = getattr(response_SR, "per_seed", None)
                final_vals = response_SR.f_lambda_vals
                final_comp = getattr(response_SR, "final_pysr_complexity", None)
                use_sf_pred = getattr(response_SR, "use_sf_for_predictions", False)
                runner_y_scaling = getattr(response_SR, "y_scaling", None)
                x0 = response_SR.symbol

            if show_nn_ensemble:
                plt.plot(
                    u_vals,
                    f_vals_np,
                    color="red",
                    linewidth=2,
                    label="NN ensemble",
                )

            if show_per_seed and per_seed is not None:
                for run in per_seed:
                    sym_expr_post = run["sym_expr_post"]
                    f_lambda_seed = lambdify(x0, sym_expr_post, modules="numpy")
                    comp = run.get("pysr_complexity", None)

                    plt.plot(
                        u_vals,
                        f_lambda_seed(u_vals),
                        alpha=0.5,
                        linestyle="--",
                        label=f"seed {run['seed']} (C={comp})",
                    )

            if show_final:
                plt.plot(
                    u_vals,
                    final_vals,
                    color="blue",
                    linewidth=2,
                    label=f"Final SR (C={final_comp})",
                )

            plt.title(
                f"Key {key} | label={self.label}, "
                f"density_scaling={self.density_scaling}, "
                f"y_scaling={runner_y_scaling}, pred_sf={use_sf_pred}"
            )
            plt.xlabel("u")
            plt.ylabel(self.response_function_label)

            if legend:
                plt.legend()

            plt.show()

    def get_results(self) -> Dict[str, Any]:
        """
        Return accumulated outputs in a single dictionary.

        Returns both:
        - generic keys independent of label
        - dynamic keys using D_* when label='diff' and G_* when label='grow'

        Primary storage is now keyed by dataset key. Ordered list views are also
        included for backward compatibility.
        """
        self._refresh_ordered_views()

        results = {
            "label": self.label,
            "response_symbol": self.response_symbol,
            "response_function_label": self.response_function_label,
            "persist_hof_files": self.persist_hof_files,
            "keys": list(self.keys),
            "response_SR_runs_by_key": self.response_SR_runs_by_key,
            "response_sym_simp_rounded_by_key": self.response_sym_simp_rounded_by_key,
            "response_lambdas_by_key": self.response_lambdas_by_key,
            "response_vals_by_key": self.response_vals_by_key,
            "u_vals_np_by_key": self.u_vals_np_by_key,
            "response_sym_vals_by_key": self.response_sym_vals_by_key,
            "u_scale_SRs": self.u_scale_SRs,
            "y_scale_SRs": self.y_scale_SRs,
            "load_times": self.load_times,
            "compute_times": self.compute_times,
            # backward-compatible ordered views
            "response_SR_runs": self.response_SR_runs,
            "response_sym_simp_rounded_list": self.response_sym_simp_rounded_list,
            "response_lambdas": self.response_lambdas,
            "response_vals_list": self.response_vals_list,
            "u_vals_np_list": self.u_vals_np_list,
            "response_sym_vals_list": self.response_sym_vals_list,
        }

        results[self.response_run_key] = self.response_SR_runs_by_key
        results[self.response_sym_expr_key] = self.response_sym_simp_rounded_by_key
        results[self.response_lambda_key] = self.response_lambdas_by_key
        results[self.response_vals_key] = self.response_vals_by_key
        results[self.response_sym_vals_key] = self.response_sym_vals_by_key

        return results