# Pedagogical Header
# Module: SR_run.py
# Purpose: Run single symbolic-regression fitting workflows.
# Used by: experim.ipynb (JN) and sibling helper modules.
# Behavior: intended to match original JN implementation.



import copy
import os
import re
import shutil
import time
import numpy as np
import torch
import sympy as sp
import math
from pathlib import Path

from sympy import symbols, Float, preorder_traversal, lambdify

# Assumes these already exist in your codebase:
# - hist_properties
# - _extract_float_constants
# - _replace_float_constants
# - _round_constants_by_variance
# - _rounding_step_from_variance
# - sym_conv
# - sym_conv_rounded



from ..io.paths import (
    _get_species_u,
    build_save_path,
    build_save_base,
    should_skip_all,
    save_figure,
    to_torch,
    hist_properties,
    hist_properties_wrapper
)


def sym_conv(sym_expr):
    """
    Replace x0 -> u and print simplified expression.
    """
    x0, u = symbols('x0 u')
    sym_expr_sub_simplified = sym_expr.subs(x0, u).simplify()
    print("Final symbolic expression for D(u):")
    print(sp.expand(sym_expr_sub_simplified))
    return sym_expr_sub_simplified



def round_numbers(expr, sig_figs=2):
    """Round all Float numbers in a SymPy expression to given significant figures."""
    def round_sf(x, s):
        return float(f"{x:.{s}g}")  # convert to float with s significant figs

    return expr.xreplace({
        n: sp.Float(round_sf(n, sig_figs))
        for n in expr.atoms(sp.Float)
    })

def sym_conv_rounded(sym_expr, sig_figs=2):
    """
    Replace x0 -> u, simplify, and round numeric values to 2 significant figures.
    """
    x0, u = symbols('x0 u')
    sym_expr_sub_simplified = sym_expr.subs(x0, u).simplify()
    sym_expr_rounded = round_numbers(sym_expr_sub_simplified, sig_figs)

    print("Final symbolic expression for D(u):")
    print(sp.expand(sym_expr_rounded))
    return sym_expr_rounded



def _extract_float_constants(expr):
    """Return all numeric Float constants from a SymPy expression, in traversal order."""
    return [float(node) for node in preorder_traversal(expr) if isinstance(node, Float)]


def _replace_float_constants(expr, new_constants):
    """Replace Float constants in traversal order with new values."""
    old_constants = [node for node in preorder_traversal(expr) if isinstance(node, Float)]
    if len(old_constants) != len(new_constants):
        raise ValueError("Mismatch between number of old and new constants.")

    repl = {
        old: Float(new)
        for old, new in zip(old_constants, new_constants)
    }
    return expr.xreplace(repl)


def _rounding_step_from_variance(var):
    """
    Choose a rounding step one order of magnitude larger than the variance.
    Example:
        var = 0.001  -> step = 0.01
        var = 0.02   -> step = 0.1
        var = 1e-5   -> step = 1e-4
    """
    if var <= 0 or not np.isfinite(var):
        return None
    return 10 ** (math.floor(math.log10(var)) + 1)


def _round_constants_by_variance(expr, const_vars):
    """
    Round each constant using a step determined by its variance across seeds.
    """
    constants = _extract_float_constants(expr)
    if len(constants) != len(const_vars):
        raise ValueError("Number of constants and variances do not match.")

    rounded = []
    for c, v in zip(constants, const_vars):
        step = _rounding_step_from_variance(v)
        if step is None:
            rounded.append(c)
        else:
            rounded.append(round(c / step) * step)

    return _replace_float_constants(expr, rounded)


class SymbolicRegressionRunner:
    """
    Run symbolic regression across multiple seeds and store all information used.

    Scaling behavior
    ----------------
    density_scaling : {"dimensionless", "dimensionfull"} or numeric
        Controls the input-density scaling convention used internally.
        - "dimensionless" -> scale factor = 1.0
        - "dimensionfull" -> scale factor = 1e6 * dataobj.u_green_max
        - numeric         -> use that numeric value directly

    y_scaling : {"dimensionless", "max_abs", "mean_abs", "std", "rms"} or numeric
        Controls the target/output scaling used internally before SR fitting.
        The fit is performed on:
            y_fit = y_raw / y_sf

        Options:
        - "dimensionless" -> scale factor = 1.0
        - "max_abs"       -> max(abs(y_raw))
        - "mean_abs"      -> mean(abs(y_raw))
        - "std"           -> std(y_raw)
        - "rms"           -> sqrt(mean(y_raw^2))
        - numeric         -> use that numeric value directly

    use_sf_for_predictions : bool
        Backward-compatible control for evaluating the fit-space callable.
        This only affects the fit-space diagnostic callable (`f_lambda_fit`).
        The main callable (`f_lambda`) is built from the postprocessed raw-space
        expression and always expects raw input u.

    u_norm : None or numeric
        The input scaling written back into the postprocessed symbolic expression.
        Default: u_sf

    y_norm : None or numeric
        The output scaling written back into the postprocessed symbolic expression.
        Default: y_sf

    persist_hof_files : bool
        If True, keep PySR hall-of-fame CSV artifacts on disk.
        If False (default), remove hall-of-fame CSV outputs after each seed fit
        to avoid accumulating many intermediate files.
    """

    def __init__(
        self,
        SRmodel,
        dataobj,
        orig_dic,
        num_u: int = 200,
        device: str = "cpu",
        label: str = "diff",
        speciesLabel: str = "green",
        sig_figs: int = 2,
        density_scaling="dimensionfull",
        y_scaling="dimensionless",
        use_sf_for_predictions: bool = True,
        u_norm=None,
        y_norm=None,
        percentiles=(5, 95),
        seeds=(0, 1, 2),
        persist_hof_files: bool = False,
    ):
        # --------
        # Inputs / config
        # --------
        self.SRmodel = SRmodel
        self.dataobj = dataobj
        self.orig_dic = orig_dic
        self.num_u = num_u
        self.device = device
        self.label = label
        self.speciesLabel = speciesLabel
        self.sig_figs = sig_figs

        self.density_scaling = density_scaling
        self.y_scaling = y_scaling
        self.use_sf_for_predictions = use_sf_for_predictions

        self.u_sf = self._resolve_density_scale(density_scaling)
        self.y_sf = None

        self.u_norm = self.u_sf if u_norm is None else u_norm
        self.y_norm = y_norm  # finalized later once y_sf is known

        self.percentiles = percentiles
        self.seeds = tuple(seeds)
        self.persist_hof_files = bool(persist_hof_files)

        # --------
        # Stored setup / metadata
        # --------
        self.start_time = None
        self.end_time = None
        self.runtime_seconds = None
        self.runtime_hms = None

        self.model_wrappers = list(orig_dic.values())
        self.num_funcs = len(orig_dic)

        # --------
        # Histogram / u-range
        # --------
        self.hist_info = None
        self.low_u = None
        self.high_u = None

        # --------
        # Grids / arrays
        # --------
        self.u_vals_torch = None
        self.u_vals_np = None               # raw x-grid
        self.u_vals_np_fit = None           # x-grid used for fitting
        self.u_vals_np_predict = None       # fit-space diagnostic evaluation grid

        # --------
        # Ensemble predictions / weighting
        # --------
        self.weights = None
        self.f_vals_np = None               # backward-compatible alias to raw y
        self.f_vals_np_raw = None           # raw target values
        self.f_vals_np_fit = None           # scaled target values used for fitting
        self.wrapper_details = []

        # --------
        # Per-seed SR results
        # --------
        self.symbol = symbols("x0")
        self.per_seed = []

        # --------
        # PySR-only timing
        # --------
        self.seed_fit_runtimes_seconds = []
        self.seed_fit_runtimes_hms = []
        self.total_pysr_runtime_seconds = None
        self.mean_pysr_runtime_seconds = None
        self.min_pysr_runtime_seconds = None
        self.max_pysr_runtime_seconds = None

        # --------
        # Expression comparison / outputs
        # --------
        self.reference_expr = None
        self.reference_constants = None
        self.same_num_constants = None
        self.structure_matches = None
        self.const_matrix = None
        self.constant_variances = None
        self.mean_constants = None

        self.sym_expr = None                # fit-space expression
        self.sym_expr_post = None           # raw-space expression
        self.sym_expr_simplified = None
        self.sym_expr_simplified_rounded = None

        # --------
        # Callable output
        # --------
        self.f_lambda_fit = None            # fit-space callable
        self.f_lambda_fit_vals = None       # fit-space callable values
        self.f_lambda = None                # raw-space callable
        self.f_lambda_vals = None           # raw-space callable values

        # --------
        # Complexity bookkeeping
        # --------
        self.seed_pysr_complexities = []
        self.seed_selected_equations = []
        self.seed_equation_tables = []

        self.final_pysr_complexity = None
        self.final_pysr_complexity_source = None

        self.final_sympy_node_complexity = None
        self.final_simplified_sympy_node_complexity = None
        self.final_rounded_sympy_node_complexity = None

    def _cleanup_hof_artifacts(self, model) -> None:
        """
        Remove PySR hall-of-fame disk artifacts when persistence is disabled.

        PySR typically writes per-fit artifacts under outputs/<timestamp>/.
        This method deletes hall_of_fame*.csv files and prunes their timestamped
        run directories, while leaving unrelated project outputs untouched.
        """
        if self.persist_hof_files:
            return

        def _safe_getattr(obj, name):
            """Best-effort attribute access for PySR versions with deprecated properties."""
            try:
                return getattr(obj, name)
            except Exception:
                return None

        candidates = []
        for attr in ("hall_of_fame_file", "hall_of_fame_path", "equation_file", "equation_file_"):
            val = _safe_getattr(model, attr)
            if isinstance(val, str) and val.strip():
                p = Path(val).expanduser()
                if not p.is_absolute():
                    p = (Path.cwd() / p).resolve()
                candidates.append(p)

        # PySR >= 1.5: prefer output_directory_ + run_id_.
        output_directory = _safe_getattr(model, "output_directory_")
        run_id = _safe_getattr(model, "run_id_")
        if isinstance(output_directory, str) and output_directory.strip() and isinstance(run_id, str) and run_id.strip():
            run_dir = Path(output_directory).expanduser() / run_id
            if not run_dir.is_absolute():
                run_dir = (Path.cwd() / run_dir).resolve()
            candidates.extend(run_dir.glob("hall_of_fame*"))

        seen_dirs = set()
        for p in candidates:
            try:
                if p.exists() and p.name.startswith("hall_of_fame") and p.suffix.lower() == ".csv":
                    p.unlink(missing_ok=True)

                run_dir = p.parent
                if run_dir.exists():
                    for f in run_dir.glob("hall_of_fame*"):
                        if f.is_file():
                            f.unlink(missing_ok=True)
                    seen_dirs.add(run_dir)
            except Exception:
                # Best-effort cleanup should never interrupt fitting.
                continue

        # Remove only timestamped PySR run directories directly under outputs/.
        ts_pattern = re.compile(r"^\d{8}_\d{6}_[A-Za-z0-9]+$")
        for run_dir in seen_dirs:
            try:
                if (
                    run_dir.exists()
                    and run_dir.is_dir()
                    and ts_pattern.match(run_dir.name)
                    and run_dir.parent.name == "outputs"
                ):
                    shutil.rmtree(run_dir, ignore_errors=True)
            except Exception:
                continue

    @staticmethod
    def _seconds_to_hms(seconds):
        seconds_int = int(round(float(seconds)))
        hours, remainder = divmod(seconds_int, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _extract_pysr_complexity(model, sym_expr=None):
        """
        Robustly try to extract the complexity of the selected/best PySR equation.
        """
        equations_df = getattr(model, "equations_", None)
        selected_row = None
        complexity = None

        # 1) Try get_best()
        if hasattr(model, "get_best"):
            try:
                best = model.get_best()
                if isinstance(best, dict):
                    selected_row = best
                    complexity = best.get("complexity", None)
                else:
                    try:
                        selected_row = dict(best)
                        complexity = selected_row.get("complexity", None)
                    except Exception:
                        pass
            except Exception:
                pass

        # 2) Try selected model index
        if complexity is None and equations_df is not None:
            try:
                selected_idx = getattr(model, "idx_model_selection_", None)
                if selected_idx is not None:
                    row = equations_df.iloc[int(selected_idx)]
                    selected_row = dict(row)
                    complexity = row.get("complexity", None)
            except Exception:
                pass

        # 3) Try exact match against sympy string
        if complexity is None and equations_df is not None and sym_expr is not None:
            try:
                sym_str = str(sym_expr)
                for _, row in equations_df.iterrows():
                    row_sym = row.get("sympy_format", None)
                    if row_sym is not None and str(row_sym) == sym_str:
                        selected_row = dict(row)
                        complexity = row.get("complexity", None)
                        break
            except Exception:
                pass

        return complexity, selected_row, equations_df

    @staticmethod
    def _sympy_node_complexity(expr):
        """
        Fallback complexity for any SymPy expression.
        Counts total nodes in the expression tree.
        """
        if expr is None:
            return None

        try:
            return sum(1 for _ in expr.preorder_traversal())
        except Exception:
            try:
                from sympy import preorder_traversal
                return sum(1 for _ in preorder_traversal(expr))
            except Exception:
                return None

    def _resolve_density_scale(self, density_scaling):
        if isinstance(density_scaling, (int, float)):
            return float(density_scaling)

        if density_scaling == "dimensionless":
            return 1.0

        if density_scaling == "dimensionfull":
            if not hasattr(self.dataobj, "u_green_max"):
                raise AttributeError(
                    "density_scaling='dimensionfull' requires dataobj.u_green_max."
                )
            return 1e6 * self.dataobj.u_green_max

        raise ValueError(
            f"Unknown density_scaling: {density_scaling}. "
            "Expected 'dimensionless', 'dimensionfull', or numeric."
        )

    @staticmethod
    def _resolve_output_scale(y_scaling, y_values):
        y_values = np.asarray(y_values, dtype=float).ravel()

        if y_values.size == 0:
            raise ValueError("Cannot resolve y scaling from an empty target array.")

        if isinstance(y_scaling, (int, float)):
            sf = float(y_scaling)
        elif y_scaling == "dimensionless":
            sf = 1.0
        elif y_scaling == "max_abs":
            sf = float(np.max(np.abs(y_values)))
        elif y_scaling == "mean_abs":
            sf = float(np.mean(np.abs(y_values)))
        elif y_scaling == "std":
            sf = float(np.std(y_values))
        elif y_scaling == "rms":
            sf = float(np.sqrt(np.mean(y_values ** 2)))
        else:
            raise ValueError(
                f"Unknown y_scaling: {y_scaling}. "
                "Expected 'dimensionless', 'max_abs', 'mean_abs', 'std', 'rms', or numeric."
            )

        if not np.isfinite(sf) or sf == 0.0:
            raise ValueError(
                f"Resolved y scaling factor must be finite and nonzero, got {sf}."
            )

        return sf

    def _compute_u_range(self):
        self.hist_info = hist_properties(
            self.dataobj,
            speciesLabel=self.speciesLabel,
            num_bins=100,
            low=self.percentiles[0],
            high=self.percentiles[1],
        )
        self.low_u = self.hist_info["low_count"]
        self.high_u = self.hist_info["high_count"]

    def _build_u_grid(self):
        self.u_vals_torch = (
            torch.linspace(self.low_u, self.high_u, self.num_u)
            .view(-1, 1)
            .to(self.device)
        )
        self.u_vals_np = self.u_vals_torch.detach().cpu().numpy()   # raw x

        self.u_vals_np_fit = self.u_vals_np * self.u_sf             # fit-space x
        self.u_vals_np_predict = (
            self.u_vals_np_fit if self.use_sf_for_predictions else self.u_vals_np
        )

    def _compute_ensemble_predictions_and_weights(self):
        self.weights = np.zeros(self.num_u, dtype=float)
        self.f_vals_np_raw = np.zeros_like(self.u_vals_np, dtype=float)
        self.wrapper_details = []

        for wrapper in self.model_wrappers:
            if self.label == "diff":
                func = wrapper.model.diffusion
                sf = wrapper.model.D_scale
                func_name = "diffusion"
            elif self.label == "grow":
                func = wrapper.model.growth
                sf = wrapper.model.G_scale
                func_name = "growth"
            else:
                raise ValueError(
                    f"Unknown label '{self.label}'. Expected 'diff' or 'grow'."
                )

            func.to(self.device)

            with torch.no_grad():
                pred_raw = func(self.u_vals_torch).detach().cpu().numpy() * sf
                self.f_vals_np_raw += pred_raw

            u_train = np.asarray(wrapper.y_train, dtype=float).ravel()
            hist, bin_edges = np.histogram(u_train, bins=20, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            interp_weights = np.interp(
                self.u_vals_np[:, 0],
                bin_centers,
                hist,
                left=0.0,
                right=0.0,
            )
            self.weights += interp_weights

            self.wrapper_details.append(
                {
                    "wrapper": wrapper,
                    "function_name": func_name,
                    "scale_factor": sf,
                    "u_train": u_train,
                    "hist": hist,
                    "bin_edges": bin_edges,
                    "bin_centers": bin_centers,
                    "predictions_raw": pred_raw,
                    "interp_weights": interp_weights,
                }
            )

        self.f_vals_np_raw /= max(self.num_funcs, 1)
        self.weights /= max(self.num_funcs, 1)

        # Resolve target scaling only once raw target values are known
        self.y_sf = self._resolve_output_scale(self.y_scaling, self.f_vals_np_raw)
        if self.y_norm is None:
            self.y_norm = self.y_sf

        # Build fit target
        self.f_vals_np_fit = self.f_vals_np_raw / self.y_sf

        # Backward-compatible alias
        self.f_vals_np = self.f_vals_np_raw

    def _fit_per_seed_models(self):
        U = self.symbol
        self.per_seed = []
        self.seed_fit_runtimes_seconds = []
        self.seed_fit_runtimes_hms = []

        for seed in self.seeds:
            model_i = copy.deepcopy(self.SRmodel)

            if hasattr(model_i, "random_state"):
                model_i.random_state = seed
            if hasattr(model_i, "random_seed"):
                model_i.random_seed = seed

            fit_start = time.time()
            model_i.fit(
                self.u_vals_np_fit,
                self.f_vals_np_fit.ravel(),
                weights=self.weights,
            )
            fit_end = time.time()

            fit_runtime_seconds = float(fit_end - fit_start)
            fit_runtime_hms = self._seconds_to_hms(fit_runtime_seconds)

            # sym_expr is fit-space: y_fit = sym_expr(x_fit)
            sym_expr = model_i.sympy()

            # raw-space: y_raw = y_norm * sym_expr(x_raw * u_norm)
            sym_expr_post = self.y_norm * sym_expr.subs(U, U * self.u_norm)

            complexity, selected_row, equations_df = self._extract_pysr_complexity(
                model_i,
                sym_expr=sym_expr,
            )
            sympy_nodes = self._sympy_node_complexity(sym_expr_post)

            self._cleanup_hof_artifacts(model_i)

            run_info = {
                "seed": seed,
                "model": model_i,
                "sym_expr": sym_expr,                   # fit-space
                "sym_expr_post": sym_expr_post,         # raw-space
                "constants": _extract_float_constants(sym_expr_post),
                "runtime_seconds": fit_runtime_seconds,
                "runtime_hms": fit_runtime_hms,
                "pysr_complexity": complexity,
                "selected_equation": selected_row,
                "equations_df": equations_df,
                "sympy_node_complexity": sympy_nodes,
            }

            self.per_seed.append(run_info)
            self.seed_fit_runtimes_seconds.append(fit_runtime_seconds)
            self.seed_fit_runtimes_hms.append(fit_runtime_hms)

        if len(self.seed_fit_runtimes_seconds) > 0:
            runtimes = np.asarray(self.seed_fit_runtimes_seconds, dtype=float)
            self.total_pysr_runtime_seconds = float(np.sum(runtimes))
            self.mean_pysr_runtime_seconds = float(np.mean(runtimes))
            self.min_pysr_runtime_seconds = float(np.min(runtimes))
            self.max_pysr_runtime_seconds = float(np.max(runtimes))
        else:
            self.total_pysr_runtime_seconds = None
            self.mean_pysr_runtime_seconds = None
            self.min_pysr_runtime_seconds = None
            self.max_pysr_runtime_seconds = None

    def _compare_constants_across_seeds(self):
        self.reference_expr = self.per_seed[0]["sym_expr_post"]
        self.reference_constants = self.per_seed[0]["constants"]

        self.same_num_constants = all(
            len(run["constants"]) == len(self.reference_constants)
            for run in self.per_seed
        )

        if self.same_num_constants:
            self.structure_matches = all(
                str(
                    run["sym_expr_post"].subs(
                        {
                            c: 1
                            for c in preorder_traversal(run["sym_expr_post"])
                            if isinstance(c, Float)
                        }
                    )
                )
                == str(
                    self.reference_expr.subs(
                        {
                            c: 1
                            for c in preorder_traversal(self.reference_expr)
                            if isinstance(c, Float)
                        }
                    )
                )
                for run in self.per_seed
            )
        else:
            self.structure_matches = False

        if (
            self.same_num_constants
            and self.structure_matches
            and len(self.reference_constants) > 0
        ):
            self.const_matrix = np.array(
                [run["constants"] for run in self.per_seed],
                dtype=float,
            )
            self.constant_variances = np.var(self.const_matrix, axis=0)
            self.mean_constants = np.mean(self.const_matrix, axis=0)

            mean_expr = _replace_float_constants(
                self.reference_expr,
                self.mean_constants,
            )

            self.sym_expr_post = mean_expr
            self.sym_expr_simplified = sym_conv(mean_expr)

            sym_expr_var_rounded = _round_constants_by_variance(
                mean_expr,
                self.constant_variances,
            )
            self.sym_expr_simplified_rounded = sym_conv(sym_expr_var_rounded)

        else:
            self.const_matrix = None
            self.constant_variances = None
            self.mean_constants = None

            self.sym_expr_post = self.reference_expr
            self.sym_expr_simplified = sym_conv(self.reference_expr)
            self.sym_expr_simplified_rounded = sym_conv_rounded(
                self.reference_expr,
                sig_figs=self.sig_figs,
            )

    def _build_lambda(self):
        # Fit-space expression from reference seed
        self.sym_expr = self.per_seed[0]["sym_expr"]

        # Fit-space callable: expects x_fit, returns y_fit
        self.f_lambda_fit = lambdify(self.symbol, self.sym_expr, modules="numpy")
        self.f_lambda_fit_vals = self.f_lambda_fit(self.u_vals_np_predict)

        # Raw-space callable: expects raw x, returns raw y
        self.f_lambda = lambdify(self.symbol, self.sym_expr_post, modules="numpy")
        self.f_lambda_vals = self.f_lambda(self.u_vals_np)

    def _store_complexity_summary(self):
        self.seed_pysr_complexities = [
            run.get("pysr_complexity", None) for run in self.per_seed
        ]
        self.seed_selected_equations = [
            run.get("selected_equation", None) for run in self.per_seed
        ]
        self.seed_equation_tables = [
            run.get("equations_df", None) for run in self.per_seed
        ]

        self.final_pysr_complexity = (
            self.per_seed[0].get("pysr_complexity", None)
            if len(self.per_seed) > 0 else None
        )
        self.final_pysr_complexity_source = (
            f"reference_seed_{self.per_seed[0]['seed']}"
            if len(self.per_seed) > 0 else None
        )

        self.final_sympy_node_complexity = self._sympy_node_complexity(
            self.sym_expr_post
        )
        self.final_simplified_sympy_node_complexity = self._sympy_node_complexity(
            self.sym_expr_simplified
        )
        self.final_rounded_sympy_node_complexity = self._sympy_node_complexity(
            self.sym_expr_simplified_rounded
        )

    def _finalize_runtime(self):
        self.end_time = time.time()
        self.runtime_seconds = int(self.end_time - self.start_time)

        hours, remainder = divmod(self.runtime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.runtime_hms = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def fit(self, verbose: bool = True):
        self.start_time = time.time()

        self._compute_u_range()
        self._build_u_grid()
        self._compute_ensemble_predictions_and_weights()
        self._fit_per_seed_models()
        self._compare_constants_across_seeds()
        self._build_lambda()
        self._store_complexity_summary()
        self._finalize_runtime()

        if verbose:
            print("Per-seed symbolic expressions:")
            for run in self.per_seed:
                print(
                    f"Seed {run['seed']}: {run['sym_expr_post']} "
                    f"[PySR fit time: {run['runtime_seconds']:.3f} s, "
                    f"PySR complexity: {run.get('pysr_complexity', None)}, "
                    f"SymPy nodes: {run.get('sympy_node_complexity', None)}]"
                )

            print("\nFinal symbolic expression:")
            print(self.sym_expr_post)

            print("\nRounded using constant variances:")
            print(self.sym_expr_simplified_rounded)

            print("\nScaling configuration:")
            print(f"  density_scaling        = {self.density_scaling}")
            print(f"  resolved density scale = {self.u_sf}")
            print(f"  y_scaling              = {self.y_scaling}")
            print(f"  resolved output scale  = {self.y_sf}")
            print(f"  use_sf_for_predictions = {self.use_sf_for_predictions}")
            print(f"  u_norm                 = {self.u_norm}")
            print(f"  y_norm                 = {self.y_norm}")

            print("\nFinal complexity summary:")
            print(f"  final_pysr_complexity                   = {self.final_pysr_complexity}")
            print(f"  final_sympy_node_complexity             = {self.final_sympy_node_complexity}")
            print(f"  final_simplified_sympy_node_complexity  = {self.final_simplified_sympy_node_complexity}")
            print(f"  final_rounded_sympy_node_complexity     = {self.final_rounded_sympy_node_complexity}")

            if self.constant_variances is not None:
                print("\nConstant variances:")
                for i, v in enumerate(self.constant_variances):
                    step = _rounding_step_from_variance(v)
                    print(f"  c{i}: var = {v:.6g}, rounding step = {step}")

            if self.mean_pysr_runtime_seconds is not None:
                print("\nPySR per-seed runtime summary:")
                print(f"  Mean: {self.mean_pysr_runtime_seconds:.3f} s")
                print(f"  Min : {self.min_pysr_runtime_seconds:.3f} s")
                print(f"  Max : {self.max_pysr_runtime_seconds:.3f} s")
                print(f"  Total across seeds: {self.total_pysr_runtime_seconds:.3f} s")

            print(f"\nFitting complete. Total pipeline runtime = {self.runtime_hms}")

        return self

    def to_dict(self):
        return {
            "SRmodel": self.SRmodel,
            "dataobj": self.dataobj,
            "orig_dic": self.orig_dic,
            "num_u": self.num_u,
            "device": self.device,
            "label": self.label,
            "speciesLabel": self.speciesLabel,
            "sig_figs": self.sig_figs,

            "density_scaling": self.density_scaling,
            "y_scaling": self.y_scaling,
            "use_sf_for_predictions": self.use_sf_for_predictions,

            "u_sf": self.u_sf,
            "y_sf": self.y_sf,
            "u_norm": self.u_norm,
            "y_norm": self.y_norm,

            "percentiles": self.percentiles,
            "seeds": self.seeds,
            "persist_hof_files": self.persist_hof_files,

            "hist_info": self.hist_info,
            "low_u": self.low_u,
            "high_u": self.high_u,

            # x grids
            "u_vals": self.u_vals_torch * self.u_sf if self.u_vals_torch is not None else None,
            "u_vals_torch": self.u_vals_torch,
            "u_vals_np": self.u_vals_np,                 # raw x grid
            "u_vals_np_raw": self.u_vals_np,
            "u_vals_np_fit": self.u_vals_np_fit,
            "u_vals_np_predict": self.u_vals_np_predict,

            # y arrays
            "f_vals": self.f_vals_np,                    # backward-compatible alias to raw y
            "f_vals_np": self.f_vals_np,
            "f_vals_np_raw": self.f_vals_np_raw,
            "f_vals_np_fit": self.f_vals_np_fit,

            "weights": self.weights,
            "model_wrappers": self.model_wrappers,
            "num_funcs": self.num_funcs,
            "wrapper_details": self.wrapper_details,

            "runs": self.per_seed,
            "per_seed": self.per_seed,

            "seed_fit_runtimes_seconds": self.seed_fit_runtimes_seconds,
            "seed_fit_runtimes_hms": self.seed_fit_runtimes_hms,
            "total_pysr_runtime_seconds": self.total_pysr_runtime_seconds,
            "mean_pysr_runtime_seconds": self.mean_pysr_runtime_seconds,
            "min_pysr_runtime_seconds": self.min_pysr_runtime_seconds,
            "max_pysr_runtime_seconds": self.max_pysr_runtime_seconds,

            # expressions
            "sym_expr": self.sym_expr,                   # fit-space
            "sym_expr_post": self.sym_expr_post,         # raw-space
            "sym_expr_simplified": self.sym_expr_simplified,
            "sym_expr_simplified_rounded": self.sym_expr_simplified_rounded,

            "constant_variances": self.constant_variances,
            "const_matrix": self.const_matrix,
            "mean_constants": self.mean_constants,
            "same_num_constants": self.same_num_constants,
            "structure_matches": self.structure_matches,

            # callables
            "f_lambda_fit": self.f_lambda_fit,
            "f_lambda_fit_vals": self.f_lambda_fit_vals,
            "f_lambda": self.f_lambda,
            "f_lambda_vals": self.f_lambda_vals,

            "seed_pysr_complexities": self.seed_pysr_complexities,
            "seed_selected_equations": self.seed_selected_equations,
            "seed_equation_tables": self.seed_equation_tables,

            "final_pysr_complexity": self.final_pysr_complexity,
            "final_pysr_complexity_source": self.final_pysr_complexity_source,
            "final_sympy_node_complexity": self.final_sympy_node_complexity,
            "final_simplified_sympy_node_complexity": self.final_simplified_sympy_node_complexity,
            "final_rounded_sympy_node_complexity": self.final_rounded_sympy_node_complexity,

            "runtime_seconds": self.runtime_seconds,
            "runtime_hms": self.runtime_hms,
        }