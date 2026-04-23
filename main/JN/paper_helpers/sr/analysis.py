from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


x0 = sp.symbols("x0")


def weighted_mse(y_true, y_pred, w=None, weights: bool = False) -> float:
    """
    Compute MSE safely.

    If weights=False:
        mean((y_true - y_pred)^2)

    If weights=True:
        sum(w * (y_true - y_pred)^2) / sum(w)

    Falls back to unweighted MSE if weights are invalid.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if w is None:
        w = np.ones_like(y_true, dtype=float)
    else:
        w = np.asarray(w, dtype=float).ravel()

    n = min(len(y_true), len(y_pred), len(w))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    w = w[:n]

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    w = w[mask]

    if y_true.size == 0:
        return np.inf

    sq_err = (y_true - y_pred) ** 2

    if not weights:
        return float(np.mean(sq_err))

    w_sum = np.sum(w)
    if (not np.isfinite(w_sum)) or (w_sum <= 0):
        return float(np.mean(sq_err))

    return float(np.sum(w * sq_err) / w_sum)


def summarize_mean_minmax(values, decimals: int = 1, unit: str = "") -> str:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "N/A"
    mean_val = np.mean(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    suffix = f" {unit}" if unit else ""
    return (
        f"{mean_val:.{decimals}f} ± "
        f"[{min_val:.{decimals}f}, {max_val:.{decimals}f}]{suffix}"
    )


def summarize_mean_std(values, decimals: int = 1, unit: str = "") -> str:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "N/A"
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    suffix = f" {unit}" if unit else ""
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}{suffix}"


class SymbolicRegressionSeedReporter:
    """
    Analyze and plot per-seed symbolic-regression results in raw space.

    Parameters
    ----------
    SR_runs
        Symbolic-regression outputs, either:
        - mapping: key -> symbolic-regression output
        - sequence aligned with ``keys``
    keys
        Optional sequence of keys. If ``SR_runs`` is a mapping and ``keys`` is
        omitted, keys are inferred from ``SR_runs``.
    exclusions_map
        Optional mapping from key -> list of seeds to exclude from ranking
        and retained-seed statistics.
    symbol
        Default SymPy symbol used when a run does not provide one.
    label
        Either ``"diff"`` or ``"grow"``. Controls whether printed/plot labels
        use ``D`` / ``D(u)`` or ``G`` / ``G(u)``.
    """

    def __init__(
        self,
        SR_runs: Union[Sequence[Any], Mapping[Any, Any]],
        keys: Optional[Sequence[Any]] = None,
        exclusions_map: Optional[Mapping[Any, Sequence[int]]] = None,
        symbol: Any = None,
        label: str = "diff",
    ) -> None:
        self.SR_runs_by_key = self._normalize_runs_by_key(SR_runs=SR_runs, keys=keys)
        self.keys = list(self.SR_runs_by_key.keys())
        self.exclusions_map = {
            k: list(v) for k, v in (exclusions_map or {}).items()
        }
        self.default_symbol = x0 if symbol is None else symbol
        self.label = label

        self._validate_label()

        self.results_by_key: Dict[Any, Dict[str, Any]] = {}
        self.all_seed_runtimes: List[float] = []
        self.all_kept_seed_runtimes: List[float] = []

    @staticmethod
    def _normalize_runs_by_key(
        SR_runs: Union[Sequence[Any], Mapping[Any, Any]],
        keys: Optional[Sequence[Any]],
    ) -> Dict[Any, Any]:
        """
        Normalize SR_runs into a dictionary keyed by dataset key.
        """
        if isinstance(SR_runs, Mapping):
            sr_runs_by_key = dict(SR_runs)

            if keys is None:
                return sr_runs_by_key

            missing = [key for key in keys if key not in sr_runs_by_key]
            if missing:
                raise KeyError(
                    f"Keys {missing} were requested but not found in SR_runs. "
                    f"Available keys: {list(sr_runs_by_key.keys())}"
                )

            return {key: sr_runs_by_key[key] for key in keys}

        if keys is None:
            raise ValueError(
                "keys must be provided when SR_runs is a sequence."
            )

        if len(SR_runs) != len(keys):
            raise ValueError(
                f"SR_runs and keys must have the same length, got "
                f"{len(SR_runs)} and {len(keys)}."
            )

        return {key: SR for key, SR in zip(keys, SR_runs)}

    def _validate_label(self) -> None:
        valid_labels = {"diff", "grow"}
        if self.label not in valid_labels:
            raise ValueError(
                f"Unknown label: {self.label}. "
                f"Expected one of {sorted(valid_labels)}."
            )

    @property
    def response_symbol(self) -> str:
        return "D" if self.label == "diff" else "G"

    @property
    def response_function_label(self) -> str:
        return f"{self.response_symbol}(x0)"

    @staticmethod
    def _safe_eval_raw_expr(expr, symbol, x_vals):
        """
        Evaluate a raw-space SymPy expression on raw x values.
        Returns a 1D float array aligned to x_vals.
        """
        x_vals = np.asarray(x_vals, dtype=float).ravel()
        f = sp.lambdify(symbol, expr, modules="numpy")
        y = f(x_vals)
        y = np.asarray(y, dtype=float)

        if y.ndim == 0:
            y = np.full_like(x_vals, float(y), dtype=float)
        else:
            y = y.ravel()

        if y.size == 1 and x_vals.size > 1:
            y = np.full_like(x_vals, float(y[0]), dtype=float)

        if y.size != x_vals.size:
            raise ValueError(
                f"Prediction length mismatch: got {y.size}, expected {x_vals.size}"
            )

        return y

    @staticmethod
    def _expression_form(expr, symbol=None) -> str:
        """
        Convert a SymPy expression into a canonical symbolic 'form' that ignores
        fitted constant values but preserves structure.

        Examples
        --------
        a*exp(b*x) + c   -> C1*exp(C2*x0) + C3
        a*exp(b*x)       -> C1*exp(C2*x0)
        3.1              -> C1
        """
        if symbol is None:
            symbol = x0

        expr = sp.simplify(expr)
        counter = {"n": 0}

        def next_const():
            counter["n"] += 1
            return sp.Symbol(f"C{counter['n']}")

        def canon(node):
            node = sp.simplify(node)

            if node == symbol:
                return symbol

            if not node.has(symbol):
                return next_const()

            if node.is_Atom:
                return node

            new_args = tuple(canon(arg) for arg in node.args)
            rebuilt = node.func(*new_args)
            return rebuilt

        return sp.sstr(canon(expr))

    def _extract_raw_run_data(self, SR: Any) -> Dict[str, Any]:
        """
        Extract raw-space arrays and metadata from either dict-style or
        object-style runner output.

        Everything returned here is in raw physical space, so MSE and plots are
        consistent.
        """
        if isinstance(SR, dict):
            u_vals = np.asarray(SR["u_vals_np"], dtype=float).ravel()
            f_target = np.asarray(
                SR.get("f_vals_np_raw", SR.get("f_vals")),
                dtype=float,
            ).ravel()
            weights = np.asarray(
                SR.get("weights", np.ones_like(u_vals)),
                dtype=float,
            ).ravel()
            per_seed = SR["per_seed"]
            final_vals = np.asarray(SR["f_lambda_vals"], dtype=float).ravel()
            symbol = SR.get("symbol", self.default_symbol)
            final_expr = SR.get(
                "sym_expr_post",
                SR.get("sym_expr_simplified_rounded", None),
            )
        else:
            u_vals = np.asarray(SR.u_vals_np, dtype=float).ravel()
            f_target = np.asarray(
                getattr(SR, "f_vals_np_raw", SR.f_vals_np),
                dtype=float,
            ).ravel()
            weights = np.asarray(
                getattr(SR, "weights", np.ones_like(u_vals)),
                dtype=float,
            ).ravel()
            per_seed = SR.per_seed
            final_vals = np.asarray(SR.f_lambda_vals, dtype=float).ravel()
            symbol = getattr(SR, "symbol", self.default_symbol)
            final_expr = getattr(
                SR,
                "sym_expr_post",
                getattr(SR, "sym_expr_simplified_rounded", None),
            )

        n = min(len(u_vals), len(f_target), len(weights), len(final_vals))
        u_vals = u_vals[:n]
        f_target = f_target[:n]
        weights = weights[:n]
        final_vals = final_vals[:n]

        mask = (
            np.isfinite(u_vals)
            & np.isfinite(f_target)
            & np.isfinite(weights)
            & np.isfinite(final_vals)
        )
        u_vals = u_vals[mask]
        f_target = f_target[mask]
        weights = weights[mask]
        final_vals = final_vals[mask]

        sort_idx = np.argsort(u_vals)
        u_vals = u_vals[sort_idx]
        f_target = f_target[sort_idx]
        weights = weights[sort_idx]
        final_vals = final_vals[sort_idx]

        return {
            "u_vals": u_vals,
            "f_target": f_target,
            "weights": weights,
            "per_seed": per_seed,
            "final_vals": final_vals,
            "symbol": symbol,
            "final_expr": final_expr,
        }

    def _analyze_single_run(
        self,
        key: Any,
        SR: Any,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        current_exclusions = self.exclusions_map.get(key, [])

        extracted = self._extract_raw_run_data(SR)
        u_vals_fit = extracted["u_vals"]
        f_target = extracted["f_target"]
        weights = extracted["weights"]
        per_seed = extracted["per_seed"]
        symbol = extracted["symbol"]

        key_rows = []

        if verbose:
            print(f"\n>>> KEY: {key}")
            print(f"    Excluding seeds from stats: {current_exclusions}")

        for run in per_seed:
            seed_val = run["seed"]
            runtime_seconds = float(run.get("runtime_seconds", np.nan))

            try:
                expr_raw = run.get("sym_expr_post", run.get("sym_expr"))
                expr_raw = sp.simplify(expr_raw)

                seed_pred = self._safe_eval_raw_expr(expr_raw, symbol, u_vals_fit)

                fit_unweighted = weighted_mse(
                    f_target,
                    seed_pred,
                    weights,
                    weights=False,
                )
                fit_weighted = weighted_mse(
                    f_target,
                    seed_pred,
                    weights,
                    weights=True,
                )

                row = {
                    "seed": seed_val,
                    "expr": expr_raw,
                    "expr_form": self._expression_form(expr_raw, symbol),
                    "excluded": seed_val in current_exclusions,
                    "fit_unweighted": fit_unweighted,
                    "fit_weighted": fit_weighted,
                    "runtime_seconds": runtime_seconds,
                }
                key_rows.append(row)

                if np.isfinite(runtime_seconds):
                    self.all_seed_runtimes.append(runtime_seconds)

                if verbose:
                    print(
                        f"  [Seed {seed_val:2}] "
                        f"{self.response_function_label} = {expr_raw}"
                    )

            except Exception as e:
                if verbose:
                    print(f"  [Seed {seed_val:2}] Skipping due to error: {e}")

        valid_rows = [row for row in key_rows if not row["excluded"]]
        ranked_rows = sorted(valid_rows, key=lambda r: r["fit_weighted"])
        valid_runtimes = [
            row["runtime_seconds"]
            for row in valid_rows
            if np.isfinite(row["runtime_seconds"])
        ]
        self.all_kept_seed_runtimes.extend(valid_runtimes)

        if verbose:
            print("\n  --- Final Ranked Seeds (Best to Worst Weighted Fit) ---")
            if not ranked_rows:
                print("    No valid seeds remaining after exclusions.")
            else:
                for rank, row in enumerate(ranked_rows, 1):
                    runtime_str = (
                        f"{row['runtime_seconds']:.1f} s"
                        if np.isfinite(row["runtime_seconds"])
                        else "N/A"
                    )
                    print(
                        f"    Rank {rank:2}: Seed {row['seed']:2} | "
                        f"W-MSE: {row['fit_weighted']:.4e} | "
                        f"MSE: {row['fit_unweighted']:.4e} | "
                        f"Runtime: {runtime_str} | "
                        f"Expr: {row['expr']}"
                    )

            print(
                "\n  Runtime summary (kept seeds): "
                f"{summarize_mean_minmax(valid_runtimes, decimals=1, unit='s')}"
            )
            print("-" * 75)

        return {
            "key": key,
            "label": self.label,
            "response_symbol": self.response_symbol,
            "response_function_label": self.response_function_label,
            "excluded_seeds": list(current_exclusions),
            "rows": key_rows,
            "valid_rows": valid_rows,
            "ranked_rows": ranked_rows,
            "runtime_summary_kept": summarize_mean_minmax(
                valid_runtimes,
                decimals=1,
                unit="s",
            ),
            "extracted": extracted,
        }

    def analyze(self, verbose: bool = True) -> Dict[Any, Dict[str, Any]]:
        """
        Analyze all runs, compute rankings, and optionally print summaries.
        """
        self.results_by_key = {}
        self.all_seed_runtimes = []
        self.all_kept_seed_runtimes = []

        if verbose:
            print("\n" + "=" * 75)
            print(
                f"CLEANED {self.response_symbol} SYMBOLIC EXPRESSIONS "
                f"& WEIGHTED-FIT RANKING"
            )
            print("=" * 75)

        for key in self.keys:
            SR = self.SR_runs_by_key[key]
            result = self._analyze_single_run(key, SR, verbose=verbose)
            self.results_by_key[key] = result

        if verbose:
            summary = self.global_summary()
            print("\n" + "=" * 75)
            print("GLOBAL RUNTIME SUMMARY ACROSS ALL KEYS")
            print("=" * 75)
            print(f"All processed seeds: {summary['all_processed_seeds']}")
            print(f"All kept seeds:      {summary['all_kept_seeds']}")
            print("\nProcessing Complete.")

        return self.results_by_key

    def global_summary(self) -> Dict[str, str]:
        """
        Return global runtime summaries across all analyzed keys.
        """
        return {
            "all_processed_seeds": summarize_mean_std(
                self.all_seed_runtimes,
                decimals=1,
                unit="s",
            ),
            "all_kept_seeds": summarize_mean_std(
                self.all_kept_seed_runtimes,
                decimals=1,
                unit="s",
            ),
        }

    def expression_form_counts(
        self,
        *,
        include_excluded: bool = False,
        per_key: bool = False,
        sort_by: str = "count",
        verbose: bool = True,
    ) -> Union[Dict[Any, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        Count symbolic-expression forms across analyzed runs.

        Parameters
        ----------
        include_excluded
            If False, only retained seeds are counted. If True, all analyzed
            seeds are counted.
        per_key
            If True, return one table per key. If False, return a global table.
        sort_by
            Either 'count' or 'form'.
        verbose
            If True, print a formatted table.

        Returns
        -------
        If per_key=False:
            list of dict rows with keys:
                'form', 'count', 'seeds'
        If per_key=True:
            dict:
                key -> list of dict rows
        """
        if not self.results_by_key:
            raise RuntimeError(
                "No analysis results found. Run analyze() before calling "
                "expression_form_counts()."
            )

        if sort_by not in {"count", "form"}:
            raise ValueError("sort_by must be either 'count' or 'form'.")

        def build_rows(rows):
            counts = {}
            for row in rows:
                form = row.get("expr_form", self._expression_form(row["expr"]))
                if form not in counts:
                    counts[form] = {"form": form, "count": 0, "seeds": []}
                counts[form]["count"] += 1
                counts[form]["seeds"].append(row["seed"])

            out = list(counts.values())

            if sort_by == "count":
                out.sort(key=lambda r: (-r["count"], r["form"]))
            else:
                out.sort(key=lambda r: r["form"])

            return out

        if per_key:
            tables = {}
            for key, result in self.results_by_key.items():
                rows = result["rows"] if include_excluded else result["valid_rows"]
                tables[key] = build_rows(rows)

                if verbose:
                    print(f"\nExpression-form counts | key={key}")
                    print("-" * 75)
                    if not tables[key]:
                        print("No expressions available.")
                    else:
                        print(f"{'Count':>5}  {'Seeds':<20}  Form")
                        print("-" * 75)
                        for r in tables[key]:
                            seeds_str = ", ".join(map(str, r["seeds"]))
                            print(f"{r['count']:>5}  {seeds_str:<20}  {r['form']}")
            return tables

        all_rows = []
        for result in self.results_by_key.values():
            rows = result["rows"] if include_excluded else result["valid_rows"]
            all_rows.extend(rows)

        table = build_rows(all_rows)

        if verbose:
            print("\nGlobal expression-form counts")
            print("-" * 75)
            if not table:
                print("No expressions available.")
            else:
                print(f"{'Count':>5}  {'Seeds':<20}  Form")
                print("-" * 75)
                for r in table:
                    seeds_str = ", ".join(map(str, r["seeds"]))
                    print(f"{r['count']:>5}  {seeds_str:<20}  {r['form']}")

        return table

    def best_expression_per_form(
        self,
        *,
        include_excluded: bool = False,
        per_key: bool = False,
        sort_by: str = "count",
        verbose: bool = True,
    ) -> Union[Dict[Any, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """
        For each symbolic-expression form, return the best representative
        expression, where 'best' is defined by lowest unweighted MSE.

        Parameters
        ----------
        include_excluded
            If False, only retained seeds are considered. If True, all analyzed
            seeds are considered.
        per_key
            If True, return one table per key. If False, return a global table.
        sort_by
            Either:
                - 'count' : descending form frequency, then ascending best MSE
                - 'form'  : alphabetical by form
                - 'mse'   : ascending best unweighted MSE
        verbose
            If True, print a formatted table.

        Returns
        -------
        If per_key=False:
            list of dict rows with keys:
                'form', 'count', 'best_seed', 'best_mse', 'best_weighted_mse',
                'best_expr', 'seeds'
        If per_key=True:
            dict:
                key -> list of dict rows
        """
        if not self.results_by_key:
            raise RuntimeError(
                "No analysis results found. Run analyze() before calling "
                "best_expression_per_form()."
            )

        if sort_by not in {"count", "form", "mse"}:
            raise ValueError("sort_by must be one of {'count', 'form', 'mse'}.")

        def build_rows(rows):
            grouped = {}

            for row in rows:
                form = row.get("expr_form", self._expression_form(row["expr"]))
                if form not in grouped:
                    grouped[form] = {
                        "form": form,
                        "count": 0,
                        "seeds": [],
                        "best_row": row,
                    }

                grouped[form]["count"] += 1
                grouped[form]["seeds"].append(row["seed"])

                current_best = grouped[form]["best_row"]
                if row["fit_unweighted"] < current_best["fit_unweighted"]:
                    grouped[form]["best_row"] = row

            out = []
            for form, info in grouped.items():
                best = info["best_row"]
                out.append(
                    {
                        "form": form,
                        "count": info["count"],
                        "seeds": info["seeds"],
                        "best_seed": best["seed"],
                        "best_mse": best["fit_unweighted"],
                        "best_weighted_mse": best["fit_weighted"],
                        "best_expr": best["expr"],
                    }
                )

            if sort_by == "count":
                out.sort(key=lambda r: (-r["count"], r["best_mse"], r["form"]))
            elif sort_by == "form":
                out.sort(key=lambda r: r["form"])
            elif sort_by == "mse":
                out.sort(key=lambda r: (r["best_mse"], r["form"]))

            return out

        def print_table(title, table):
            print(f"\n{title}")
            print("-" * 140)
            if not table:
                print("No expressions available.")
                return

            header = (
                f"{'Count':>5}  "
                f"{'Best seed':>9}  "
                f"{'Best MSE':>12}  "
                f"{'Best W-MSE':>12}  "
                f"{'Seeds':<20}  "
                f"{'Form':<35}  "
                f"Best expression"
            )
            print(header)
            print("-" * 140)

            for r in table:
                seeds_str = ", ".join(map(str, r["seeds"]))
                print(
                    f"{r['count']:>5}  "
                    f"{r['best_seed']:>9}  "
                    f"{r['best_mse']:>12.4e}  "
                    f"{r['best_weighted_mse']:>12.4e}  "
                    f"{seeds_str:<20}  "
                    f"{r['form']:<35}  "
                    f"{r['best_expr']}"
                )

        if per_key:
            tables = {}
            for key, result in self.results_by_key.items():
                rows = result["rows"] if include_excluded else result["valid_rows"]
                tables[key] = build_rows(rows)

                if verbose:
                    print_table(f"Best expression per form | key={key}", tables[key])

            return tables

        all_rows = []
        for result in self.results_by_key.values():
            rows = result["rows"] if include_excluded else result["valid_rows"]
            all_rows.extend(rows)

        table = build_rows(all_rows)

        if verbose:
            print_table("Best expression per form | global", table)

        return table

    def plot_runs(
        self,
        *,
        figsize_main=(7, 4.5),
        figsize_residual=(7, 3.5),
        show_seed_predictions: bool = True,
        show_final: bool = True,
        show_reference: bool = True,
        show_residuals: bool = True,
        legend_fontsize: int = 8,
    ) -> None:
        """
        Plot all runs in raw space, consistent with the MSE calculations.
        """
        for key in self.keys:
            SR = self.SR_runs_by_key[key]
            extracted = self._extract_raw_run_data(SR)

            u_vals = extracted["u_vals"]
            f_target = extracted["f_target"]
            weights = extracted["weights"]
            per_seed = extracted["per_seed"]
            final_vals = extracted["final_vals"]
            symbol = extracted["symbol"]

            plt.figure(figsize=figsize_main)

            if show_reference:
                plt.plot(
                    u_vals,
                    f_target,
                    color="red",
                    linewidth=2,
                    label="NN ensemble",
                )

            if show_seed_predictions:
                for run in per_seed:
                    expr_raw = run.get("sym_expr_post", run.get("sym_expr"))
                    comp = run.get("pysr_complexity", None)

                    try:
                        seed_pred = self._safe_eval_raw_expr(expr_raw, symbol, u_vals)
                        plt.plot(
                            u_vals,
                            seed_pred,
                            alpha=0.5,
                            linestyle="--",
                            label=f"seed {run['seed']} (C={comp})",
                        )
                    except Exception as e:
                        print(f"[Plot] Key {key} seed {run['seed']} skipped: {e}")

            if show_final:
                plt.plot(
                    u_vals,
                    final_vals,
                    color="blue",
                    linewidth=2,
                    label="Final SR",
                )

            plt.title(f"Key {key} | label={self.label}")
            plt.xlabel("u")
            plt.ylabel(self.response_function_label)
            plt.legend(fontsize=legend_fontsize)
            plt.tight_layout()
            plt.show()

            if show_residuals:
                plt.figure(figsize=figsize_residual)
                residual = final_vals - f_target
                plt.plot(
                    u_vals,
                    residual,
                    linewidth=1.5,
                    label="Final SR residual",
                )
                plt.plot(
                    u_vals,
                    np.sqrt(np.maximum(weights, 0)) * residual,
                    linewidth=1.5,
                    label=r"$\sqrt{w}$ residual",
                )
                plt.axhline(0.0, linewidth=1)
                plt.title(f"Residuals | Key {key} | label={self.label}")
                plt.xlabel("u")
                plt.ylabel("Residual")
                plt.legend(fontsize=legend_fontsize)
                plt.tight_layout()
                plt.show()

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the current reporter state as a dictionary.
        """
        return {
            "label": self.label,
            "response_symbol": self.response_symbol,
            "response_function_label": self.response_function_label,
            "keys": self.keys,
            "SR_runs_by_key": self.SR_runs_by_key,
            "exclusions_map": self.exclusions_map,
            "results_by_key": self.results_by_key,
            "all_seed_runtimes": self.all_seed_runtimes,
            "all_kept_seed_runtimes": self.all_kept_seed_runtimes,
            "global_summary": self.global_summary(),
        }