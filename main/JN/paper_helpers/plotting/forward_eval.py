from __future__ import annotations

# =========================
# Imports
# =========================
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..io.paths import build_save_base, predict_u_pred
from ..simulation.forward_batch import build_forward_sr_settings_tag


# =========================
# Plot style
# =========================
COLORS = {
    "true": "#111111",
    "binn": "#1f77b4",
    "fwd": "#8B4513",
    "fwd_fill": "#CD853F",
    "sr": "#d62728",
}

BINN_COLORS = [
    "#1f77b4",
    "#4a90d9",
    "#76b7eb",
    "#0b3c6d",
]

FWD_COLORS = [
    "#8c564b",
    "#a97463",
    "#c8a68c",
    "#5c4033",
]

SR_COLORS = [
    "#d62728",
    "#e15759",
    "#ff9896",
    "#8c1d18",
]

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    }
)


# =========================
# Helpers
# =========================
def mean_percentage_difference(arr: np.ndarray, u_true: np.ndarray) -> float:
    """
    Compute mean absolute percentage difference across seeds and time.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (S, T), one time series per seed.
    u_true : np.ndarray
        Ground-truth counts of shape (T,).

    Returns
    -------
    float
        Mean absolute percentage difference in percent.
    """
    pe_per_seed = np.mean(
        np.abs(arr - u_true[None, :]) / np.maximum(u_true[None, :], 1e-12),
        axis=1,
    ) * 100.0
    return float(pe_per_seed.mean())


def summarize_pe(pe_t: np.ndarray) -> Tuple[float, float, float]:
    """
    Summarize a per-time percentage-difference curve.

    Parameters
    ----------
    pe_t : np.ndarray
        Shape (T,) percentage-difference curve.

    Returns
    -------
    tuple
        (mean_pe, plus, minus), where plus and minus are distances from the mean
        to the max and min respectively.
    """
    mean_pe = float(pe_t.mean())
    min_pe = float(pe_t.min())
    max_pe = float(pe_t.max())
    plus = max_pe - mean_pe
    minus = mean_pe - min_pe
    return mean_pe, plus, minus


def safe_std(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Standard deviation with sensible behavior for a single seed.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    axis : int, default=0
        Reduction axis.

    Returns
    -------
    np.ndarray
        Standard deviation along the given axis. Returns zeros when there is only
        one sample along that axis.
    """
    if arr.shape[axis] <= 1:
        shape = list(arr.shape)
        del shape[axis]
        return np.zeros(shape, dtype=float)
    return arr.std(axis=axis, ddof=1)


@dataclass
class LegendConfig:
    loc: str = "best"
    ncols: int = 2
    fontsize: int = 10


# =========================
# Main class
# =========================
class ForwardSimulationAnalysis:
    """
    Analyze BINN, forward PDE, and optional SR-forward simulation counts.

    Parameters
    ----------
    keys : Sequence[Any]
        Dataset keys.
    dataInfo_dict : Mapping[Any, Mapping[str, Any]]
        Per-key metadata dictionary containing at least "hoursRange".
    dataObj_dict : Mapping[Any, Any]
        Maps keys to dataset/data objects.
    speciesLabels : str or Mapping[Any, str]
        Either one species label for all keys or a per-key mapping.
    binn_ES_list : Sequence[Any]
        Experimental-setting values to compare.
    binn_models_dics_ES : Mapping
        Nested mapping indexed by [ES][key][seed].
    binn_exts : Mapping
        Nested mapping indexed by [ES][key][seed] used to build filenames.
    binnSplitSeeds : Sequence[Any]
        Split seeds to aggregate over.
    init_type : str
        Forward-simulation initial-condition type, e.g. "pred" or "data".
    save_dir : str
        Primary output directory for saved figures.
    save_dir2 : str or None
        Optional secondary directory under "present/" for copied figures.
    device : str, default="cpu"
        Device used for BINN predictions.
    sr_save_params : dict or None, default=None
        Settings used to reconstruct SR forward-simulation filenames.
    fig_size : tuple, default=(7, 5)
        Default figure size.
    legend_config : LegendConfig or None, default=None
        Legend styling.
    show : bool, default=True
        Whether to display figures with plt.show().
    dpi : int, default=150
        Figure save dpi.
    counts_log_scale : bool, default=False
        Whether counts plots should use logarithmic y-scale.
    shift_time_to_zero : bool, default=False
        If True, shift each plotted time array so that it starts at 0.
    """

    VALID_PLOT_TYPES = {
        "counts_per_key",
        "diff_abs_per_key",
        "diff_pct_per_key",
        "all_keys_counts",
        "counts_grid_multi_es",
        "difference_grid_multi_es",
    }

    def __init__(
        self,
        *,
        keys: Sequence[Any],
        dataInfo_dict: Mapping[Any, Mapping[str, Any]],
        dataObj_dict: Mapping[Any, Any],
        speciesLabels: Any,
        binn_ES_list: Sequence[Any],
        binn_models_dics_ES: Mapping[Any, Mapping[Any, Mapping[Any, Any]]],
        binn_exts: Mapping[Any, Mapping[Any, Mapping[Any, Any]]],
        binnSplitSeeds: Sequence[Any],
        init_type: str,
        save_dir: str,
        save_dir2: Optional[str] = None,
        device: str = "cpu",
        sr_save_params: Optional[Mapping[str, Any]] = None,
        fig_size: Tuple[float, float] = (7, 5),
        legend_config: Optional[LegendConfig] = None,
        show: bool = True,
        dpi: int = 150,
        counts_log_scale: bool = False,
        base_dir: str = "fwd",
        base_dir_SR: str = "fwd/SR",
        xlabel_fontsize: float = 11.0,
        ylabel_fontsize: float = 11.0,
        xtick_labelsize: float = 10.0,
        ytick_labelsize: float = 10.0,
        title_fontsize: float = 11.0,
        annotation_fontsize: float = 10.0,
        major_tick_length: float = 4.0,
        major_tick_width: float = 1.0,
        minor_tick_length: float = 2.5,
        minor_tick_width: float = 0.8,
        shift_time_to_zero: bool = False,
    ) -> None:
        self.keys = list(keys)
        self.dataInfo_dict = dataInfo_dict
        self.dataObj_dict = dataObj_dict
        self.speciesLabels = speciesLabels
        self.binn_ES_list = list(binn_ES_list)
        self.binn_models_dics_ES = binn_models_dics_ES
        self.binn_exts = binn_exts
        self.binnSplitSeeds = list(binnSplitSeeds)
        self.init_type = init_type
        self.save_dir = save_dir
        self.save_dir2 = save_dir2
        self.device = device
        self.sr_save_params = dict(sr_save_params) if sr_save_params is not None else None
        self.fig_size = fig_size
        self.legend_config = legend_config or LegendConfig()
        self.show = show
        self.dpi = dpi
        self.counts_log_scale = counts_log_scale
        self.base_dir = base_dir
        self.base_dir_SR = base_dir_SR

        self.xlabel_fontsize = xlabel_fontsize
        self.ylabel_fontsize = ylabel_fontsize
        self.xtick_labelsize = xtick_labelsize
        self.ytick_labelsize = ytick_labelsize
        self.title_fontsize = title_fontsize
        self.annotation_fontsize = annotation_fontsize
        self.major_tick_length = major_tick_length
        self.major_tick_width = major_tick_width
        self.minor_tick_length = minor_tick_length
        self.minor_tick_width = minor_tick_width
        self.shift_time_to_zero = shift_time_to_zero

        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------
    def _get_species_label(self, key: Any) -> str:
        if isinstance(self.speciesLabels, Mapping):
            return self.speciesLabels[key]
        return self.speciesLabels

    def _get_true_counts(self, data_obj: Any) -> np.ndarray:
        counts = {dkey: dval.shape[0] for dkey, dval in data_obj.cdgs.items()}
        return np.asarray(list(counts.values()), dtype=float)

    def _get_count_scale_factor(self, data_obj: Any) -> float:
        area = data_obj.dx * data_obj.dy
        return area / 1e6

    def _get_plot_time(self, hours_range: np.ndarray) -> np.ndarray:
        """
        Return the plotting time array, optionally shifted so the first point is 0.
        """
        t = np.asarray(hours_range, dtype=float).copy()
        if self.shift_time_to_zero and t.size > 0:
            t = t - t[0]
        return t

    def _get_time_xlabel(self) -> str:
        return "Time since first measurement [hours]" if self.shift_time_to_zero else "Time [hours]"

    def _save_figure(self, fig: plt.Figure, filename: str) -> str:
        out_path = os.path.join(self.save_dir, filename)
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")

        if self.save_dir2 is not None:
            secondary_dir = os.path.join("present", self.save_dir2)
            os.makedirs(secondary_dir, exist_ok=True)
            out_path2 = os.path.join(secondary_dir, filename)
            shutil.copyfile(out_path, out_path2)
            print(f"[SAVE COPY] {out_path2}")

        print(f"[SAVE] {out_path}")
        return out_path

    def _finalize_figure(self, fig: plt.Figure) -> None:
        fig.tight_layout()
        if self.show:
            plt.show()
        plt.close(fig)

    def _style_axes(self, ax: plt.Axes) -> None:
        ax.tick_params(
            axis="x",
            which="major",
            labelsize=self.xtick_labelsize,
            length=self.major_tick_length,
            width=self.major_tick_width,
        )
        ax.tick_params(
            axis="y",
            which="major",
            labelsize=self.ytick_labelsize,
            length=self.major_tick_length,
            width=self.major_tick_width,
        )
        ax.tick_params(
            axis="x",
            which="minor",
            length=self.minor_tick_length,
            width=self.minor_tick_width,
        )
        ax.tick_params(
            axis="y",
            which="minor",
            length=self.minor_tick_length,
            width=self.minor_tick_width,
        )

    def _build_standard_forward_filename(self, es_val: Any, key: Any, seed: Any) -> str:
        binn_ext = self.binn_exts[es_val][key][seed].copy()
        binn_ext["binnTVsplitSeed"] = seed
        return build_save_base(
            binn_ext,
            f"ufwd_init{self.init_type}.npy",
            base_dir=self.base_dir,
        )

    def _build_sr_forward_filename(self, es_val: Any, key: Any, seed: Any) -> str:
        if self.sr_save_params is None:
            raise ValueError("sr_save_params must be provided to load SR forward simulations.")

        save_tag = build_forward_sr_settings_tag(
            population_size=self.sr_save_params["population_size"],
            niter=self.sr_save_params["niter"],
            label="DG",
            suffix=self.sr_save_params.get("save_suffix", "updated_mean"),
        )

        binn_ext = self.binn_exts[es_val][key][seed].copy()
        binn_ext["binnTVsplitSeed"] = seed
        return build_save_base(
            binn_ext,
            f"{save_tag}.npy",
            base_dir=self.base_dir_SR,
        )

    def _load_seed_timeseries(
        self,
        *,
        key: Any,
        es_val: Any,
        include_sr: bool,
    ) -> Dict[str, Any]:
        """
        Load per-seed count trajectories and compute aggregate statistics.

        Returns
        -------
        dict
            Aggregated stats and raw arrays for one (key, ES) pair.
        """
        data_obj = self.dataObj_dict[key]
        hours_range = np.asarray(self.dataInfo_dict[key]["hoursRange"], dtype=float)
        u_true = self._get_true_counts(data_obj)
        sf = self._get_count_scale_factor(data_obj)

        species_label = self._get_species_label(key)

        u_binn_list: List[np.ndarray] = []
        u_fwd_list: List[np.ndarray] = []
        u_fwd_sr_list: List[np.ndarray] = []

        for seed in self.binnSplitSeeds:
            u_pred_seed = (
                predict_u_pred(
                    self.binn_models_dics_ES[es_val][key][seed],
                    dataobj=data_obj,
                    device=self.device,
                )
                * data_obj.u_green_max
                * 1e6
            )

            fwd_filename = self._build_standard_forward_filename(es_val, key, seed)
            if not os.path.exists(fwd_filename):
                raise FileNotFoundError(
                    f"Missing forward simulation file for key={key}, ES={es_val}, seed={seed}:\n"
                    f"  {fwd_filename}"
                )

            u_fwd_seed = (
                np.load(fwd_filename)
                * data_obj.K
                * data_obj.u_green_max
                * 1e6
            )

            u_binn_t = u_pred_seed.sum(axis=(0, 1))
            u_fwd_t = u_fwd_seed.sum(axis=(0, 1))

            u_binn_list.append(u_binn_t)
            u_fwd_list.append(u_fwd_t)

            if include_sr:
                try:
                    sr_filename = self._build_sr_forward_filename(es_val, key, seed)
                except Exception as exc:
                    raise RuntimeError(
                        "Could not build SR forward filename. "
                        "Make sure sr_save_params matches the settings used in run_forward_sr_batch()."
                    ) from exc

                if os.path.exists(sr_filename):
                    u_fwd_sr_seed = (
                        np.load(sr_filename)
                        * data_obj.K
                        * data_obj.u_green_max
                        * 1e6
                    )
                    u_fwd_sr_t = u_fwd_sr_seed.sum(axis=(0, 1))
                    u_fwd_sr_list.append(u_fwd_sr_t)
                else:
                    print(
                        f"[WARN] Missing SR forward file for key={key}, ES={es_val}, seed={seed}:\n"
                        f"  {sr_filename}"
                    )

        u_binn_arr = np.vstack(u_binn_list) * sf
        u_fwd_arr = np.vstack(u_fwd_list) * sf
        sr_available = len(u_fwd_sr_list) > 0
        u_fwd_sr_arr = np.vstack(u_fwd_sr_list) * sf if sr_available else None

        binn_mean = u_binn_arr.mean(axis=0)
        binn_std = safe_std(u_binn_arr, axis=0)

        fwd_mean = u_fwd_arr.mean(axis=0)
        fwd_std = safe_std(u_fwd_arr, axis=0)

        denom = np.maximum(u_true, 1e-12)

        err_binn = np.abs(u_binn_arr - u_true[None, :])
        err_fwd = np.abs(u_fwd_arr - u_true[None, :])

        err_binn_m = err_binn.mean(axis=0)
        err_binn_s = safe_std(err_binn, axis=0)

        err_fwd_m = err_fwd.mean(axis=0)
        err_fwd_s = safe_std(err_fwd, axis=0)

        pe_binn_arr = np.abs(u_binn_arr - u_true[None, :]) / denom[None, :] * 100.0
        pe_fwd_arr = np.abs(u_fwd_arr - u_true[None, :]) / denom[None, :] * 100.0

        pe_binn = np.abs(binn_mean - u_true) / denom * 100.0
        pe_fwd = np.abs(fwd_mean - u_true) / denom * 100.0

        out: Dict[str, Any] = {
            "key": key,
            "es_val": es_val,
            "species_label": species_label,
            "hoursRange": hours_range,
            "u_true": u_true,
            "u_binn_arr": u_binn_arr,
            "u_fwd_arr": u_fwd_arr,
            "binn_mean": binn_mean,
            "binn_std": binn_std,
            "fwd_mean": fwd_mean,
            "fwd_std": fwd_std,
            "err_binn_m": err_binn_m,
            "err_binn_s": err_binn_s,
            "err_fwd_m": err_fwd_m,
            "err_fwd_s": err_fwd_s,
            "pe_binn": pe_binn,
            "pe_fwd": pe_fwd,
            "pe_binn_m": pe_binn_arr.mean(axis=0),
            "pe_binn_s": safe_std(pe_binn_arr, axis=0),
            "pe_fwd_m": pe_fwd_arr.mean(axis=0),
            "pe_fwd_s": safe_std(pe_fwd_arr, axis=0),
            "PE_binn_scalar": mean_percentage_difference(u_binn_arr, u_true),
            "PE_fwd_scalar": mean_percentage_difference(u_fwd_arr, u_true),
            "sr_available": sr_available,
        }

        if sr_available and u_fwd_sr_arr is not None:
            fwd_sr_mean = u_fwd_sr_arr.mean(axis=0)
            fwd_sr_std = safe_std(u_fwd_sr_arr, axis=0)

            err_fwd_sr = np.abs(u_fwd_sr_arr - u_true[None, :])
            err_fwd_sr_m = err_fwd_sr.mean(axis=0)
            err_fwd_sr_s = safe_std(err_fwd_sr, axis=0)

            pe_fwd_sr_arr = np.abs(u_fwd_sr_arr - u_true[None, :]) / denom[None, :] * 100.0
            pe_fwd_sr = np.abs(fwd_sr_mean - u_true) / denom * 100.0

            out.update(
                {
                    "u_fwd_sr_arr": u_fwd_sr_arr,
                    "fwd_sr_mean": fwd_sr_mean,
                    "fwd_sr_std": fwd_sr_std,
                    "err_fwd_sr_m": err_fwd_sr_m,
                    "err_fwd_sr_s": err_fwd_sr_s,
                    "pe_fwd_sr": pe_fwd_sr,
                    "pe_fwd_sr_m": pe_fwd_sr_arr.mean(axis=0),
                    "pe_fwd_sr_s": safe_std(pe_fwd_sr_arr, axis=0),
                    "PE_fwd_sr_scalar": mean_percentage_difference(u_fwd_sr_arr, u_true),
                }
            )

        print(
            f"[INFO] key={key}, ES={es_val} | "
            f"PE_binn={out['PE_binn_scalar']:.2f}% | "
            f"PE_fwd={out['PE_fwd_scalar']:.2f}%"
            + (
                f" | PE_fwd_sr={out['PE_fwd_sr_scalar']:.2f}%"
                if out["sr_available"]
                else ""
            )
        )

        return out

    def _collect_all_stats(self, include_sr: bool) -> Dict[Any, Dict[Any, Dict[str, Any]]]:
        """
        Collect stats for all keys and ES values.

        Returns
        -------
        dict
            Nested mapping stats[key][es_val] = stats_dict.
        """
        stats: Dict[Any, Dict[Any, Dict[str, Any]]] = {}
        for key in self.keys:
            stats[key] = {}
            for es_val in self.binn_ES_list:
                stats[key][es_val] = self._load_seed_timeseries(
                    key=key,
                    es_val=es_val,
                    include_sr=include_sr,
                )
        return stats

    # ------------------------------------------------------------
    # Plotting methods
    # ------------------------------------------------------------
    def _apply_counts_axis_style(self, ax: plt.Axes) -> None:
        ax.set_xlabel(self._get_time_xlabel(), fontsize=self.xlabel_fontsize)
        ax.set_ylabel("Cell counts", fontsize=self.ylabel_fontsize)
        if self.counts_log_scale:
            ax.set_yscale("log")
        ax.set_facecolor("white")
        self._style_axes(ax)

    def _apply_standard_axis_style(
        self,
        ax: plt.Axes,
        xlabel: str,
        ylabel: str,
    ) -> None:
        ax.set_xlabel(xlabel, fontsize=self.xlabel_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.ylabel_fontsize)
        ax.set_facecolor("white")
        self._style_axes(ax)

    def _plot_counts_per_key(self, stats_all: Dict[Any, Dict[Any, Dict[str, Any]]]) -> None:
        for key, per_es in stats_all.items():
            for es_val, stats in per_es.items():
                fig, ax = plt.subplots(figsize=self.fig_size)

                t = self._get_plot_time(stats["hoursRange"])
                ax.plot(
                    t,
                    stats["u_true"],
                    marker="o",
                    linestyle="-",
                    lw=1.0,
                    color=COLORS["true"],
                    label=r"$N_{data}$",
                )

                ax.plot(
                    t,
                    stats["binn_mean"],
                    linestyle="-",
                    lw=1.5,
                    color=COLORS["binn"],
                    label=r"$N_u$",
                )
                ax.fill_between(
                    t,
                    np.clip(stats["binn_mean"] - stats["binn_std"], 1e-12, None),
                    stats["binn_mean"] + stats["binn_std"],
                    alpha=0.25,
                    color=COLORS["binn"],
                    linewidth=0,
                )

                ax.plot(
                    t,
                    stats["fwd_mean"],
                    linestyle="--",
                    lw=1.5,
                    color=COLORS["fwd"],
                    label=r"$N_{fwd}$",
                )
                ax.fill_between(
                    t,
                    np.clip(stats["fwd_mean"] - stats["fwd_std"], 1e-12, None),
                    stats["fwd_mean"] + stats["fwd_std"],
                    alpha=0.25,
                    color=COLORS["fwd_fill"],
                    linewidth=0,
                )

                if stats["sr_available"]:
                    ax.plot(
                        t,
                        stats["fwd_sr_mean"],
                        linestyle="-.",
                        lw=1.5,
                        color=COLORS["sr"],
                        label=r"$N_{SR}$",
                    )
                    ax.fill_between(
                        t,
                        np.clip(stats["fwd_sr_mean"] - stats["fwd_sr_std"], 1e-12, None),
                        stats["fwd_sr_mean"] + stats["fwd_sr_std"],
                        alpha=0.20,
                        color=COLORS["sr"],
                        linewidth=0,
                    )

                self._apply_counts_axis_style(ax)
                ax.legend(
                    loc=self.legend_config.loc,
                    ncols=self.legend_config.ncols,
                    fontsize=self.legend_config.fontsize,
                )

                fig.tight_layout()
                self._save_figure(fig, f"{key}_ES{es_val}_counts.png")
                self._finalize_figure(fig)

    def _plot_diff_abs_per_key(self, stats_all: Dict[Any, Dict[Any, Dict[str, Any]]]) -> None:
        for key, per_es in stats_all.items():
            for es_val, stats in per_es.items():
                fig, ax = plt.subplots(figsize=self.fig_size)

                t = self._get_plot_time(stats["hoursRange"])

                ax.errorbar(
                    t,
                    stats["err_binn_m"],
                    yerr=stats["err_binn_s"],
                    fmt="-o",
                    capsize=3,
                    lw=1.0,
                    color=COLORS["binn"],
                    label="|BINN - data|",
                )
                ax.errorbar(
                    t,
                    stats["err_fwd_m"],
                    yerr=stats["err_fwd_s"],
                    fmt="--s",
                    capsize=3,
                    lw=1.0,
                    color=COLORS["fwd"],
                    label="|forward - data|",
                )

                if stats["sr_available"]:
                    ax.errorbar(
                        t,
                        stats["err_fwd_sr_m"],
                        yerr=stats["err_fwd_sr_s"],
                        fmt="-.^",
                        capsize=3,
                        lw=1.0,
                        color=COLORS["sr"],
                        label="|forward SR - data|",
                    )

                self._apply_standard_axis_style(
                    ax,
                    xlabel=self._get_time_xlabel(),
                    ylabel="Absolute difference (counts)",
                )
                ax.legend(
                    loc=self.legend_config.loc,
                    ncols=self.legend_config.ncols,
                    fontsize=self.legend_config.fontsize,
                )

                fig.tight_layout()
                self._save_figure(fig, f"{key}_ES{es_val}_diffs_abs.png")
                self._finalize_figure(fig)

    def _plot_diff_pct_per_key(self, stats_all: Dict[Any, Dict[Any, Dict[str, Any]]]) -> None:
        for key, per_es in stats_all.items():
            for es_val, stats in per_es.items():
                fig, ax = plt.subplots(figsize=self.fig_size)

                t = self._get_plot_time(stats["hoursRange"])

                ax.errorbar(
                    t,
                    stats["pe_binn_m"],
                    yerr=stats["pe_binn_s"],
                    fmt="-o",
                    capsize=3,
                    lw=1.0,
                    color=COLORS["binn"],
                    label="BINN",
                )
                ax.errorbar(
                    t,
                    stats["pe_fwd_m"],
                    yerr=stats["pe_fwd_s"],
                    fmt="--s",
                    capsize=3,
                    lw=1.0,
                    color=COLORS["fwd"],
                    label="forward",
                )

                if stats["sr_available"]:
                    ax.errorbar(
                        t,
                        stats["pe_fwd_sr_m"],
                        yerr=stats["pe_fwd_sr_s"],
                        fmt="-.^",
                        capsize=3,
                        lw=1.0,
                        color=COLORS["sr"],
                        label="forward SR",
                    )

                self._apply_standard_axis_style(
                    ax,
                    xlabel=self._get_time_xlabel(),
                    ylabel="Percentage difference [%]",
                )
                ax.legend(
                    loc=self.legend_config.loc,
                    ncols=self.legend_config.ncols,
                    fontsize=self.legend_config.fontsize,
                )

                fig.tight_layout()
                self._save_figure(fig, f"{key}_ES{es_val}_diffs_pct.png")
                self._finalize_figure(fig)

    def _plot_all_keys_counts(self, stats_all: Dict[Any, Dict[Any, Dict[str, Any]]]) -> None:
        for es_val in self.binn_ES_list:
            fig, ax = plt.subplots(figsize=self.fig_size)

            legend_handles = {}
            legend_labels = {}

            Rep_num = {
                "2_3": 2,
                "2_5": 1,
                "3_1": 3,
            }

            for key_idx, key in enumerate(self.keys):
                stats = stats_all[key][es_val]
                t = self._get_plot_time(stats["hoursRange"])

                h_data, = ax.plot(
                    t,
                    stats["u_true"],
                    marker="o",
                    linestyle="-",
                    markersize=3,
                    lw=1.0,
                    color=COLORS["true"],
                )
                h_binn, = ax.plot(
                    t,
                    stats["binn_mean"],
                    linestyle="-",
                    lw=1.0,
                    color=COLORS["binn"],
                )
                ax.fill_between(
                    t,
                    stats["u_binn_arr"].min(axis=0),
                    stats["u_binn_arr"].max(axis=0),
                    color=COLORS["binn"],
                    alpha=0.12,
                    linewidth=0,
                )

                h_fwd, = ax.plot(
                    t,
                    stats["fwd_mean"],
                    linestyle="--",
                    lw=1.0,
                    color=COLORS["fwd"],
                )
                ax.fill_between(
                    t,
                    stats["u_fwd_arr"].min(axis=0),
                    stats["u_fwd_arr"].max(axis=0),
                    color=COLORS["fwd"],
                    alpha=0.12,
                    linewidth=0,
                )

                h_sr = None
                if stats["sr_available"]:
                    h_sr, = ax.plot(
                        t,
                        stats["fwd_sr_mean"],
                        linestyle="-.",
                        lw=1.0,
                        color=COLORS["sr"],
                    )
                    ax.fill_between(
                        t,
                        stats["u_fwd_sr_arr"].min(axis=0),
                        stats["u_fwd_sr_arr"].max(axis=0),
                        color=COLORS["sr"],
                        alpha=0.10,
                        linewidth=0,
                    )

                if "data" not in legend_handles:
                    legend_handles["data"] = h_data
                    legend_labels["data"] = r"$N_{data}$"
                if "BINN" not in legend_handles:
                    legend_handles["BINN"] = h_binn
                    legend_labels["BINN"] = r"$N_u$"
                if "forward" not in legend_handles:
                    legend_handles["forward"] = h_fwd
                    legend_labels["forward"] = r"$N_{fwd}$"
                if h_sr is not None and "forward SR" not in legend_handles:
                    legend_handles["forward SR"] = h_sr
                    legend_labels["forward SR"] = r"$N_{SR}$"

                ax.text(
                    t[-1],
                    stats["u_true"][-1] + 45,
                    f"Repl. {Rep_num[key]}",
                    fontsize=self.annotation_fontsize,
                    ha="center",
                    va="center",
                    color="black",
                )

            self._apply_counts_axis_style(ax)
            order = ["data", "BINN", "forward", "forward SR"]
            handles = [legend_handles[m] for m in order if m in legend_handles]
            labels = [legend_labels[m] for m in order if m in legend_labels]
            ax.legend(
                handles,
                labels,
                loc=self.legend_config.loc,
                ncols=self.legend_config.ncols,
                fontsize=self.legend_config.fontsize,
            )

            fig.tight_layout()
            self._save_figure(fig, f"ALLKEYS_ES{es_val}_counts.png")
            self._finalize_figure(fig)

    def _plot_counts_grid_multi_es(self, stats_all: Dict[Any, Dict[Any, Dict[str, Any]]]) -> None:
        for key, per_es in stats_all.items():
            n_es = len(self.binn_ES_list)
            fig, axes = plt.subplots(
                n_es,
                1,
                figsize=(8, 4 + 3 * max(n_es - 1, 0)),
                sharex=True,
            )
            if n_es == 1:
                axes = [axes]

            for idx, (ax, es_val) in enumerate(zip(axes, self.binn_ES_list)):
                stats = per_es[es_val]
                t = self._get_plot_time(stats["hoursRange"])

                binn_color = BINN_COLORS[idx % len(BINN_COLORS)]
                fwd_color = FWD_COLORS[idx % len(FWD_COLORS)]
                sr_color = SR_COLORS[idx % len(SR_COLORS)]

                ax.plot(
                    t,
                    stats["u_true"],
                    marker="o",
                    lw=1.0,
                    ms=4,
                    color=COLORS["true"],
                    label="True",
                )

                ax.plot(
                    t,
                    stats["binn_mean"],
                    lw=1.5,
                    color=binn_color,
                    label=f"BINN | ES={es_val}",
                )
                ax.fill_between(
                    t,
                    np.clip(stats["binn_mean"] - stats["binn_std"], 1e-12, None),
                    stats["binn_mean"] + stats["binn_std"],
                    alpha=0.20,
                    color=binn_color,
                    linewidth=0,
                )

                ax.plot(
                    t,
                    stats["fwd_mean"],
                    lw=1.5,
                    color=fwd_color,
                    label=f"Fwd Sim | ES={es_val}",
                )
                ax.fill_between(
                    t,
                    np.clip(stats["fwd_mean"] - stats["fwd_std"], 1e-12, None),
                    stats["fwd_mean"] + stats["fwd_std"],
                    alpha=0.20,
                    color=fwd_color,
                    linewidth=0,
                )

                if stats["sr_available"]:
                    ax.plot(
                        t,
                        stats["fwd_sr_mean"],
                        lw=1.5,
                        linestyle="-.",
                        color=sr_color,
                        label=f"Fwd SR | ES={es_val}",
                    )
                    ax.fill_between(
                        t,
                        np.clip(stats["fwd_sr_mean"] - stats["fwd_sr_std"], 1e-12, None),
                        stats["fwd_sr_mean"] + stats["fwd_sr_std"],
                        alpha=0.18,
                        color=sr_color,
                        linewidth=0,
                    )

                binn_mean_pe, binn_plus, binn_minus = summarize_pe(stats["pe_binn"])
                fwd_mean_pe, fwd_plus, fwd_minus = summarize_pe(stats["pe_fwd"])

                title = (
                    f"{stats['species_label']}{key} | ES={es_val} | "
                    f"BINN {binn_mean_pe:.2f}% (+{binn_plus:.2f}/-{binn_minus:.2f}), "
                    f"Fwd {fwd_mean_pe:.2f}% (+{fwd_plus:.2f}/-{fwd_minus:.2f})"
                )
                if stats["sr_available"]:
                    sr_mean_pe, sr_plus, sr_minus = summarize_pe(stats["pe_fwd_sr"])
                    title += f", SR {sr_mean_pe:.2f}% (+{sr_plus:.2f}/-{sr_minus:.2f})"

                ax.set_title(title, fontsize=self.title_fontsize)
                ax.set_ylabel("Counts", fontsize=self.ylabel_fontsize)
                if self.counts_log_scale:
                    ax.set_yscale("log")
                self._style_axes(ax)
                ax.legend(
                    loc=self.legend_config.loc,
                    ncols=self.legend_config.ncols,
                    fontsize=self.legend_config.fontsize,
                )

            axes[-1].set_xlabel(self._get_time_xlabel(), fontsize=self.xlabel_fontsize)
            self._style_axes(axes[-1])

            fig.tight_layout()
            self._save_figure(fig, f"{key}_counts_grid_multi_es.png")
            self._finalize_figure(fig)

    def _plot_difference_grid_multi_es(self, stats_all: Dict[Any, Dict[Any, Dict[str, Any]]]) -> None:
        for key, per_es in stats_all.items():
            n_es = len(self.binn_ES_list)
            fig, axes = plt.subplots(
                2,
                n_es,
                figsize=(4 * n_es + 2, 8),
                sharex=True,
                gridspec_kw={"height_ratios": [2, 1]},
            )

            if n_es == 1:
                axes = np.array(axes).reshape(2, 1)

            for idx, es_val in enumerate(self.binn_ES_list):
                stats = per_es[es_val]
                t = self._get_plot_time(stats["hoursRange"])

                binn_color = BINN_COLORS[idx % len(BINN_COLORS)]
                fwd_color = FWD_COLORS[idx % len(FWD_COLORS)]
                sr_color = SR_COLORS[idx % len(SR_COLORS)]

                ax1 = axes[0, idx]
                ax2 = axes[1, idx]

                ax1.plot(t, stats["err_binn_m"], lw=2, color=binn_color, label=f"BINN | ES={es_val}")
                ax1.fill_between(
                    t,
                    np.clip(stats["err_binn_m"] - stats["err_binn_s"], 1e-12, None),
                    stats["err_binn_m"] + stats["err_binn_s"],
                    alpha=0.20,
                    color=binn_color,
                    linewidth=0,
                )

                ax1.plot(t, stats["err_fwd_m"], lw=2, color=fwd_color, label=f"Fwd Sim | ES={es_val}")
                ax1.fill_between(
                    t,
                    np.clip(stats["err_fwd_m"] - stats["err_fwd_s"], 1e-12, None),
                    stats["err_fwd_m"] + stats["err_fwd_s"],
                    alpha=0.20,
                    color=fwd_color,
                    linewidth=0,
                )

                if stats["sr_available"]:
                    ax1.plot(
                        t,
                        stats["err_fwd_sr_m"],
                        lw=2,
                        linestyle="-.",
                        color=sr_color,
                        label=f"Fwd SR | ES={es_val}",
                    )
                    ax1.fill_between(
                        t,
                        np.clip(stats["err_fwd_sr_m"] - stats["err_fwd_sr_s"], 1e-12, None),
                        stats["err_fwd_sr_m"] + stats["err_fwd_sr_s"],
                        alpha=0.18,
                        color=sr_color,
                        linewidth=0,
                    )

                if idx == 0:
                    ax1.set_ylabel("Absolute count difference", fontsize=self.ylabel_fontsize)
                self._style_axes(ax1)
                ax1.legend(
                    loc=self.legend_config.loc,
                    ncols=1,
                    fontsize=self.legend_config.fontsize,
                )

                ax2.plot(t, stats["pe_binn"], lw=2, color=binn_color, label=f"BINN | ES={es_val}")
                ax2.plot(t, stats["pe_fwd"], lw=2, color=fwd_color, label=f"Fwd Sim | ES={es_val}")

                if stats["sr_available"]:
                    ax2.plot(
                        t,
                        stats["pe_fwd_sr"],
                        lw=2,
                        linestyle="-.",
                        color=sr_color,
                        label=f"Fwd SR | ES={es_val}",
                    )

                if idx == 0:
                    ax2.set_ylabel("Absolute % difference", fontsize=self.ylabel_fontsize)
                ax2.set_xlabel(self._get_time_xlabel(), fontsize=self.xlabel_fontsize)
                self._style_axes(ax2)
                ax2.legend(
                    loc=self.legend_config.loc,
                    ncols=1,
                    fontsize=self.legend_config.fontsize,
                )

            fig.tight_layout()
            self._save_figure(fig, f"{key}_difference_grid_multi_es.png")
            self._finalize_figure(fig)

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def run(
        self,
        *,
        include_sr: bool = False,
        plot_types: Optional[Sequence[str]] = None,
    ) -> Dict[Any, Dict[Any, Dict[str, Any]]]:
        """
        Run the analysis and generate only the requested plot types.

        Parameters
        ----------
        include_sr : bool, default=False
            Whether to attempt loading and plotting SR forward simulations.
        plot_types : sequence of str or None, default=None
            Exact plot types to generate. If None, defaults to:
                ["counts_per_key", "diff_abs_per_key", "diff_pct_per_key"]

        Returns
        -------
        dict
            Nested statistics dictionary:
                stats[key][es_val] = stats_dict
        """
        if plot_types is None:
            plot_types = [
                "counts_per_key",
                "diff_abs_per_key",
                "diff_pct_per_key",
            ]

        plot_types = list(plot_types)
        invalid = [p for p in plot_types if p not in self.VALID_PLOT_TYPES]
        if invalid:
            raise ValueError(
                f"Unknown plot_types: {invalid}. "
                f"Valid options are: {sorted(self.VALID_PLOT_TYPES)}"
            )

        stats_all = self._collect_all_stats(include_sr=include_sr)

        if "counts_per_key" in plot_types:
            self._plot_counts_per_key(stats_all)

        if "diff_abs_per_key" in plot_types:
            self._plot_diff_abs_per_key(stats_all)

        if "diff_pct_per_key" in plot_types:
            self._plot_diff_pct_per_key(stats_all)

        if "all_keys_counts" in plot_types:
            self._plot_all_keys_counts(stats_all)

        if "counts_grid_multi_es" in plot_types:
            self._plot_counts_grid_multi_es(stats_all)

        if "difference_grid_multi_es" in plot_types:
            self._plot_difference_grid_multi_es(stats_all)

        return stats_all