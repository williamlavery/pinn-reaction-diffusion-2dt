from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..io.paths import (
    _get_species_u,
    build_save_base,
    build_save_path,
    should_skip_all,
    to_torch,
)


class MidlinePlotter:
    """Plot mid-slice line comparisons for average BINN predictions."""

    def __init__(
        self,
        device: str = "cpu",
        pred_colors: Optional[Sequence[str]] = None,
        data_colors: Optional[Sequence[str]] = None,
        fwd_colors: Optional[Sequence[str]] = None,
    ) -> None:
        self.device = device

        self.pred_colors = list(pred_colors) if pred_colors is not None else [
            "#1f77b4",  # t=0
            "#4f9fd8",  # t=final
        ]
        self.data_colors = list(data_colors) if data_colors is not None else [
            "#003300",
            "#006600",
            "#009933",
            "#66CC66",
            "#CCFFCC",
        ]
        self.fwd_colors = list(fwd_colors) if fwd_colors is not None else [
            "#8B4513",  # t=0
            "#C19A6B",  # t=final
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_save_target(
        save_dic: Optional[Dict[str, Any]],
        save_name: Optional[str],
        base_dir: str,
        overwrite: bool,
    ) -> Optional[str]:
        """Resolve save path and respect overwrite rules."""
        target = build_save_path(save_dic, save_name, base_dir) if save_dic is not None else None
        out_paths = [target] if target else []
        if should_skip_all(out_paths, overwrite):
            return None
        return target

    @staticmethod
    def _save_two_figures(
        fig1: plt.Figure,
        fig2: plt.Figure,
        save_name: Optional[str],
        target: Optional[str],
    ) -> None:
        """Save the x2-mid and x1-mid figures with consistent suffixes."""
        if not save_name:
            return

        base = target if target else save_name
        root, ext = os.path.splitext(base)
        if not ext:
            ext = ".png"

        fig1.savefig(
            root + "_x2mid" + ext,
            dpi=100,
            bbox_inches="tight",
            facecolor="None",
        )
        fig2.savefig(
            root + "_x1mid" + ext,
            dpi=100,
            bbox_inches="tight",
            facecolor="None",
        )

    def _get_scaled_data_field(
        self,
        dataobj: Any,
        species_label: str,
    ) -> np.ndarray:
        """Get the reference data field in physical units."""
        return _get_species_u(dataobj, species_label) * dataobj.u_green_max * 10**6

    def _collect_prediction_average(
        self,
        model_wrappers: Dict[Any, Any],
        dataobj: Any,
        K: float,
    ) -> np.ndarray:
        """Collect average BINN predictions as [M, Nx1, Nx2, Nt] in physical units."""
        t, x1, x2 = dataobj.t, dataobj.x1, dataobj.x2
        Nt, Nx1, Nx2 = len(t), len(x1), len(x2)

        preds = []
        for wrapper in model_wrappers.values():
            model = wrapper.model
            model.eval()
            with torch.no_grad():
                pred_flat = model(to_torch(dataobj.inputs, self.device)) * K
                pred = pred_flat.reshape(Nx1, Nx2, Nt).cpu().numpy()
            preds.append(pred)

        return np.stack(preds, axis=0) * dataobj.u_green_max * 10**6

    def _collect_forward_average(
        self,
        model_wrappers: Dict[Any, Any],
        binn_exts_key: Dict[Any, Dict[str, Any]],
        dataobj: Any,
        init_type: str,
    ) -> np.ndarray:
        """Load average forward simulations as [M, Nx1, Nx2, Nt] in physical units."""
        u_fwd_list = []

        for split_seed in sorted(model_wrappers.keys()):
            binn_ext = copy.deepcopy(binn_exts_key[split_seed])
            binn_ext["binnTVsplitSeed"] = split_seed

            filename = build_save_base(
                binn_ext,
                f"ufwd_init{init_type}.npy",
                base_dir="plots",
            )
            u_fwd_seed = np.load(filename) * dataobj.K * dataobj.u_green_max * 10**6
            u_fwd_list.append(u_fwd_seed)

        return np.stack(u_fwd_list, axis=0)

    @staticmethod
    def _average_stats(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return mean, min, max across average axis."""
        return U.mean(axis=0), U.min(axis=0), U.max(axis=0)

    @staticmethod
    def _format_coord(value: float) -> str:
        """Format fixed-coordinate values cleanly for legend labels."""
        return f"{value:.3g}"

    @staticmethod
    def _format_time_as_int(value: float) -> str:
        """Format time values as hours for legend labels."""
        return f"{int(round(value * 24))}"

    def _make_slice_labels(
        self,
        fixed_axis: str,
        fixed_value: float,
        final_time: float,
    ) -> Dict[str, str]:
        """Build legend labels with fixed ordering x_1, x_2, t."""
        fixed_value_str = self._format_coord(fixed_value)
        final_time_str = self._format_time_as_int(final_time)

        if fixed_axis == "x_1":
            x1_term = f"x_1={fixed_value_str}"
            x2_term = "x_2"
        elif fixed_axis == "x_2":
            x1_term = "x_1"
            x2_term = f"x_2={fixed_value_str}"
        else:
            raise ValueError(f"Unsupported fixed_axis: {fixed_axis}")

        pred_t0 = rf"$\hat{{u}}_1({x1_term},\tilde{{x}}_2,\,t_0)$"
        pred_tT = rf"$\hat{{u}}_1({x1_term},\tilde{{x}}_2,\,t_f)$"

        fwd_t0 = rf"$u_{{\mathrm{{fwd}}}}({x1_term},\tilde{{x}}_2,\,t_0)$"
        fwd_tT = rf"$u_{{\mathrm{{fwd}}}}({x1_term}, \tilde{{x}}_2,\,t_f)$"

        data_t0 = rf"$u_{{\mathrm{{data}},1}}({x1_term},\tilde{{x}}_2,\,t_0)$"
        data_tT = rf"$u_{{\mathrm{{data}},1}}({x1_term},\tilde{{x}}_2,\,t_f)$"

        return {
            "pred_t0": pred_t0,
            "pred_tT": pred_tT,
            "fwd_t0": fwd_t0,
            "fwd_tT": fwd_tT,
            "data_t0": data_t0,
            "data_tT": data_tT,
        }

    @staticmethod
    def _style_axes(
        ax: plt.Axes,
        xlabel: str,
        ylabel: str,
        xlabel_fontsize: int,
        ylabel_fontsize: int,
        xtick_labelsize: int,
        ytick_labelsize: int,
        tick_length: float,
        tick_width: float,
        minor_tick_length: Optional[float] = None,
        minor_tick_width: Optional[float] = None,
    ) -> None:
        """Apply all axis font and tick styling explicitly."""
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        ax.tick_params(
            axis="x",
            which="major",
            labelsize=xtick_labelsize,
            length=tick_length,
            width=tick_width,
        )
        ax.tick_params(
            axis="y",
            which="major",
            labelsize=ytick_labelsize,
            length=tick_length,
            width=tick_width,
        )

        if minor_tick_length is not None or minor_tick_width is not None:
            ax.minorticks_on()
            ax.tick_params(
                axis="x",
                which="minor",
                length=tick_length if minor_tick_length is None else minor_tick_length,
                width=tick_width if minor_tick_width is None else minor_tick_width,
            )
            ax.tick_params(
                axis="y",
                which="minor",
                length=tick_length if minor_tick_length is None else minor_tick_length,
                width=tick_width if minor_tick_width is None else minor_tick_width,
            )

    def _plot_single_midline(
        self,
        ax: plt.Axes,
        axis_values: np.ndarray,
        pred_mean_t0: np.ndarray,
        pred_min_t0: np.ndarray,
        pred_max_t0: np.ndarray,
        pred_mean_tT: np.ndarray,
        pred_min_tT: np.ndarray,
        pred_max_tT: np.ndarray,
        data_t0: np.ndarray,
        data_tT: np.ndarray,
        xlabel: str,
        legend_loc: str,
        legend_ncols: int,
        legend_fontsize: int,
        labels: Dict[str, str],
        xlabel_fontsize: int,
        ylabel_fontsize: int,
        xtick_labelsize: int,
        ytick_labelsize: int,
        tick_length: float,
        tick_width: float,
        minor_tick_length: Optional[float] = None,
        minor_tick_width: Optional[float] = None,
        fwd_mean_t0: Optional[np.ndarray] = None,
        fwd_min_t0: Optional[np.ndarray] = None,
        fwd_max_t0: Optional[np.ndarray] = None,
        fwd_mean_tT: Optional[np.ndarray] = None,
        fwd_min_tT: Optional[np.ndarray] = None,
        fwd_max_tT: Optional[np.ndarray] = None,
        ylim: Optional[int] = None,
    ) -> None:
        """Draw one midline plot with prediction, optional forward sim, and data."""
        line_pred_0, = ax.plot(
            axis_values,
            pred_mean_t0,
            "-",
            lw=2,
            alpha=0.9,
            color=self.pred_colors[0],
            marker=".",
            label=labels["pred_t0"],
        )
        ax.fill_between(
            axis_values,
            pred_min_t0,
            pred_max_t0,
            alpha=0.25,
            color=self.pred_colors[0],
            label="_nolegend_",
        )

        line_pred_T, = ax.plot(
            axis_values,
            pred_mean_tT,
            "-",
            lw=2,
            alpha=0.9,
            color=self.pred_colors[1],
            marker=".",
            label=labels["pred_tT"],
        )
        ax.fill_between(
            axis_values,
            pred_min_tT,
            pred_max_tT,
            alpha=0.25,
            color=self.pred_colors[1],
            label="_nolegend_",
        )

        handles = [line_pred_0, line_pred_T]

        if fwd_mean_t0 is not None and fwd_mean_tT is not None:
            line_fwd_0, = ax.plot(
                axis_values,
                fwd_mean_t0,
                marker="s",
                markersize=3,
                linestyle="--",
                lw=2,
                alpha=0.9,
                color=self.fwd_colors[0],
                label=labels["fwd_t0"],
            )
            ax.fill_between(
                axis_values,
                fwd_min_t0,
                fwd_max_t0,
                alpha=0.25,
                color=self.fwd_colors[0],
                label="_nolegend_",
            )

            line_fwd_T, = ax.plot(
                axis_values,
                fwd_mean_tT,
                marker="s",
                markersize=3,
                linestyle="--",
                lw=2,
                alpha=0.9,
                color=self.fwd_colors[1],
                label=labels["fwd_tT"],
            )
            ax.fill_between(
                axis_values,
                fwd_min_tT,
                fwd_max_tT,
                alpha=0.25,
                color=self.fwd_colors[1],
                label="_nolegend_",
            )

            handles.extend([line_fwd_0, line_fwd_T])

        line_data_0, = ax.plot(
            axis_values,
            data_t0,
            marker=".",
            color=self.data_colors[2],
            lw=2,
            label=labels["data_t0"],
        )
        line_data_T, = ax.plot(
            axis_values,
            data_tT,
            marker=".",
            color=self.data_colors[0],
            lw=2,
            label=labels["data_tT"],
        )
        handles.extend([line_data_0, line_data_T])

        ax.set_facecolor("white")
        self._style_axes(
            ax=ax,
            xlabel=xlabel,
            ylabel="Cell density [cells mm$^{-2}$]",
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            tick_length=tick_length,
            tick_width=tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
        )

        ax.legend(
            handles=handles,
            loc=legend_loc,
            ncol=legend_ncols,
            fontsize=legend_fontsize,
            framealpha=0.5,
            facecolor="white",
        )
        
        ax.set_ylim(ylim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_midline_prediction_vs_data(
        self,
        model_wrappers: Dict[Any, Any],
        dataobj: Any,
        save_dic: Optional[Dict[str, Any]] = None,
        save_name: Optional[str] = None,
        base_dir: str = "plots",
        species_label: str = "green",
        K: float = 1.0,
        overwrite: bool = False,
        legend_ncols: int = 2,
        legend_fontsize: int = 10,
        figsize: Tuple[float, float] = (6, 5),
        legend_loc_x2: str = "upper left",
        legend_loc_x1: str = "upper left",
        x1_idx: Optional[int] = None,
        x2_idx: Optional[int] = None,
        ylim: Optional[int] = None,
        xlabel_fontsize: int = 11,
        ylabel_fontsize: int = 11,
        xtick_labelsize: int = 10,
        ytick_labelsize: int = 10,
        tick_length: float = 4.0,
        tick_width: float = 1.0,
        minor_tick_length: Optional[float] = None,
        minor_tick_width: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Plot average BINN prediction vs data along the two spatial midlines.

        Produces:
        1. x2-varying midline at fixed x1 midpoint
        2. x1-varying midline at fixed x2 midpoint

        Font/tick controls
        ------------------
        xlabel_fontsize : int
            Fontsize of x-axis label.
        ylabel_fontsize : int
            Fontsize of y-axis label.
        xtick_labelsize : int
            Fontsize of x tick labels.
        ytick_labelsize : int
            Fontsize of y tick labels.
        tick_length : float
            Major tick length.
        tick_width : float
            Major tick width.
        minor_tick_length : float, optional
            Minor tick length. If None, minor ticks are left off unless width is given.
        minor_tick_width : float, optional
            Minor tick width.

        Returns
        -------
        np.ndarray or None
            average mean prediction field with shape [Nx1, Nx2, Nt].
        """
        target = self._resolve_save_target(save_dic, save_name, base_dir, overwrite)
        if save_dic is not None and save_name is not None and target is None:
            return None

        u_data = self._get_scaled_data_field(dataobj, species_label)
        U_pred = self._collect_prediction_average(model_wrappers, dataobj, K)
        u_pred_mean, u_pred_min, u_pred_max = self._average_stats(U_pred)

        t, x1, x2 = dataobj.t, dataobj.x1, dataobj.x2
        Nt, Nx1, Nx2 = len(t), len(x1), len(x2)

        t0_idx = 0
        tT_idx = -1

        x1_idx = x1_idx if x1_idx is not None else Nx1 // 2
        x2_idx = x2_idx if x2_idx is not None else Nx2 // 2

        labels_x2mid = self._make_slice_labels(
            fixed_axis="x_1",
            fixed_value=x1[x1_idx],
            final_time=t[tT_idx],
        )

        labels_x1mid = self._make_slice_labels(
            fixed_axis="x_2",
            fixed_value=x2[x2_idx],
            final_time=t[tT_idx],
        )

        fig1, ax1 = plt.subplots(figsize=figsize)
        self._plot_single_midline(
            ax=ax1,
            axis_values=x2,
            pred_mean_t0=u_pred_mean[x1_idx, :, t0_idx],
            pred_min_t0=u_pred_min[x1_idx, :, t0_idx],
            pred_max_t0=u_pred_max[x1_idx, :, t0_idx],
            pred_mean_tT=u_pred_mean[x1_idx, :, tT_idx],
            pred_min_tT=u_pred_min[x1_idx, :, tT_idx],
            pred_max_tT=u_pred_max[x1_idx, :, tT_idx],
            data_t0=u_data[x1_idx, :, t0_idx],
            data_tT=u_data[x1_idx, :, tT_idx],
            xlabel="x2 [mm]",
            legend_loc=legend_loc_x2,
            legend_ncols=legend_ncols,
            legend_fontsize=legend_fontsize,
            labels=labels_x2mid,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            tick_length=tick_length,
            tick_width=tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
            ylim = ylim,
        )
        fig1.tight_layout()

        fig2, ax2 = plt.subplots(figsize=figsize)
        self._plot_single_midline(
            ax=ax2,
            axis_values=x1,
            pred_mean_t0=u_pred_mean[:, x2_idx, t0_idx],
            pred_min_t0=u_pred_min[:, x2_idx, t0_idx],
            pred_max_t0=u_pred_max[:, x2_idx, t0_idx],
            pred_mean_tT=u_pred_mean[:, x2_idx, tT_idx],
            pred_min_tT=u_pred_min[:, x2_idx, tT_idx],
            pred_max_tT=u_pred_max[:, x2_idx, tT_idx],
            data_t0=u_data[:, x2_idx, t0_idx],
            data_tT=u_data[:, x2_idx, tT_idx],
            xlabel=r"$x_1$ [mm]",
            legend_loc=legend_loc_x1,
            legend_ncols=legend_ncols,
            legend_fontsize=legend_fontsize,
            labels=labels_x1mid,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            tick_length=tick_length,
            tick_width=tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
            ylim = ylim,
        )
        fig2.tight_layout()

        self._save_two_figures(fig1, fig2, save_name, target)
        plt.show()

        return u_pred_mean

    def plot_midline_prediction_forward_data(
        self,
        model_wrappers: Dict[Any, Any],
        binn_exts_key: Dict[Any, Dict[str, Any]],
        dataobj: Any,
        save_dic: Optional[Dict[str, Any]] = None,
        save_name: Optional[str] = None,
        base_dir: str = "plots",
        species_label: str = "green",
        K: float = 1.0,
        overwrite: bool = False,
        init_type: str = "pred",
        legend_loc_x2: str = "upper left",
        legend_loc_x1: str = "upper left",
        figsize: Tuple[float, float] = (7, 5),
        legend_ncols: int = 6,
        legend_fontsize: int = 10,
        plot_forward: bool = True,
        x1_idx: Optional[int] = None,
        x2_idx: Optional[int] = None,
        xlabel_fontsize: int = 11,
        ylabel_fontsize: int = 11,
        xtick_labelsize: int = 10,
        ytick_labelsize: int = 10,
        tick_length: float = 4.0,
        tick_width: float = 1.0,
        minor_tick_length: Optional[float] = None,
        minor_tick_width: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Plot average BINN prediction vs optional forward simulation vs data
        along the two spatial midlines.

        Produces:
        1. x2-varying midline at fixed x1 midpoint
        2. x1-varying midline at fixed x2 midpoint

        Font/tick controls
        ------------------
        xlabel_fontsize : int
            Fontsize of x-axis label.
        ylabel_fontsize : int
            Fontsize of y-axis label.
        xtick_labelsize : int
            Fontsize of x tick labels.
        ytick_labelsize : int
            Fontsize of y tick labels.
        tick_length : float
            Major tick length.
        tick_width : float
            Major tick width.
        minor_tick_length : float, optional
            Minor tick length.
        minor_tick_width : float, optional
            Minor tick width.

        Returns
        -------
        (u_pred_mean, u_fwd_mean)
            average mean prediction and average mean forward simulation.
            If plot_forward=False, u_fwd_mean is None.
        """
        target = self._resolve_save_target(save_dic, save_name, base_dir, overwrite)
        if save_dic is not None and save_name is not None and target is None:
            return None, None

        u_data = self._get_scaled_data_field(dataobj, species_label)

        U_pred = self._collect_prediction_average(model_wrappers, dataobj, K)
        u_pred_mean, u_pred_min, u_pred_max = self._average_stats(U_pred)

        if plot_forward:
            U_fwd = self._collect_forward_average(
                model_wrappers=model_wrappers,
                binn_exts_key=binn_exts_key,
                dataobj=dataobj,
                init_type=init_type,
            )
            u_fwd_mean, u_fwd_min, u_fwd_max = self._average_stats(U_fwd)
        else:
            u_fwd_mean = u_fwd_min = u_fwd_max = None

        t, x1, x2 = dataobj.t, dataobj.x1, dataobj.x2
        Nt, Nx1, Nx2 = len(t), len(x1), len(x2)

        t0_idx = 0
        tT_idx = -1

        x1_idx = x1_idx if x1_idx is not None else Nx1 // 2
        x2_idx = x2_idx if x2_idx is not None else Nx2 // 2

        labels_x2mid = self._make_slice_labels(
            fixed_axis="x_1",
            fixed_value=x1[x1_idx],
            final_time=t[tT_idx],
        )

        labels_x1mid = self._make_slice_labels(
            fixed_axis="x_2",
            fixed_value=x2[x2_idx],
            final_time=t[tT_idx],
        )

        fig1, ax1 = plt.subplots(figsize=figsize)
        self._plot_single_midline(
            ax=ax1,
            axis_values=x2,
            pred_mean_t0=u_pred_mean[x1_idx, :, t0_idx],
            pred_min_t0=u_pred_min[x1_idx, :, t0_idx],
            pred_max_t0=u_pred_max[x1_idx, :, t0_idx],
            pred_mean_tT=u_pred_mean[x1_idx, :, tT_idx],
            pred_min_tT=u_pred_min[x1_idx, :, tT_idx],
            pred_max_tT=u_pred_max[x1_idx, :, tT_idx],
            data_t0=u_data[x1_idx, :, t0_idx],
            data_tT=u_data[x1_idx, :, tT_idx],
            fwd_mean_t0=None if u_fwd_mean is None else u_fwd_mean[x1_idx, :, t0_idx],
            fwd_min_t0=None if u_fwd_min is None else u_fwd_min[x1_idx, :, t0_idx],
            fwd_max_t0=None if u_fwd_max is None else u_fwd_max[x1_idx, :, t0_idx],
            fwd_mean_tT=None if u_fwd_mean is None else u_fwd_mean[x1_idx, :, tT_idx],
            fwd_min_tT=None if u_fwd_min is None else u_fwd_min[x1_idx, :, tT_idx],
            fwd_max_tT=None if u_fwd_max is None else u_fwd_max[x1_idx, :, tT_idx],
            xlabel="x2 [mm]",
            legend_loc=legend_loc_x2,
            legend_ncols=legend_ncols,
            legend_fontsize=legend_fontsize,
            labels=labels_x2mid,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            tick_length=tick_length,
            tick_width=tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
        )
        fig1.tight_layout()

        fig2, ax2 = plt.subplots(figsize=figsize)
        self._plot_single_midline(
            ax=ax2,
            axis_values=x1,
            pred_mean_t0=u_pred_mean[:, x2_idx, t0_idx],
            pred_min_t0=u_pred_min[:, x2_idx, t0_idx],
            pred_max_t0=u_pred_max[:, x2_idx, t0_idx],
            pred_mean_tT=u_pred_mean[:, x2_idx, tT_idx],
            pred_min_tT=u_pred_min[:, x2_idx, tT_idx],
            pred_max_tT=u_pred_max[:, x2_idx, tT_idx],
            data_t0=u_data[:, x2_idx, t0_idx],
            data_tT=u_data[:, x2_idx, tT_idx],
            fwd_mean_t0=None if u_fwd_mean is None else u_fwd_mean[:, x2_idx, t0_idx],
            fwd_min_t0=None if u_fwd_min is None else u_fwd_min[:, x2_idx, t0_idx],
            fwd_max_t0=None if u_fwd_max is None else u_fwd_max[:, x2_idx, t0_idx],
            fwd_mean_tT=None if u_fwd_mean is None else u_fwd_mean[:, x2_idx, tT_idx],
            fwd_min_tT=None if u_fwd_min is None else u_fwd_min[:, x2_idx, tT_idx],
            fwd_max_tT=None if u_fwd_max is None else u_fwd_max[:, x2_idx, tT_idx],
            xlabel=r"$x_1$ [mm]",
            legend_loc=legend_loc_x1,
            legend_ncols=legend_ncols,
            legend_fontsize=legend_fontsize,
            labels=labels_x1mid,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            tick_length=tick_length,
            tick_width=tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
        )
        fig2.tight_layout()

        self._save_two_figures(fig1, fig2, save_name, target)
        plt.show()

        return u_pred_mean, u_fwd_mean