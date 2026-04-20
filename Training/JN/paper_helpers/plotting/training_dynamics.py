# Pedagogical Header
# Module: paper_plots/D_G_with_epochs.py
# Purpose: Plot D(u)/G(u) curves across training epochs.
# Used by: experim.ipynb (JN) and sibling helper modules.
# Behavior: intended to match original JN implementation.

"""
D_G_with_epochs.py

Plot growth and diffusion curves as functions of cell density across training.

This module provides utilities for visualizing how a selected model evolves over
training. For each group of models, one run is selected and the saved growth and
diffusion curves across epochs are plotted against cell density.

Main use case
-------------
This is useful when you want to inspect:
- how growth(u) changes during training
- how diffusion(u) changes during training
- which saved epoch corresponds to the best model
- how the learned curves relate to the central density range in the data

For each group, the class produces:
- one growth-vs-density figure
- one diffusion-vs-density figure

The selected "best model" curve is overlaid in black, and earlier/later saved
epochs are coloured to show training progression.

Update
------
This version exposes explicit controls for:
- x-axis label fontsize
- y-axis label fontsize
- x tick-label fontsize
- y tick-label fontsize
- major tick length/width
- minor tick length/width
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from ..io.paths import build_save_base, hist_properties


class TrainingDynamicsPlotter:
    """Plot growth and diffusion curves across saved training epochs."""

    def __init__(
        self,
        device: str = "cpu",
        diffusion_base_color: str = "#d35400",
        diffusion_best_color: str = "#a04000",
        growth_base_color: str = "#2e7d32",
        growth_best_color: str = "#006400",
        gray_base_color: str = "#808080",
    ) -> None:
        self.device = device
        self.diffusion_base_color = diffusion_base_color
        self.diffusion_best_color = diffusion_best_color
        self.growth_base_color = growth_base_color
        self.growth_best_color = growth_best_color
        self.gray_base_color = gray_base_color

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_numpy(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().flatten().numpy()
        return np.asarray(x).flatten()

    @staticmethod
    def _blend_with_white(base_color: str, t: float) -> Tuple[float, float, float, float]:
        br, bg, bb, ba = to_rgba(base_color)
        r = (1 - t) * 1.0 + t * br
        g = (1 - t) * 1.0 + t * bg
        b = (1 - t) * 1.0 + t * bb
        return (r, g, b, ba)

    def _gray_color(self, ep: int, gray_epochs: Sequence[int]) -> Tuple[float, float, float, float]:
        if not gray_epochs:
            return to_rgba(self.gray_base_color)

        gray_sorted = sorted(gray_epochs)
        idx = gray_sorted.index(ep)

        if len(gray_sorted) == 1:
            frac = 1.0
        else:
            frac = 1.0 - (idx / (len(gray_sorted) - 1))

        t = 0.4 + 0.6 * frac
        r, g, b, _ = self._blend_with_white(self.gray_base_color, t)
        alpha = 0.3 + 0.6 * frac
        return (r, g, b, alpha)

    def _epoch_color(
        self,
        ep: int,
        colored_epochs: Sequence[int],
        gray_epochs: Sequence[int],
        base_color: str,
        best_color: str,
    ) -> Tuple[float, float, float, float]:
        if ep in colored_epochs:
            colored_sorted = sorted(colored_epochs)
            best_ep = max(colored_sorted)
            if ep == best_ep:
                return to_rgba(best_color)

            idx = colored_sorted.index(ep)
            if len(colored_sorted) == 1:
                t_raw = 0.0
            else:
                t_raw = idx / (len(colored_sorted) - 1)

            t = 0.2 + 0.7 * t_raw
            return self._blend_with_white(base_color, t)

        if ep in gray_epochs:
            return self._gray_color(ep, gray_epochs)

        return to_rgba(self.gray_base_color)

    @staticmethod
    def _get_density_scale(wrapper: Any, species_label: str) -> float:
        model = wrapper.model
        val = None

        if species_label.lower() == "red" and hasattr(model, "u_red_max"):
            val = float(model.u_red_max) * 1e6
        elif species_label.lower() == "green" and hasattr(model, "u_green_max"):
            val = float(model.u_green_max) * 1e6
        elif hasattr(model, "u_max"):
            val = float(model.u_max) * 1e6

        if val is None or not np.isfinite(val) or val <= 0:
            return 1.0
        return val

    @staticmethod
    def _select_one_wrapper_per_group(
        model_wrapper_groups: Dict[Any, Dict[Any, Any]],
        model_index: int,
    ) -> Dict[Any, Any]:
        selected = {}
        for group_key, group_dict in model_wrapper_groups.items():
            if not group_dict:
                raise ValueError(f"Group '{group_key}' has no models.")
            inner_keys = sorted(group_dict.keys())
            if model_index < 0 or model_index >= len(inner_keys):
                raise IndexError(
                    f"model_index={model_index} out of range for group '{group_key}' "
                    f"(has {len(inner_keys)} models)."
                )
            selected[group_key] = group_dict[inner_keys[model_index]]
        return selected

    @staticmethod
    def _max_saved_epochs(selected_wrappers: Dict[Any, Any], attr: str) -> int:
        return max((len(getattr(w, attr)) for w in selected_wrappers.values()), default=0)

    @staticmethod
    def _build_epoch_list(max_epochs_all: int, epoch_step: int) -> Sequence[int]:
        if max_epochs_all <= 0:
            raise ValueError("No saved epochs available.")
        epochs = list(range(0, max_epochs_all, max(1, int(epoch_step))))
        if epochs[-1] != max_epochs_all - 1:
            epochs.append(max_epochs_all - 1)
        return epochs

    def _get_prediction_or_zeros(
        self,
        wrapper: Any,
        attr: str,
        ep: int,
        target_len: int,
    ) -> np.ndarray:
        seq = getattr(wrapper, attr)
        if ep < len(seq):
            arr = self._as_numpy(seq[ep])
            if arr.size < target_len:
                out = np.zeros(target_len, dtype=float)
                out[:arr.size] = arr
                return out
            if arr.size > target_len:
                return arr[:target_len]
            return arr
        return np.zeros(target_len, dtype=float)

    def _global_y_limits(
        self,
        selected_wrappers: Dict[Any, Any],
        x_vals_by_group: Dict[Any, np.ndarray],
        attr: str,
        epochs: Sequence[int],
    ) -> Tuple[float, float]:
        y_min, y_max = np.inf, -np.inf
        for ep in epochs:
            for group_key, wrapper in selected_wrappers.items():
                xg = x_vals_by_group[group_key]
                arr = self._get_prediction_or_zeros(wrapper, attr, ep, target_len=len(xg))
                y_min = min(y_min, float(arr.min()))
                y_max = max(y_max, float(arr.max()))
        return (
            y_min if np.isfinite(y_min) else 0.0,
            y_max if np.isfinite(y_max) else 1.0,
        )

    @staticmethod
    def _last_best_training_epoch(wrapper: Any, fallback_epoch: int) -> int:
        if hasattr(wrapper, "last_improved"):
            return int(wrapper.last_improved)
        return fallback_epoch

    @staticmethod
    def _build_legend_handles(
        epochs: Sequence[int],
        colored_epochs: Sequence[int],
        gray_epochs: Sequence[int],
        epochs_sf: int,
        color_fn,
        best_model_handle: Optional[Line2D],
    ) -> Sequence[Line2D]:
        handles = []

        if colored_epochs:
            colored_sorted = sorted(colored_epochs)
            n_col = len(colored_sorted)
            num_demo = min(3, n_col)
            demo_positions = sorted({int(round(x)) for x in np.linspace(0, n_col - 1, num_demo)})

            for pos in demo_positions:
                ep_demo = colored_sorted[pos]
                train_ep_demo = ep_demo * epochs_sf
                color_demo = color_fn(ep_demo)
                handles.append(
                    Line2D([0], [0], color=color_demo, lw=3, label=f"{train_ep_demo}")
                )

        if gray_epochs:
            gray_sorted = sorted(gray_epochs)
            ep_demo = gray_sorted[0]
            train_ep_demo = ep_demo * epochs_sf
            color_demo = color_fn(ep_demo)
            handles.append(
                Line2D([0], [0], color=color_demo, lw=3, label=f"> best (≥ {train_ep_demo})")
            )

        if best_model_handle is not None:
            handles.append(best_model_handle)

        return handles

    @staticmethod
    def _save_secondary_copy(image_path: str, save_dir2: str, figsize: Tuple[float, float], dpi: int) -> None:
        import matplotlib.image as mpimg

        os.makedirs(save_dir2, exist_ok=True)
        out_path = os.path.join(save_dir2, os.path.basename(image_path))

        img = mpimg.imread(image_path)
        fig_copy, ax_copy = plt.subplots(figsize=figsize)
        ax_copy.imshow(img)
        ax_copy.axis("off")
        fig_copy.tight_layout()
        fig_copy.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_copy)

    @staticmethod
    def _resolve_plot_xlim(
        gray: bool,
        x_lim: Optional[Tuple[float, float]],
        low_x: float,
        high_x: float,
    ) -> Optional[Tuple[float, float]]:
        """
        Determine x-limits for the plot.

        Rules:
        - If x_lim is provided, always use it.
        - If x_lim is None and gray=False, use the central 90% range [low_x, high_x].
        - If x_lim is None and gray=True, keep the full matplotlib x-range.
        """
        if x_lim is not None:
            return x_lim
        if not gray:
            return (low_x, high_x)
        return None

    @staticmethod
    def _restrict_to_central_range(
        x: np.ndarray,
        y: np.ndarray,
        low_x: float,
        high_x: float,
        gray: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restrict plotted predictions to the central 90% range when gray=False.

        This means:
        - gray=True: return the full curve unchanged
        - gray=False: only keep points with low_x <= x <= high_x

        If the caller also sets x_lim wider than [low_x, high_x], matplotlib will
        show white space padding outside the restricted prediction range.
        """
        if gray:
            return x, y

        mask = (x >= low_x) & (x <= high_x)
        return x[mask], y[mask]

    @staticmethod
    def _style_axes(
        ax: Any,
        *,
        xtick_labelsize: float,
        ytick_labelsize: float,
        major_tick_length: float,
        major_tick_width: float,
        minor_tick_length: float,
        minor_tick_width: float,
    ) -> None:
        ax.tick_params(
            axis="x",
            which="major",
            labelsize=xtick_labelsize,
            length=major_tick_length,
            width=major_tick_width,
        )
        ax.tick_params(
            axis="y",
            which="major",
            labelsize=ytick_labelsize,
            length=major_tick_length,
            width=major_tick_width,
        )
        ax.tick_params(
            axis="x",
            which="minor",
            length=minor_tick_length,
            width=minor_tick_width,
        )
        ax.tick_params(
            axis="y",
            which="minor",
            length=minor_tick_length,
            width=minor_tick_width,
        )

    def _plot_training_curves_for_quantity(
        self,
        *,
        quantity_name: str,
        pred_attr: str,
        model_eval_attr: str,
        model_scale_attr: str,
        selected_wrappers: Dict[Any, Any],
        outer_keys: Sequence[Any],
        labels: Sequence[str],
        x_vals_by_group: Dict[Any, np.ndarray],
        sf_by_group: Dict[Any, float],
        epochs: Sequence[int],
        last_best_raw_by_group: Dict[Any, int],
        low_u: float,
        high_u: float,
        hist_min: float,
        hist_max: float,
        K: float,
        y_lim: Tuple[float, float],
        out_root: str,
        model_index: int,
        figsize: Tuple[float, float],
        dpi: int,
        axis_labels: bool,
        xlabel_fontsize: float,
        ylabel_fontsize: float,
        xtick_labelsize: float,
        ytick_labelsize: float,
        major_tick_length: float,
        major_tick_width: float,
        minor_tick_length: float,
        minor_tick_width: float,
        x_lim: Optional[Tuple[float, float]],
        gray: bool,
        legend_loc: Any,
        legend_ncols: int,
        legend_fontsize: int,
        epochs_sf: int,
        overwrite: bool,
        save_dir2: Optional[str],
        base_color: str,
        best_color: str,
        ylabel: str,
    ) -> None:
        for group_key, label in zip(outer_keys, labels):
            xg = x_vals_by_group[group_key]
            sf = sf_by_group[group_key]
            wrapper = selected_wrappers[group_key]
            last_best_raw = last_best_raw_by_group[group_key]

            colored_epochs = [ep for ep in epochs if (ep * epochs_sf) <= last_best_raw + 1e-9]
            gray_epochs = [ep for ep in epochs if (ep * epochs_sf) > last_best_raw + 1e-9]

            out_png = f"{out_root}_{quantity_name}_{group_key}_model{model_index}.png"
            if (not overwrite) and os.path.isfile(out_png):
                print(f"{quantity_name.capitalize()} file exists and overwrite=False; skipping: {out_png}")
                continue

            fig, ax = plt.subplots(figsize=figsize)

            color_fn = lambda ep: self._epoch_color(
                ep=ep,
                colored_epochs=colored_epochs,
                gray_epochs=gray_epochs,
                base_color=base_color,
                best_color=best_color,
            )

            best_model_handle = None

            low_x = low_u * sf * K
            high_x = high_u * sf * K
            xmin = hist_min * sf * K
            xmax = hist_max * sf * K

            for ep in epochs:
                arr = self._get_prediction_or_zeros(wrapper, pred_attr, ep, target_len=len(xg))
                x_plot, y_plot = self._restrict_to_central_range(
                    x=xg,
                    y=arr,
                    low_x=low_x,
                    high_x=high_x,
                    gray=gray,
                )
                col = color_fn(ep)
                zorder = 1 if ep in gray_epochs else 2
                ax.plot(x_plot, y_plot, lw=3, color=col, zorder=zorder)

            sample_model = wrapper.model
            u_vals_torch = getattr(sample_model, "u_vals_torch", None)

            if u_vals_torch is not None and hasattr(sample_model, model_eval_attr):
                sample_model.eval()
                with torch.no_grad():
                    raw_eval = getattr(sample_model, model_eval_attr)(u_vals_torch).flatten()
                    if hasattr(sample_model, model_scale_attr):
                        raw_eval = getattr(sample_model, model_scale_attr) * raw_eval

                eval_np = self._as_numpy(raw_eval)
                x_best, y_best = self._restrict_to_central_range(
                    x=xg,
                    y=eval_np,
                    low_x=low_x,
                    high_x=high_x,
                    gray=gray,
                )
                ax.plot(x_best, y_best, lw=3, ls="-", color="k", zorder=5, label="_nolegend_")
                best_model_handle = Line2D(
                    [0], [0],
                    lw=3,
                    ls="-",
                    color="k",
                    label=f"best model ({last_best_raw})",
                )

            if gray:
                if low_x > xmin:
                    ax.axvspan(xmin, low_x, color="gray", alpha=0.12, zorder=0)
                if high_x < xmax:
                    ax.axvspan(high_x, xmax, color="gray", alpha=0.12, zorder=0)

            ax.set_facecolor("white")

            resolved_xlim = self._resolve_plot_xlim(
                gray=gray,
                x_lim=x_lim,
                low_x=low_x,
                high_x=high_x,
            )
            if resolved_xlim is not None:
                ax.set_xlim(resolved_xlim)

            ax.set_ylim(y_lim)

            if axis_labels:
                ax.set_xlabel(r"Cell density [cells mm$^{-2}$]", fontsize=xlabel_fontsize)
                ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

            self._style_axes(
                ax,
                xtick_labelsize=xtick_labelsize,
                ytick_labelsize=ytick_labelsize,
                major_tick_length=major_tick_length,
                major_tick_width=major_tick_width,
                minor_tick_length=minor_tick_length,
                minor_tick_width=minor_tick_width,
            )

            legend_handles = self._build_legend_handles(
                epochs=epochs,
                colored_epochs=colored_epochs,
                gray_epochs=gray_epochs,
                epochs_sf=epochs_sf,
                color_fn=color_fn,
                best_model_handle=best_model_handle,
            )
            if legend_handles:
                ax.legend(
                    handles=legend_handles,
                    fontsize=legend_fontsize,
                    ncol=legend_ncols,
                    frameon=False,
                    facecolor="white",
                    framealpha=1.0,
                    loc=legend_loc,
                    title_fontsize=legend_fontsize+1,
                    title = "Epoch "
                )

            ax.grid(False)
            fig.tight_layout()

            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
            plt.show()
            plt.close(fig)

            print(f"Saved {quantity_name} plot for {group_key}, model_index={model_index}: {out_png}")

            if save_dir2:
                self._save_secondary_copy(out_png, save_dir2, figsize, dpi)
                print(f"Also saved second copy of {quantity_name} plot for {group_key}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_single_run_growth_diffusion_over_training(
        self,
        model_wrapper_groups: Dict[Any, Dict[Any, Any]],
        dataobj: Any,
        species_label: str,
        save_dic: Dict[str, Any],
        save_name: str,
        labels: Optional[Sequence[str]] = None,
        num_bins: int = 50,
        K: float = 1.0,
        base_dir: str = "plots",
        overwrite: bool = False,
        legend_pos_diffusion: Any = (0.5, 0.5),
        legend_pos_growth: Any = (0.5, 0.5),
        legend_ncols: int = 1,
        legend_fontsize: int = 10,
        figsize: Tuple[float, float] = (8, 8),
        x_lim: Optional[Tuple[float, float]] = None,
        y_lim_growth: Optional[Tuple[float, float]] = None,
        y_lim_diffusion: Optional[Tuple[float, float]] = None,
        axis_labels: bool = True,
        xlabel_fontsize: float = 11.0,
        ylabel_fontsize: float = 11.0,
        xtick_labelsize: float = 10.0,
        ytick_labelsize: float = 10.0,
        major_tick_length: float = 4.0,
        major_tick_width: float = 1.0,
        minor_tick_length: float = 2.5,
        minor_tick_width: float = 0.8,
        epoch_step: int = 1,
        dpi: int = 100,
        epochs_sf: int = 100,
        save_dir2: Optional[str] = None,
        model_index: int = 0,
        gray: bool = False,
    ) -> None:
        """
        Plot growth(u) and diffusion(u) across saved training epochs for one
        selected model per group.

        Parameters
        ----------
        model_wrapper_groups
            Nested dictionary of the form {group_key: {run_key: wrapper}}.
        dataobj
            Data object used to compute density percentile reference regions.
        species_label
            Typically "green" or "red".
        save_dic, save_name
            Used to build the base output path.
        labels
            Optional plot labels for each outer group.
        model_index
            Which run to select within each group.
        epochs_sf
            Number of raw training epochs between stored predictions.
        gray
            If True, shade the regions outside the central 90% in gray.
            If False, do not show gray shading, and the plotted prediction curves
            are restricted to the central 90% density range.
            If x_lim is wider than that range, the extra region appears as white
            space padding.

        Produces
        --------
        For each group:
        - one growth-vs-density PNG
        - one diffusion-vs-density PNG
        """
        out_base = build_save_base(save_dic, save_name, base_dir=base_dir)
        out_root, _ = os.path.splitext(out_base)

        outer_keys = list(model_wrapper_groups.keys())
        if not outer_keys:
            raise ValueError("No groups in model_wrapper_groups.")
        if labels is None:
            labels = [str(k) for k in outer_keys]

        selected_wrappers = self._select_one_wrapper_per_group(model_wrapper_groups, model_index)

        h_props = hist_properties(dataobj, speciesLabel=species_label, num_bins=num_bins)
        low_u = h_props["low_count"]
        high_u = h_props["high_count"]
        hist_min = h_props["min_count"]
        hist_max = h_props["max_count"]

        x_vals_by_group = {}
        sf_by_group = {}
        for group_key, wrapper in selected_wrappers.items():
            u_vals_np = np.asarray(wrapper.model.u_vals).flatten()
            sf = self._get_density_scale(wrapper, species_label)
            sf_by_group[group_key] = sf
            x_vals_by_group[group_key] = u_vals_np * sf * K

        max_epochs_growth = self._max_saved_epochs(selected_wrappers, "growth_preds")
        max_epochs_diffusion = self._max_saved_epochs(selected_wrappers, "diffusion_preds")
        max_epochs_all = max(max_epochs_growth, max_epochs_diffusion)
        epochs = self._build_epoch_list(max_epochs_all, epoch_step)

        growth_ylim = self._global_y_limits(
            selected_wrappers, x_vals_by_group, "growth_preds", epochs
        )
        diffusion_ylim = self._global_y_limits(
            selected_wrappers, x_vals_by_group, "diffusion_preds", epochs
        )

        if y_lim_growth is not None:
            growth_ylim = y_lim_growth
        if y_lim_diffusion is not None:
            diffusion_ylim = y_lim_diffusion

        fallback_best_epoch = (max_epochs_all - 1) * epochs_sf
        last_best_raw_by_group = {
            group_key: self._last_best_training_epoch(wrapper, fallback_best_epoch)
            for group_key, wrapper in selected_wrappers.items()
        }

        self._plot_training_curves_for_quantity(
            quantity_name="growth",
            pred_attr="growth_preds",
            model_eval_attr="growth",
            model_scale_attr="G_scale",
            selected_wrappers=selected_wrappers,
            outer_keys=outer_keys,
            labels=labels,
            x_vals_by_group=x_vals_by_group,
            sf_by_group=sf_by_group,
            epochs=epochs,
            last_best_raw_by_group=last_best_raw_by_group,
            low_u=low_u,
            high_u=high_u,
            hist_min=hist_min,
            hist_max=hist_max,
            K=K,
            y_lim=growth_ylim,
            out_root=out_root,
            model_index=model_index,
            figsize=figsize,
            dpi=dpi,
            axis_labels=axis_labels,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            major_tick_length=major_tick_length,
            major_tick_width=major_tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
            x_lim=x_lim,
            gray=gray,
            legend_loc=legend_pos_growth,
            legend_ncols=legend_ncols,
            legend_fontsize=legend_fontsize,
            epochs_sf=epochs_sf,
            overwrite=overwrite,
            save_dir2=save_dir2,
            base_color=self.growth_base_color,
            best_color=self.growth_best_color,
            ylabel=r"Growth [days$^{-1}$]",
        )

        self._plot_training_curves_for_quantity(
            quantity_name="diffusion",
            pred_attr="diffusion_preds",
            model_eval_attr="diffusion",
            model_scale_attr="D_scale",
            selected_wrappers=selected_wrappers,
            outer_keys=outer_keys,
            labels=labels,
            x_vals_by_group=x_vals_by_group,
            sf_by_group=sf_by_group,
            epochs=epochs,
            last_best_raw_by_group=last_best_raw_by_group,
            low_u=low_u,
            high_u=high_u,
            hist_min=hist_min,
            hist_max=hist_max,
            K=K,
            y_lim=diffusion_ylim,
            out_root=out_root,
            model_index=model_index,
            figsize=figsize,
            dpi=dpi,
            axis_labels=axis_labels,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            major_tick_length=major_tick_length,
            major_tick_width=major_tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
            x_lim=x_lim,
            gray=gray,
            legend_loc=legend_pos_diffusion,
            legend_ncols=legend_ncols,
            legend_fontsize=legend_fontsize,
            epochs_sf=epochs_sf,
            overwrite=overwrite,
            save_dir2=save_dir2,
            base_color=self.diffusion_base_color,
            best_color=self.diffusion_best_color,
            ylabel=r"Diffusion [mm$^2$ days$^{-1}$]",
        )