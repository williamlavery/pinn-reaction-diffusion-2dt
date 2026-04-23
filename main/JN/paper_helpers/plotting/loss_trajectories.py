from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


class EarlyStoppingTrajectoryPlotter:
    """Plot repeated loss trajectories across multiple early-stopping settings."""

    def __init__(
        self,
        tableau_colors: Optional[Sequence[str]] = None,
    ) -> None:
        self.tableau_colors = list(tableau_colors) if tableau_colors is not None else [
            "#1F77B4",  # strong blue
            "#FF7F0E",  # vivid orange
            "#2CA02C",  # green
            "#D62728",  # red
            "#9467BD",  # purple
            "#8C564B",  # brown
            "#E377C2",  # pink
            "#7F7F7F",  # gray
            "#BCBD22",  # yellow/green
            "#17BECF",  # teal
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_es_marker_map(es_list: Sequence[Any]) -> Dict[Any, str]:
        default_markers = ["o", "s", "^", "D", "v"]
        return {es: default_markers[i % len(default_markers)] for i, es in enumerate(es_list)}

    @staticmethod
    def _marker_sizes_from_shapes(es_marker_map: Dict[Any, str]) -> Dict[Any, float]:
        marker_size_by_shape = {
            "o": 100,   # circle
            "s": 50,    # square
            "^": 40,    # triangle up
            "v": 40,    # triangle down
            "D": 60,    # diamond
        }
        return {
            es: marker_size_by_shape.get(marker, 60)
            for es, marker in es_marker_map.items()
        }

    @staticmethod
    def _make_broken_axes(figsize: Tuple[float, float]) -> Tuple[Figure, Axes, Axes]:
        fig, (ax_left, ax_right) = plt.subplots(
            1,
            2,
            figsize=figsize,
            sharey=True,
            gridspec_kw={"wspace": 0.05},
        )

        ax_left.spines["right"].set_visible(False)
        ax_right.spines["left"].set_visible(False)
        ax_right.yaxis.set_tick_params(labelleft=False)

        d = 0.012
        kwargs_left = dict(transform=ax_left.transAxes, color="k", clip_on=False)
        ax_left.plot((1 - d, 1 + d), (0 - d, 0 + d), **kwargs_left)
        ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_left)

        kwargs_right = dict(transform=ax_right.transAxes, color="k", clip_on=False)
        ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs_right)
        ax_right.plot((-d, +d), (0 - d, 0 + d), **kwargs_right)

        return fig, ax_left, ax_right

    @staticmethod
    def _extract_single_label(model_dict: Dict[str, Dict[Any, Any]]) -> str:
        if len(model_dict) != 1:
            raise ValueError("Expected each models_dics to contain one label.")
        return list(model_dict.keys())[0]

    @staticmethod
    def _best_epoch_from_total_loss(
        wrapper: Any,
        use_val_loss: bool,
    ) -> int:
        if use_val_loss and len(wrapper.val_loss_list):
            total_losses = np.array(wrapper.val_loss_list, dtype=float)
        else:
            total_losses = np.array(wrapper.train_loss_list, dtype=float)

        if len(total_losses) == 0:
            return 0

        best_loss = wrapper.best_val_loss
        return int(np.argmin(np.abs(total_losses - best_loss)))

    @staticmethod
    def _select_loss_array(
        wrapper: Any,
        loss_kind: str,
        use_val_loss: bool,
    ) -> Optional[np.ndarray]:
        if loss_kind == "total":
            if use_val_loss and len(wrapper.val_loss_list):
                return np.array(wrapper.val_loss_list, dtype=float)
            return np.array(wrapper.train_loss_list, dtype=float)

        if loss_kind == "data":
            if hasattr(wrapper, "val_data_loss_list") and len(wrapper.val_data_loss_list) > 0:
                return np.array(wrapper.val_data_loss_list, dtype=float)
            return None

        if loss_kind == "pde":
            if hasattr(wrapper, "val_pde_loss_list") and len(wrapper.val_pde_loss_list) > 0:
                return np.array(wrapper.val_pde_loss_list, dtype=float)
            return None

        raise ValueError(f"Unknown loss_kind: {loss_kind}")

    def _repeat_color_map(
        self,
        repeat_indices: Sequence[int],
        base_cmap: str,
    ) -> Dict[int, Any]:
        _ = cm.get_cmap(base_cmap, len(repeat_indices) + 2)
        return {
            r: self.tableau_colors[r % len(self.tableau_colors)]
            for r in repeat_indices
        }

    @staticmethod
    def _repeat_legend_handles(
        repeat_indices: Sequence[int],
        repeat_colors: Dict[int, Any],
    ) -> Sequence[Line2D]:
        return [
            Line2D([0], [0], color=repeat_colors[r], lw=2, label=f"{r}")
            for r in repeat_indices
        ]

    @staticmethod
    def _es_legend_handles(
        es_list: Sequence[Any],
        es_marker_map: Dict[Any, str],
        es_marker_size: Dict[Any, float],
    ) -> Sequence[Line2D]:
        return [
            Line2D(
                [0],
                [0],
                marker=es_marker_map[es],
                linestyle="",
                markeredgecolor="k",
                markerfacecolor="gray",
                markersize=np.sqrt(es_marker_size[es]),
                label=f"{es}",
            )
            for es in es_list
        ]

    @staticmethod
    def _style_ticks(
        ax: Axes,
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

    def _style_broken_axes(
        self,
        ax_left: Axes,
        ax_right: Axes,
        *,
        xlim_left: Tuple[float, float],
        xlim_right: Tuple[float, float],
        yscale: Optional[str],
        xscale: Optional[str],
        ylim: Optional[Tuple[float, float]],
        ylabel: str,
        xlabel_fontsize: float,
        ylabel_fontsize: float,
        xtick_labelsize: float,
        ytick_labelsize: float,
        major_tick_length: float,
        major_tick_width: float,
        minor_tick_length: float,
        minor_tick_width: float,
    ) -> None:
        ax_left.set_xlim(*xlim_left)
        ax_right.set_xlim(*xlim_right)

        if yscale:
            ax_left.set_yscale(yscale)

        if xscale:
            ax_left.set_xscale(xscale)
            ax_right.set_xscale(xscale)

        if ylim:
            ax_left.set_ylim(ylim)

        # Remove per-axis labels
        ax_left.set_xlabel("")
        ax_right.set_xlabel("")

        # Add centered label across both axes
        ax_left.figure.supxlabel("Epoch", fontsize=xlabel_fontsize)
        ax_left.set_ylabel(ylabel, fontsize=ylabel_fontsize)

        ax_left.grid(True, ls="--", alpha=0.3)
        ax_right.grid(True, ls="--", alpha=0.3)

        self._style_ticks(
            ax_left,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            major_tick_length=major_tick_length,
            major_tick_width=major_tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
        )
        self._style_ticks(
            ax_right,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            major_tick_length=major_tick_length,
            major_tick_width=major_tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
        )

    def _plot_one_loss_family(
        self,
        *,
        models_dics_list: Sequence[Dict[str, Dict[Any, Any]]],
        es_list: Sequence[Any],
        repeat_indices: Sequence[int],
        repeat_colors: Dict[int, Any],
        es_marker_map: Dict[Any, str],
        es_marker_size: Dict[Any, float],
        use_val_loss: bool,
        loss_kind: str,
        ylabel: str,
        ylim: Optional[Tuple[float, float]],
        figsize: Tuple[float, float],
        yscale: Optional[str],
        xscale: Optional[str],
        xlim_left: Tuple[float, float],
        xlim_right: Tuple[float, float],
        repeat_handles: Sequence[Line2D],
        es_handles: Sequence[Line2D],
        legend_loc_repeat: str,
        legend_loc_es: str,
        repeat_legend_fontsize: float,
        repeat_legend_title_fontsize: float,
        es_legend_fontsize: float,
        es_legend_title_fontsize: float,
        xlabel_fontsize: float,
        ylabel_fontsize: float,
        xtick_labelsize: float,
        ytick_labelsize: float,
        major_tick_length: float,
        major_tick_width: float,
        minor_tick_length: float,
        minor_tick_width: float,
        base_label: str,
        title: Optional[str],
        save_path: Optional[str],
    ) -> Tuple[Figure, Axes, Axes]:
        fig, ax_left, ax_right = self._make_broken_axes(figsize)

        for models_dics, es in zip(models_dics_list, es_list):
            model_dic = models_dics[base_label]
            marker = es_marker_map[es]
            marker_size = es_marker_size[es]

            for r in repeat_indices:
                wrapper = model_dic[r]
                losses = self._select_loss_array(
                    wrapper,
                    loss_kind=loss_kind,
                    use_val_loss=use_val_loss,
                )
                if losses is None or len(losses) == 0:
                    continue

                epochs = np.arange(len(losses))
                color = repeat_colors[r]

                for ax in (ax_left, ax_right):
                    ax.plot(
                        epochs,
                        losses,
                        lw=1,
                        alpha=0.5,
                        color=color,
                    )

                idx_es = self._best_epoch_from_total_loss(wrapper, use_val_loss=use_val_loss)
                idx_es = min(idx_es, len(losses) - 1)

                for ax in (ax_left, ax_right):
                    ax.scatter(
                        epochs[idx_es],
                        losses[idx_es],
                        marker=marker,
                        s=marker_size,
                        edgecolor="k",
                        linewidth=0.75,
                        color=color,
                        zorder=4,
                    )

        self._style_broken_axes(
            ax_left,
            ax_right,
            xlim_left=xlim_left,
            xlim_right=xlim_right,
            yscale=yscale,
            xscale=xscale,
            ylim=ylim,
            ylabel=ylabel,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            major_tick_length=major_tick_length,
            major_tick_width=major_tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
        )

        if loss_kind == "total":
            print(title if title is not None else f"Total loss trajectories for {base_label}")
        elif loss_kind == "data":
            print(f"Data loss trajectories for {base_label}")
        elif loss_kind == "pde":
            print(f"PDE loss trajectories for {base_label}")

        legend_repeat = ax_right.legend(
            handles=repeat_handles,
            title="TV split",
            loc=legend_loc_repeat,
            fontsize=repeat_legend_fontsize,
            title_fontsize=repeat_legend_title_fontsize,
        )
        ax_right.add_artist(legend_repeat)

        ax_left.legend(
            handles=es_handles,
            title=r"$\mathit{ES}$",
            loc=legend_loc_es,
            fontsize=es_legend_fontsize,
            title_fontsize=es_legend_title_fontsize,
        )

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="None")

        plt.show()
        return fig, ax_left, ax_right

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_loss_trajectories_across_early_stopping(
        self,
        models_dics_list: Sequence[Dict[str, Dict[Any, Any]]],
        es_list: Sequence[Any],
        use_val_loss: bool = True,
        yscale: Optional[str] = "log",
        xscale: Optional[str] = None,
        ylim_total: Optional[Tuple[float, float]] = (1e-2, 1e2),
        ylim_data: Optional[Tuple[float, float]] = (1e-2, 1e2),
        ylim_pde: Optional[Tuple[float, float]] = (1e-2, 1e2),
        figsize: Tuple[float, float] = (9, 5),
        base_cmap: str = "Blues",
        es_marker_map: Optional[Dict[Any, str]] = None,
        title: Optional[str] = None,
        xlim_left: Tuple[float, float] = (1, 300),
        xlim_right: Tuple[float, float] = (800, 2000),
        base_dir: Optional[str] = None,
        xlabel_fontsize: float = 11.0,
        ylabel_fontsize: float = 11.0,
        xtick_labelsize: float = 10.0,
        ytick_labelsize: float = 10.0,
        major_tick_length: float = 4.0,
        major_tick_width: float = 1.0,
        minor_tick_length: float = 2.5,
        minor_tick_width: float = 0.8,
        total_legend_loc_repeat: str = "upper right",
        total_legend_loc_es: str = "lower left",
        data_legend_loc_repeat: str = "upper right",
        data_legend_loc_es: str = "lower left",
        pde_legend_loc_repeat: str = "lower right",
        pde_legend_loc_es: str = "lower left",
        total_repeat_legend_fontsize: float = 9.0,
        total_repeat_legend_title_fontsize: float = 10.0,
        total_es_legend_fontsize: float = 10.0,
        total_es_legend_title_fontsize: float = 10.5,
        data_repeat_legend_fontsize: float = 10.0,
        data_repeat_legend_title_fontsize: float = 10.5,
        data_es_legend_fontsize: float = 10.0,
        data_es_legend_title_fontsize: float = 10.5,
        pde_repeat_legend_fontsize: float = 10.0,
        pde_repeat_legend_title_fontsize: float = 10.5,
        pde_es_legend_fontsize: float = 10.0,
        pde_es_legend_title_fontsize: float = 10.5,
    ) -> None:
        """
        Plot total, data, and PDE loss trajectories across ES settings.

        Parameters
        ----------
        models_dics_list
            List of nested dictionaries, one per ES setting:
            [
              {label: {repeat_idx: ModelWrapper, ...}},
              {label: {repeat_idx: ModelWrapper, ...}},
              ...
            ]
        es_list
            List of ES values, one for each element of models_dics_list.
        use_val_loss
            If True, use validation loss when available for the total-loss plot.
            Otherwise fall back to training loss.
        total_legend_loc_repeat, total_legend_loc_es
            Legend locations for the total-loss plot.
        data_legend_loc_repeat, data_legend_loc_es
            Legend locations for the data-loss plot.
        pde_legend_loc_repeat, pde_legend_loc_es
            Legend locations for the PDE-loss plot.
        """
        if len(models_dics_list) != len(es_list):
            raise ValueError("ES_list must match models_dics_list length")

        if es_marker_map is None:
            es_marker_map = self._default_es_marker_map(es_list)

        es_marker_size = self._marker_sizes_from_shapes(es_marker_map)

        first_dic = models_dics_list[0]
        base_label = self._extract_single_label(first_dic)

        repeat_indices = sorted(first_dic[base_label].keys())
        repeat_colors = self._repeat_color_map(repeat_indices, base_cmap=base_cmap)

        repeat_handles = self._repeat_legend_handles(repeat_indices, repeat_colors)
        es_handles = self._es_legend_handles(es_list, es_marker_map, es_marker_size)

        if base_dir:
            os.makedirs(base_dir, exist_ok=True)

        self._plot_one_loss_family(
            models_dics_list=models_dics_list,
            es_list=es_list,
            repeat_indices=repeat_indices,
            repeat_colors=repeat_colors,
            es_marker_map=es_marker_map,
            es_marker_size=es_marker_size,
            use_val_loss=use_val_loss,
            loss_kind="total",
            ylabel="Total loss",
            ylim=ylim_total,
            figsize=figsize,
            yscale=yscale,
            xscale=xscale,
            xlim_left=xlim_left,
            xlim_right=xlim_right,
            repeat_handles=repeat_handles,
            es_handles=es_handles,
            legend_loc_repeat=total_legend_loc_repeat,
            legend_loc_es=total_legend_loc_es,
            repeat_legend_fontsize=total_repeat_legend_fontsize,
            repeat_legend_title_fontsize=total_repeat_legend_title_fontsize,
            es_legend_fontsize=total_es_legend_fontsize,
            es_legend_title_fontsize=total_es_legend_title_fontsize,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            major_tick_length=major_tick_length,
            major_tick_width=major_tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
            base_label=base_label,
            title=title,
            save_path=None if base_dir is None else os.path.join(base_dir, "total_loss.png"),
        )

        self._plot_one_loss_family(
            models_dics_list=models_dics_list,
            es_list=es_list,
            repeat_indices=repeat_indices,
            repeat_colors=repeat_colors,
            es_marker_map=es_marker_map,
            es_marker_size=es_marker_size,
            use_val_loss=use_val_loss,
            loss_kind="data",
            ylabel="Data loss [a.u.]",
            ylim=ylim_data,
            figsize=figsize,
            yscale=yscale,
            xscale=xscale,
            xlim_left=xlim_left,
            xlim_right=xlim_right,
            repeat_handles=repeat_handles,
            es_handles=es_handles,
            legend_loc_repeat=data_legend_loc_repeat,
            legend_loc_es=data_legend_loc_es,
            repeat_legend_fontsize=data_repeat_legend_fontsize,
            repeat_legend_title_fontsize=data_repeat_legend_title_fontsize,
            es_legend_fontsize=data_es_legend_fontsize,
            es_legend_title_fontsize=data_es_legend_title_fontsize,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            major_tick_length=major_tick_length,
            major_tick_width=major_tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
            base_label=base_label,
            title=None,
            save_path=None if base_dir is None else os.path.join(base_dir, "data_loss.png"),
        )

        self._plot_one_loss_family(
            models_dics_list=models_dics_list,
            es_list=es_list,
            repeat_indices=repeat_indices,
            repeat_colors=repeat_colors,
            es_marker_map=es_marker_map,
            es_marker_size=es_marker_size,
            use_val_loss=use_val_loss,
            loss_kind="pde",
            ylabel="PDE loss [a.u.]",
            ylim=ylim_pde,
            figsize=figsize,
            yscale=yscale,
            xscale=xscale,
            xlim_left=xlim_left,
            xlim_right=xlim_right,
            repeat_handles=repeat_handles,
            es_handles=es_handles,
            legend_loc_repeat=pde_legend_loc_repeat,
            legend_loc_es=pde_legend_loc_es,
            repeat_legend_fontsize=pde_repeat_legend_fontsize,
            repeat_legend_title_fontsize=pde_repeat_legend_title_fontsize,
            es_legend_fontsize=pde_es_legend_fontsize,
            es_legend_title_fontsize=pde_es_legend_title_fontsize,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            major_tick_length=major_tick_length,
            major_tick_width=major_tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
            base_label=base_label,
            title=None,
            save_path=None if base_dir is None else os.path.join(base_dir, "pde_loss.png"),
        )