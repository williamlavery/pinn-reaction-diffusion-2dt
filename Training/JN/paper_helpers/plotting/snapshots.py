# Pedagogical Header
# Module: paper_plots/snapshots_plotter.py
# Purpose: Generate snapshot heatmap/scatter figures for paper panels.
# Used by: experim.ipynb (JN) and sibling helper modules.
# Behavior: intended to match original JN implementation.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
density_snapshot_plotter.py
===========================

Utilities for creating snapshot visualizations from 2D density fields and
cell-position data.

This module provides the ``DensitySnapshotPlotter`` class, which creates:

    1. density heatmaps with optional slice lines
    2. scatter plots of cell positions with optional slice lines
    3. a 1xN panel figure of green/parental cell positions at selected time points
    4. one saved scatter figure per key for a user-supplied time array
    5. one saved wide horizontal scatter figure per requested time, with
        subplots spanning the selected keys
    6. a relative time equivalent of the above, with optional density threshold highlighting.

The class is designed to work with project data objects containing:
    - u_red, u_green
    - u_red_max, u_green_max
    - x1, x2
    - hoursRange
    - optionally cdgs, cdrs for scatter snapshots
"""

from __future__ import annotations

# =========================
# Imports
# =========================
from pathlib import Path
import re
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle


class DensitySnapshotPlotter:
    """
    Plot density heatmaps and scatter snapshots from a project data object.

    Parameters
    ----------
    out_dir : str or Path
        Output directory where figures will be saved.
    overwrite_existing : bool, default=True
        Whether to overwrite existing output files.
    show : bool, default=True
        Whether to display figures with ``plt.show()``.
    save_dpi : int, default=100
        DPI used when saving figures.
    """

    def __init__(
        self,
        out_dir: Union[str, Path],
        *,
        overwrite_existing: bool = True,
        show: bool = True,
        save_dpi: int = 100,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.overwrite_existing = overwrite_existing
        self.show = show
        self.save_dpi = save_dpi

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    @staticmethod
    def _validate_species(species: str) -> None:
        valid = {"red", "green", "total"}
        if species not in valid:
            raise ValueError(f"`species` must be one of {sorted(valid)}.")

    @staticmethod
    def _resolve_time_index(nt: int, t_index: Optional[int]) -> int:
        if t_index is None:
            return nt - 1
        return int(t_index) % nt

    @staticmethod
    def _validate_optional_index(idx: Optional[int], n: int, name: str) -> None:
        if idx is not None and not (0 <= idx < n):
            raise IndexError(f"`{name}` must be in [0, {n - 1}] or None.")

    @staticmethod
    def _compute_grid_geometry(
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, list]:
        nx1 = len(x1)
        nx2 = len(x2)

        dx1 = float((x1[-1] - x1[0]) / (nx1 - 1)) if nx1 > 1 else 1.0
        dx2 = float((x2[-1] - x2[0]) / (nx2 - 1)) if nx2 > 1 else 1.0

        x1_lo = float(x1[0] - dx1 / 2)
        x1_hi = float(x1[-1] + dx1 / 2)
        x2_lo = float(x2[0] - dx2 / 2)
        x2_hi = float(x2[-1] + dx2 / 2)

        x1_edges = x1_lo + np.arange(nx1 + 1) * dx1
        x2_edges = x2_lo + np.arange(nx2 + 1) * dx2
        extent = [x1_lo, x1_hi, x2_lo, x2_hi]

        return x1_lo, x1_hi, x2_lo, x2_hi, x1_edges, x2_edges, extent

    @staticmethod
    def _resolve_heatmap_field(
        u_red: np.ndarray,
        u_green: np.ndarray,
        species: Literal["red", "green", "total"],
    ):
        vmax_red = float(np.nanmax(u_red))
        vmax_green = float(np.nanmax(u_green))
        vmax_total = float(np.nanmax(u_red + u_green))

        if species == "red":
            hm_vmax = max(vmax_red, 1e-12)
            default_cmap = "Reds"

            def hm_field(k: int) -> np.ndarray:
                return u_red[:, :, k]

        elif species == "green":
            hm_vmax = max(vmax_green, 1e-12)
            default_cmap = "Greens"

            def hm_field(k: int) -> np.ndarray:
                return u_green[:, :, k]

        else:
            hm_vmax = max(vmax_total, 1e-12)
            default_cmap = "viridis"

            def hm_field(k: int) -> np.ndarray:
                return u_red[:, :, k] + u_green[:, :, k]

        return hm_field, hm_vmax, default_cmap

    @staticmethod
    def _get_scatter_positions(data_obj, hour_t: float):
        """Resolve nearest-time scatter positions from data_obj.cdgs / data_obj.cdrs."""
        cdg = None
        cdr = None

        if hasattr(data_obj, "cdgs") and hasattr(data_obj, "cdrs"):
            keys = list(data_obj.cdgs.keys())
            if keys:
                arr = np.asarray(keys, dtype=float)
                best = keys[int(np.argmin(np.abs(arr - hour_t)))]
                cdg = np.asarray(data_obj.cdgs[best]) / 1000.0
                if best in data_obj.cdrs:
                    cdr = np.asarray(data_obj.cdrs[best]) / 1000.0

        return cdg, cdr

    @staticmethod
    def _sanitize_for_filename(value) -> str:
        s = str(value)
        safe = []
        for ch in s:
            if ch.isalnum() or ch in ("-", "_"):
                safe.append(ch)
            elif ch == ".":
                safe.append("p")
            else:
                safe.append("_")
        return "".join(safe)

    @staticmethod
    def _format_time_array_for_filename(time_array: Sequence[float]) -> str:
        return "__".join(DensitySnapshotPlotter._sanitize_for_filename(v) for v in time_array)

    @staticmethod
    def _coerce_time_values(values: Sequence[object], *, label: str) -> np.ndarray:
        """Convert heterogeneous time labels to float values for robust matching."""
        parsed = []
        bad_values = []

        for raw in values:
            try:
                parsed.append(float(raw))
                continue
            except (TypeError, ValueError):
                pass

            text = str(raw)
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            if match is None:
                bad_values.append(raw)
            else:
                parsed.append(float(match.group(0)))

        if bad_values:
            sample = ", ".join(repr(v) for v in bad_values[:3])
            raise ValueError(
                f"Could not parse numeric times from `{label}`. "
                f"Examples: {sample}"
            )

        return np.asarray(parsed, dtype=float)

    def _maybe_show_and_close(self, fig: plt.Figure) -> None:
        if self.show:
            plt.show()
        plt.close(fig)

    def _resolve_overwrite(self, overwrite_existing: Optional[bool]) -> bool:
        return self.overwrite_existing if overwrite_existing is None else overwrite_existing

    @staticmethod
    def _style_axes(
        ax: plt.Axes,
        *,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlabel_fontsize: float = 10.0,
        ylabel_fontsize: float = 10.0,
        xtick_labelsize: float = 9.0,
        ytick_labelsize: float = 9.0,
        major_tick_length: float = 4.0,
        major_tick_width: float = 1.0,
        minor_tick_length: Optional[float] = 2.5,
        minor_tick_width: Optional[float] = 0.8,
    ) -> None:
        """Apply explicit axis-label and tick styling."""
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

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

        if minor_tick_length is not None and minor_tick_width is not None:
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

    def _plot_green_scatter_panel(
        self,
        ax: plt.Axes,
        *,
        cdg: np.ndarray,
        x1_lo: float,
        x1_hi: float,
        x2_lo: float,
        x2_hi: float,
        x1_edges: np.ndarray,
        x2_edges: np.ndarray,
        psize: float,
        alpha: float,
        col_parental: str,
        xlabel_fontsize: float,
        ylabel_fontsize: float,
        xtick_labelsize: float,
        ytick_labelsize: float,
        major_tick_length: float,
        major_tick_width: float,
        minor_tick_length: float,
        minor_tick_width: float,
        ix: Optional[int],
        iy: Optional[int],
        x0: Optional[float],
        y0: Optional[float],
        show_slice_lines: bool,
        title: Optional[str] = None,
        title_fontsize: float = 10.0,
        show_xlabel: bool = True,
        show_ylabel: bool = True,
        highlight_dense_squares: bool = False,
        density_slice: Optional[np.ndarray] = None,
        density_threshold: float = 1000.0,
    ) -> None:
        """Draw one green/parental-cell scatter panel onto an existing axis."""
        if cdg.size:
            
            cdg = cdg / 1000.0
            ax.scatter(
                cdg[:, 0],
                cdg[:, 1],
                s=psize,
                c=col_parental,
                alpha=alpha,
                edgecolors="none",
                zorder=3,
            )

        # Highlight dense squares with purple rectangles
        if highlight_dense_squares and density_slice is not None:
            nx1, nx2 = density_slice.shape
            for i in range(nx1):
                for j in range(nx2):
                    if density_slice[i, j] > density_threshold:
                        width = x1_edges[i+1] - x1_edges[i]
                        height = x2_edges[j+1] - x2_edges[j]
                        rect = Rectangle(
                            (x1_edges[i], x2_edges[j]), width, height,
                            linewidth=1.5, edgecolor='purple', facecolor='none', zorder=4
                        )
                        ax.add_patch(rect)

        if show_slice_lines:
            if x0 is not None:
                ax.axvline(x=x0, lw=2, ls="-", alpha=1, color="r", zorder=5)
            if y0 is not None:
                ax.axhline(y=y0, lw=2, ls="--", alpha=1, color="r", zorder=5)

        ax.set_xlim(x1_lo, x1_hi)
        ax.set_ylim(x2_lo, x2_hi)
        ax.set_aspect("equal")

        if title is not None:
            ax.set_title(title, fontsize=title_fontsize)

        self._style_axes(
            ax,
            xlabel=r"$x_1$ [mm]" if show_xlabel else None,
            ylabel=r"$x_2$ [mm]" if show_ylabel else None,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            major_tick_length=major_tick_length,
            major_tick_width=major_tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
        )

        ax.grid(False)
        ax.set_xticks(x1_edges, minor=True)
        ax.set_yticks(x2_edges, minor=True)
        ax.grid(which="minor", color="0.7", linestyle="-", linewidth=0.5)

    def _plot_black_domain_panel(
        self,
        ax: plt.Axes,
        *,
        x1_lo: float,
        x1_hi: float,
        x2_lo: float,
        x2_hi: float,
        x1_edges: np.ndarray,
        x2_edges: np.ndarray,
        xlabel_fontsize: float,
        ylabel_fontsize: float,
        xtick_labelsize: float,
        ytick_labelsize: float,
        major_tick_length: float,
        major_tick_width: float,
        minor_tick_length: float,
        minor_tick_width: float,
        ix: Optional[int],
        iy: Optional[int],
        x0: Optional[float],
        y0: Optional[float],
        show_slice_lines: bool,
        title: Optional[str] = None,
        title_fontsize: float = 10.0,
        show_xlabel: bool = True,
        show_ylabel: bool = True,
    ) -> None:
        """Helper to plot a black empty domain when time data is missing."""
        ax.set_facecolor("black")
        ax.set_xlim(x1_lo, x1_hi)
        ax.set_ylim(x2_lo, x2_hi)
        ax.set_aspect("equal")

        if show_slice_lines:
            if x0 is not None:
                ax.axvline(x=x0, lw=2, ls="-", alpha=1, color="r")
            if y0 is not None:
                ax.axhline(y=y0, lw=2, ls="--", alpha=1, color="r")

        if title is not None:
            ax.set_title(title, fontsize=title_fontsize)

        self._style_axes(
            ax,
            xlabel=r"$x_1$ [mm]" if show_xlabel else None,
            ylabel=r"$x_2$ [mm]" if show_ylabel else None,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            major_tick_length=major_tick_length,
            major_tick_width=major_tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
        )

        ax.grid(False)
        ax.set_xticks(x1_edges, minor=True)
        ax.set_yticks(x2_edges, minor=True)
        ax.grid(which="minor", color="0.7", linestyle="-", linewidth=0.5)


    # ------------------------------------------------------------
    # Plotting methods
    # ------------------------------------------------------------
    def make_density_heatmap(
        self,
        data_obj,
        *,
        t_index: Optional[int] = None,
        ix: Optional[int] = None,
        iy: Optional[int] = None,
        species: Literal["red", "green", "total"] = "green",
        cmap: Optional[str] = None,
        figsize: Tuple[float, float] = (7, 5),
        xlabel_fontsize: float = 10.0,
        ylabel_fontsize: float = 10.0,
        xtick_labelsize: float = 9.0,
        ytick_labelsize: float = 9.0,
        title_fontsize: float = 10.0,
        legend_fontsize: float = 9.0,
        legend_loc: str = "upper right",
        legend_kwargs: Optional[dict] = None,
        major_tick_length: float = 4.0,
        major_tick_width: float = 1.0,
        minor_tick_length: float = 2.5,
        minor_tick_width: float = 0.8,
        show_colorbar: bool = True,
        cbar_label: str = "cell density [mm$^{-2}$]",
        cbar_label_fontsize: Optional[float] = None,
        cbar_tick_fontsize: Optional[float] = None,
        cbar_fraction: float = 0.046,
        cbar_pad: float = 0.04,
        cbar_shrink: float = 1.0,
        cbar_aspect: float = 20,
        cbar_ticks: Optional[np.ndarray] = None,
        cbar_format: Optional[str] = None,
        cbar_extend: str = "neither",
        overwrite_existing: Optional[bool] = None,
    ) -> Path:
        """Create and save a density heatmap snapshot."""
        self._validate_species(species)
        overwrite = self._resolve_overwrite(overwrite_existing)

        u_red = np.asarray(data_obj.u_red) * data_obj.u_red_max * 1e6
        u_green = np.asarray(data_obj.u_green) * data_obj.u_green_max * 1e6
        x1 = np.asarray(data_obj.x1)
        x2 = np.asarray(data_obj.x2)

        nx1, nx2, nt = u_red.shape
        k = self._resolve_time_index(nt, t_index)

        self._validate_optional_index(ix, nx1, "ix")
        self._validate_optional_index(iy, nx2, "iy")

        x0 = None if ix is None else float(x1[ix])
        y0 = None if iy is None else float(x2[iy])

        x1_lo, x1_hi, x2_lo, x2_hi, x1_edges, x2_edges, extent = self._compute_grid_geometry(x1, x2)
        hm_field, hm_vmax, default_cmap = self._resolve_heatmap_field(u_red, u_green, species)
        cmap = cmap or default_cmap

        heat_png = self.out_dir / f"density_heatmap_slice_{species}_t{k:04d}.png"
        if heat_png.exists() and not overwrite:
            return heat_png

        legend_kwargs = dict(legend_kwargs or {})
        legend_kwargs.setdefault("frameon", True)
        legend_kwargs.setdefault("facecolor", "white")

        norm = Normalize(vmin=0, vmax=hm_vmax, clip=True)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        im = ax.imshow(
            hm_field(k).T,
            origin="lower",
            extent=extent,
            norm=norm,
            cmap=cmap,
        )

        if x0 is not None:
            ax.axvline(
                x=x0,
                lw=2,
                ls="--",
                alpha=0.85,
                label=fr"$x_1 = {np.round(x0,3)}$ mm",
                color="r",
            )
        if y0 is not None:
            ax.axhline(
                y=y0,
                lw=2,
                ls="-",
                alpha=0.85,
                label=fr"$x_2 = {np.round(y0,3)}$ mm",
                color="r",
            )

        self._style_axes(
            ax,
            xlabel=r"$x_1$ [mm]",
            ylabel=r"$x_2$ [mm]",
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            major_tick_length=major_tick_length,
            major_tick_width=major_tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
        )

        ax.grid(False)
        ax.set_xticks(x1_edges, minor=True)
        ax.set_yticks(x2_edges, minor=True)
        ax.grid(which="minor", color="0.7", linestyle="-", linewidth=1)

        handles, _ = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(
                loc=legend_loc,
                fontsize=legend_fontsize,
                **legend_kwargs,
            )
            leg.set_zorder(10)

        if show_colorbar:
            label_fs = xlabel_fontsize if cbar_label_fontsize is None else cbar_label_fontsize
            tick_fs = xtick_labelsize if cbar_tick_fontsize is None else cbar_tick_fontsize

            cbar = fig.colorbar(
                im,
                ax=ax,
                location="top",
                fraction=cbar_fraction,
                pad=cbar_pad,
                shrink=cbar_shrink,
                aspect=cbar_aspect,
                extend=cbar_extend,
            )

            cbar.set_label(cbar_label, fontsize=label_fs)
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.tick_params(
                labelsize=tick_fs,
                length=major_tick_length,
                width=major_tick_width,
            )

            if cbar_ticks is not None:
                cbar.set_ticks(cbar_ticks)

            if cbar_format is not None:
                cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(cbar_format))

        fig.savefig(heat_png, dpi=self.save_dpi, bbox_inches="tight")
        self._maybe_show_and_close(fig)

        return heat_png

    def make_positions_scatter(
        self,
        data_obj,
        *,
        t_index: Optional[int] = None,
        ix: Optional[int] = None,
        iy: Optional[int] = None,
        figsize: Tuple[float, float] = (7, 5),
        psize: float = 10.0,
        col_parental: str = "tab:green",
        col_resistant: str = "tab:red",
        show_slice_lines: bool = True,
        xlabel_fontsize: float = 10.0,
        ylabel_fontsize: float = 10.0,
        xtick_labelsize: float = 9.0,
        ytick_labelsize: float = 9.0,
        title_fontsize: float = 10.0,
        legend_fontsize: float = 9.0,
        legend_loc: str = "upper right",
        legend_kwargs: Optional[dict] = None,
        major_tick_length: float = 4.0,
        major_tick_width: float = 1.0,
        minor_tick_length: float = 2.5,
        minor_tick_width: float = 0.8,
        overwrite_existing: Optional[bool] = None,
    ) -> Path:
        """Create and save a scatter plot snapshot of cell positions."""
        overwrite = self._resolve_overwrite(overwrite_existing)

        x1 = np.asarray(data_obj.x1)
        x2 = np.asarray(data_obj.x2)
        t = np.asarray(data_obj.hoursRange, dtype=float)

        nx1 = len(x1)
        nx2 = len(x2)
        nt = len(t)

        k = self._resolve_time_index(nt, t_index)

        self._validate_optional_index(ix, nx1, "ix")
        self._validate_optional_index(iy, nx2, "iy")

        x0 = None if ix is None else float(x1[ix])
        y0 = None if iy is None else float(x2[iy])
        hour_t = float(t[k])

        x1_lo, x1_hi, x2_lo, x2_hi, x1_edges, x2_edges, _ = self._compute_grid_geometry(x1, x2)
        cdg, cdr = self._get_scatter_positions(data_obj, hour_t)

        scatter_png = self.out_dir / f"positions_scatter_t{k:04d}.png"
        if scatter_png.exists() and not overwrite:
            return scatter_png

        legend_kwargs = dict(legend_kwargs or {})
        legend_kwargs.setdefault("frameon", True)
        legend_kwargs.setdefault("facecolor", "white")

        fig, ax = plt.subplots(figsize=figsize)

        legend_handles = []
        legend_labels = []

        if cdg is not None and cdg.size:
            scg = ax.scatter(
                cdg[:, 0],
                cdg[:, 1],
                s=psize,
                c=col_parental,
                alpha=0.9,
                edgecolors="none",
                zorder=3,
            )
            legend_handles.append(scg)
            legend_labels.append("cells")

        if cdr is not None and cdr.size:
            scr = ax.scatter(
                cdr[:, 0],
                cdr[:, 1],
                s=psize,
                c=col_resistant,
                alpha=0.9,
                edgecolors="none",
                zorder=3,
            )
            legend_handles.append(scr)
            legend_labels.append("resistant")

        if show_slice_lines:
            if x0 is not None:
                lx = ax.axvline(x=x0, lw=2, ls="-", alpha=1, color="r")
                legend_handles.append(lx)
                legend_labels.append(fr"$x_1 = {np.round(x0,3)}$")
            if y0 is not None:
                ly = ax.axhline(y=y0, lw=2, ls="--", alpha=1, color="r")
                legend_handles.append(ly)
                legend_labels.append(fr"$x_2 = {np.round(y0,3)}$")

        ax.set_xlim(x1_lo, x1_hi)
        ax.set_ylim(x2_lo, x2_hi)
        ax.set_aspect("equal")

        self._style_axes(
            ax,
            xlabel=r"$x_1$ [mm]",
            ylabel=r"$x_2$ [mm]",
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            major_tick_length=major_tick_length,
            major_tick_width=major_tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
        )

        ax.grid(False)
        ax.set_xticks(x1_edges, minor=True)
        ax.set_yticks(x2_edges, minor=True)
        ax.grid(which="minor", color="0.7", linestyle="-", linewidth=0.5)

        if legend_handles:
            leg = ax.legend(
                legend_handles,
                legend_labels,
                loc=legend_loc,
                fontsize=legend_fontsize,
                **legend_kwargs,
            )
            leg.set_zorder(10)

        fig.tight_layout()
        fig.savefig(scatter_png, dpi=self.save_dpi, bbox_inches="tight")
        self._maybe_show_and_close(fig)

        return scatter_png

    def make_green_cell_scatter_snapshots_5_times(
        self,
        data_obj,
        *,
        overwrite_existing: Optional[bool] = None,
        psize: float = 10.0,
        alpha: float = 0.8,
        col_parental: str = "tab:green",
        figsize_per_panel: Tuple[float, float] = (4.0, 4.0),
        xlabel_fontsize: float = 10.0,
        ylabel_fontsize: float = 10.0,
        xtick_labelsize: float = 9.0,
        ytick_labelsize: float = 9.0,
        title_fontsize: float = 10.0,
        major_tick_length: float = 4.0,
        major_tick_width: float = 1.0,
        minor_tick_length: float = 2.5,
        minor_tick_width: float = 0.8,
        ix: Optional[int] = None,
        iy: Optional[int] = None,
        show_slice_lines: bool = False,
        tnum: int = 5,
    ) -> Path:
        """Create a 1xN series of subplots for 5 times."""
        overwrite = self._resolve_overwrite(overwrite_existing)

        x1 = np.asarray(data_obj.x1, dtype=float)
        x2 = np.asarray(data_obj.x2, dtype=float)
        t = np.asarray(data_obj.hoursRange, dtype=float)

        if t.size == 0:
            raise ValueError("`data_obj.hoursRange` is empty.")
        if not hasattr(data_obj, "cdgs"):
            raise AttributeError("`data_obj` must have a `cdgs` attribute.")

        nx1 = len(x1)
        nx2 = len(x2)

        self._validate_optional_index(ix, nx1, "ix")
        self._validate_optional_index(iy, nx2, "iy")

        x0 = None if ix is None else float(x1[ix])
        y0 = None if iy is None else float(x2[iy])

        x1_lo, x1_hi, x2_lo, x2_hi, x1_edges, x2_edges, _ = self._compute_grid_geometry(x1, x2)

        time_indices = np.linspace(0, len(t) - 1, tnum, dtype=int)
        time_indices = np.unique(time_indices)
        chosen_times = t[time_indices]

        cdg_keys = list(data_obj.cdgs.keys())
        if len(cdg_keys) == 0:
            raise ValueError("`data_obj.cdgs` is empty.")

        cdg_keys_arr = np.asarray(cdg_keys, dtype=float)

        scatter_png = self.out_dir / "green_cell_scatter_5_times.png"
        if scatter_png.exists() and not overwrite:
            return scatter_png

        n_panels = len(chosen_times)
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
            squeeze=False,
        )
        axes = axes.ravel()

        for ax, hour_t in zip(axes, chosen_times):
            best_key = cdg_keys[int(np.argmin(np.abs(cdg_keys_arr - hour_t)))]
            cdg = np.asarray(data_obj.cdgs.get(best_key, []), dtype=float)

            self._plot_green_scatter_panel(
                ax,
                cdg=cdg,
                x1_lo=x1_lo,
                x1_hi=x1_hi,
                x2_lo=x2_lo,
                x2_hi=x2_hi,
                x1_edges=x1_edges,
                x2_edges=x2_edges,
                psize=psize,
                alpha=alpha,
                col_parental=col_parental,
                xlabel_fontsize=xlabel_fontsize,
                ylabel_fontsize=ylabel_fontsize,
                xtick_labelsize=xtick_labelsize,
                ytick_labelsize=ytick_labelsize,
                major_tick_length=major_tick_length,
                major_tick_width=major_tick_width,
                minor_tick_length=minor_tick_length,
                minor_tick_width=minor_tick_width,
                ix=ix,
                iy=iy,
                x0=x0,
                y0=y0,
                show_slice_lines=show_slice_lines,
                title=f"t = {best_key:g} h",
                title_fontsize=title_fontsize,
            )

        fig.tight_layout()
        fig.savefig(scatter_png, dpi=self.save_dpi, bbox_inches="tight")
        self._maybe_show_and_close(fig)

        return scatter_png

    def make_green_cell_scatter_snapshots_for_times(
        self,
        data_obj,
        *,
        time_array: Sequence[float],
        figure_name: Optional[str] = None,
        overwrite_existing: Optional[bool] = None,
        psize: float = 10.0,
        alpha: float = 0.8,
        col_parental: str = "tab:green",
        figsize_per_panel: Tuple[float, float] = (4.0, 4.0),
        xlabel_fontsize: float = 10.0,
        ylabel_fontsize: float = 10.0,
        xtick_labelsize: float = 9.0,
        ytick_labelsize: float = 9.0,
        title_fontsize: float = 10.0,
        major_tick_length: float = 4.0,
        major_tick_width: float = 1.0,
        minor_tick_length: float = 2.5,
        minor_tick_width: float = 0.8,
        ix: Optional[int] = None,
        iy: Optional[int] = None,
        show_slice_lines: bool = False,
    ) -> Path:
        """Create and save a 1xN figure of scatter plots at the user-supplied times."""
        overwrite = self._resolve_overwrite(overwrite_existing)

        x1 = np.asarray(data_obj.x1, dtype=float)
        x2 = np.asarray(data_obj.x2, dtype=float)
        requested_times = np.asarray(time_array, dtype=float)

        if requested_times.size == 0:
            raise ValueError("`time_array` must contain at least one time point.")
        if not hasattr(data_obj, "cdgs"):
            raise AttributeError("`data_obj` must have a `cdgs` attribute.")
        if len(data_obj.cdgs) == 0:
            raise ValueError("`data_obj.cdgs` is empty.")

        nx1 = len(x1)
        nx2 = len(x2)

        self._validate_optional_index(ix, nx1, "ix")
        self._validate_optional_index(iy, nx2, "iy")

        x0 = None if ix is None else float(x1[ix])
        y0 = None if iy is None else float(x2[iy])

        x1_lo, x1_hi, x2_lo, x2_hi, x1_edges, x2_edges, _ = self._compute_grid_geometry(x1, x2)

        cdg_keys = list(data_obj.cdgs.keys())
        cdg_keys_arr = np.asarray(cdg_keys, dtype=float)

        if figure_name is None:
            times_tag = self._format_time_array_for_filename(requested_times)
            figure_name = f"green_cell_scatter_times_{times_tag}"

        scatter_png = self.out_dir / f"{figure_name}.png"
        if scatter_png.exists() and not overwrite:
            return scatter_png

        n_panels = len(requested_times)
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
            squeeze=False,
        )
        axes = axes.ravel()

        for ax, hour_t in zip(axes, requested_times):
            best_idx = int(np.argmin(np.abs(cdg_keys_arr - hour_t)))
            
            if np.abs(cdg_keys_arr[best_idx] - hour_t) > 1e-3:
                # Black domain if no strict match
                self._plot_black_domain_panel(
                    ax,
                    x1_lo=x1_lo, x1_hi=x1_hi, x2_lo=x2_lo, x2_hi=x2_hi,
                    x1_edges=x1_edges, x2_edges=x2_edges,
                    xlabel_fontsize=xlabel_fontsize, ylabel_fontsize=ylabel_fontsize,
                    xtick_labelsize=xtick_labelsize, ytick_labelsize=ytick_labelsize,
                    major_tick_length=major_tick_length, major_tick_width=major_tick_width,
                    minor_tick_length=minor_tick_length, minor_tick_width=minor_tick_width,
                    ix=ix, iy=iy, x0=x0, y0=y0, show_slice_lines=show_slice_lines,
                    title=f"req = {hour_t:g} h\nNo data", title_fontsize=title_fontsize
                )
            else:
                best_key = cdg_keys[best_idx]
                cdg = np.asarray(data_obj.cdgs.get(best_key, []), dtype=float)

                self._plot_green_scatter_panel(
                    ax,
                    cdg=cdg,
                    x1_lo=x1_lo, x1_hi=x1_hi, x2_lo=x2_lo, x2_hi=x2_hi,
                    x1_edges=x1_edges, x2_edges=x2_edges,
                    psize=psize, alpha=alpha, col_parental=col_parental,
                    xlabel_fontsize=xlabel_fontsize, ylabel_fontsize=ylabel_fontsize,
                    xtick_labelsize=xtick_labelsize, ytick_labelsize=ytick_labelsize,
                    major_tick_length=major_tick_length, major_tick_width=major_tick_width,
                    minor_tick_length=minor_tick_length, minor_tick_width=minor_tick_width,
                    ix=ix, iy=iy, x0=x0, y0=y0, show_slice_lines=show_slice_lines,
                    title=f"req = {hour_t:g} h\nused = {float(best_key):g} h",
                    title_fontsize=title_fontsize,
                )

        fig.tight_layout()
        fig.savefig(scatter_png, dpi=self.save_dpi, bbox_inches="tight")
        self._maybe_show_and_close(fig)

        return scatter_png

    def make_green_cell_scatter_snapshots_for_keys_and_times(
        self,
        data_obj_dict: Dict[str, object],
        *,
        keys: Iterable[str],
        time_array: Sequence[float],
        mapping_dict: Optional[Dict[str, str]] = None,
        ix: Optional[int] = None,
        iy: Optional[int] = None,
        overwrite_existing: Optional[bool] = None,
        psize: float = 10.0,
        alpha: float = 0.8,
        col_parental: str = "tab:green",
        figsize_per_panel: Tuple[float, float] = (4.0, 4.0),
        xlabel_fontsize: float = 10.0,
        ylabel_fontsize: float = 10.0,
        xtick_labelsize: float = 9.0,
        ytick_labelsize: float = 9.0,
        title_fontsize: float = 10.0,
        major_tick_length: float = 4.0,
        major_tick_width: float = 1.0,
        minor_tick_length: float = 2.5,
        minor_tick_width: float = 0.8,
        show_slice_lines: bool = False,
        subdir: Optional[str] = "per_key_time_arrays",
    ) -> Dict[str, Path]:
        """
        For each key, create and save one separate scatter figure using the same
        supplied `time_array`. Evaluates STRICTLY per time point: if the 
        requested time is missing for a key, it plots a black domain.
        """
        overwrite = self._resolve_overwrite(overwrite_existing)
        results: Dict[str, Path] = {}

        original_out_dir = self.out_dir
        if subdir is not None:
            self.out_dir = self.out_dir / subdir
            self.out_dir.mkdir(parents=True, exist_ok=True)

        try:
            time_tag = self._format_time_array_for_filename(time_array)
            requested_times = np.asarray(time_array, dtype=float)

            if requested_times.size == 0:
                raise ValueError("`time_array` must contain at least one time point.")

            for key in keys:
                if key not in data_obj_dict:
                    raise KeyError(f"Key {key!r} not found in `data_obj_dict`.")

                data_obj = data_obj_dict[key]
                figure_name = f"green_cell_scatter_key_{self._sanitize_for_filename(key)}__times_{time_tag}.png"
                scatter_png = self.out_dir / figure_name

                if scatter_png.exists() and not overwrite:
                    results[key] = scatter_png
                    continue

                x1 = np.asarray(data_obj.x1, dtype=float)
                x2 = np.asarray(data_obj.x2, dtype=float)

                nx1 = len(x1)
                nx2 = len(x2)

                self._validate_optional_index(ix, nx1, "ix")
                self._validate_optional_index(iy, nx2, "iy")

                x0 = None if ix is None else float(x1[ix])
                y0 = None if iy is None else float(x2[iy])

                x1_lo, x1_hi, x2_lo, x2_hi, x1_edges, x2_edges, _ = self._compute_grid_geometry(x1, x2)

                n_panels = len(requested_times)
                fig, axes = plt.subplots(
                    1,
                    n_panels,
                    figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
                    squeeze=False,
                )
                axes = axes.ravel()

                mapped_key = mapping_dict[key] if (mapping_dict and key in mapping_dict) else key

                has_time_data = hasattr(data_obj, "cdgs") and len(data_obj.cdgs) > 0
                if has_time_data:
                    cdg_keys_list = list(data_obj.cdgs.keys())
                    cdg_keys_arr = np.asarray(cdg_keys_list, dtype=float)
                else:
                    cdg_keys_list = []
                    cdg_keys_arr = np.array([])

                for ax, hour_t in zip(axes, requested_times):
                    
                    if has_time_data:
                        match_idx = int(np.argmin(np.abs(cdg_keys_arr - hour_t)))
                        is_match = np.abs(cdg_keys_arr[match_idx] - hour_t) <= 1e-3
                    else:
                        is_match = False

                    if not is_match:
                        # Black domain
                        self._plot_black_domain_panel(
                            ax,
                            x1_lo=x1_lo, x1_hi=x1_hi, x2_lo=x2_lo, x2_hi=x2_hi,
                            x1_edges=x1_edges, x2_edges=x2_edges,
                            xlabel_fontsize=xlabel_fontsize, ylabel_fontsize=ylabel_fontsize,
                            xtick_labelsize=xtick_labelsize, ytick_labelsize=ytick_labelsize,
                            major_tick_length=major_tick_length, major_tick_width=major_tick_width,
                            minor_tick_length=minor_tick_length, minor_tick_width=minor_tick_width,
                            ix=ix, iy=iy, x0=x0, y0=y0, show_slice_lines=show_slice_lines,
                            title=f"{mapped_key}\nreq = {hour_t:g} h\nNo data", title_fontsize=title_fontsize
                        )
                    else:
                        # Real data
                        best_key = cdg_keys_list[match_idx]
                        cdg = np.asarray(data_obj.cdgs.get(best_key, []), dtype=float)

                        self._plot_green_scatter_panel(
                            ax,
                            cdg=cdg,
                            x1_lo=x1_lo, x1_hi=x1_hi, x2_lo=x2_lo, x2_hi=x2_hi,
                            x1_edges=x1_edges, x2_edges=x2_edges,
                            psize=psize, alpha=alpha, col_parental=col_parental,
                            xlabel_fontsize=xlabel_fontsize, ylabel_fontsize=ylabel_fontsize,
                            xtick_labelsize=xtick_labelsize, ytick_labelsize=ytick_labelsize,
                            major_tick_length=major_tick_length, major_tick_width=major_tick_width,
                            minor_tick_length=minor_tick_length, minor_tick_width=minor_tick_width,
                            ix=ix, iy=iy, x0=x0, y0=y0, show_slice_lines=show_slice_lines,
                            title=f"{mapped_key}\nreq = {hour_t:g} h\nused = {float(best_key):g} h",
                            title_fontsize=title_fontsize,
                        )

                fig.tight_layout()
                fig.savefig(scatter_png, dpi=self.save_dpi, bbox_inches="tight")
                self._maybe_show_and_close(fig)

                results[key] = scatter_png

        finally:
            self.out_dir = original_out_dir

        return results

    def make_green_cell_scatter_snapshots_individually_for_keys_and_times(
        self,
        data_obj_dict: Dict[str, object],
        *,
        keys: Iterable[str],
        time_array: Sequence[float],
        ix: Optional[int] = None,
        iy: Optional[int] = None,
        overwrite_existing: Optional[bool] = None,
        psize: float = 10.0,
        alpha: float = 0.8,
        col_parental: str = "tab:green",
        figsize: Tuple[float, float] = (4.0, 4.0),
        xlabel_fontsize: float = 10.0,
        ylabel_fontsize: float = 10.0,
        xtick_labelsize: float = 9.0,
        ytick_labelsize: float = 9.0,
        title_fontsize: float = 10.0,
        major_tick_length: float = 4.0,
        major_tick_width: float = 1.0,
        minor_tick_length: float = 2.5,
        minor_tick_width: float = 0.8,
        show_slice_lines: bool = False,
        subdir: Optional[str] = "per_key_individual_times",
    ) -> Dict[str, Dict[float, Path]]:
        """
        For each key and each requested time, create and save one separate scatter
        figure. (Note: Only saves if data actually exists or matches).
        """
        overwrite = self._resolve_overwrite(overwrite_existing)
        results: Dict[str, Dict[float, Path]] = {}

        original_out_dir = self.out_dir
        if subdir is not None:
            self.out_dir = self.out_dir / subdir
            self.out_dir.mkdir(parents=True, exist_ok=True)

        try:
            requested_times = np.asarray(time_array, dtype=float)
            if requested_times.size == 0:
                raise ValueError("`time_array` must contain at least one time point.")

            for key in keys:
                if key not in data_obj_dict:
                    raise KeyError(f"Key {key!r} not found in `data_obj_dict`.")

                data_obj = data_obj_dict[key]

                x1 = np.asarray(data_obj.x1, dtype=float)
                x2 = np.asarray(data_obj.x2, dtype=float)

                if not hasattr(data_obj, "cdgs"):
                    raise AttributeError(f"`data_obj_dict[{key!r}]` must have a `cdgs` attribute.")
                if len(data_obj.cdgs) == 0:
                    raise ValueError(f"`data_obj_dict[{key!r}].cdgs` is empty.")

                nx1 = len(x1)
                nx2 = len(x2)

                self._validate_optional_index(ix, nx1, "ix")
                self._validate_optional_index(iy, nx2, "iy")

                x0 = None if ix is None else float(x1[ix])
                y0 = None if iy is None else float(x2[iy])

                x1_lo, x1_hi, x2_lo, x2_hi, x1_edges, x2_edges, _ = self._compute_grid_geometry(x1, x2)

                cdg_keys = list(data_obj.cdgs.keys())
                cdg_keys_arr = self._coerce_time_values(cdg_keys, label=f"data_obj_dict[{key!r}].cdgs keys")

                key_results: Dict[float, Path] = {}
                safe_key = self._sanitize_for_filename(key)

                for hour_t in requested_times:
                    match_idx = int(np.argmin(np.abs(cdg_keys_arr - hour_t)))
                    
                    if np.abs(cdg_keys_arr[match_idx] - hour_t) > 1e-3:
                        # Skip generating individual files for times with no data
                        continue

                    best_key = cdg_keys[match_idx]
                    cdg = np.asarray(data_obj.cdgs.get(best_key, []), dtype=float)

                    figure_name = (
                        f"green_cell_scatter_key_{safe_key}"
                        f"__req_{self._sanitize_for_filename(hour_t)}h"
                        f"__used_{self._sanitize_for_filename(best_key)}h.png"
                    )
                    scatter_png = self.out_dir / figure_name

                    if scatter_png.exists() and not overwrite:
                        key_results[float(hour_t)] = scatter_png
                        continue

                    fig, ax = plt.subplots(figsize=figsize)

                    self._plot_green_scatter_panel(
                        ax,
                        cdg=cdg,
                        x1_lo=x1_lo,
                        x1_hi=x1_hi,
                        x2_lo=x2_lo,
                        x2_hi=x2_hi,
                        x1_edges=x1_edges,
                        x2_edges=x2_edges,
                        psize=psize,
                        alpha=alpha,
                        col_parental=col_parental,
                        xlabel_fontsize=xlabel_fontsize,
                        ylabel_fontsize=ylabel_fontsize,
                        xtick_labelsize=xtick_labelsize,
                        ytick_labelsize=ytick_labelsize,
                        major_tick_length=major_tick_length,
                        major_tick_width=major_tick_width,
                        minor_tick_length=minor_tick_length,
                        minor_tick_width=minor_tick_width,
                        ix=ix,
                        iy=iy,
                        x0=x0,
                        y0=y0,
                        show_slice_lines=show_slice_lines,
                        title=None,
                        title_fontsize=title_fontsize,
                    )

                    fig.tight_layout()
                    fig.savefig(scatter_png, dpi=self.save_dpi, bbox_inches="tight")
                    self._maybe_show_and_close(fig)

                    key_results[float(hour_t)] = scatter_png

                results[key] = key_results

        finally:
            self.out_dir = original_out_dir

        return results

    def make_green_cell_scatter_snapshots_for_times_with_keys_as_subplots(
        self,
        data_obj_dict: Dict[str, object],
        *,
        keys: Iterable[str],
        time_array: Sequence[float],
        mapping_dict: Optional[Dict[str, str]] = None,
        ix: Optional[int] = None,
        iy: Optional[int] = None,
        overwrite_existing: Optional[bool] = None,
        psize: float = 10.0,
        alpha: float = 0.8,
        col_parental: str = "tab:green",
        figsize_per_panel: Tuple[float, float] = (4.0, 4.0),
        xlabel_fontsize: float = 10.0,
        ylabel_fontsize: float = 10.0,
        xtick_labelsize: float = 9.0,
        ytick_labelsize: float = 9.0,
        title_fontsize: float = 10.0,
        major_tick_length: float = 4.0,
        major_tick_width: float = 1.0,
        minor_tick_length: float = 2.5,
        minor_tick_width: float = 0.8,
        show_slice_lines: bool = False,
        subdir: Optional[str] = "per_time_across_keys",
        share_axes: bool = False,
        include_requested_and_used_time_in_filename: bool = True,
    ) -> Dict[Union[float, str], Path]:
        """
        For each requested time, create one separate wide horizontal figure with
        one subplot per key. STRICT per-time matching: plots black domain if 
        time data is missing for a key at that hour.
        
        Additionally saves one massive vertically long master figure containing 
        all times (rows) and all keys (columns).
        """
        overwrite = self._resolve_overwrite(overwrite_existing)
        requested_times = np.asarray(time_array, dtype=float)
        key_list = list(keys)

        if requested_times.size == 0:
            raise ValueError("`time_array` must contain at least one time point.")
        if len(key_list) == 0:
            raise ValueError("`keys` must contain at least one key.")

        results: Dict[Union[float, str], Path] = {}

        original_out_dir = self.out_dir
        if subdir is not None:
            self.out_dir = self.out_dir / subdir
            self.out_dir.mkdir(parents=True, exist_ok=True)

        try:
            plot_info: Dict[str, dict] = {}

            # Pre-compute data characteristics for all keys
            for key in key_list:
                if key not in data_obj_dict:
                    raise KeyError(f"Key {key!r} not found in `data_obj_dict`.")

                data_obj = data_obj_dict[key]
                has_time_data = hasattr(data_obj, "cdgs") and len(data_obj.cdgs) > 0

                x1 = np.asarray(data_obj.x1, dtype=float)
                x2 = np.asarray(data_obj.x2, dtype=float)

                nx1 = len(x1)
                nx2 = len(x2)

                self._validate_optional_index(ix, nx1, "ix")
                self._validate_optional_index(iy, nx2, "iy")

                x0 = None if ix is None else float(x1[ix])
                y0 = None if iy is None else float(x2[iy])

                x1_lo, x1_hi, x2_lo, x2_hi, x1_edges, x2_edges, _ = self._compute_grid_geometry(x1, x2)

                if has_time_data:
                    cdg_keys = list(data_obj.cdgs.keys())
                    cdg_keys_arr = self._coerce_time_values(cdg_keys, label=f"data_obj_dict[{key!r}].cdgs keys")
                else:
                    cdg_keys = []
                    cdg_keys_arr = np.array([])

                plot_info[key] = {
                    "data_obj": data_obj,
                    "x0": x0,
                    "y0": y0,
                    "x1_lo": x1_lo,
                    "x1_hi": x1_hi,
                    "x2_lo": x2_lo,
                    "x2_hi": x2_hi,
                    "x1_edges": x1_edges,
                    "x2_edges": x2_edges,
                    "has_time_data": has_time_data,
                    "cdg_keys": cdg_keys,
                    "cdg_keys_arr": cdg_keys_arr,
                }

            # Setup the massive master grid figure
            n_rows = len(requested_times)
            n_cols = len(key_list)
            
            fig_master, axes_master = plt.subplots(
                n_rows,
                n_cols,
                figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
                squeeze=False,
                sharex=share_axes,
                sharey=share_axes,
            )

            # Loop over requested times (rows)
            for row_idx, hour_t in enumerate(requested_times):
                used_times_for_filename = []

                # Determine file naming parameters
                for key in key_list:
                    info = plot_info[key]
                    if info["has_time_data"]:
                        match_idx = int(np.argmin(np.abs(info["cdg_keys_arr"] - hour_t)))
                        if np.abs(info["cdg_keys_arr"][match_idx] - hour_t) <= 1e-3:
                            used_times_for_filename.append(float(info["cdg_keys"][match_idx]))
                        else:
                            used_times_for_filename.append("nodata")
                    else:
                        used_times_for_filename.append("nodata")

                if include_requested_and_used_time_in_filename:
                    used_tag = self._format_time_array_for_filename(used_times_for_filename)
                    figure_name = (
                        f"green_cell_scatter_all_keys"
                        f"__req_{self._sanitize_for_filename(hour_t)}h"
                        f"__used_{used_tag}.png"
                    )
                else:
                    figure_name = (
                        f"green_cell_scatter_all_keys"
                        f"__req_{self._sanitize_for_filename(hour_t)}h.png"
                    )

                scatter_png = self.out_dir / figure_name
                
                # Setup the single horizontal row figure
                fig_single, axes_single = plt.subplots(
                    1,
                    n_cols,
                    figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1]),
                    squeeze=False,
                    sharex=share_axes,
                    sharey=share_axes,
                )
                axes_single = axes_single.ravel()

                for col_idx, key in enumerate(key_list):
                    ax_single = axes_single[col_idx]
                    ax_master = axes_master[row_idx, col_idx]
                    
                    info = plot_info[key]
                    mapped_key = mapping_dict[key] if (mapping_dict and key in mapping_dict) else key

                    is_match = False
                    match_idx = -1
                    if info["has_time_data"]:
                        match_idx = int(np.argmin(np.abs(info["cdg_keys_arr"] - hour_t)))
                        is_match = np.abs(info["cdg_keys_arr"][match_idx] - hour_t) <= 1e-3

                    # Reusable formatting dictionary
                    plot_kwargs = dict(
                        x1_lo=info["x1_lo"], x1_hi=info["x1_hi"], 
                        x2_lo=info["x2_lo"], x2_hi=info["x2_hi"],
                        x1_edges=info["x1_edges"], x2_edges=info["x2_edges"],
                        xlabel_fontsize=xlabel_fontsize, ylabel_fontsize=ylabel_fontsize,
                        xtick_labelsize=xtick_labelsize, ytick_labelsize=ytick_labelsize,
                        major_tick_length=major_tick_length, major_tick_width=major_tick_width,
                        minor_tick_length=minor_tick_length, minor_tick_width=minor_tick_width,
                        ix=ix, iy=iy, x0=info["x0"], y0=info["y0"], show_slice_lines=show_slice_lines,
                        show_xlabel=True, show_ylabel=True,
                    )

                    if not is_match:
                        title_str = f"{mapped_key}\nreq = {float(hour_t):g} h\nNo data"
                        
                        self._plot_black_domain_panel(
                            ax_single, title=title_str, title_fontsize=title_fontsize, **plot_kwargs
                        )
                        self._plot_black_domain_panel(
                            ax_master, title=title_str, title_fontsize=title_fontsize, **plot_kwargs
                        )
                    else:
                        best_key = info["cdg_keys"][match_idx]
                        cdg = np.asarray(info["data_obj"].cdgs.get(best_key, []), dtype=float)
                        title_str = f"{mapped_key}\nreq = {float(hour_t):g} h\nused = {float(best_key):g} h"

                        self._plot_green_scatter_panel(
                            ax_single, cdg=cdg, psize=psize, alpha=alpha, col_parental=col_parental,
                            title=title_str, title_fontsize=title_fontsize, **plot_kwargs
                        )
                        self._plot_green_scatter_panel(
                            ax_master, cdg=cdg, psize=psize, alpha=alpha, col_parental=col_parental,
                            title=title_str, title_fontsize=title_fontsize, **plot_kwargs
                        )

                # Save the single horizontal figure
                if not (scatter_png.exists() and not overwrite):
                    fig_single.tight_layout()
                    fig_single.savefig(scatter_png, dpi=self.save_dpi, bbox_inches="tight")
                
                self._maybe_show_and_close(fig_single)
                results[float(hour_t)] = scatter_png

            # Save the massive master grid figure
            fig_master.tight_layout()
            
            time_range_tag = f"from_{self._sanitize_for_filename(requested_times[0])}h_to_{self._sanitize_for_filename(requested_times[-1])}h"
            master_png = self.out_dir / f"green_cell_scatter_MASTER_all_keys_all_times_{time_range_tag}.png"
            
            if not (master_png.exists() and not overwrite):
                fig_master.savefig(master_png, dpi=self.save_dpi, bbox_inches="tight")
                
            self._maybe_show_and_close(fig_master)
            results["master"] = master_png

        finally:
            self.out_dir = original_out_dir

        return results
    def make_green_cell_scatter_snapshots_for_relative_times_with_keys_as_subplots(
        self,
        data_obj_dict: Dict[str, object],
        *,
        keys: Iterable[str],
        relative_time_array: Sequence[float],
        mapping_dict: Optional[Dict[str, str]] = None,
        ix: Optional[int] = None,
        iy: Optional[int] = None,
        overwrite_existing: Optional[bool] = None,
        psize: float = 10.0,
        alpha: float = 0.8,
        col_parental: str = "tab:green",
        figsize_per_panel: Tuple[float, float] = (4.0, 4.0),
        xlabel_fontsize: float = 10.0,
        ylabel_fontsize: float = 10.0,
        xtick_labelsize: float = 9.0,
        ytick_labelsize: float = 9.0,
        title_fontsize: float = 10.0,
        major_tick_length: float = 4.0,
        major_tick_width: float = 1.0,
        minor_tick_length: float = 2.5,
        minor_tick_width: float = 0.8,
        show_slice_lines: bool = False,
        highlight_dense_squares: bool = False,  
        density_threshold: float = 1000.0,      
        subdir: Optional[str] = "per_relative_time_across_keys",
        share_axes: bool = False,
        include_requested_and_used_time_in_filename: bool = True,
    ) -> Dict[Union[float, str], Path]:
        """
        Plots relative time samples with an optional purple grid square 
        highlight for areas exceeding the density threshold.
        """
        overwrite = self._resolve_overwrite(overwrite_existing)
        requested_relative_times = np.asarray(relative_time_array, dtype=float)
        key_list = list(keys)

        if requested_relative_times.size == 0:
            raise ValueError("`relative_time_array` must contain at least one time point.")
        if len(key_list) == 0:
            raise ValueError("`keys` must contain at least one key.")

        results: Dict[Union[float, str], Path] = {}

        original_out_dir = self.out_dir
        if subdir is not None:
            self.out_dir = self.out_dir / subdir
            self.out_dir.mkdir(parents=True, exist_ok=True)

        try:
            plot_info: Dict[str, dict] = {}

            # Pre-compute data characteristics and t0 for all keys
            for key in key_list:
                if key not in data_obj_dict:
                    raise KeyError(f"Key {key!r} not found in `data_obj_dict`.")

                data_obj = data_obj_dict[key]
                has_time_data = hasattr(data_obj, "cdgs") and len(data_obj.cdgs) > 0

                x1 = np.asarray(data_obj.x1, dtype=float)
                x2 = np.asarray(data_obj.x2, dtype=float)

                nx1 = len(x1)
                nx2 = len(x2)

                self._validate_optional_index(ix, nx1, "ix")
                self._validate_optional_index(iy, nx2, "iy")

                x0 = None if ix is None else float(x1[ix])
                y0 = None if iy is None else float(x2[iy])

                x1_lo, x1_hi, x2_lo, x2_hi, x1_edges, x2_edges, _ = self._compute_grid_geometry(x1, x2)

                if has_time_data:
                    cdg_keys = list(data_obj.cdgs.keys())
                    cdg_keys_arr = self._coerce_time_values(cdg_keys, label=f"data_obj_dict[{key!r}].cdgs keys")
                    t0 = np.min(cdg_keys_arr)
                else:
                    cdg_keys = []
                    cdg_keys_arr = np.array([])
                    t0 = None

                # Extract density field if highlights are enabled
                if highlight_dense_squares and hasattr(data_obj, "u_green") and hasattr(data_obj, "hoursRange"):
                    u_green_max = float(getattr(data_obj, "u_green_max", 1.0) or 1.0)
                    u_green = np.asarray(data_obj.u_green) * u_green_max * 1e6
                    t_array = self._coerce_time_values(data_obj.hoursRange, label=f"data_obj_dict[{key!r}].hoursRange")
                else:
                    u_green = np.array([])
                    t_array = np.array([])

                plot_info[key] = {
                    "data_obj": data_obj,
                    "x0": x0,
                    "y0": y0,
                    "x1_lo": x1_lo,
                    "x1_hi": x1_hi,
                    "x2_lo": x2_lo,
                    "x2_hi": x2_hi,
                    "x1_edges": x1_edges,
                    "x2_edges": x2_edges,
                    "has_time_data": has_time_data,
                    "cdg_keys": cdg_keys,
                    "cdg_keys_arr": cdg_keys_arr,
                    "t0": t0,
                    "u_green": u_green,
                    "t_array": t_array,
                }

            # Setup the massive master grid figure
            n_rows = len(requested_relative_times)
            n_cols = len(key_list)
            
            fig_master, axes_master = plt.subplots(
                n_rows,
                n_cols,
                figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
                squeeze=False,
                sharex=share_axes,
                sharey=share_axes,
            )

            # Loop over requested relative times (rows)
            for row_idx, rel_hour_t in enumerate(requested_relative_times):
                used_times_for_filename = []

                # Determine file naming parameters
                for key in key_list:
                    info = plot_info[key]
                    if info["has_time_data"]:
                        target_abs_t = info["t0"] + rel_hour_t
                        match_idx = int(np.argmin(np.abs(info["cdg_keys_arr"] - target_abs_t)))
                        if np.abs(info["cdg_keys_arr"][match_idx] - target_abs_t) <= 1e-3:
                            used_times_for_filename.append(float(info["cdg_keys"][match_idx]))
                        else:
                            used_times_for_filename.append("nodata")
                    else:
                        used_times_for_filename.append("nodata")

                if include_requested_and_used_time_in_filename:
                    used_tag = self._format_time_array_for_filename(used_times_for_filename)
                    figure_name = (
                        f"green_cell_scatter_all_keys"
                        f"__req_rel_{self._sanitize_for_filename(rel_hour_t)}h"
                        f"__used_abs_{used_tag}.png"
                    )
                else:
                    figure_name = (
                        f"green_cell_scatter_all_keys"
                        f"__req_rel_{self._sanitize_for_filename(rel_hour_t)}h.png"
                    )

                scatter_png = self.out_dir / figure_name
                
                # Setup the single horizontal row figure
                fig_single, axes_single = plt.subplots(
                    1,
                    n_cols,
                    figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1]),
                    squeeze=False,
                    sharex=share_axes,
                    sharey=share_axes,
                )
                axes_single = axes_single.ravel()

                for col_idx, key in enumerate(key_list):
                    ax_single = axes_single[col_idx]
                    ax_master = axes_master[row_idx, col_idx]
                    
                    info = plot_info[key]
                    mapped_key = mapping_dict[key] if (mapping_dict and key in mapping_dict) else key

                    is_match = False
                    match_idx = -1
                    density_slice = None

                    if info["has_time_data"]:
                        target_abs_t = info["t0"] + rel_hour_t
                        match_idx = int(np.argmin(np.abs(info["cdg_keys_arr"] - target_abs_t)))
                        is_match = np.abs(info["cdg_keys_arr"][match_idx] - target_abs_t) <= 1e-3
                        
                        # Grab the matching density slice if highlighting is enabled
                        if is_match and highlight_dense_squares and info["u_green"].size > 0:
                            density_match_idx = int(np.argmin(np.abs(info["t_array"] - target_abs_t)))
                            density_match_idx = max(0, min(density_match_idx, info["u_green"].shape[-1] - 1))
                            density_slice = info["u_green"][:, :, density_match_idx]

                    # Reusable formatting dictionary (WITHOUT density highlight args)
                    plot_kwargs = dict(
                        x1_lo=info["x1_lo"], x1_hi=info["x1_hi"], 
                        x2_lo=info["x2_lo"], x2_hi=info["x2_hi"],
                        x1_edges=info["x1_edges"], x2_edges=info["x2_edges"],
                        xlabel_fontsize=xlabel_fontsize, ylabel_fontsize=ylabel_fontsize,
                        xtick_labelsize=xtick_labelsize, ytick_labelsize=ytick_labelsize,
                        major_tick_length=major_tick_length, major_tick_width=major_tick_width,
                        minor_tick_length=minor_tick_length, minor_tick_width=minor_tick_width,
                        ix=ix, iy=iy, x0=info["x0"], y0=info["y0"], show_slice_lines=show_slice_lines,
                        show_xlabel=True, show_ylabel=True,
                    )

                    if not is_match:
                        title_str = f"{mapped_key}\nrel = {float(rel_hour_t):g} h\nNo data"
                        
                        self._plot_black_domain_panel(
                            ax_single, title=title_str, title_fontsize=title_fontsize, **plot_kwargs
                        )
                        self._plot_black_domain_panel(
                            ax_master, title=title_str, title_fontsize=title_fontsize, **plot_kwargs
                        )
                    else:
                        best_key = info["cdg_keys"][match_idx]
                        cdg = np.asarray(info["data_obj"].cdgs.get(best_key, []), dtype=float)
                        title_str = f"{mapped_key}\nrel = {float(rel_hour_t):g} h\nabs = {float(best_key):g} h"

                        # Pass the density args ONLY to the green scatter panel
                        self._plot_green_scatter_panel(
                            ax_single, cdg=cdg, psize=psize, alpha=alpha, col_parental=col_parental,
                            title=title_str, title_fontsize=title_fontsize, 
                            highlight_dense_squares=highlight_dense_squares,
                            density_slice=density_slice,
                            density_threshold=density_threshold,
                            **plot_kwargs
                        )
                        self._plot_green_scatter_panel(
                            ax_master, cdg=cdg, psize=psize, alpha=alpha, col_parental=col_parental,
                            title=title_str, title_fontsize=title_fontsize, 
                            highlight_dense_squares=highlight_dense_squares,
                            density_slice=density_slice,
                            density_threshold=density_threshold,
                            **plot_kwargs
                        )

                # Save the single horizontal figure if not overwritten
                if not (scatter_png.exists() and not overwrite):
                    fig_single.tight_layout()
                    fig_single.savefig(scatter_png, dpi=self.save_dpi, bbox_inches="tight")
                
                self._maybe_show_and_close(fig_single)
                results[float(rel_hour_t)] = scatter_png

            # Save the master grid figure
            fig_master.tight_layout()
            
            time_range_tag = f"from_{self._sanitize_for_filename(requested_relative_times[0])}h_to_{self._sanitize_for_filename(requested_relative_times[-1])}h"
            master_png = self.out_dir / f"green_cell_scatter_MASTER_rel_all_keys_{time_range_tag}.png"
            
            if not (master_png.exists() and not overwrite):
                fig_master.savefig(master_png, dpi=self.save_dpi, bbox_inches="tight")
                
            self._maybe_show_and_close(fig_master)
            results["master"] = master_png

        finally:
            self.out_dir = original_out_dir

        return results
    def make_density_heatmap_and_scatter_snapshots(
        self,
        data_obj,
        *,
        t_index: Optional[int] = None,
        ix: Optional[int] = None,
        iy: Optional[int] = None,
        species: Literal["red", "green", "total"] = "green",
        cmap: Optional[str] = None,
        overwrite_existing: Optional[bool] = None,
        psize: float = 10.0,
        col_parental: str = "tab:green",
        col_resistant: str = "tab:red",
        show_slice_lines_on_scatter: bool = True,
        figsizes: Tuple[Tuple[float, float], Tuple[float, float]] = ((7, 5), (7, 5)),
        xlabel_fontsize: float = 10.0,
        ylabel_fontsize: float = 10.0,
        xtick_labelsize: float = 9.0,
        ytick_labelsize: float = 9.0,
        title_fontsize: float = 10.0,
        legend_fontsize: float = 9.0,
        legend_loc_heatmap: str = "upper right",
        legend_loc_scatter: str = "upper right",
        legend_kwargs: Optional[dict] = None,
        major_tick_length: float = 4.0,
        major_tick_width: float = 1.0,
        minor_tick_length: float = 2.5,
        minor_tick_width: float = 0.8,
        show_colorbar: bool = True,
        cbar_label: str = "Cell density [cells mm$^{-2}$]",
        cbar_label_fontsize: Optional[float] = None,
        cbar_tick_fontsize: Optional[float] = None,
        cbar_fraction: float = 0.046,
        cbar_pad: float = 0.04,
        cbar_shrink: float = 1.0,
        cbar_aspect: float = 20,
        cbar_ticks: Optional[np.ndarray] = None,
        cbar_format: Optional[str] = None,
        cbar_extend: str = "neither",
    ) -> Tuple[Path, Path]:
        """
        Create:
          1. a density heatmap with optional slice lines
          2. a scatter plot of positions with optional slice lines
        """
        figsize_heatmap, figsize_scatter = figsizes

        heat_png = self.make_density_heatmap(
            data_obj=data_obj,
            t_index=t_index,
            ix=ix,
            iy=iy,
            species=species,
            cmap=cmap,
            figsize=figsize_heatmap,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            title_fontsize=title_fontsize,
            legend_fontsize=legend_fontsize,
            legend_loc=legend_loc_heatmap,
            legend_kwargs=legend_kwargs,
            major_tick_length=major_tick_length,
            major_tick_width=major_tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
            show_colorbar=show_colorbar,
            cbar_label=cbar_label,
            cbar_label_fontsize=cbar_label_fontsize,
            cbar_tick_fontsize=cbar_tick_fontsize,
            cbar_fraction=cbar_fraction,
            cbar_pad=cbar_pad,
            cbar_shrink=cbar_shrink,
            cbar_aspect=cbar_aspect,
            cbar_ticks=cbar_ticks,
            cbar_format=cbar_format,
            cbar_extend=cbar_extend,
            overwrite_existing=overwrite_existing,
        )

        scatter_png = self.make_positions_scatter(
            data_obj=data_obj,
            t_index=t_index,
            ix=ix,
            iy=iy,
            figsize=figsize_scatter,
            psize=psize,
            col_parental=col_parental,
            col_resistant=col_resistant,
            show_slice_lines=show_slice_lines_on_scatter,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            xtick_labelsize=xtick_labelsize,
            ytick_labelsize=ytick_labelsize,
            title_fontsize=title_fontsize,
            legend_fontsize=legend_fontsize,
            legend_loc=legend_loc_scatter,
            legend_kwargs=legend_kwargs,
            major_tick_length=major_tick_length,
            major_tick_width=major_tick_width,
            minor_tick_length=minor_tick_length,
            minor_tick_width=minor_tick_width,
            overwrite_existing=overwrite_existing,
        )

        return heat_png, scatter_png