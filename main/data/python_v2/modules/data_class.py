"""
dataClass.py
============

Description
-----------
Data structures and helper utilities for converting experimental point-cloud
cell data (e.g. parental vs resistant positions) into regular 2D density
grids and time-series stacks, and for wrapping those stacks into convenient
experiment classes.

The module supports:
- Reading CSV point clouds (x, y) from disk.
- Rasterising points into 2D density grids on a regular lattice.
- Building time-resolved density stacks over experimental conditions.
- Container classes that store these stacks in a consistent format and
  generate standardised (x1, x2, t) input grids for downstream models.

Contents
--------
Helper functions
----------------
- read_points_csv(path)
    Read a CSV of (x, y[, ...]) into an (N, 2) float array.

- fft_cross_covariance_radial(grid_a, grid_b, bin_size, max_r_bins)
    Radially averaged cross-covariance between two 2D grids via FFT.

- clip_to_domain(pts, xmin, xmax, ymin, ymax)
    Remove points outside a rectangular domain.

- rasterize(points, xmin, xmax, ymin, ymax, bin_size)
    Rasterise 2D points into a count grid with given bin size.

- generate_loop_combinations(x2_opts, x3_opts, x4_opts, x5_opts, x7_opts, plate_hours)
    Build a DataFrame of all combinations of experimental settings and hours.

- points_to_density(points, xmin, xmax, ymin, ymax, dx, dy)
    Convert (x, y) points into a 2D density grid on a regular lattice.

- _compute_global_density_max(...)
    Scan all timepoints for a given experimental combo and return the maximum
    total density encountered.

- run_density_stack(...)
    For sets of (dose, IC, reps, plate) options and hours, build 3D density
    stacks (in time) and return a dictionary of payloads.

- generate_inputs_2d(x1, x2, t)
    Generate a full tensor grid of (x1, x2, t) triplets.

Classes
-------
- Data_experiment
    Construct density stacks for one (dose, IC, reps, plate) from CSV files
    and expose normalised densities, coordinates, and (x1, x2, t) inputs.

- Data_experiment_manual
    Similar to Data_experiment, but accept precomputed density arrays and
    coordinate grids instead of reading from CSV.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Literal, Optional, Union

import imageio.v2 as imageio  # reserved for potential image I/O
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


# -----------------------------
# Helper functions
# -----------------------------


def read_points_csv(path: Path) -> np.ndarray:
    """
    Read a CSV of (x, y[, ...]) into an (N, 2) float array.

    Parameters
    ----------
    path : pathlib.Path
        Path to the CSV file containing at least two columns representing
        x- and y-coordinates.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) containing point coordinates. If the file is
        missing or has fewer than 2 columns, returns an empty (0, 2) array.
    """
    if not path.exists():
        return np.empty((0, 2), dtype=float)

    # Use pandas to be robust to extra columns; take first two.
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    pts = df.iloc[:, :2].to_numpy(dtype=float)
    return pts


def fft_cross_covariance_radial(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    bin_size: float,
    max_r_bins: int,
) -> np.ndarray:
    """
    Compute radially averaged cross-covariance between two 2D count grids.

    The procedure is:
      1) Subtract the mean from each grid to obtain zero-mean fields.
      2) Compute the circular cross-correlation via FFT:
         ifft2(FFT(grid_a) * conj(FFT(grid_b))).
      3) Normalize by number of pixels.
      4) Compute a radial profile by averaging over integer radius bins.
      5) Return profile values for radii 0 .. max_r_bins - 1.

    Parameters
    ----------
    grid_a : np.ndarray
        First 2D array.
    grid_b : np.ndarray
        Second 2D array; must be the same shape as ``grid_a``.
    bin_size : float
        Bin size in physical units (not directly used in the FFT domain, but
        conceptually corresponds to the spatial grid size).
    max_r_bins : int
        Number of radial bins to compute.

    Returns
    -------
    np.ndarray
        1D array of length ``max_r_bins`` containing the radially averaged
        cross-covariance profile.
    """
    fa = grid_a - grid_a.mean()
    fb = grid_b - grid_b.mean()

    # FFT-based circular correlation: ifft2(Fa * conj(Fb))
    Fa = np.fft.rfft2(fa)
    Fb = np.fft.rfft2(fb)
    corr = np.fft.irfft2(Fa * np.conj(Fb), s=fa.shape)
    corr = np.fft.fftshift(corr)  # center the zero-lag

    # Normalize by number of pixels so magnitudes aren't grid-size-dependent
    corr /= (fa.size if fa.size > 0 else 1.0)

    # Radial bins
    ny, nx = corr.shape
    y = np.arange(-ny // 2, ny // 2 + (ny % 2))
    x = np.arange(-nx // 2, nx // 2 + (nx % 2))
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X * X + Y * Y)

    # Bin edges at integer radii; we only need up to max_r_bins - 1
    rbins = np.arange(0, max_r_bins + 1, 1)  # edges: 0,1,2,...,max_r_bins
    # Digitize R into bins [k, k+1)
    which = np.digitize(R, rbins) - 1  # => 0..max_r_bins-1 or >= max

    profile = np.zeros(max_r_bins, dtype=float)
    for k in range(max_r_bins):
        mask = which == k
        if np.any(mask):
            profile[k] = corr[mask].mean()
        else:
            profile[k] = 0.0

    return profile


def clip_to_domain(
    pts: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> np.ndarray:
    """
    Clip a set of 2D points to a rectangular domain.

    Parameters
    ----------
    pts : np.ndarray
        Array of shape (N, 2) with x- and y-coordinates.
    xmin, xmax : float
        Minimum and maximum x-values defining the domain.
    ymin, ymax : float
        Minimum and maximum y-values defining the domain.

    Returns
    -------
    np.ndarray
        Subset of ``pts`` lying in [xmin, xmax] × [ymin, ymax].
    """
    if pts.size == 0:
        return pts
    mask = (
        (pts[:, 0] >= xmin)
        & (pts[:, 0] <= xmax)
        & (pts[:, 1] >= ymin)
        & (pts[:, 1] <= ymax)
    )
    return pts[mask]


def rasterize(
    points: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    bin_size: float,
) -> np.ndarray:
    """
    Convert 2D points into a 2D count grid over a rectangular domain.

    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of x- and y-coordinates.
    xmin, xmax : float
        Domain limits in x.
    ymin, ymax : float
        Domain limits in y.
    bin_size : float
        Edge length of each square bin (in the same units as x, y).

    Returns
    -------
    np.ndarray
        2D array of shape (ny, nx) giving point counts in each bin.
    """
    width = max(0.0, xmax - xmin)
    height = max(0.0, ymax - ymin)

    nx = int(np.ceil(width / bin_size)) if bin_size > 0 else 1
    ny = int(np.ceil(height / bin_size)) if bin_size > 0 else 1

    nx = max(1, nx)
    ny = max(1, ny)

    grid = np.zeros((ny, nx), dtype=float)
    if points.size == 0 or width == 0 or height == 0:
        return grid

    # Shift coordinates so (xmin, ymin) -> (0,0), then bin
    shifted_x = points[:, 0] - xmin
    shifted_y = points[:, 1] - ymin

    ix = np.floor(shifted_x / bin_size).astype(int)
    iy = np.floor(shifted_y / bin_size).astype(int)

    # Guard: points on max boundary should fall just inside
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    # Accumulate
    for xbin, ybin in zip(ix, iy):
        grid[ybin, xbin] += 1.0

    return grid


def generate_loop_combinations(
    x2_opts,
    x3_opts,
    x4_opts,
    x5_opts,
    x7_opts,
    plate_hours: dict[int, range] = {1: range(0, 97, 4), 2: range(0, 93, 4)},
) -> pd.DataFrame:
    """
    Generate a DataFrame listing all combinations of experimental settings.

    The layout mirrors nested MATLAB-style loops over dose, IC, replicate
    indices, and plate number, with an inner loop over hours based on
    a plate-specific hours schedule.

    Parameters
    ----------
    x2_opts, x3_opts, x4_opts, x5_opts, x7_opts :
        Iterables of candidate values for each experimental parameter:
        - dose (x2)
        - IC (x3)
        - replicate 1 (x4)
        - replicate 2 (x5)
        - plate (x7)
    plate_hours : dict[int, range], optional
        Mapping plate number -> range of hours.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per combination and columns:
        ["dose", "ic", "rep1", "rep2", "hours", "plate"].
    """
    rows = []
    for x2, x3, x4, x5, x7 in itertools.product(
        x2_opts, x3_opts, x4_opts, x5_opts, x7_opts
    ):
        hours = plate_hours[x7]
        for x6 in hours:
            rows.append(
                {
                    "dose": x2,
                    "ic": x3,
                    "rep1": x4,
                    "rep2": x5,
                    "hours": x6,
                    "plate": x7,
                }
            )
    return pd.DataFrame(rows)


# -----------------------------
# Discrete points -> density grid on regular lattice
# -----------------------------


def points_to_density(
    points: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert discrete (x, y) points into a 2D density grid on a regular lattice.

    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of x- and y-coordinates.
    xmin, xmax : float
        Domain limits in x.
    ymin, ymax : float
        Domain limits in y.
    dx, dy : float
        Bin sizes in x and y.

    Returns
    -------
    density : np.ndarray
        (ny, nx) density grid with units "cells per unit area"
        (counts / (dx * dy)).
    x_centers : np.ndarray
        1D array of length nx containing x-bin centers.
    y_centers : np.ndarray
        1D array of length ny containing y-bin centers.
    """
    width = max(0.0, xmax - xmin)
    height = max(0.0, ymax - ymin)

    nx = max(1, int(np.ceil(width / dx))) if dx > 0 else 1
    ny = max(1, int(np.ceil(height / dy))) if dy > 0 else 1

    x_edges = np.linspace(xmin, xmax, nx + 1)
    y_edges = np.linspace(ymin, ymax, ny + 1)

    if points.size == 0 or width == 0 or height == 0:
        H = np.zeros((ny, nx), dtype=float)
    else:
        pts = clip_to_domain(points, xmin, xmax, ymin, ymax)
        if pts.size == 0:
            H = np.zeros((ny, nx), dtype=float)
        else:
            # Note: np.histogram2d expects x first then y; bins list is [x_edges, y_edges]
            Hxy, _, _ = np.histogram2d(
                pts[:, 0], pts[:, 1], bins=[x_edges, y_edges]
            )
            H = Hxy.T  # rows=y, cols=x

    cell_area = dx * dy if dx > 0 and dy > 0 else 1.0
    density = H / cell_area

    x_centers = (x_edges[:-1] + x_edges[1:]) * 0.5
    y_centers = (y_edges[:-1] + y_edges[1:]) * 0.5

    return density, x_centers, y_centers


# -----------------------------
# Global max density pre-pass (optional)
# -----------------------------


def _compute_global_density_max(
    x2,
    x3,
    x4,
    x5,
    x7,
    hoursRange,
    data_root: Path,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    dx: float,
    dy: float,
) -> float:
    """
    Compute global max total density across all timepoints for one setting.

    For a given combination (x2, x3, x4, x5) and plate x7, this function:
      - reads the parental (green) and resistant (red) CSV files for each hour
        in ``hoursRange`` from ``data_root``,
      - rasterises them to density grids, and
      - returns the maximum of (parental + resistant) over all grids.

    Parameters
    ----------
    x2, x3, x4, x5 :
        Experimental settings (dose, IC, replicates).
    x7 :
        Plate number.
    hoursRange :
        Iterable of hour values.
    data_root : pathlib.Path
        Root directory containing the CSV files.
    xmin, xmax, ymin, ymax : float
        Spatial domain used for clipping and rasterisation.
    dx, dy : float
        Lattice spacings for density computation.

    Returns
    -------
    float
        Global maximum of the total density over all timepoints.
        If no density is found, returns 1.0 (to avoid a degenerate 0 scale).
    """
    global_max = 0.0
    for x6 in hoursRange:
        suffix = (
            f"_dose_{x2}_ic_{x3}_rep1_{x4}_rep2_{x5}_hours_{x6}_plate_{x7}.csv"
        )
        
        print(suffix)
        path_green = data_root / f"color_green{suffix}"
        path_red = data_root / f"color_red{suffix}"

        cdg = read_points_csv(path_green)
        cdr = read_points_csv(path_red)

        cdg = clip_to_domain(cdg, xmin, xmax, ymin, ymax)
        cdr = clip_to_domain(cdr, xmin, xmax, ymin, ymax)

        dens_g, _, _ = points_to_density(
            cdg, xmin, xmax, ymin, ymax, dx, dy
        )
        dens_r, _, _ = points_to_density(
            cdr, xmin, xmax, ymin, ymax, dx, dy
        )
        dens_total = dens_g + dens_r

        if dens_total.size:
            m = float(np.max(dens_total))
            if m > global_max:
                global_max = m

    # avoid degenerate 0 scale
    return global_max if global_max > 0 else 1.0


def run_density_stack(
    x2_opts,
    x3_opts,
    x4_opts,
    x5_opts,
    *,
    x7: int = 1,  # plate number
    hoursRange=range(0, 97, 4),
    # Lattice spacings
    dx: float = 10.0,
    dy: float = 10.0,
    # Domain
    xmin: float = 0.0,
    xmax: float = 1000.0,
    ymin: float = 0.0,
    ymax: float = 1000.0,
    # I/O
    data_root: Union[str, Path] = "split_csvs",
    # Options
    return_components: bool = True,
    compute_vmax: bool = False,
):
    """
    Build time-resolved density stacks for multiple experimental combinations.

    For each (x2, x3, x4, x5) combination and each time in ``hoursRange``:
      - Read green/red CSV point clouds.
      - Clip them to the domain [xmin, xmax] × [ymin, ymax].
      - Convert to densities on the (dx, dy) grid.
      - Compute total density as parental + resistant.

    The result is a dictionary mapping each (x2, x3, x4, x5, x7) tuple to
    a payload containing density stacks and metadata.

    Parameters
    ----------
    x2_opts, x3_opts, x4_opts, x5_opts :
        Iterables of candidate values for each experimental parameter.
    x7 : int, optional
        Plate number (default is 1).
    hoursRange :
        Iterable of hours for which CSV files exist (default range(0, 97, 4)).
    dx, dy : float, optional
        Lattice spacings in x and y (default 10.0).
    xmin, xmax, ymin, ymax : float, optional
        Domain limits for clipping and rasterisation.
    data_root : str or pathlib.Path, optional
        Root directory containing the CSV point-cloud files.
    return_components : bool, optional
        If True, include separate parental and resistant density stacks.
    compute_vmax : bool, optional
        If True, include a 'vmax' entry with the global max total density.

    Returns
    -------
    dict
        Dictionary mapping (x2, x3, x4, x5, x7) -> payload, where payload
        is a dict containing:
            - "times"            : list[int] hours
            - "x_centers"        : np.ndarray (Nx,)
            - "y_centers"        : np.ndarray (Ny,)
            - "density_total"    : np.ndarray (Nx, Ny, T)
            - "density_parental" : np.ndarray (Nx, Ny, T) (optional)
            - "density_resistant": np.ndarray (Nx, Ny, T) (optional)
            - "cdgs"             : dict[hour -> parental points]
            - "cdrs"             : dict[hour -> resistant points]
            - "vmax"             : float (optional, if compute_vmax=True)
    """
    data_root = Path(data_root)

    tasks = [
        (x2, x3, x4, x5)
        for x2 in x2_opts
        for x3 in x3_opts
        for x4 in x4_opts
        for x5 in x5_opts
    ]
    total_outer = len(tasks)

    out: dict = {}
    start_time = time.time()

    for outer_idx, (x2, x3, x4, x5) in enumerate(tasks, start=1):
        total_inner = len(hoursRange)

        dens_tot_frames = []
        dens_g_frames = [] if return_components else None
        dens_r_frames = [] if return_components else None
        x_centers = None
        y_centers = None
        cdgs: dict = {}
        cdrs: dict = {}

        for inner_idx, x6 in enumerate(hoursRange, start=1):
            suffix = (
                f"_dose_{x2}_ic_{x3}_rep1_{x4}_rep2_{x5}_hours_{x6}_plate_{x7}.csv"
            )

            path_green = data_root / f"color_green{suffix}"
            path_red = data_root / f"color_red{suffix}"

            cdg = read_points_csv(path_green)
            cdr = read_points_csv(path_red)

            cdg = clip_to_domain(cdg, xmin, xmax, ymin, ymax)
            cdgs[x6] = cdg
            cdr = clip_to_domain(cdr, xmin, xmax, ymin, ymax)
            cdrs[x6] = cdr

            dens_g, xc, yc = points_to_density(
                cdg, xmin, xmax, ymin, ymax, dx, dy
            )
            dens_r, _, _ = points_to_density(
                cdr, xmin, xmax, ymin, ymax, dx, dy
            )
            dens_total = dens_g + dens_r

            if x_centers is None:
                x_centers = xc
                y_centers = yc

            # Transpose to (Nx, Ny, 1) so later concatenate gives (Nx, Ny, Nt)
            dens_tot_frames.append(dens_total.T[..., np.newaxis])
            if return_components:
                dens_g_frames.append(dens_g.T[..., np.newaxis])
                dens_r_frames.append(dens_r.T[..., np.newaxis])

            done = ((outer_idx - 1) * total_inner) + inner_idx
            total = total_outer * total_inner
            pct = 100.0 * done / max(1, total)

            print(
                f"[{pct:5.1f}%] dose={x2} ic={x3} rep1={x4} rep2={x5} "
                f"plate={x7} | t={x6:3d}h",
                end="\r",
            )

        # Concatenate along time axis -> (Nx, Ny, Nt)
        density_total = np.concatenate(dens_tot_frames, axis=2)
        payload: dict = {
            "times": list(hoursRange),
            "x_centers": x_centers,
            "y_centers": y_centers,
            "density_total": density_total,
            "cdgs": cdgs,
            "cdrs": cdrs,
        }

        if return_components and dens_g_frames is not None and dens_r_frames is not None:
            payload["density_parental"] = np.concatenate(
                dens_g_frames, axis=2
            )
            payload["density_resistant"] = np.concatenate(
                dens_r_frames, axis=2
            )

        if compute_vmax:
            payload["vmax"] = float(np.nanmax(density_total))

        out[(x2, x3, x4, x5, x7)] = payload

    print(" " * 120, end="\r")
    print(
        f"Built density stacks for {total_outer} combo(s) "
        f"in {time.time() - start_time:.1f}s."
    )
    return out


# -----------------------------
# Experiment data containers
# -----------------------------


class Data_experiment:
    """
    Container for one experiment defined by (dose, IC, replicates, plate).

    The class wraps the density stacks produced by :func:`run_density_stack`
    for a single combination (x2_opt, x3_opt, x4_opt, x5_opt, x7). It stores:

    - Normalised densities (red, green, total) on a (x1, x2, t) grid.
    - Physical coordinates x1_orig, x2_orig and their normalised variants
      x1, x2 (scaled to ~[0, 1] for downstream models).
    - Time array t (in days) and hoursRange (in hours).
    - Dictionaries of original point clouds (cdgs/cdrs).
    - A precomputed (x1, x2, t) input grid for modelling.

    Parameters
    ----------
    x2_opt, x3_opt, x4_opt, x5_opt :
        Experimental settings for dose, IC, and replicates.
    x7 : int, optional
        Plate number (default is 1).
    hoursRange :
        Iterable of hours at which data are available.
    dx, dy : float, optional
        Lattice spacings for density construction.
    xmin, xmax, ymin, ymax : float, optional
        Spatial domain for clipping and density grids.
    gamma : float, optional
        Optional parameter stored for downstream models.
    data_root : str or pathlib.Path, optional
        Root directory containing CSV point clouds.
    return_components : bool, optional
        If True, keep parental and resistant stacks separate.
    compute_vmax : bool, optional
        If True, compute and store a global vmax for total density.
    """

    def __init__(
        self,
        x2_opt,
        x3_opt,
        x4_opt,
        x5_opt,
        *,
        x7: int = 1,  # plate number
        hoursRange=range(0, 97, 4),
        # Lattice spacings
        dx: float = 10.0,
        dy: float = 10.0,
        # Domain
        xmin: float = 0.0,
        xmax: float = 1000.0,
        ymin: float = 0.0,
        ymax: float = 1000.0,
        gamma: float = 0.0,
        # I/O
        data_root: Union[str, Path] = "split_csvs",
        # Options
        return_components: bool = True,
        compute_vmax: bool = False,
    ):
        outputs = run_density_stack(
            x2_opts=[x2_opt],
            x3_opts=[x3_opt],
            x4_opts=[x4_opt],
            x5_opts=[x5_opt],
            x7=x7,
            hoursRange=hoursRange,
            dx=dx,
            dy=dy,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            data_root=data_root,
            return_components=return_components,
            compute_vmax=compute_vmax,
        )

        key = (x2_opt, x3_opt, x4_opt, x5_opt, x7)

        # Store configuration
        self.x2_opt = x2_opt
        self.x3_opt = x3_opt
        self.x4_opt = x4_opt
        self.x5_opt = x5_opt
        self.x7 = x7
        self.dx = dx
        self.dy = dy
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.hoursRange = np.array(hoursRange)

        # Stacks (note: shapes are (Nx, Ny, T) after our transposes above)
        self.u_red = (
            outputs[key]["density_resistant"] if return_components else None
        )
        self.u_green = (
            outputs[key]["density_parental"] if return_components else None
        )
        self.u_total = outputs[key]["density_total"]  # shape (Nx, Ny, T)

        self.cdgs = outputs[key]["cdgs"]
        self.cdrs = outputs[key]["cdrs"]
        self.x1_orig = outputs[key]["x_centers"]  # physical coords (x)
        self.x2_orig = outputs[key]["y_centers"]  # physical coords (y)
        # Store time in days for modelling convenience
        self.t = np.array(hoursRange) / 24.0

        # Normalize stacks by their own maxima (if desired)
        if self.u_red is not None:
            self.u_red_max = (
                float(np.max(self.u_red)) if self.u_red.size else 0.0
            )
            if self.u_red_max > 0:
                self.u_red = self.u_red / self.u_red_max
        else:
            self.u_red_max = None

        if self.u_green is not None:
            self.u_green_max = (
                float(np.max(self.u_green)) if self.u_green.size else 0.0
            )
            if self.u_green_max > 0:
                self.u_green = self.u_green / self.u_green_max
        else:
            self.u_green_max = None

        if self.u_total is not None:
            self.u_total_max = (
                float(np.max(self.u_total)) if self.u_total.size else 0.0
            )
            if self.u_total_max > 0:
                self.u_total = self.u_total / self.u_total_max
        else:
            self.u_total_max = None

        # Normalise physical coordinates for downstream code (approx [0, 1])
        self.x1 = self.x1_orig / 1000.0
        self.x2 = self.x2_orig / 1000.0

        self.x1min = float(np.min(self.x1))
        self.x1max = float(np.max(self.x1))
        self.L1 = self.x1max - self.x1min
        self.nx1 = len(self.x1)

        self.x2min = float(np.min(self.x2))
        self.x2max = float(np.max(self.x2))
        self.L2 = self.x2max - self.x2min
        self.nx2 = len(self.x2)

        # Full tensor grid of (x1, x2, t) triplets
        self.inputs = generate_inputs_2d(self.x1, self.x2, self.t)

        self.gamma = gamma
        self.K = 1


def generate_inputs_2d(
    x1: np.ndarray,
    x2: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Generate a full tensor grid of (x1, x2, t) triplets.

    Parameters
    ----------
    x1 : np.ndarray
        1D array of x1 coordinates.
    x2 : np.ndarray
        1D array of x2 coordinates.
    t : np.ndarray
        1D array of time points.

    Returns
    -------
    np.ndarray
        Array of shape (len(x1) * len(x2) * len(t), 3) where each row is
        [x1_i, x2_j, t_k] from the full tensor product of grids.
    """
    X1, X2, T = np.meshgrid(x1, x2, t, indexing="ij")
    return np.column_stack([X1.ravel(), X2.ravel(), T.ravel()])


class Data_experiment_manual:
    """
    Container for manually supplied experiment densities and coordinates.

    This class mirrors the structure of :class:`Data_experiment` but assumes
    that the user already has the density stacks (u_red, u_green, u_total)
    and coordinate arrays (x1, x2, t). It normalises the stacks and
    precomputes a full (x1, x2, t) input grid but does not read any CSVs.

    Parameters
    ----------
    x2_opt, x3_opt, x4_opt, x5_opt :
        Experimental settings for dose, IC, and replicates.
    x7 : int, optional
        Plate number (default is 1).
    hoursRange :
        Iterable of hours corresponding to the time indices, stored as metadata.
    dx, dy : float, optional
        Lattice spacings used to generate the supplied densities.
    xmin, xmax, ymin, ymax : float, optional
        Underlying physical domain.
    gamma : float, optional
        Optional parameter stored for downstream models.
    u_red, u_green, u_total : np.ndarray, optional
        Precomputed density stacks (Nx, Ny, T).
    x1, x2 : np.ndarray, optional
        Coordinate arrays corresponding to the first two dimensions.
    t : np.ndarray, optional
        Time array of length T.
    return_components : bool, optional
        Kept for API compatibility; not used directly.
    compute_vmax : bool, optional
        Kept for API compatibility; not used directly.
    """

    def __init__(
        self,
        x2_opt,
        x3_opt,
        x4_opt,
        x5_opt,
        *,
        x7: int = 1,  # plate number
        hoursRange=range(0, 97, 4),
        # Lattice spacings
        dx: float = 10.0,
        dy: float = 10.0,
        # Domain
        xmin: float = 0.0,
        xmax: float = 1000.0,
        ymin: float = 0.0,
        ymax: float = 1000.0,
        gamma: float = 0.0,
        u_red: Optional[np.ndarray] = None,
        u_green: Optional[np.ndarray] = None,
        u_total: Optional[np.ndarray] = None,
        x1: Optional[np.ndarray] = None,
        x2: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
        # Options (kept for interface symmetry)
        return_components: bool = True,
        compute_vmax: bool = False,
    ):
        key = (x2_opt, x3_opt, x4_opt, x5_opt, x7)

        self.x2_opt = x2_opt
        self.x3_opt = x3_opt
        self.x4_opt = x4_opt
        self.x5_opt = x5_opt
        self.x7 = x7
        self.dx = dx
        self.dy = dy
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.hoursRange = np.array(hoursRange)
        self.u_red = u_red
        self.u_green = u_green
        self.u_total = u_total
        self.x1 = x1
        self.x2 = x2
        self.t = t

        # Normalize stacks by their own maxima (if desired)
        if self.u_red is not None:
            self.u_red_max = (
                float(np.max(self.u_red)) if self.u_red.size else 0.0
            )
            if self.u_red_max > 0:
                self.u_red = self.u_red / self.u_red_max
        else:
            self.u_red_max = None

        if self.u_green is not None:
            self.u_green_max = (
                float(np.max(self.u_green)) if self.u_green.size else 0.0
            )
            if self.u_green_max > 0:
                self.u_green = self.u_green / self.u_green_max
        else:
            self.u_green_max = None

        if self.u_total is not None:
            self.u_total_max = (
                float(np.max(self.u_total)) if self.u_total.size else 0.0
            )
            if self.u_total_max > 0:
                self.u_total = self.u_total / self.u_total_max
        else:
            self.u_total_max = None

        # Coordinate metadata
        self.x1min = float(np.min(self.x1))
        self.x1max = float(np.max(self.x1))
        self.L1 = self.x1max - self.x1min
        self.nx1 = len(self.x1)

        self.x2min = float(np.min(self.x2))
        self.x2max = float(np.max(self.x2))
        self.L2 = self.x2max - self.x2min
        self.nx2 = len(self.x2)

        # Full tensor grid of (x1, x2, t) triplets
        self.inputs = generate_inputs_2d(self.x1, self.x2, self.t)

        self.gamma = gamma
        self.K = 1
