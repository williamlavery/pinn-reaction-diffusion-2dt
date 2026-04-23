import os
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Union, Literal, Optional, Tuple
from pathlib import Path
import os, glob


# ----------------------------
# ====== SAVE UTILITIES ======
# ----------------------------

def to_torch(x, device="cpu", dtype=torch.float32):
    """Convert numpy array to torch tensor on specified device."""
    return torch.as_tensor(x, dtype=dtype, device=device)

def _get_species_u(dataobj, speciesLabel):
    """Helper to select species data array from dataobj."""
    if speciesLabel == "green":
        return dataobj.u_green
    elif speciesLabel == "red":
        return dataobj.u_red
    else:
        raise ValueError(f"Unknown speciesLabel '{speciesLabel}', expected 'green' or 'red'.")
    

def build_save_path(save_dic=None, save_name=None, base_dir="plots"):
    """
    Build a save path as base_dir/key1_val1/key2_val2/.../save_name.
    The order of keys is preserved as given in save_dic.
    """
    if save_dic is None or save_name is None:
        return None  # caller may fall back to previous 'name' parameter or just not save
    parts = [base_dir] + [f"{k}_{v}" for k, v in save_dic.items()]
    dir_path = os.path.join(*parts) if parts else base_dir
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, save_name)

def should_skip_all(paths, overwrite=False):
    """
    If all paths exist and overwrite=False, skip/return early.
    If any path is missing, proceed.
    """
    if not paths:
        return False
    if not overwrite and all(os.path.exists(p) for p in paths):
        for p in paths:
            print(f"File exists (overwrite=False): {p}")
        print("All target files exist and overwrite=False; returning without plotting.")
        return True
    return False
def save_figure(fig, path, overwrite=False, dpi=100, mkdir_dir=False):
    """Save a Matplotlib figure respecting overwrite."""
    if path is None:
        return
    if os.path.exists(path) and not overwrite:
        print(f"Skipping existing file (overwrite=False): {path}")
        return
    if mkdir_dir:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="None")

def save_animation(anim, path, overwrite=False, fps=6, dpi=100):
    """Save a Matplotlib animation as GIF respecting overwrite."""
    if path is None:
        return
    if os.path.exists(path) and not overwrite:
        print(f"Skipping existing file (overwrite=False): {path}")
        return
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)


# ---------- unified save path helpers ----------
def build_save_dir_from_dic(save_dic: dict, base_dir: str = "plots") -> str:
    """
    Create a directory path like:
      base_dir/key1_val1/key2_val2/...
    using the order of keys as given by save_dic (Py3.7+ preserves insertion order).
    """
    parts = [f"{k}_{v}" for k, v in save_dic.items()]
    dir_path = os.path.join(base_dir, *parts) if parts else base_dir
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def build_save_base(save_dic: dict, save_name: str, base_dir: str = "plots") -> str:
    """
    Returns the *base* (no extension) full path:
      base_dir/key1_val1/key2_val2/.../save_path
    You can append suffixes + extensions to this base.
    """
    
    dir_path = build_save_dir_from_dic(save_dic, base_dir=base_dir)
    return os.path.join(dir_path, save_name)
# ------------------------------------------------



# ---------------------------------
# ====== ANALYSIS UTILITIES =======
# ---------------------------------

def hist_properties(dataobj, speciesLabel="green", num_bins=100, low=5, high=95):
    """
    Compute histogram-based properties for the given data and species.
    """
    u = _get_species_u(dataobj, speciesLabel)
    u_flat = u.flatten()

    # Histogram with torch for consistency
    hist, bin_edges = torch.histogram(torch.as_tensor(u_flat, dtype=torch.float32), bins=num_bins)
    hist_np = hist.numpy()

    # Compute percentile thresholds on histogram counts
    low_count_thresh = np.percentile(hist_np, low)
    high_count_thresh = np.percentile(hist_np, high)

    # Bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {
        "hist": hist,
        "bin_edges": bin_edges,
        "bin_indices": torch.bucketize(torch.as_tensor(u_flat), bin_edges[1:-1]),
        "bin_centers": bin_centers,
        "low_count_thresh": low_count_thresh,
        "high_count_thresh": high_count_thresh,
        "low_count": np.percentile(u, low),
        "high_count": np.percentile(u, high),
        "min_count": np.min(u_flat),
        "max_count": np.max(u_flat)}

def hist_properties_wrapper(mw, num_bins=1000, low=5, high=95):
    """
    Compute histogram-based properties for the given data and species.
    """

    u = np.array(mw.y_train, copy=True)

    # Histogram with torch for consistency
    hist, bin_edges = torch.histogram(torch.as_tensor(u.flatten(), dtype=torch.float32), bins=num_bins)
    hist_np = hist.numpy()

    # Compute percentile thresholds on histogram counts
    low_count_thresh = np.percentile(hist_np, low)
    high_count_thresh = np.percentile(hist_np, high)
    low_count = np.percentile(u.flatten(), low)
    high_count = np.percentile(u.flatten(), high)

    # Bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {
        "hist": hist,
        "bin_edges": bin_edges,
        "bin_indices": torch.bucketize(torch.as_tensor(u.flatten()), bin_edges[1:-1]),
        "bin_centers": bin_centers,
        "low_count_thresh": low_count_thresh,
        "high_count_thresh": high_count_thresh,
        "low_count": low_count,
        "high_count": high_count,
    }


def predict_u_pred(
    model_or_wrappers,
    dataobj,
    *,
    K: float = 1.0,
    device: str = "cpu",
    return_numpy: bool = True,
):
    """
    Run the model to produce u_pred with shape (Nx1, Nx2, Nt), scaled by K.

    Args:
        model_or_wrappers: Either
            - a dict of wrappers (takes the first .model),
            - a single wrapper with attribute .model, or
            - a torch.nn.Module directly.
        dataobj: object with attributes x1, x2, t, and (optionally) inputs
                 where inputs is an array of shape (Nx1*Nx2*Nt, 3) in (x1,x2,t) order.
                 If `inputs` is missing, they’re generated from (x1, x2, t).
        K (float): multiply predictions by this scale factor.
        device (str): torch device for inference.
        chunk_size (int|None): if set, evaluate in chunks to save memory.
        return_numpy (bool): if True return np.ndarray on CPU; else return torch.Tensor on CPU.

    Returns:
        u_pred: array/tensor of shape (Nx1, Nx2, Nt)
    """

    # ---- resolve model ----
    model = None
    if isinstance(model_or_wrappers, dict):
        # take first wrapper's .model
        wrapper = next(iter(model_or_wrappers.values()))
        model = getattr(wrapper, "model", wrapper)
    else:
        model = getattr(model_or_wrappers, "model", model_or_wrappers)

    if not hasattr(model, "eval"):
        raise TypeError("Could not resolve a torch.nn.Module from 'model_or_wrappers'.")

    model = model.to(device)
    model.eval()

    chunk_size = None  # disable chunking for now

    # ---- resolve grid sizes ----
    x1 = np.asarray(dataobj.x1)
    x2 = np.asarray(dataobj.x2)
    t  = np.asarray(dataobj.t)
    Nx1, Nx2, Nt = len(x1), len(x2), len(t)

    # ---- resolve inputs (N,3) in (x1, x2, t) order ----
    inputs = getattr(dataobj, "inputs", None)
    if inputs is None:
        # generate (x1, x2, t) triplets with ij indexing
        X1, X2, T = np.meshgrid(x1, x2, t, indexing="ij")
        inputs = np.column_stack([X1.ravel(), X2.ravel(), T.ravel()])
    else:
        inputs = np.asarray(inputs)

    # ---- torch inference ----
    with torch.no_grad():
        x = torch.as_tensor(inputs, dtype=next(model.parameters()).dtype, device=device)

        if chunk_size is None or chunk_size <= 0:
            y = model(x)
        else:
            # evaluate in chunks to reduce peak memory
            outs = []
            for i in range(0, x.shape[0], chunk_size):
                outs.append(model(x[i:i+chunk_size]))
            y = torch.cat(outs, dim=0)

        y = y * K  # scale
        y = y.reshape(Nx1, Nx2, Nt)

        if return_numpy:
            return y.detach().cpu().numpy()
        else:
            return y.detach().cpu()


