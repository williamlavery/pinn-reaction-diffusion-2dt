"""Standalone data modules for the publication pipeline."""

from .data_class import Data_experiment


# Optional plot utility for compatibility with v1 call sites.
def plot_experiment_panels_single_figure(exp):
    """Compatibility shim; plotting is intentionally minimal in this package."""
    return None
