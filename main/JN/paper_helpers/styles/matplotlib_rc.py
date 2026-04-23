import matplotlib as mpl


def apply_default_style() -> None:
    """Apply consistent plotting defaults across paper figures."""
    mpl.rcParams["axes.spines.top"] = True
    mpl.rcParams["axes.spines.right"] = True
    mpl.rcParams["axes.spines.bottom"] = True
    mpl.rcParams["axes.spines.left"] = True
    mpl.rcParams["axes.linewidth"] = 0.5
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.grid"] = "False"
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["grid.linewidth"] = 0.5
