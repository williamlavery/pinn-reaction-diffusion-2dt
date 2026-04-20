from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from .data_build import build_data_object
from .pathing import dict_to_ordered_path


ARTIFACT_FILENAME = "data_obj.npy"


def _artifact_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        _ = np.load(path, allow_pickle=True).item()
    except Exception:
        return False
    return True


def run_single_simulation(data_obj_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    store = data_obj_params["RDEq_params_store"]
    additional = data_obj_params["additional_params"]

    out_dir = Path(additional["initial_path"]) / dict_to_ordered_path(store)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / ARTIFACT_FILENAME

    overwrite = bool(additional["overwrite_bool"])
    should_skip = _artifact_is_valid(out_file) and not overwrite

    if should_skip:
        status = "skipped_existing"
    else:
        data_obj = build_data_object(data_obj_params=data_obj_params)
        np.save(out_file, arr=data_obj, allow_pickle=True)
        status = "generated"

    if bool(additional["plot_bool"]):
        # Optional plotting stays lazy so dependency is only needed when used.
        from data.python_v2.modules import plot_experiment_panels_single_figure

        data_obj = np.load(out_file, allow_pickle=True).item()
        plot_experiment_panels_single_figure(exp=data_obj)

    return {
        "status": status,
        "artifact": str(out_file),
        "artifact_valid": _artifact_is_valid(out_file),
        "path_key": str(dict_to_ordered_path(store)),
    }
