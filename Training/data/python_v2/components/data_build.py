from __future__ import annotations

from typing import Any, Dict


def build_data_object(data_obj_params: Dict[str, Dict[str, Any]]) -> Any:
    # Local import keeps package import side effects small and explicit.
    from data.python_v2.modules import Data_experiment

    store = data_obj_params["RDEq_params_store"]
    additional = data_obj_params["additional_params"]

    return Data_experiment(
        x2_opt=store["x2opt"],
        x3_opt=store["x3opt"],
        x4_opt=store["x4opt"],
        x5_opt=store["x5opt"],
        x7=store["x7"],
        hoursRange=store["hoursRange"],
        dx=store["dx"],
        dy=store["dy"],
        xmax=store["xmax"],
        ymax=store["ymax"],
        xmin=store["xmin"],
        ymin=store["ymin"],
        gamma=store["gamma"],
        data_root=additional["data_root"],
    )
