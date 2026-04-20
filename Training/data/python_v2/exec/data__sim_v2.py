from __future__ import annotations

import argparse
import copy
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict

# Ensure the Training directory is importable regardless of cwd.
TRAINING_ROOT = Path(__file__).resolve().parents[3]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from data.python_v2.components.run_metadata import (
    append_jsonl,
    environment_snapshot,
    get_git_commit,
    utc_now_iso,
    write_json,
)
from data.python_v2.components.simulate import run_single_simulation
from data.python_v2.config.experiment_config import build_default_config


def print_run_summary(run_number: int, total: int, payload: Dict[str, Any]) -> None:
    store = payload["RDEq_params_store"]
    model = payload["RDEq_params"]

    print("=" * 88)
    print(f"RUN {run_number}/{total}")
    print("=" * 88)
    print(
        "settings:",
        f"x2={store['x2opt']}",
        f"x3={store['x3opt']}",
        f"x4={store['x4opt']}",
        f"x5={store['x5opt']}",
        f"dx={store['dx']}",
        f"dy={store['dy']}",
        f"hours={store['hoursRange']}",
    )
    print(
        "grid:",
        f"dataX1num={model['dataX1num']}",
        f"dataX2num={model['dataX2num']}",
        f"dataTnum={model['dataTnum']}",
    )


def build_base_params(config) -> Dict[str, Dict[str, Any]]:
    store = {
        "speciesLabel": config.fixed_params.species_label,
        "x2opt": config.parameter_grid.x2_opts[0],
        "x3opt": config.parameter_grid.x3_opts[0],
        "x4opt": config.parameter_grid.x4_opts[0],
        "x5opt": config.parameter_grid.x5_opts[0],
        "x7": config.parameter_grid.x7_opts[0],
        "dx": config.parameter_grid.dx_opts[0],
        "dy": config.parameter_grid.dy_opts[0],
        "xmin": config.domain.xmin,
        "ymin": config.domain.ymin,
        "xmax": config.domain.xmax,
        "ymax": config.domain.ymax,
        "hoursRange": list(config.hours_range_map.values())[0],
        "gamma": config.fixed_params.gamma,
        "K": config.fixed_params.K,
    }

    model = {
        "dataX1num": int(config.domain.xmax // config.parameter_grid.dx_opts[0]),
        "dataX2num": int(config.domain.ymax // config.parameter_grid.dy_opts[0]),
        "dataTnum": len(store["hoursRange"]),
        "dataK": config.fixed_params.K,
    }

    additional = {
        "data_root": str(config.data_root),
        "initial_path": str(config.dataobj_root),
        "plot_bool": bool(config.run_control.plot_bool),
        "overwrite_bool": bool(config.run_control.overwrite_bool),
        "binn_path": str(config.training_root / "binn"),
    }

    return {
        "RDEq_params_store": store,
        "RDEq_params": model,
        "additional_params": additional,
    }


def run_pipeline(force_overwrite: bool | None = None, force_plot: bool | None = None) -> Dict[str, Any]:
    cfg = build_default_config()

    if force_overwrite is not None:
        cfg = copy.deepcopy(cfg)
        cfg.run_control = type(cfg.run_control)(
            plot_bool=cfg.run_control.plot_bool,
            overwrite_bool=force_overwrite,
        )

    if force_plot is not None:
        cfg = copy.deepcopy(cfg)
        cfg.run_control = type(cfg.run_control)(
            plot_bool=force_plot,
            overwrite_bool=cfg.run_control.overwrite_bool,
        )

    cfg.runs_root.mkdir(parents=True, exist_ok=True)
    cfg.dataobj_root.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.runs_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.time()
    header = {
        "run_id": run_id,
        "started_at_utc": utc_now_iso(),
        "git_commit": get_git_commit(cfg.training_root),
        "training_root": str(cfg.training_root),
        "data_root": str(cfg.data_root),
        "dataobj_root": str(cfg.dataobj_root),
        "environment": environment_snapshot(),
        "hours_range_map": {f"{k[0]}_{k[1]}": v for k, v in cfg.hours_range_map.items()},
        "parameter_grid": {
            "x2_opts": cfg.parameter_grid.x2_opts,
            "x3_opts": cfg.parameter_grid.x3_opts,
            "x4_opts": cfg.parameter_grid.x4_opts,
            "x5_opts": cfg.parameter_grid.x5_opts,
            "x7_opts": cfg.parameter_grid.x7_opts,
            "dx_opts": cfg.parameter_grid.dx_opts,
            "dy_opts": cfg.parameter_grid.dy_opts,
        },
        "run_control": {
            "plot_bool": cfg.run_control.plot_bool,
            "overwrite_bool": cfg.run_control.overwrite_bool,
        },
    }
    write_json(run_dir / "run_header.json", header)

    base_params = build_base_params(cfg)

    param_lists = [
        cfg.parameter_grid.x2_opts,
        cfg.parameter_grid.x3_opts,
        cfg.parameter_grid.x4_opts,
        cfg.parameter_grid.x5_opts,
        cfg.parameter_grid.x7_opts,
        cfg.parameter_grid.dx_opts,
        cfg.parameter_grid.dy_opts,
    ]

    total_candidate = 1
    for items in param_lists:
        total_candidate *= len(items)

    records_path = run_dir / "records.jsonl"
    total_executed = 0
    total_generated = 0
    total_skipped = 0

    for combo_idx, (x2opt, x3opt, x4opt, x5opt, x7, dx, dy) in enumerate(product(*param_lists), start=1):
        schedule_key = (x4opt, x5opt)
        if schedule_key not in cfg.hours_range_map:
            append_jsonl(
                records_path,
                {
                    "combo_index": combo_idx,
                    "status": "not_scheduled",
                    "params": {
                        "x2opt": x2opt,
                        "x3opt": x3opt,
                        "x4opt": x4opt,
                        "x5opt": x5opt,
                        "x7": x7,
                        "dx": dx,
                        "dy": dy,
                    },
                },
            )
            continue

        hours_range = cfg.hours_range_map[schedule_key]
        payload = copy.deepcopy(base_params)
        payload["RDEq_params_store"]["x2opt"] = x2opt
        payload["RDEq_params_store"]["x3opt"] = x3opt
        payload["RDEq_params_store"]["x4opt"] = x4opt
        payload["RDEq_params_store"]["x5opt"] = x5opt
        payload["RDEq_params_store"]["x7"] = x7
        payload["RDEq_params_store"]["dx"] = dx
        payload["RDEq_params_store"]["dy"] = dy
        payload["RDEq_params_store"]["hoursRange"] = list(hours_range)

        payload["RDEq_params"]["dataX1num"] = int(cfg.domain.xmax // dx)
        payload["RDEq_params"]["dataX2num"] = int(cfg.domain.ymax // dy)
        payload["RDEq_params"]["dataTnum"] = len(hours_range)

        total_executed += 1
        print_run_summary(total_executed, total_candidate, payload)

        started = time.time()
        result = run_single_simulation(payload)
        runtime_sec = round(time.time() - started, 4)

        if result["status"] == "generated":
            total_generated += 1
        elif result["status"] == "skipped_existing":
            total_skipped += 1

        append_jsonl(
            records_path,
            {
                "combo_index": combo_idx,
                "executed_index": total_executed,
                "runtime_sec": runtime_sec,
                "status": result["status"],
                "artifact": result["artifact"],
                "artifact_valid": result["artifact_valid"],
                "path_key": result["path_key"],
                "params": payload["RDEq_params_store"],
                "grid": payload["RDEq_params"],
            },
        )

    summary = {
        "run_id": run_id,
        "finished_at_utc": utc_now_iso(),
        "duration_sec": round(time.time() - run_start, 3),
        "candidate_combinations": total_candidate,
        "executed_combinations": total_executed,
        "generated": total_generated,
        "skipped_existing": total_skipped,
        "records_file": str(records_path),
    }
    write_json(run_dir / "summary.json", summary)

    append_jsonl(
        cfg.runs_root / "run_index.jsonl",
        {
            **summary,
            "run_dir": str(run_dir),
            "run_control": {
                "plot_bool": cfg.run_control.plot_bool,
                "overwrite_bool": cfg.run_control.overwrite_bool,
            },
        },
    )

    print("-" * 88)
    print("Run complete")
    print(f"run_dir        : {run_dir}")
    print(f"generated      : {total_generated}")
    print(f"skipped        : {total_skipped}")
    print(f"records        : {records_path}")
    print("-" * 88)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run publication-grade data simulation pipeline.")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite of existing artifacts.")
    parser.add_argument("--no-overwrite", action="store_true", help="Never overwrite existing valid artifacts.")
    parser.add_argument("--plot", action="store_true", help="Enable optional plotting after generation.")
    parser.add_argument("--no-plot", action="store_true", help="Disable optional plotting.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    force_overwrite = None
    if args.overwrite and args.no_overwrite:
        raise ValueError("Use either --overwrite or --no-overwrite, not both.")
    if args.overwrite:
        force_overwrite = True
    if args.no_overwrite:
        force_overwrite = False

    force_plot = None
    if args.plot and args.no_plot:
        raise ValueError("Use either --plot or --no-plot, not both.")
    if args.plot:
        force_plot = True
    if args.no_plot:
        force_plot = False

    run_pipeline(force_overwrite=force_overwrite, force_plot=force_plot)


if __name__ == "__main__":
    main()
