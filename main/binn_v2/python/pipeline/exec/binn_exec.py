from __future__ import annotations

import argparse
import copy
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict

# Ensure main root is importable regardless of current working directory.
MAIN_ROOT = Path(__file__).resolve().parents[4]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from binn_v2.python.pipeline.components.run_metadata import (
    append_jsonl,
    environment_snapshot,
    get_git_commit,
    utc_now_iso,
    write_json,
)
from binn_v2.python.pipeline.components.simulate import run_single_binn_simulation
from binn_v2.python.pipeline.config.experiment_config import build_default_config, valid_hours_range


def print_run_summary(run_number: int, total: int, payload: Dict[str, Dict[str, Any]], tv_params, model_params, fit_params) -> None:
    store = payload["RDEq_params_store"]
    model = payload["RDEq_params"]
    bfit = fit_params["binn_fit_params"]

    print("=" * 96)
    print(f"RUN {run_number}/{total}")
    print("=" * 96)
    print(
        "data:",
        f"species={store['speciesLabel']}",
        f"x2={store['x2opt']}",
        f"x3={store['x3opt']}",
        f"x4={store['x4opt']}",
        f"x5={store['x5opt']}",
        f"dx={store['dx']}",
        f"dy={store['dy']}",
        f"hours={store['hoursRange']}",
    )
    print(
        "model:",
        f"U={model_params['binn_model_params']['binn_construction_params']['binnUsize']}",
        f"D={model_params['binn_model_params']['binn_construction_params']['binnDsize']}",
        f"G={model_params['binn_model_params']['binn_construction_params']['binnGsize']}",
        f"VF={tv_params['binnTV_params']['binnVF']}",
        f"seed={tv_params['binnTV_params']['binnGenerateIndicesArgs']['binnTVsplitSeed']}",
        f"ES={bfit['binnES']}",
        f"label={bfit['binnModelLabel']}",
    )
    print("grid:", f"Nx={model['dataX1num']}", f"Ny={model['dataX2num']}", f"Nt={model['dataTnum']}")


def build_base_payload(cfg):
    b = cfg.binn_params
    d = cfg.data_params

    data_store = {
        "speciesLabel": d.species_labels[0],
        "x2opt": d.x2_opts[0],
        "x3opt": d.x3_opts[0],
        "x4opt": d.valid_x4x5s[0][0],
        "x5opt": d.valid_x4x5s[0][1],
        "x7": d.x7_opts[0],
        "dx": d.valid_dxdys[0][0],
        "dy": d.valid_dxdys[0][1],
        "xmin": cfg.domain.xmin,
        "ymin": cfg.domain.ymin,
        "xmax": cfg.domain.xmax,
        "ymax": cfg.domain.ymax,
        "hoursRange": [0],
        "gamma": b.gamma,
        "K": b.K,
    }

    data_model = {
        "dataX1num": int(cfg.domain.xmax // d.valid_dxdys[0][0]),
        "dataX2num": int(cfg.domain.ymax // d.valid_dxdys[0][1]),
        "dataTnum": 1,
    }

    additional = {
        "initial_path": str(cfg.dataobj_root),
        "binn_path": str(cfg.model_root),
        "plot_bool": bool(cfg.run_control.plot_bool),
        "overwrite_bool": bool(cfg.run_control.overwrite_bool),
    }

    data_obj_params = {
        "RDEq_params_store": data_store,
        "RDEq_params": data_model,
        "additional_params": additional,
    }

    tv_params = {
        "binnTV_params": {
            "binnVF": b.vf_values[0],
            "binnGenerateIndicesLabel": b.generate_indices_label,
            "binnGenerateIndicesArgs": {"binnTVsplitSeed": b.tv_split_seeds[0]},
        }
    }

    model_params = {
        "binn_model_params": {
            "binn_construction_params": {
                "binnUsize": b.binn_usizes[0],
                "binnDsize": b.binn_dsizes[0],
                "binnGsize": b.binn_gsizes[0],
                "DoneParamBool": b.done_param_bool,
                "binnDevice": b.device,
                "binnInitializeDenoiseBool": b.initialize_denoise_bool,
                "allConstraints": b.all_constraints,
                "D_constraint_bool": b.d_constraint_bools[0],
                "D_bound": b.d_bounds[0],
            },
            "BNdata_loss_params": {"BNdataLossFuncLabel": b.data_loss_labels[0]},
            "pde_loss_params": {"BCbool": b.bc_bool, "numPDEsamples": b.num_pde_samples[0]},
        }
    }

    fit_params = {
        "binn_fit_params": {
            "binnLR": b.lr,
            "binnBatchSize": b.batch_size,
            "binnRelUpdateThresh": b.rel_update_thresh,
            "binnRelSaveThresh": b.rel_save_thresh,
            "binnES": b.es_values[0],
            "binnModelLabel": b.model_labels[0],
        },
        "binn_fit_params_additionals": {
            "binnEpochs": b.epochs,
            "binnES_check": b.es_check,
            "printFreq": cfg.run_control.print_freq,
        },
    }

    return data_obj_params, tv_params, model_params, fit_params


def run_pipeline(max_runs: int | None = None) -> Dict[str, Any]:
    cfg = build_default_config()
    cfg.dataobj_root.mkdir(parents=True, exist_ok=True)
    cfg.model_root.mkdir(parents=True, exist_ok=True)
    cfg.runs_root.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.runs_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.time()
    header = {
        "run_id": run_id,
        "started_at_utc": utc_now_iso(),
        "git_commit": get_git_commit(cfg.main_root),
        "main_root": str(cfg.main_root),
        "dataobj_root": str(cfg.dataobj_root),
        "model_root": str(cfg.model_root),
        "environment": environment_snapshot(),
    }
    write_json(run_dir / "run_header.json", header)

    data_obj_base, tv_base, model_base, fit_base = build_base_payload(cfg)

    d = cfg.data_params
    b = cfg.binn_params

    param_lists = [
        d.x2_opts,
        d.x3_opts,
        d.valid_x4x5s,
        d.x7_opts,
        d.valid_dxdys,
        d.tends,
        d.species_labels,
        b.binn_usizes,
        b.binn_dsizes,
        b.binn_gsizes,
        b.es_values,
        b.tv_split_seeds,
        b.vf_values,
        b.model_labels,
        b.num_pde_samples,
        b.data_loss_labels,
        b.d_constraint_bools,
        b.d_bounds,
    ]

    records_path = run_dir / "records.jsonl"
    
    # 1. Generate all raw combinations
    all_combinations = list(product(*param_lists))
    valid_combinations = []
    
    # 2. Pre-filter and log the invalid combinations
    for combo_idx, combo in enumerate(all_combinations, start=1):
        x4opt, x5opt = combo[2] # valid_x4x5s is at index 2
        tend = combo[5]         # tends is at index 5
        
        hours_range = valid_hours_range(x4opt=x4opt, x5opt=x5opt, tend=tend)
        
        if hours_range is None:
            append_jsonl(
                records_path,
                {
                    "combo_index": combo_idx,
                    "status": "not_scheduled",
                    "params": {
                        "x2opt": combo[0],
                        "x3opt": combo[1],
                        "x4opt": x4opt,
                        "x5opt": x5opt,
                        "tend": tend,
                    },
                },
            )
        else:
            # Store the computed hours_range alongside the combo so we don't recalculate it
            valid_combinations.append((combo_idx, combo, hours_range))

    # 3. Set the total candidate count to ONLY the valid runs
    total_candidate = len(valid_combinations)
    
    executed = 0
    generated = 0
    retrained = 0

    # 4. Iterate over the pre-filtered valid combinations
    for combo_idx, combo, hours_range in valid_combinations:
        (
            x2opt,
            x3opt,
            valid_x4x5,
            x7,
            valid_dxdy,
            tend,
            species_label,
            u_size,
            d_size,
            g_size,
            es,
            split_seed,
            vf,
            model_label,
            num_pde_sample,
            data_loss_label,
            d_constraint_bool,
            d_bound,
        ) = combo

        x4opt, x5opt = valid_x4x5

        data_obj_params = copy.deepcopy(data_obj_base)
        tv_params = copy.deepcopy(tv_base)
        model_params = copy.deepcopy(model_base)
        fit_params = copy.deepcopy(fit_base)

        dx, dy = valid_dxdy
        x1num = int(cfg.domain.xmax // dx)
        x2num = int(cfg.domain.ymax // dy)

        store = data_obj_params["RDEq_params_store"]
        store["speciesLabel"] = species_label
        store["x2opt"] = x2opt
        store["x3opt"] = x3opt
        store["x4opt"] = x4opt
        store["x5opt"] = x5opt
        store["x7"] = x7
        store["dx"] = dx
        store["dy"] = dy
        store["hoursRange"] = list(hours_range)

        data_obj_params["RDEq_params"]["dataX1num"] = x1num
        data_obj_params["RDEq_params"]["dataX2num"] = x2num
        data_obj_params["RDEq_params"]["dataTnum"] = len(hours_range)

        tv_params["binnTV_params"]["binnVF"] = vf
        tv_params["binnTV_params"]["binnGenerateIndicesArgs"] = {"binnTVsplitSeed": split_seed}

        mcon = model_params["binn_model_params"]["binn_construction_params"]
        mcon["binnUsize"] = u_size
        mcon["binnDsize"] = d_size
        mcon["binnGsize"] = g_size
        mcon["D_constraint_bool"] = d_constraint_bool
        mcon["D_bound"] = d_bound
        model_params["binn_model_params"]["pde_loss_params"]["numPDEsamples"] = num_pde_sample
        model_params["binn_model_params"]["BNdata_loss_params"]["BNdataLossFuncLabel"] = data_loss_label

        fit_params["binn_fit_params"]["binnES"] = es
        fit_params["binn_fit_params"]["binnModelLabel"] = model_label

        executed += 1
        print_run_summary(executed, total_candidate, data_obj_params, tv_params, model_params, fit_params)

        start_one = time.time()
        result = run_single_binn_simulation(data_obj_params, tv_params, model_params, fit_params)
        runtime_sec = round(time.time() - start_one, 4)

        if result["status"] == "trained_new":
            generated += 1
        elif result["status"] == "retrained":
            retrained += 1

        append_jsonl(
            records_path,
            {
                "combo_index": combo_idx,
                "executed_index": executed,
                "runtime_sec": runtime_sec,
                "status": result["status"],
                "model_dir": result["model_dir"],
                "model_path": result["model_path"],
                "params": data_obj_params["RDEq_params_store"],
                "tv": tv_params["binnTV_params"],
                "model": model_params["binn_model_params"],
                "fit": fit_params["binn_fit_params"],
            },
        )

        if max_runs is not None and executed >= max_runs:
            break

    summary = {
        "run_id": run_id,
        "finished_at_utc": utc_now_iso(),
        "duration_sec": round(time.time() - run_start, 3),
        "candidate_combinations": total_candidate,
        "executed_combinations": executed,
        "trained_new": generated,
        "retrained": retrained,
        "records_file": str(records_path),
    }
    write_json(run_dir / "summary.json", summary)

    append_jsonl(
        cfg.runs_root / "run_index.jsonl",
        {
            **summary,
            "run_dir": str(run_dir),
        },
    )

    print("-" * 96)
    print("BINN run complete")
    print(f"run_dir     : {run_dir}")
    print(f"trained_new : {generated}")
    print(f"retrained   : {retrained}")
    print(f"records     : {records_path}")
    print("-" * 96)

    return summary

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone publication-grade BINN smart pipeline.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional limit for quick validation runs.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(max_runs=args.max_runs)


if __name__ == "__main__":
    main()
