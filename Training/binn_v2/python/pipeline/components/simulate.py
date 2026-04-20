from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from binn_v2.python.Modules.Models.BuildBINNs_2D import (
    BINN_2d,
    data_loss_GLS,
    data_loss_MSE,
    pde_loss_with_bc_2d,
    pde_loss_without_bc_2d,
)
from binn_v2.python.Modules.Utils.ModelWrapper import ModelWrapper
from binn_v2.python.Modules.dataClass import Data_experiment_manual

from .utils import dict_to_path, flatten_one_level, to_torch_grad


def bn_load_raw_func_info(data_obj_params: Dict[str, Any]) -> Dict[str, Any]:
    return data_obj_params["RDEq_params_store"].copy()


def bn_load_raw_data(data_obj_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    additional = data_obj_params["additional_params"]
    data_obj_path = additional["initial_path"]

    func_info = bn_load_raw_func_info(data_obj_params)
    info_path = dict_to_path(func_info)
    load_path = os.path.join(data_obj_path, info_path, "data_obj.npy")

    data_obj = np.load(load_path, allow_pickle=True).item()
    return func_info, data_obj


def bn_random_val_indices(inputs, vf: float, split_seed: int):
    n = len(inputs)
    num_val = int(vf * n)
    np.random.seed(split_seed)
    p = np.random.permutation(n)
    return p[-num_val:]


def bn_tvsplit(data_obj, data_obj_params: Dict[str, Any], tv_params: Dict[str, Any], model_params: Dict[str, Any]):
    species = data_obj_params["RDEq_params_store"]["speciesLabel"]
    if species == "red":
        u = data_obj.u_red.copy()
    else:
        u = data_obj.u_green.copy()

    inputs = data_obj.inputs
    outputs = u.reshape(-1)[:, None]

    binn_tv = tv_params["binnTV_params"]
    vf = binn_tv["binnVF"]
    split_seed = binn_tv["binnGenerateIndicesArgs"]["binnTVsplitSeed"]

    if vf:
        val_indices = bn_random_val_indices(inputs, vf, split_seed)
        all_indices = np.arange(len(inputs))
        train_indices = np.setdiff1d(all_indices, val_indices)

        x_train_np = inputs[train_indices]
        y_train_np = outputs[train_indices]
        x_val_np = inputs[val_indices]
        y_val_np = outputs[val_indices]
        validation_data = [x_val_np, y_val_np]
    else:
        x_train_np = inputs
        y_train_np = outputs
        x_val_np = []
        y_val_np = []
        validation_data = None

    tv_dic = {
        "train_dic": {"x_train_np": x_train_np, "y_train_np": y_train_np},
        "val_dic": {
            "x_val_np": x_val_np,
            "y_val_np": y_val_np,
            "validation_data": validation_data,
        },
    }

    func_info = flatten_one_level(binn_tv) if vf else {"binnVF": vf}
    return func_info, tv_dic


def bn_model_data_loss_func(model_params: Dict[str, Any]):
    label = model_params["binn_model_params"]["BNdata_loss_params"]["BNdataLossFuncLabel"]
    if label == "GLS":
        return {"BNdataLossFuncLabel": label}, data_loss_GLS
    return {"BNdataLossFuncLabel": label}, data_loss_MSE


def bn_model_pde_loss_func(model_params: Dict[str, Any]):
    pde_params = model_params["binn_model_params"]["pde_loss_params"]
    if pde_params["BCbool"]:
        return pde_params.copy(), pde_loss_with_bc_2d
    return pde_params.copy(), pde_loss_without_bc_2d


def bn_model_construction(data_obj_orig, data_obj_params: Dict[str, Any], model_params: Dict[str, Any]):
    bparams = model_params["binn_model_params"]["binn_construction_params"]

    data_obj_params["additional_params"]["x1"] = data_obj_orig.x1
    data_obj_params["additional_params"]["x2"] = data_obj_orig.x2
    data_obj_params["additional_params"]["u_red_max"] = data_obj_orig.u_red_max
    data_obj_params["additional_params"]["u_green_max"] = data_obj_orig.u_green_max

    _, data_loss_func = bn_model_data_loss_func(model_params)
    _, pde_loss_func = bn_model_pde_loss_func(model_params)

    model = BINN_2d(
        data_obj_params=data_obj_params,
        model_params=model_params,
        data_loss_func=data_loss_func,
        pde_loss_func=pde_loss_func,
    )
    return bparams.copy(), model


def bn_model_fit_first_smart(model_save_dir: str, nn_binn, fit_params: Dict[str, Any], tv_dic: Dict[str, Any], device: str):
    train_dic = tv_dic["train_dic"]
    val_dic = tv_dic["val_dic"]

    x_train_np = train_dic["x_train_np"]
    y_train_np = train_dic["y_train_np"]
    x_val_np = val_dic["x_val_np"]
    y_val_np = val_dic["y_val_np"]
    validation_data = val_dic["validation_data"]

    x_train = to_torch_grad(x_train_np, device)
    y_train = to_torch_grad(y_train_np, device)

    if validation_data is not None:
        x_val = to_torch_grad(x_val_np, device)
        y_val = to_torch_grad(y_val_np, device)
        validation_data = [x_val, y_val]

    bfit = fit_params["binn_fit_params"]
    bfit_add = fit_params["binn_fit_params_additionals"]

    model_label = bfit["binnModelLabel"]
    opt = torch.optim.Adam(nn_binn.parameters(), lr=bfit["binnLR"])

    weights_dir = os.path.join(model_save_dir, f"Weights_binn_num{model_label}")
    os.makedirs(weights_dir, exist_ok=True)

    modelw = ModelWrapper(
        model=nn_binn,
        optimizer=opt,
        loss=nn_binn.loss,
        save_name=f"{weights_dir}/test",
    )

    modelw.model_save_dir = model_save_dir
    modelw.binnModelLabel = model_label
    modelw.x_train = x_train_np
    modelw.y_train = y_train_np
    modelw.x_val = x_val_np
    modelw.y_val = y_val_np
    modelw.x_train_torch = x_train
    modelw.y_train_torch = y_train
    modelw.validation_data = validation_data

    modelw.batch_size = bfit["binnBatchSize"]
    modelw.early_stopping = bfit["binnES"]
    modelw.rel_update_thresh = bfit["binnRelUpdateThresh"]
    modelw.rel_save_thresh = bfit["binnRelSaveThresh"]
    modelw.verbose = 1

    modelw.fit(
        x_tr_input=modelw.x_train_torch,
        y_tr_input=modelw.y_train_torch,
        batch_size=modelw.batch_size,
        epochs=int(bfit_add["binnEpochs"]),
        verbose=modelw.verbose,
        validation_data=modelw.validation_data,
        early_stopping=modelw.early_stopping,
        rel_update_thresh=modelw.rel_update_thresh,
        rel_save_thresh=modelw.rel_save_thresh,
        print_freq=bfit_add["printFreq"],
    )

    return bfit.copy(), modelw


def bn_first_train_smart(model_save_dir: str, data_obj_params, tv_params, model_params, fit_params):
    device = model_params["binn_model_params"]["binn_construction_params"]["binnDevice"]

    _, data_obj = bn_load_raw_data(data_obj_params)
    _, tv_dic = bn_tvsplit(data_obj, data_obj_params, tv_params, model_params)
    _, nn_binn = bn_model_construction(data_obj, data_obj_params, model_params)

    _, modelw = bn_model_fit_first_smart(
        model_save_dir=model_save_dir,
        nn_binn=nn_binn,
        fit_params=fit_params,
        tv_dic=tv_dic,
        device=device,
    )

    return data_obj, modelw


def bn_model_finder(path_intro: str, data_obj_params, tv_params, model_params, fit_params):
    all_func_info: Dict[str, Any] = {}
    all_func_info.update(bn_load_raw_func_info(data_obj_params))

    binn_tv = tv_params["binnTV_params"]
    all_func_info.update(flatten_one_level(binn_tv) if binn_tv["binnVF"] else {"binnVF": binn_tv["binnVF"]})

    all_func_info.update(flatten_one_level(model_params["binn_model_params"]))
    all_func_info.update(fit_params["binn_fit_params"])

    full_path = os.path.join(path_intro, dict_to_path(all_func_info))
    if os.path.exists(full_path) and os.listdir(full_path):
        return full_path, True

    os.makedirs(full_path, exist_ok=True)
    return full_path, False


def bn_model_check_es(path_intro: str, data_obj_params, tv_params, model_params, fit_params):
    current_dir, exists = bn_model_finder(path_intro, data_obj_params, tv_params, model_params, fit_params)
    if exists:
        return False

    fit_params_to_update = copy.deepcopy(fit_params)
    current_es = fit_params["binn_fit_params"]["binnES"]

    sorted_es = np.array(sorted(fit_params["binn_fit_params_additionals"]["binnES_check"], reverse=True))
    for es in sorted_es[sorted_es < current_es]:
        fit_params_to_update["binn_fit_params"]["binnES"] = int(es)
        old_dir, old_exists = bn_model_finder(path_intro, data_obj_params, tv_params, model_params, fit_params_to_update)
        if not old_exists:
            continue

        shutil.copytree(old_dir, current_dir, dirs_exist_ok=True)

        model_label = fit_params["binn_fit_params"]["binnModelLabel"]
        model_path = os.path.join(current_dir, f"binnModel{model_label}.pth")
        modelw = torch.load(model_path, weights_only=False)
        modelw.early_stopping = current_es
        modelw.save_name = modelw.save_name.replace(f"binnES_{es}", f"binnES_{current_es}")
        torch.save(modelw, model_path)
        return True

    return False


def bn_model_fit_again(model_dir_path: str, fit_params, load_es_bool: bool = True):
    model_label = fit_params["binn_fit_params"]["binnModelLabel"]
    model_path = os.path.join(model_dir_path, f"binnModel{model_label}.pth")

    bfit_add = fit_params["binn_fit_params_additionals"]
    modelw = torch.load(model_path, weights_only=False)

    total_train_losses = len(modelw.train_loss_list)
    if modelw.early_stopping is not None and total_train_losses - modelw.last_improved >= modelw.early_stopping:
        return modelw

    if load_es_bool:
        modelw.load_ES()
    else:
        modelw.load_expired()

    modelw.fit(
        x_tr_input=modelw.x_train_torch,
        y_tr_input=modelw.y_train_torch,
        batch_size=modelw.batch_size,
        epochs=int(bfit_add["binnEpochs"]),
        verbose=modelw.verbose,
        validation_data=modelw.validation_data,
        early_stopping=modelw.early_stopping,
        rel_update_thresh=modelw.rel_update_thresh,
        rel_save_thresh=modelw.rel_save_thresh,
        print_freq=bfit_add["printFreq"],
    )
    return modelw


def bn_retrain(model_dir_path: str, tv_params, fit_params, load_es_bool: bool = True):
    _ = tv_params  # retained for interface parity
    return bn_model_fit_again(model_dir_path, fit_params, load_es_bool=load_es_bool)


def bn_save_model_generated_dataobj(model_loaded, dataobj, data_obj_params, model_label: int, path_to_dataobj_dir: str, device: str):
    nt, nx1, nx2 = len(dataobj.t), len(dataobj.x1), len(dataobj.x2)
    model_loaded.load_best_val()

    with torch.no_grad():
        u_pred_flat = model_loaded.model(to_torch_grad(dataobj.inputs, device))
        u = u_pred_flat.reshape(nx1, nx2, nt).detach().cpu().numpy()

    data_obj_binn = Data_experiment_manual(
        x2_opt=dataobj.x2_opt,
        x3_opt=dataobj.x3_opt,
        x4_opt=dataobj.x4_opt,
        x5_opt=dataobj.x5_opt,
        x7=dataobj.x7,
        hoursRange=dataobj.hoursRange,
        x1=dataobj.x1,
        x2=dataobj.x2,
        t=dataobj.t,
    )

    if data_obj_params["RDEq_params_store"]["speciesLabel"] == "red":
        data_obj_binn.u_red = u
    else:
        data_obj_binn.u_green = u

    save_path = os.path.join(path_to_dataobj_dir, f"data_binn_num{model_label}")
    np.save(save_path, data_obj_binn)


def bn_save_model(modelw, model_save_dir: str, data_obj, data_obj_params, tv_params, model_params, fit_params):
    bfit = fit_params["binn_fit_params"]
    model_label = bfit["binnModelLabel"]
    device = model_params["binn_model_params"]["binn_construction_params"]["binnDevice"]

    model_save_path = os.path.join(model_save_dir, f"binnModel{model_label}.pth")
    torch.save(modelw, model_save_path)

    bn_save_model_generated_dataobj(
        model_loaded=modelw,
        dataobj=data_obj,
        data_obj_params=data_obj_params,
        model_label=model_label,
        path_to_dataobj_dir=model_save_dir,
        device=device,
    )


def run_single_binn_simulation(data_obj_params, tv_params, model_params, fit_params):
    bfit = fit_params["binn_fit_params"]
    bconstruct = model_params["binn_model_params"]["binn_construction_params"]
    additional = data_obj_params["additional_params"]

    model_label = bfit["binnModelLabel"]
    binn_path = additional["binn_path"]

    model_save_dir, _ = bn_model_finder(
        path_intro=binn_path,
        data_obj_params=data_obj_params,
        tv_params=tv_params,
        model_params=model_params,
        fit_params=fit_params,
    )

    model_path = os.path.join(model_save_dir, f"binnModel{model_label}.pth")
    utilize_pretrained = bn_model_check_es(
        path_intro=binn_path,
        data_obj_params=data_obj_params,
        tv_params=tv_params,
        model_params=model_params,
        fit_params=fit_params,
    )
    load_es_bool = True if utilize_pretrained else False

    if os.path.isfile(model_path):
        modelw = bn_retrain(model_save_dir, tv_params, fit_params, load_es_bool=load_es_bool)
        _, data_obj = bn_load_raw_data(data_obj_params)
        status = "retrained"
    else:
        data_obj, modelw = bn_first_train_smart(model_save_dir, data_obj_params, tv_params, model_params, fit_params)
        status = "trained_new"

    # Store dataset views for downstream consistency.
    modelw.u_red = data_obj.u_red
    modelw.u_green = data_obj.u_green
    modelw.u_total = data_obj.u_total

    bn_save_model(modelw, model_save_dir, data_obj, data_obj_params, tv_params, model_params, fit_params)

    return {
        "status": status,
        "model_dir": str(Path(model_save_dir)),
        "model_path": str(Path(model_path)),
        "device": bconstruct["binnDevice"],
    }
