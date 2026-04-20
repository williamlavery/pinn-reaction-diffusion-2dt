from __future__ import annotations

import os
from itertools import product
from typing import Any, Dict

import torch


def to_torch_grad(ndarray, device: str):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    return arr.to(device)


def flatten_one_level(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(v)
        else:
            out[k] = v
    return out


def dict_to_path(arg_dict: Dict[str, Any], sep: str = "/", kv_delim: str = "_") -> str:
    # Preserves legacy behavior for list values (str(list)).
    parts = [f"{k}{kv_delim}{v}" for k, v in arg_dict.items()]
    return os.path.join(*parts) if sep == "/" else sep.join(parts)


def constrained_product(params, constrained_groups, independent_keys, order):
    if any(len(choices) == 0 for _, choices in constrained_groups):
        return
    for group_tuple_selection in product(*(choices for _, choices in constrained_groups)):
        fixed = {}
        for (names, _), values in zip(constrained_groups, group_tuple_selection):
            for k, v in zip(names, values):
                fixed[k] = v
        indep_lists = [params[k] for k in independent_keys]
        for indep_values in product(*indep_lists):
            full = dict(fixed)
            full.update({k: v for k, v in zip(independent_keys, indep_values)})
            yield tuple(full[k] for k in order)
