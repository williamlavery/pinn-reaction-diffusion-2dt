"""Parsing and utility helpers shared across BINN configuration code.

These helpers support function introspection, path construction from parameter
dictionaries, and lightweight dictionary/product transformations.
"""

import inspect
import os
from itertools import product

import numpy as np
import torch

def parse_function(func):
    """
    Parses a function and returns its name and keyword arguments with default values.

    Parameters:
        func (function): The function object to parse.

    Returns:
        dict: {"function_name": {arg_name: default_value or None}}
    """
    sig = inspect.signature(func)
    arg_info = {}
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            arg_info[name] = None  # No default = required positional or keyword arg
        else:
            arg_info[name] = param.default

    return {func.__name__: arg_info}


def dictToPath(arg_dict, sep='/', kv_delim='_'):
    """
    Converts a dictionary into a path string like arg1_val1/arg2_val2.

    Parameters:
        arg_dict (dict): Dictionary of key-value pairs.
        sep (str): Separator between key-value pairs (default: '/').
        kv_delim (str): Delimiter between key and value (default: '_').

    Returns:
        str: Generated path string.
    """
    parts = [f"{key}{kv_delim}{value}" for key, value in arg_dict.items()]
    return os.path.join(*parts) if sep == '/' else sep.join(parts)

    
def clone_empty_instance(obj):
    """
    Create a new instance of the same class as `obj`,
    but with no attributes (clean __dict__).
    """
    cls = obj.__class__
    new_obj = cls.__new__(cls)  # bypass __init__
    return new_obj

def to_torch_grad(ndarray, device):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr


def unravel_one_level(d):
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result.update(value)  # flatten one level
        else:
            result[key] = value
    return result


def constrained_product(params, constrained_groups, independent_keys, order):
    """
    Yields dictionaries with one value per key in `order`, honoring valid tuples for each constrained group.
    """
    if any(len(choices) == 0 for _, choices in constrained_groups):
        return  # nothing to yield

    # product over constrained groups (each pick is a tuple of values for that group)
    for group_tuple_selection in product(*(choices for _, choices in constrained_groups)):
        # build partial assignment from constrained groups
        fixed = {}
        for (names, _), values in zip(constrained_groups, group_tuple_selection):
            for k, v in zip(names, values):
                fixed[k] = v

        # now product over independent keys
        indep_lists = [params[k] for k in independent_keys]
        for indep_values in product(*indep_lists):
            full = dict(fixed)
            full.update({k: v for k, v in zip(independent_keys, indep_values)})
            # yield in positional order
            yield tuple(full[k] for k in order)