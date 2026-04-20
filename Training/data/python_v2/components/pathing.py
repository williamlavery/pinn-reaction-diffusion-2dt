from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable


PATH_KEY_ORDER = [
    "speciesLabel",
    "x2opt",
    "x3opt",
    "x4opt",
    "x5opt",
    "x7",
    "dx",
    "dy",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "hoursRange",
    "gamma",
    "K",
]


def _value_to_token(value: Any) -> str:
    if isinstance(value, list):
        # Keep legacy behavior from v1 dictToPath where lists are stringified
        # directly (e.g. "[32, 36, 40, 44, 48, 52]") for hoursRange paths.
        return str(value)
    return str(value)


def dict_to_ordered_path(params: Dict[str, Any], key_order: Iterable[str] = PATH_KEY_ORDER) -> Path:
    parts = []
    for key in key_order:
        if key not in params:
            continue
        token = _value_to_token(params[key])
        parts.append(f"{key}_{token}")
    return Path(*parts)
