from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class DomainConfig:
    xmin: float = 0.0
    xmax: float = 1410.0
    ymin: float = 0.0
    ymax: float = 1050.0


@dataclass(frozen=True)
class RunControl:
    plot_bool: bool = False
    overwrite_bool: bool = True


@dataclass(frozen=True)
class ParameterGrid:
    x2_opts: List[str] = field(default_factory=lambda: ["0p0"])
    x3_opts: List[str] = field(default_factory=lambda: ["0p0"])
    x4_opts: List[int] = field(default_factory=lambda: [2, 3])
    x5_opts: List[int] = field(default_factory=lambda: [1, 3, 5])
    x7_opts: List[int] = field(default_factory=lambda: [1])
    dx_opts: List[float] = field(default_factory=lambda: [100.0])
    dy_opts: List[float] = field(default_factory=lambda: [100.0])


@dataclass(frozen=True)
class FixedExperimentParams:
    gamma: int = 0
    K: int = 1
    species_label: str = "green"


@dataclass(frozen=True)
class PipelineConfig:
    training_root: Path
    data_root: Path
    dataobj_root: Path
    runs_root: Path
    domain: DomainConfig
    run_control: RunControl
    fixed_params: FixedExperimentParams
    parameter_grid: ParameterGrid
    hours_range_map: Dict[Tuple[int, int], List[int]]


def build_default_config() -> PipelineConfig:
    # Resolve from this file so execution is independent of cwd.
    training_root = Path(__file__).resolve().parents[3]
    data_root = training_root / "data" / "split_csvs"
    dataobj_root = training_root / "dataObj_v2"
    runs_root = training_root / "data" / "python_v2" / "runs"

    # Preserves the current v1 execution behavior where schedules are keyed by (x4, x5).
    hours_range_map = {
        (2, 3): list(range(8, 53, 4)),
        (2, 5): list(range(24, 57, 4)),
        (3, 1): list(range(4, 65, 4)),
    }

    return PipelineConfig(
        training_root=training_root,
        data_root=data_root,
        dataobj_root=dataobj_root,
        runs_root=runs_root,
        domain=DomainConfig(),
        run_control=RunControl(),
        fixed_params=FixedExperimentParams(),
        parameter_grid=ParameterGrid(),
        hours_range_map=hours_range_map,
    )
