from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    print_freq: int = 10


@dataclass(frozen=True)
class DataParams:
    species_labels: List[str] = field(default_factory=lambda: ["green"])
    x2_opts: List[str] = field(default_factory=lambda: ["0p0"])
    x3_opts: List[str] = field(default_factory=lambda: ["0p0"])
    x7_opts: List[int] = field(default_factory=lambda: [1])
    valid_x4x5s: List[Tuple[int, int]] = field(default_factory=lambda: [(2,3),(2,5),(3, 1)])
    valid_dxdys: List[Tuple[float, float]] = field(default_factory=lambda: [(100.0, 100.0)])
    tends: List[int] = field(default_factory=lambda: [52, 56, 64])


@dataclass(frozen=True)
class BinnParams:
    # low-level architecture
    binn_usizes: List[int] = field(default_factory=lambda: [64])
    binn_dsizes: List[int] = field(default_factory=lambda: [4])
    binn_gsizes: List[int] = field(default_factory=lambda: [4])
    model_labels: List[int] = field(default_factory=lambda: [0])

    # training control
    es_values: List[int] = field(default_factory=lambda: [500, 1000, 2000])
    tv_split_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    vf_values: List[float] = field(default_factory=lambda: [0.2])

    # loss and constraints
    num_pde_samples: List[int] = field(default_factory=lambda: [100])
    data_loss_labels: List[str] = field(default_factory=lambda: ["MSE"])
    d_constraint_bools: List[int] = field(default_factory=lambda: [0])
    d_bounds: List[int] = field(default_factory=lambda: [1])

    # fixed scalar settings
    gamma: int = 0
    K: int = 1
    device: str = "cpu"
    lr: float = 1e-3
    batch_size: int = 38
    rel_update_thresh: float = 0.05
    rel_save_thresh: float = 0.05
    initialize_denoise_bool: int = 0
    done_param_bool: bool = False
    all_constraints: int = 0
    bc_bool: int = 0
    generate_indices_label: str = "random"
    epochs: int = 10000
    es_check: List[int] = field(default_factory=lambda: [10000])


@dataclass(frozen=True)
class PipelineConfig:
    main_root: Path
    dataobj_root: Path
    model_root: Path
    runs_root: Path
    domain: DomainConfig
    run_control: RunControl
    data_params: DataParams
    binn_params: BinnParams


def valid_hours_range(x4opt: int, x5opt: int, tend: int) -> Optional[List[int]]:
    if (x4opt, x5opt, tend) == (2, 3, 52):
        return list(range(8, tend + 1, 4))
    if (x4opt, x5opt, tend) == (2, 5, 56):
        return list(range(24, tend + 1, 4))
    if (x4opt, x5opt, tend) == (3, 1, 64):
        return list(range(4, tend + 1, 4))
    return None


def build_default_config() -> PipelineConfig:
    main_root = Path(__file__).resolve().parents[4]
    return PipelineConfig(
        main_root=main_root,
        dataobj_root=main_root / "dataObj_v2",
        model_root=main_root / "binn_v2_models",
        runs_root=main_root / "binn_v2" / "runs",
        domain=DomainConfig(),
        run_control=RunControl(),
        data_params=DataParams(),
        binn_params=BinnParams(),
    )