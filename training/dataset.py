from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch


REQUIRED_TRAJECTORY_FIELDS = frozenset(
    {"states", "joint_f", "next_states", "valid"}
)


def valid_window_starts(valid: torch.Tensor, sequence_length: int) -> torch.Tensor:
    valid = torch.as_tensor(valid, dtype=torch.bool)
    if valid.ndim != 2:
        raise ValueError("valid transition mask must have shape (trajectories, steps)")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if valid.shape[1] < sequence_length:
        return torch.empty((0, 2), dtype=torch.long, device=valid.device)
    return valid.unfold(1, sequence_length, 1).all(dim=-1).nonzero()


def gather_windows(values: torch.Tensor, starts: torch.Tensor, sequence_length: int) -> torch.Tensor:
    offsets = torch.arange(sequence_length, device=starts.device)
    return values[starts[:, :1], starts[:, 1:] + offsets]


def load_trajectory_dataset(path: str | Path, expected_split: str | None = None) -> dict[str, np.ndarray]:
    path = Path(path)
    with h5py.File(path, "r", swmr=True, libver="latest") as file:
        if "data" not in file:
            raise RuntimeError(f"Dataset {path} does not contain a data group")
        group = cast(h5py.Group, file["data"])
        if group.attrs.get("mode") != "trajectory":
            raise RuntimeError(f"Dataset {path} is not a trajectory dataset")
        split = group.attrs.get("split", "unknown")
        if expected_split is not None and split not in ("unknown", expected_split):
            raise RuntimeError(f"Dataset {path} has split {split!r}; expected {expected_split!r}")
        dataset: dict[str, Any] = {
            name: np.asarray(cast(h5py.Dataset, group[name])[()])
            for name in group.keys()
        }
        dataset["_metadata"] = dict(group.attrs)

    missing = REQUIRED_TRAJECTORY_FIELDS.difference(dataset)
    if missing:
        raise RuntimeError(f"Dataset {path} is missing fields: {sorted(missing)}")
    shape = dataset["states"].shape[:2]
    if dataset["states"].ndim != 3 or not all(shape):
        raise RuntimeError(f"Dataset {path} has invalid state trajectories")
    for name, value in dataset.items():
        if name == "_metadata":
            continue
        if name == "valid":
            if value.shape != shape:
                raise RuntimeError(f"Dataset {path} has an invalid valid mask")
        elif value.shape[:2] != shape:
            raise RuntimeError(f"Dataset {path} has inconsistent {name} trajectories")
        if not np.isfinite(value).all():
            raise RuntimeError(f"Dataset {path} contains non-finite {name}")
    return dataset