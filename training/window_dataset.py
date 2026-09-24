from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from training.dataset import FEATURE_SOURCES, valid_window_starts


class TrajectoryWindowDataset(Dataset):
    """CPU dataset for worker-prefetched contiguous trajectory windows."""

    def __init__(
        self,
        dataset: Mapping[str, Any],
        input_names: Sequence[str],
        sequence_length: int,
    ) -> None:
        valid = torch.as_tensor(dataset["valid"], dtype=torch.bool)
        self.starts = valid_window_starts(valid, sequence_length).cpu().numpy()
        self.sequence_length = sequence_length
        self.input_sources = {
            name: FEATURE_SOURCES.get(name, name) or name for name in input_names
        }
        self.dataset = dataset
        self.targets = np.asarray(dataset["next_states"] - dataset["states"])

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        trajectory, start = self.starts[index]
        end = start + self.sequence_length
        inputs = {
            name: torch.from_numpy(
                np.asarray(self.dataset[source][trajectory, start:end])
            )
            for name, source in self.input_sources.items()
        }
        target = torch.from_numpy(
            np.asarray(self.targets[trajectory, start:end])
        )
        return inputs, target
