from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from training.dataset import FEATURE_SOURCES, gather_windows, valid_window_starts


@torch.no_grad()
def evaluate_rollouts(
    model,
    dataset: Mapping[str, Any],
    config: Mapping[str, Any],
    device: str,
    *,
    horizon: int,
    max_rollouts: int | None = None,
    batch_size: int = 256,
) -> dict[str, float]:
    """Measure autoregressive state drift on contiguous, valid dataset windows."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    sequence_length = int(config["sequence"]["length"])
    valid = torch.as_tensor(dataset["valid"], dtype=torch.bool, device=device)
    starts = valid_window_starts(valid, sequence_length + horizon)
    if len(starts) == 0:
        raise ValueError("Dataset has no valid rollout windows")
    if max_rollouts is not None:
        if max_rollouts < 1:
            raise ValueError("max_rollouts must be positive")
        starts = starts[:max_rollouts]

    state_values = torch.as_tensor(dataset["states"], device=device)
    target_values = torch.as_tensor(dataset["next_states"], device=device)
    model_input_names = list(config["inputs"]["low_dim"])
    feature_values = {}
    for input_name in model_input_names:
        if input_name == "states_embedding":
            continue
        source = FEATURE_SOURCES.get(input_name, input_name)
        if source not in dataset:
            raise ValueError(f"Dataset is missing rollout input {source!r}")
        feature_values[input_name] = torch.as_tensor(dataset[source], device=device)
    squared_error = 0.0
    final_squared_error = 0.0
    element_count = 0
    rollout_count = 0
    was_training = model.training
    model.eval()
    try:
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            trajectory_ids = batch_starts[:, 0]
            start_indices = batch_starts[:, 1]
            state_history = gather_windows(
                state_values, batch_starts, sequence_length
            )
            batch_error = torch.zeros((), device=device)
            for rollout_step in range(horizon):
                feature_starts = torch.stack(
                    (trajectory_ids, start_indices + rollout_step), dim=1
                )
                inputs = {}
                for input_name in model_input_names:
                    if input_name == "states_embedding":
                        inputs[input_name] = state_history
                    else:
                        inputs[input_name] = gather_windows(
                            feature_values[input_name], feature_starts, sequence_length
                        )
                delta = model(inputs)[:, -1, :]
                predicted_next = state_history[:, -1, :] + delta
                target_index = start_indices + rollout_step + sequence_length - 1
                target = target_values[trajectory_ids, target_index]
                error = (predicted_next - target).square()
                batch_error += error.sum()
                if rollout_step == horizon - 1:
                    final_squared_error += float(error.sum().cpu())
                state_history = torch.cat(
                    (state_history[:, 1:, :], predicted_next.unsqueeze(1)), dim=1
                )
            squared_error += float(batch_error.cpu())
            element_count += len(batch_starts) * horizon * target_values.shape[-1]
            rollout_count += len(batch_starts)
    finally:
        model.train(was_training)

    return {
        "rollout_mse": squared_error / element_count if element_count else np.nan,
        "rollout_final_mse": final_squared_error / (rollout_count * target_values.shape[-1]),
        "rollout_count": float(rollout_count),
    }
