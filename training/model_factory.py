from __future__ import annotations

from collections.abc import Mapping

import torch

from models.models import ModelMixedInput
from utils.running_mean_std import RunningMeanStd


def build_model(
    input_dimensions: Mapping[str, int],
    output_dim: int,
    config: Mapping[str, object],
    device: str,
) -> ModelMixedInput:
    input_cfg = config["inputs"]
    low_dim = input_cfg.get("low_dim", [])
    missing = set(low_dim).difference(input_dimensions)
    if missing:
        raise ValueError(f"Training config requires missing input dimensions: {sorted(missing)}")
    sample = {
        name: torch.zeros((1, 1, input_dimensions[name]), device=device)
        for name in low_dim
    }
    model = ModelMixedInput(
        input_sample=sample,
        output_dim=output_dim,
        input_cfg=dict(input_cfg),
        network_cfg=dict(config["network"]),
        device=device,
    )
    if model.normalize_output:
        model.set_output_rms(RunningMeanStd(shape=(output_dim,), device=device))
    model.input_dimensions = dict(input_dimensions)
    model.training_config = dict(config)
    return model