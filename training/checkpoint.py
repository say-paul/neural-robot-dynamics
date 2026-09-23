from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch


FORMAT = "nerd_sequence_v3"


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    config: Mapping[str, object],
    input_dimensions: Mapping[str, int],
    output_dim: int,
    completed_steps: int,
    dataset_paths: Mapping[str, str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "training_config": dict(config),
        "input_dimensions": dict(input_dimensions),
        "output_dim": output_dim,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_steps": completed_steps,
        "dataset_paths": dict(dataset_paths),
    }
    if model.output_rms is not None:
        payload["output_mean"] = model.output_rms.mean.detach().cpu()
        payload["output_variance"] = model.output_rms.var.detach().cpu()
    torch.save(payload, path)