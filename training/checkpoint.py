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
    scheduler=None,
    generator_state=None,
    metrics: Mapping[str, float] | None = None,
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
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if generator_state is not None:
        payload["generator_state"] = generator_state.cpu()
    if metrics:
        payload["metrics"] = dict(metrics)
    if model.output_rms is not None:
        payload["output_mean"] = model.output_rms.mean.detach().cpu()
        payload["output_variance"] = model.output_rms.var.detach().cpu()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer=None,
    scheduler=None,
    config: Mapping[str, object] | None = None,
    input_dimensions: Mapping[str, int] | None = None,
    output_dim: int | None = None,
):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise RuntimeError(
            f"Checkpoint {path} has format {payload.get('format')!r}; expected {FORMAT!r}"
        )
    if input_dimensions is not None and dict(payload.get("input_dimensions", {})) != dict(input_dimensions):
        raise RuntimeError("Checkpoint input dimensions are incompatible with the dataset")
    if output_dim is not None and payload.get("output_dim") != output_dim:
        raise RuntimeError("Checkpoint output dimension is incompatible with the dataset")
    if config is not None:
        for key in ("schema_version", "inputs", "sequence", "network", "contact", "projection"):
            if payload.get("training_config", {}).get(key) != config.get(key):
                raise RuntimeError(f"Checkpoint training configuration differs in {key!r}")

    if model.output_rms is not None and "output_mean" in payload:
        model.output_rms.mean = payload["output_mean"].to(model.output_rms.mean.device)
        model.output_rms.var = payload["output_variance"].to(model.output_rms.var.device)
    model.load_state_dict(payload["state_dict"])
    if optimizer is not None:
        if "optimizer_state_dict" not in payload:
            raise RuntimeError("Checkpoint does not contain optimizer state")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload