from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as functional

from training.dataset import gather_windows, valid_window_starts


class SequenceTrainer:
    """Train a sequence model from trajectory tensors without robot assumptions."""

    def __init__(self, model, config: Mapping[str, object], device: str):
        self.model = model
        self.config = config
        self.device = device
        self.sequence_length = int(config["sequence"]["length"])
        self.feature_sources = {"states_embedding": "states"}

    def _inputs(self, dataset, starts):
        inputs = {}
        for name in self.config["inputs"]["low_dim"]:
            source = self.feature_sources.get(name, name)
            if source not in dataset:
                raise ValueError(f"Dataset is missing model input {source!r}")
            values = torch.as_tensor(dataset[source], device=self.device)
            inputs[name] = gather_windows(values, starts, self.sequence_length)
        return inputs

    def _targets(self, dataset):
        states = torch.as_tensor(dataset["states"], device=self.device)
        next_states = torch.as_tensor(dataset["next_states"], device=self.device)
        return next_states - states

    def fit(self, dataset, epochs: int, batch_size: int, learning_rate: float, seed: int):
        targets = self._targets(dataset)
        valid = torch.as_tensor(dataset["valid"], device=self.device, dtype=torch.bool)
        starts = valid_window_starts(valid, self.sequence_length)
        if len(starts) == 0:
            raise ValueError("Dataset has no valid sequence windows")
        output = targets[valid]
        self.model.output_rms.mean = output.mean(dim=0)
        self.model.output_rms.var = output.var(dim=0, unbiased=False).clamp_min(1e-12)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.optimizer = optimizer
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.model.train()
        for _ in range(epochs):
            indices = torch.randint(len(starts), (batch_size,), generator=generator, device=self.device)
            batch_starts = starts[indices]
            prediction = self.model(self._inputs(dataset, batch_starts))
            target = gather_windows(targets, batch_starts, self.sequence_length)
            loss = functional.mse_loss(
                self.model.output_rms.normalize(prediction),
                self.model.output_rms.normalize(target),
            )
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = self.config["optimization"].get("gradient_norm")
            if gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_norm)
            optimizer.step()
        return float(loss.detach().cpu())

    @torch.no_grad()
    def evaluate(self, dataset, batch_size: int):
        targets = self._targets(dataset)
        valid = torch.as_tensor(dataset["valid"], device=self.device, dtype=torch.bool)
        starts = valid_window_starts(valid, self.sequence_length)
        if len(starts) == 0:
            raise ValueError("Dataset has no valid sequence windows")
        self.model.eval()
        total, count = 0.0, 0
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            prediction = self.model(self._inputs(dataset, batch_starts))
            target = gather_windows(targets, batch_starts, self.sequence_length)
            total += functional.mse_loss(prediction, target, reduction="sum").item()
            count += target.numel()
        return total / count if count else np.nan