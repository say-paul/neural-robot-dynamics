from __future__ import annotations

from collections.abc import Callable, Mapping

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
        self._dataset_tensors = {}

    def _tensor(self, dataset, name):
        cache = self._dataset_tensors.setdefault(id(dataset), {})
        if name not in cache:
            cache[name] = torch.as_tensor(dataset[name], device=self.device)
        return cache[name]

    def _inputs(self, dataset, starts):
        inputs = {}
        for name in self.config["inputs"]["low_dim"]:
            source = self.feature_sources.get(name, name)
            if source not in dataset:
                raise ValueError(f"Dataset is missing model input {source!r}")
            values = self._tensor(dataset, source)
            inputs[name] = gather_windows(values, starts, self.sequence_length)
        return inputs

    def _targets(self, dataset):
        states = self._tensor(dataset, "states")
        next_states = self._tensor(dataset, "next_states")
        return next_states - states

    def create_optimizer(self, learning_rate: float):
        return torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    @staticmethod
    def create_scheduler(optimizer, optimization: Mapping[str, object], total_steps: int):
        schedule = str(optimization.get("lr_schedule", "constant")).lower()
        if schedule == "constant":
            return None
        if schedule not in {"linear", "cosine"}:
            raise ValueError(f"Unsupported learning-rate schedule: {schedule!r}")
        start = float(optimization["learning_rate"])
        end = float(
            optimization.get(
                "learning_rate_end", optimization.get("lr_end", start)
            )
        )
        if total_steps < 1:
            raise ValueError("total_steps must be positive")

        def multiplier(step: int) -> float:
            progress = min(max(step, 0), total_steps) / total_steps
            if schedule == "cosine":
                progress = (1.0 - np.cos(np.pi * progress)) / 2.0
            return (start + (end - start) * progress) / start

        return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

    def fit(
        self,
        dataset,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
        *,
        optimizer=None,
        scheduler=None,
        start_step: int = 0,
        generator_state=None,
        on_step: Callable[[int, float, float, torch.Tensor], None] | None = None,
    ):
        targets = self._targets(dataset)
        valid = self._tensor(dataset, "valid").to(dtype=torch.bool)
        starts = valid_window_starts(valid, self.sequence_length)
        if len(starts) == 0:
            raise ValueError("Dataset has no valid sequence windows")
        output = targets[valid]
        self.model.output_rms.mean = output.mean(dim=0)
        self.model.output_rms.var = output.var(dim=0, unbiased=False).clamp_min(1e-12)
        if start_step < 0 or start_step > epochs:
            raise ValueError("start_step must be between zero and epochs")
        optimizer = optimizer or self.create_optimizer(learning_rate)
        self.optimizer = optimizer
        if scheduler is not None:
            self.scheduler = scheduler
        generator = torch.Generator(device=self.device).manual_seed(seed)
        if generator_state is not None:
            generator.set_state(generator_state.to(device="cpu"))
        self.model.train()
        last_loss = float("nan")
        for step in range(start_step, epochs):
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
            if scheduler is not None:
                scheduler.step()
            last_loss = float(loss.detach().cpu())
            if on_step is not None:
                on_step(
                    step + 1,
                    last_loss,
                    float(optimizer.param_groups[0]["lr"]),
                    generator.get_state(),
                )
                self.model.train()
        return {
            "loss": last_loss,
            "completed_steps": epochs,
            "generator_state": generator.get_state(),
        }

    @torch.no_grad()
    def evaluate(self, dataset, batch_size: int, max_windows: int | None = None):
        targets = self._targets(dataset)
        valid = self._tensor(dataset, "valid").to(dtype=torch.bool)
        starts = valid_window_starts(valid, self.sequence_length)
        if len(starts) == 0:
            raise ValueError("Dataset has no valid sequence windows")
        if max_windows is not None:
            if max_windows < 1:
                raise ValueError("max_windows must be positive")
            starts = starts[:max_windows]
        self.model.eval()
        total, count = 0.0, 0
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            prediction = self.model(self._inputs(dataset, batch_starts))
            target = gather_windows(targets, batch_starts, self.sequence_length)
            total += functional.mse_loss(prediction, target, reduction="sum").item()
            count += target.numel()
        return total / count if count else np.nan