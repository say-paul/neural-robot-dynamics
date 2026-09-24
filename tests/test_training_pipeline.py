import numpy as np
import pytest
import torch
from torch import nn

from training.checkpoint import load_checkpoint, save_checkpoint
from training.dataset import validate_trajectory_compatibility
from training.evaluation import evaluate_rollouts
from training.sequence_trainer import SequenceTrainer


class OutputStats:
    def __init__(self, width):
        self.mean = torch.zeros(width)
        self.var = torch.ones(width)

    def normalize(self, value):
        return value


class DummyModel(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(width))
        self.output_rms = OutputStats(width)

    def forward(self, inputs):
        return self.bias + torch.zeros_like(inputs["states_embedding"])


def training_config():
    return {
        "schema_version": 1,
        "inputs": {"low_dim": ["states_embedding", "joint_f"]},
        "sequence": {"length": 2},
        "network": {"test": True},
        "contact": {"enabled": False},
        "projection": {"enabled": False},
        "optimization": {
            "learning_rate": 1.0,
            "learning_rate_end": 0.1,
            "lr_schedule": "linear",
            "gradient_norm": 1.0,
        },
    }


def trajectory_dataset():
    states = np.zeros((1, 5, 2), dtype=np.float32)
    return {
        "states": states,
        "next_states": states.copy(),
        "joint_f": np.zeros((1, 5, 1), dtype=np.float32),
        "valid": np.ones((1, 5), dtype=bool),
        "_metadata": {
            "training_schema_version": 1,
            "training_inputs": '["states_embedding", "joint_f"]',
            "sequence_length": 2,
            "robot_id": "test",
        },
    }


def test_scheduler_reaches_configured_end_learning_rate():
    model = DummyModel(2)
    config = training_config()
    trainer = SequenceTrainer(model, config, "cpu")
    optimizer = trainer.create_optimizer(1.0)
    scheduler = trainer.create_scheduler(optimizer, config["optimization"], 4)

    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates[0] == pytest.approx(1.0)
    assert rates[-1] == pytest.approx(0.1)


def test_sequence_trainer_runs_scheduled_steps_and_callback():
    model = DummyModel(2)
    config = training_config()
    trainer = SequenceTrainer(model, config, "cpu")
    optimizer = trainer.create_optimizer(1.0)
    scheduler = trainer.create_scheduler(optimizer, config["optimization"], 3)
    observed_steps = []

    result = trainer.fit(
        trajectory_dataset(),
        epochs=3,
        batch_size=1,
        learning_rate=1.0,
        seed=7,
        optimizer=optimizer,
        scheduler=scheduler,
        on_step=lambda step, loss, learning_rate, generator_state: observed_steps.append(
            (step, learning_rate)
        ),
    )

    assert result["completed_steps"] == 3
    assert [step for step, _ in observed_steps] == [1, 2, 3]
    assert observed_steps[-1][1] == pytest.approx(0.1)


def test_checkpoint_round_trip_restores_model_and_optimizer(tmp_path):
    config = training_config()
    first = DummyModel(2)
    optimizer = torch.optim.Adam(first.parameters(), lr=0.01)
    first.bias.data[:] = 3.0
    optimizer.step()
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model=first,
        optimizer=optimizer,
        config=config,
        input_dimensions={"states_embedding": 2, "joint_f": 1},
        output_dim=2,
        completed_steps=7,
        dataset_paths={"train": "train.hdf5"},
    )

    second = DummyModel(2)
    restored_optimizer = torch.optim.Adam(second.parameters(), lr=0.01)
    payload = load_checkpoint(
        path,
        model=second,
        optimizer=restored_optimizer,
        config=config,
        input_dimensions={"states_embedding": 2, "joint_f": 1},
        output_dim=2,
    )

    assert payload["completed_steps"] == 7
    assert torch.equal(first.bias, second.bias)


def test_dataset_compatibility_rejects_wrong_input_schema():
    config = training_config()
    dataset = trajectory_dataset()
    dataset["_metadata"]["training_inputs"] = '["states_embedding"]'

    with pytest.raises(RuntimeError, match="inputs"):
        validate_trajectory_compatibility(dataset, config, path="bad.hdf5")


def test_rollout_evaluation_reports_zero_drift_for_static_model():
    config = training_config()
    metrics = evaluate_rollouts(
        DummyModel(2),
        trajectory_dataset(),
        config,
        "cpu",
        horizon=2,
    )

    assert metrics["rollout_mse"] == pytest.approx(0.0)
    assert metrics["rollout_final_mse"] == pytest.approx(0.0)
    assert metrics["rollout_count"] == 2.0
