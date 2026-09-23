from __future__ import annotations

import argparse

import torch

from training.checkpoint import save_checkpoint
from training.config import load_training_config
from training.dataset import load_trajectory_dataset
from training.model_factory import build_model
from training.sequence_trainer import SequenceTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a config-driven NeRD sequence model")
    parser.add_argument("--config", required=True, help="Training config path or robot profile")
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--test-dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()

    config = load_training_config(args.config)
    if args.batch_size is not None:
        config["optimization"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["optimization"]["learning_rate"] = args.learning_rate
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_dataset = load_trajectory_dataset(args.train_dataset, "train")
    validation_dataset = load_trajectory_dataset(args.validation_dataset, "validation")
    test_dataset = load_trajectory_dataset(args.test_dataset, "test") if args.test_dataset else None
    source_names = {"states_embedding": "states"}
    dimensions = {
        name: train_dataset[source_names.get(name, name)].shape[-1]
        for name in config["inputs"]["low_dim"]
    }
    model = build_model(dimensions, train_dataset["states"].shape[-1], config, device)
    trainer = SequenceTrainer(model, config, device)
    optimization = config["optimization"]
    loss = trainer.fit(train_dataset, args.steps, optimization["batch_size"], optimization["learning_rate"], args.seed)
    validation_mse = trainer.evaluate(validation_dataset, optimization["batch_size"])
    test_mse = trainer.evaluate(test_dataset, optimization["batch_size"]) if test_dataset else None
    save_checkpoint(args.checkpoint, model=model, optimizer=trainer.optimizer, config=config, input_dimensions=dimensions, output_dim=train_dataset["states"].shape[-1], completed_steps=args.steps, dataset_paths={"train": args.train_dataset, "validation": args.validation_dataset, **({"test": args.test_dataset} if args.test_dataset else {})})
    print(f"train_loss={loss:.6g} validation_mse={validation_mse:.6g} test_mse={test_mse}")


if __name__ == "__main__":
    main()