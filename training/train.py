from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import load_training_config
from training.dataset import load_trajectory_dataset, validate_trajectory_compatibility
from training.evaluation import evaluate_rollouts
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
    parser.add_argument("--resume", help="Resume from a compatible checkpoint")
    parser.add_argument("--log-dir", help="TensorBoard log directory")
    parser.add_argument("--validation-interval", type=int, default=5_000)
    parser.add_argument("--validation-windows", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--print-interval", type=int, default=1_000)
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=0,
        help="Evaluate autoregressive drift for this many steps after training; zero disables it",
    )
    parser.add_argument("--rollout-windows", type=int)
    args = parser.parse_args()

    config = load_training_config(args.config)
    if args.batch_size is not None:
        config["optimization"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["optimization"]["learning_rate"] = args.learning_rate
    if args.steps < 1:
        parser.error("--steps must be positive")
    if (
        args.validation_interval < 1
        or args.checkpoint_interval < 1
        or args.print_interval < 1
    ):
        parser.error("validation, checkpoint, and print intervals must be positive")
    if args.rollout_horizon < 0 or (args.rollout_windows is not None and args.rollout_windows < 1):
        parser.error("rollout horizon must be non-negative and rollout windows positive")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_dataset = load_trajectory_dataset(args.train_dataset, "train")
    validation_dataset = load_trajectory_dataset(args.validation_dataset, "validation")
    test_dataset = load_trajectory_dataset(args.test_dataset, "test") if args.test_dataset else None
    validate_trajectory_compatibility(train_dataset, config, path=args.train_dataset)
    validate_trajectory_compatibility(validation_dataset, config, path=args.validation_dataset)
    if test_dataset is not None:
        validate_trajectory_compatibility(test_dataset, config, path=args.test_dataset)
    source_names = {"states_embedding": "states"}
    dimensions = {
        name: train_dataset[source_names.get(name, name)].shape[-1]
        for name in config["inputs"]["low_dim"]
    }
    model = build_model(dimensions, train_dataset["states"].shape[-1], config, device)
    trainer = SequenceTrainer(model, config, device)
    optimization = config["optimization"]
    optimizer = trainer.create_optimizer(optimization["learning_rate"])
    scheduler = trainer.create_scheduler(optimizer, optimization, args.steps)
    start_step = 0
    generator_state = None
    if args.resume:
        payload = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            input_dimensions=dimensions,
            output_dim=train_dataset["states"].shape[-1],
        )
        start_step = int(payload.get("completed_steps", 0))
        generator_state = payload.get("generator_state")
        if start_step > args.steps:
            raise RuntimeError("Resume checkpoint is beyond the requested --steps")

    writer = SummaryWriter(args.log_dir) if args.log_dir else None
    latest_path = Path(args.checkpoint).with_suffix(".latest.pt")
    metrics = {}

    def on_step(step, loss, learning_rate, generator_state):
        if writer is not None:
            writer.add_scalar("loss/train_normalized", loss, step)
            writer.add_scalar("optimizer/learning_rate", learning_rate, step)
        should_validate = step % args.validation_interval == 0 or step == args.steps
        if should_validate:
            validation_mse = trainer.evaluate(
                validation_dataset,
                optimization["batch_size"],
                max_windows=args.validation_windows,
            )
            metrics["validation_mse"] = validation_mse
            if writer is not None:
                writer.add_scalar("loss/validation_mse", validation_mse, step)
        if step % args.print_interval == 0 or should_validate:
            validation_text = (
                ""
                if "validation_mse" not in metrics
                else f" validation_mse={metrics['validation_mse']:.6g}"
            )
            print(
                f"step={step} learning_rate={learning_rate:.6g} "
                f"train_mse={loss:.6g}{validation_text}",
                flush=True,
            )
        if step % args.checkpoint_interval == 0 or step == args.steps:
            save_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                generator_state=generator_state,
                config=config,
                input_dimensions=dimensions,
                output_dim=train_dataset["states"].shape[-1],
                completed_steps=step,
                dataset_paths={"train": args.train_dataset, "validation": args.validation_dataset},
                metrics=metrics,
            )
            print(f"checkpoint={latest_path} step={step}", flush=True)

    result = trainer.fit(
        train_dataset,
        args.steps,
        optimization["batch_size"],
        optimization["learning_rate"],
        args.seed,
        optimizer=optimizer,
        scheduler=scheduler,
        start_step=start_step,
        generator_state=generator_state,
        on_step=on_step,
    )
    validation_mse = trainer.evaluate(
        validation_dataset,
        optimization["batch_size"],
        max_windows=args.validation_windows,
    )
    test_mse = trainer.evaluate(test_dataset, optimization["batch_size"]) if test_dataset else None
    rollout_metrics = {}
    rollout_dataset = test_dataset if test_dataset is not None else validation_dataset
    if args.rollout_horizon:
        rollout_metrics = evaluate_rollouts(
            model,
            rollout_dataset,
            config,
            device,
            horizon=args.rollout_horizon,
            max_rollouts=args.rollout_windows,
            batch_size=optimization["batch_size"],
        )
        print(
            f"rollout_mse={rollout_metrics['rollout_mse']:.6g} "
            f"rollout_final_mse={rollout_metrics['rollout_final_mse']:.6g} "
            f"rollouts={int(rollout_metrics['rollout_count'])}",
            flush=True,
        )
    if writer is not None:
        writer.add_scalar("loss/validation_mse_final", validation_mse, args.steps)
        if test_mse is not None:
            writer.add_scalar("loss/test_mse", test_mse, args.steps)
        for name, value in rollout_metrics.items():
            writer.add_scalar(f"rollout/{name}", value, args.steps)
        writer.flush()
        writer.close()
    save_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        generator_state=result["generator_state"],
        config=config,
        input_dimensions=dimensions,
        output_dim=train_dataset["states"].shape[-1],
        completed_steps=args.steps,
        dataset_paths={"train": args.train_dataset, "validation": args.validation_dataset, **({"test": args.test_dataset} if args.test_dataset else {})},
        metrics={
            "validation_mse": validation_mse,
            **({"test_mse": test_mse} if test_mse is not None else {}),
            **rollout_metrics,
        },
    )
    print(f"train_loss={result['loss']:.6g} validation_mse={validation_mse:.6g} test_mse={test_mse}")


if __name__ == "__main__":
    main()