import argparse
import os
import time
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as functional

from envs.neural_environment import NeuralEnvironment
from models.models import ModelMixedInput


SEQUENCE_LENGTH = 10


def model_config():
    return {
        "encoder": {
            "low_dim": {
                "activation": "relu",
                "layer_sizes": [],
                "layernorm": False,
            }
        },
        "model": {
            "mlp": {
                "activation": "relu",
                "layer_sizes": [64],
                "layernorm": False,
            }
        },
        "transformer": {
            "n_layer": 6,
            "n_head": 12,
            "n_embd": 192,
            "block_size": 32,
            "bias": False,
            "dropout": 0.0,
        },
        "normalize_input": False,
        "normalize_output": False,
        "output_tanh": False,
    }


def create_model(state_dim, joint_f_dim, device):
    sample = {
        "states_embedding": torch.zeros((1, 1, state_dim), device=device),
        "joint_f": torch.zeros((1, 1, joint_f_dim), device=device),
    }
    return ModelMixedInput(
        input_sample=sample,
        output_dim=state_dim,
        input_cfg={"low_dim": ["states_embedding", "joint_f"]},
        network_cfg=model_config(),
        device=device,
    )


def configure_render(args):
    if args.render_backend != "rerun":
        return {}
    from envs.newton_envs import RenderMode

    os.environ.setdefault("MUJOCO_GL", "egl")
    return {
        "render_mode": RenderMode.RERUN,
        "rerun_render_settings": {
            "grpc_port": args.grpc_port,
            "web_port": args.web_port,
            "browser_host": args.browser_host,
            "native_camera": args.rerun_view == "camera",
        },
    }


def valid_transition_mask(was_terminated, action_saturation_mask, is_terminated):
    return ~(was_terminated | action_saturation_mask | is_terminated)


def valid_window_starts(valid, sequence_length=SEQUENCE_LENGTH):
    valid = torch.as_tensor(valid, dtype=torch.bool)
    if valid.ndim != 2:
        raise ValueError("valid transition mask must have shape (trajectories, steps)")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if valid.shape[1] < sequence_length:
        return torch.empty((0, 2), dtype=torch.long, device=valid.device)
    return valid.unfold(1, sequence_length, 1).all(dim=-1).nonzero()


def gather_windows(values, starts, sequence_length=SEQUENCE_LENGTH):
    offsets = torch.arange(sequence_length, device=starts.device)
    return values[starts[:, :1], starts[:, 1:] + offsets]


def collect_dataset(args, device, split, seed):
    render = args.render and split in args.render_splits
    env = NeuralEnvironment(
        env_name="Robot",
        num_envs=args.num_envs,
        newton_env_cfg={
            "robot_spec": args.robot_id,
            "seed": seed,
            "random_reset": not args.default_pose,
            **(configure_render(args) if render else {}),
        },
        default_env_mode="ground-truth",
        device=device,
        render=render,
    )
    try:
        env.reset()
        state = env.states.clone()
        generator = torch.Generator(device=env.torch_device).manual_seed(seed + 1)
        states, actions, joint_forces, next_states, valid = [], [], [], [], []
        rejected_transitions = 0
        for step in range(args.horizon):
            was_terminated = env.terminated.clone()
            action = (
                torch.rand(
                    (args.num_envs, env.action_dim),
                    generator=generator,
                    device=env.torch_device,
                )
                * 2.0
                - 1.0
            ) * args.action_scale
            next_state = env.step(action, env_mode="ground-truth").clone()
            accepted = valid_transition_mask(
                was_terminated,
                env.action_saturation_mask,
                env.terminated,
            )
            rejected_transitions += int((~accepted).sum().item())
            states.append(state.cpu())
            actions.append(action.cpu())
            joint_forces.append(env.joint_f.cpu())
            next_states.append(next_state.cpu())
            valid.append(accepted.cpu())
            if args.diagnostics and render:
                env.log_robot_diagnostics(step, action)
            if render:
                env.render()
            state = next_state

        valid_tensor = torch.stack(valid, dim=1)
        if len(valid_window_starts(valid_tensor)) == 0:
            raise RuntimeError(
                f"No physically valid {SEQUENCE_LENGTH}-step robot sequences were "
                "collected; increase --horizon, reduce --action-scale, or use more "
                "parallel environments."
            )
        dataset = {
            "states": torch.stack(states, dim=1).numpy(),
            "actions": torch.stack(actions, dim=1).numpy(),
            "joint_f": torch.stack(joint_forces, dim=1).numpy(),
            "next_states": torch.stack(next_states, dim=1).numpy(),
            "valid": valid_tensor.numpy(),
            "rejected_transitions": rejected_transitions,
        }
        if not all(np.isfinite(value).all() for value in dataset.values() if isinstance(value, np.ndarray)):
            raise RuntimeError("Collected robot data contains non-finite values")
        return dataset, env.state_dim, env.joint_f_dim, env
    except Exception:
        env.close()
        raise


def write_dataset(path, dataset, args, split, seed, state_dim, joint_f_dim):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w", libver="latest") as file:
        group = file.create_group("data")
        group.attrs["mode"] = "trajectory"
        group.attrs["num_trajectories"] = dataset["states"].shape[0]
        group.attrs["trajectory_length"] = dataset["states"].shape[1]
        group.attrs["total_transitions"] = int(dataset["valid"].sum())
        group.attrs["sequence_length"] = SEQUENCE_LENGTH
        group.attrs["robot_id"] = args.robot_id
        group.attrs["split"] = split
        group.attrs["seed"] = seed
        group.attrs["state_dim"] = state_dim
        group.attrs["joint_f_dim"] = joint_f_dim
        group.attrs["state_target"] = "next_states - states"
        group.attrs["rejected_transitions"] = dataset["rejected_transitions"]
        for name, value in dataset.items():
            if name == "rejected_transitions":
                continue
            group.create_dataset(name, data=value, compression="gzip")
    with h5py.File(path, "r", swmr=True, libver="latest") as file:
        data_group: Any = file["data"]
        if data_group["states"].shape != dataset["states"].shape:
            raise RuntimeError("HDF5 dataset validation failed")
        if data_group.attrs["split"] != split:
            raise RuntimeError("HDF5 dataset split metadata validation failed")


def load_dataset(path, expected_split=None):
    with h5py.File(path, "r", swmr=True, libver="latest") as file:
        if "data" not in file:
            raise RuntimeError(f"Dataset {path} does not contain a data group")
        group: Any = file["data"]
        if group.attrs.get("mode") != "trajectory":
            raise RuntimeError(
                f"Dataset {path} is not a trajectory dataset; regenerate it with "
                "the Transformer training script"
            )
        split = group.attrs.get("split", "unknown")
        if expected_split is not None and split not in ("unknown", expected_split):
            raise RuntimeError(
                f"Dataset {path} has split {split!r}; expected {expected_split!r}"
            )
        dataset: dict[str, np.ndarray] = {
            name: np.asarray(group[name][()]) for name in group.keys()
        }

    required = {"states", "actions", "joint_f", "next_states", "valid"}
    missing = required.difference(dataset)
    if missing:
        raise RuntimeError(f"Dataset {path} is missing fields: {sorted(missing)}")
    if not all(np.isfinite(value).all() for value in dataset.values()):
        raise RuntimeError(f"Dataset {path} contains non-finite values")
    trajectory_shape = dataset["states"].shape[:2]
    if len(dataset["states"].shape) != 3 or not all(trajectory_shape):
        raise RuntimeError(f"Dataset {path} has invalid or empty trajectories")
    for name in ("actions", "joint_f", "next_states"):
        if dataset[name].shape[:2] != trajectory_shape:
            raise RuntimeError(f"Dataset {path} has inconsistent {name} trajectories")
    if dataset["valid"].shape != trajectory_shape:
        raise RuntimeError(f"Dataset {path} has an invalid transition mask")
    return dataset


def resume_training(model, optimizer, args, state_dim, joint_f_dim, device):
    if args.resume_checkpoint is None:
        return 0

    checkpoint = torch.load(
        args.resume_checkpoint, map_location=device, weights_only=True
    )
    expected_input_cfg = {"low_dim": ["states_embedding", "joint_f"]}
    if checkpoint.get("robot_id") != args.robot_id:
        raise RuntimeError(
            f"Resume checkpoint robot_id={checkpoint.get('robot_id')!r} does not "
            f"match --robot-id={args.robot_id!r}"
        )
    if checkpoint.get("state_dim") != state_dim or checkpoint.get("joint_f_dim") != joint_f_dim:
        raise RuntimeError("Resume checkpoint input dimensions do not match the datasets")
    if checkpoint.get("input_cfg") != expected_input_cfg:
        raise RuntimeError("Resume checkpoint input configuration is incompatible")
    if checkpoint.get("network_cfg") != model_config():
        raise RuntimeError("Resume checkpoint architecture does not match model_config()")
    if checkpoint.get("num_states_history") != SEQUENCE_LENGTH:
        raise RuntimeError("Resume checkpoint history length is incompatible")
    if checkpoint.get("prediction_type") != "relative":
        raise RuntimeError("Resume checkpoint must use relative state prediction")

    model.load_state_dict(checkpoint["state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("completed_epochs", 0))


def evaluate_model(model, dataset, device, batch_size):
    states = torch.from_numpy(dataset["states"]).to(device)
    joint_f = torch.from_numpy(dataset["joint_f"]).to(device)
    targets = torch.from_numpy(dataset["next_states"] - dataset["states"]).to(device)
    starts = valid_window_starts(dataset["valid"]).to(device)
    if len(starts) == 0:
        raise ValueError(f"Dataset has no valid {SEQUENCE_LENGTH}-step sequences")

    squared_error = 0.0
    element_count = 0
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset:offset + batch_size]
            state_windows = gather_windows(states, batch_starts)
            joint_f_windows = gather_windows(joint_f, batch_starts)
            target_windows = gather_windows(targets, batch_starts)
            prediction = model(
                {
                    "states_embedding": state_windows,
                    "joint_f": joint_f_windows,
                }
            )
            squared_error += functional.mse_loss(
                prediction, target_windows, reduction="sum"
            ).item()
            element_count += target_windows.numel()
    return squared_error / element_count


def train_and_test(train_dataset, validation_dataset, test_dataset,
                   state_dim, joint_f_dim, args, device):
    train_states = torch.from_numpy(train_dataset["states"]).to(device)
    train_joint_f = torch.from_numpy(train_dataset["joint_f"]).to(device)
    train_targets = torch.from_numpy(
        train_dataset["next_states"] - train_dataset["states"]
    ).to(device)
    train_starts = valid_window_starts(train_dataset["valid"]).to(device)
    if len(train_starts) == 0:
        raise ValueError(f"Training dataset has no valid {SEQUENCE_LENGTH}-step sequences")

    model = create_model(state_dim, joint_f_dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = resume_training(
        model, optimizer, args, state_dim, joint_f_dim, device
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 10)
    model.train()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        indices = torch.randint(
            len(train_starts), (args.batch_size,), generator=generator, device=device
        )
        starts = train_starts[indices]
        state_windows = gather_windows(train_states, starts)
        joint_f_windows = gather_windows(train_joint_f, starts)
        target_windows = gather_windows(train_targets, starts)
        prediction = model(
            {
                "states_embedding": state_windows,
                "joint_f": joint_f_windows,
            }
        )
        loss = functional.mse_loss(prediction, target_windows)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.render and "train" in args.render_splits and args.render_backend == "rerun":
            import rerun as rr

            rr.set_time("training_epoch", sequence=epoch)
            rr.log("robot/training/mse", rr.Scalars(float(loss.detach().cpu())))

    model.eval()
    validation_mse = evaluate_model(
        model, validation_dataset, device, args.batch_size
    )
    if not np.isfinite(validation_mse):
        raise RuntimeError("Validation produced a non-finite loss")

    checkpoint = {
        "format": "nerd_robot_v2",
        "robot_id": args.robot_id,
        "state_dim": state_dim,
        "joint_f_dim": joint_f_dim,
        "input_cfg": {"low_dim": ["states_embedding", "joint_f"]},
        "network_cfg": model_config(),
        "prediction_type": "relative",
        "solver_name": "TransformerNeuralSolver",
        "num_states_history": SEQUENCE_LENGTH,
        "train_dataset_path": os.path.abspath(args.train_dataset_path),
        "validation_dataset_path": os.path.abspath(args.validation_dataset_path),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_epochs": start_epoch + args.epochs,
    }
    if test_dataset is not None:
        test_mse = evaluate_model(model, test_dataset, device, args.batch_size)
        if not np.isfinite(test_mse):
            raise RuntimeError("Test produced a non-finite loss")
        checkpoint["test_dataset_path"] = os.path.abspath(args.test_dataset_path)
    else:
        test_mse = None
    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint_path)), exist_ok=True)
    torch.save(checkpoint, args.checkpoint_path)
    loaded = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    reloaded_model = create_model(loaded["state_dim"], loaded["joint_f_dim"], device)
    reloaded_model.load_state_dict(loaded["state_dict"])
    reloaded_model.eval()
    validation_states = torch.from_numpy(validation_dataset["states"]).to(device)
    validation_joint_f = torch.from_numpy(validation_dataset["joint_f"]).to(device)
    validation_start = valid_window_starts(validation_dataset["valid"])[0:1].to(device)
    with torch.no_grad():
        reloaded_prediction = reloaded_model.evaluate(
            {
                "states_embedding": gather_windows(validation_states, validation_start),
                "joint_f": gather_windows(validation_joint_f, validation_start),
            }
        )
    if not torch.isfinite(reloaded_prediction).all():
        raise RuntimeError("Reloaded checkpoint produced non-finite predictions")
    return validation_mse, test_mse


def default_dataset_paths(robot_id):
    prefix = f"outputs/{robot_id}"
    return {
        "train": f"{prefix}_train.hdf5",
        "validation": f"{prefix}_validation.hdf5",
        "test": f"{prefix}_test.hdf5",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate robot HDF5 datasets and smoke-train a NeRD dynamics model"
    )
    parser.add_argument("--robot-id", default="so101")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation", "test"),
        help="Dataset splits to generate (default: train validation test)",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate the selected HDF5 splits without training a model",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Train and evaluate from existing HDF5 splits without overwriting them",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-pose", action="store_true")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--resume-checkpoint",
        help="Resume compatible model weights and optimizer state for additional epochs",
    )
    parser.add_argument("--train-dataset-path")
    parser.add_argument("--validation-dataset-path")
    parser.add_argument("--test-dataset-path")
    parser.add_argument(
        "--dataset-path",
        help="Deprecated alias for --train-dataset-path",
    )
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--render-backend", choices=("opengl", "rerun"), default="opengl")
    parser.add_argument("--rerun-view", choices=("3d", "camera"), default="camera")
    parser.add_argument("--grpc-port", type=int, default=9876)
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--browser-host", default="localhost")
    parser.add_argument(
        "--render-splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train",),
        help="Splits that should be rendered when --render is enabled",
    )
    parser.add_argument("--keep-open", action="store_true")
    args = parser.parse_args()

    paths = default_dataset_paths(args.robot_id)
    args.train_dataset_path = args.train_dataset_path or args.dataset_path or paths["train"]
    args.validation_dataset_path = args.validation_dataset_path or paths["validation"]
    args.test_dataset_path = args.test_dataset_path or paths["test"]
    args.checkpoint_path = args.checkpoint_path or f"outputs/{args.robot_id}_nerd_model.pt"
    if args.generate_only and args.skip_generation:
        parser.error("--generate-only and --skip-generation cannot be used together")
    if not args.generate_only and not {"train", "validation"}.issubset(args.splits):
        parser.error("training requires both train and validation in --splits")
    if args.keep_open and not args.render:
        parser.error("--keep-open requires --render")
    if args.keep_open and len(set(args.render_splits).intersection(args.splits)) != 1:
        parser.error("--keep-open requires exactly one rendered dataset split")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    dimensions = None
    env = None
    viewer_env = None
    try:
        if not args.skip_generation:
            split_seeds = {split: args.seed + index for index, split in enumerate(args.splits)}
            for split in args.splits:
                dataset, state_dim, joint_f_dim, env = collect_dataset(
                    args, device, split, split_seeds[split]
                )
                path = getattr(args, f"{split}_dataset_path")
                write_dataset(path, dataset, args, split, split_seeds[split], state_dim, joint_f_dim)
                dimensions = (state_dim, joint_f_dim)
                print(
                    f"generated split={split} trajectories={dataset['states'].shape[0]} "
                    f"steps={dataset['states'].shape[1]} valid={int(dataset['valid'].sum())} "
                    f"rejected={dataset['rejected_transitions']} "
                    f"dataset={path}",
                    flush=True,
                )
                if args.keep_open and split in args.render_splits:
                    viewer_env = env
                else:
                    env.close()
                env = None

        if args.generate_only:
            env = viewer_env
            if env is not None:
                print("Viewer is running. Press Ctrl+C to exit.", flush=True)
                while True:
                    time.sleep(1.0)
            return
        train_dataset = load_dataset(args.train_dataset_path, "train")
        validation_dataset = load_dataset(args.validation_dataset_path, "validation")
        test_dataset = (
            load_dataset(args.test_dataset_path, "test")
            if "test" in args.splits
            else None
        )
        dimensions = (
            train_dataset["states"].shape[-1],
            train_dataset["joint_f"].shape[-1],
        )
        if validation_dataset["states"].shape[-1] != train_dataset["states"].shape[-1] or (
            validation_dataset["joint_f"].shape[-1] != train_dataset["joint_f"].shape[-1]
        ):
            raise RuntimeError("Train and validation datasets have incompatible dimensions")
        if test_dataset is not None and (
            test_dataset["states"].shape[-1] != train_dataset["states"].shape[-1]
            or test_dataset["joint_f"].shape[-1] != train_dataset["joint_f"].shape[-1]
        ):
            raise RuntimeError("Train and test datasets have incompatible dimensions")
        validation_mse, test_mse = train_and_test(
            train_dataset,
            validation_dataset,
            test_dataset,
            dimensions[0],
            dimensions[1],
            args,
            device,
        )
        test_result = "n/a" if test_mse is None else f"{test_mse:.6g}"
        print(
            f"robot={args.robot_id} train={args.train_dataset_path} "
            f"validation={args.validation_dataset_path} "
            f"test={args.test_dataset_path if test_dataset is not None else 'not-generated'} "
            f"checkpoint={args.checkpoint_path} validation_mse={validation_mse:.6g} "
            f"test_mse={test_result}",
            flush=True,
        )
        if args.keep_open and args.render:
            env = viewer_env
            print("Viewer is running. Press Ctrl+C to exit.", flush=True)
            while True:
                time.sleep(1.0)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()