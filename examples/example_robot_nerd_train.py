import argparse
import json
import os
import time
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.tensorboard.writer import SummaryWriter

from envs.neural_environment import NeuralEnvironment
from models.models import ModelMixedInput
from training.config import load_training_config
from training.contact_backends import mujoco_warp_features
from utils.running_mean_std import RunningMeanStd


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
        "normalize_output": True,
        "output_tanh": False,
    }


def create_model(state_dim, joint_f_dim, device):
    sample = {
        "states_embedding": torch.zeros((1, 1, state_dim), device=device),
        "joint_f": torch.zeros((1, 1, joint_f_dim), device=device),
    }
    model = ModelMixedInput(
        input_sample=sample,
        output_dim=state_dim,
        input_cfg={"low_dim": ["states_embedding", "joint_f"]},
        network_cfg=model_config(),
        device=device,
    )
    set_output_statistics(
        model,
        torch.zeros(state_dim, device=device),
        torch.ones(state_dim, device=device),
    )
    return model


def set_output_statistics(model, mean, variance):
    output_rms = RunningMeanStd(shape=tuple(mean.shape), device=mean.device)
    output_rms.mean = mean.detach().clone()
    output_rms.var = variance.detach().clone().clamp_min(1e-12)
    output_rms.count = 1.0
    model.set_output_rms(output_rms)


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


def contact_transition_mask(contact_features, contact_config):
    if not contact_config["enabled"]:
        return torch.ones(
            contact_features.shape[0], dtype=torch.bool, device=contact_features.device
        )
    if contact_config.get("reject_nonfinite", True) and not torch.isfinite(contact_features).all():
        finite = torch.isfinite(contact_features).all(dim=1)
    else:
        finite = torch.ones(
            contact_features.shape[0], dtype=torch.bool, device=contact_features.device
        )
    slot_width = 8
    slots = contact_features.view(contact_features.shape[0], -1, slot_width)
    active = slots[..., 0] > 0.5
    separation = slots[..., 1]
    penetration = active & (
        separation < float(contact_config.get("reject_separation_below", -float("inf")))
    )
    return finite & ~penetration.any(dim=1)


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


def validate_dynamics_signal(dataset, velocity_dim):
    valid = dataset["valid"].astype(bool)
    state_delta = (dataset["next_states"] - dataset["states"])[valid]
    if len(state_delta) == 0:
        raise RuntimeError("Collected dataset has no valid state transitions")
    velocity_delta = state_delta[:, -velocity_dim:]
    if float(np.max(np.abs(velocity_delta))) <= 1e-8:
        raise RuntimeError(
            "Collected velocity never changes; ground-truth controls are not "
            "affecting the dynamics. Check the simulator integrator before training."
        )


def collect_dataset(args, device, split, seed):
    render = args.render and split in args.render_splits
    env = NeuralEnvironment(
        env_name="Robot",
        num_envs=args.num_envs,
        newton_env_cfg={
            "robot_spec": args.robot_id,
            "seed": seed,
            "random_reset": not args.default_pose,
            "env_offset": (
                (args.env_spacing, 0.0, args.env_spacing)
                if args.env_spacing > 0.0
                else (0.0, 0.0, 0.0)
            ),
            **(configure_render(args) if render else {}),
        },
        default_env_mode="ground-truth",
        device=device,
        render=render,
    )
    try:
        contact_config = args.training_config_data["contact"]
        collect_contacts = bool(contact_config["enabled"])
        if collect_contacts and contact_config["backend"] != "mujoco_warp":
            raise ValueError(f"Unsupported contact backend: {contact_config['backend']}")
        num_trajectories = args.num_trajectories or args.num_envs
        generator = torch.Generator(device=env.torch_device).manual_seed(seed + 1)
        batches = {
            "states": [],
            "actions": [],
            "joint_f": [],
            "next_states": [],
            "valid": [],
        }
        if collect_contacts:
            batches["self_contact"] = []
        rejected_transitions = 0
        collected_trajectories = 0
        next_progress = min(args.generation_print_interval, num_trajectories)
        print(
            f"collecting split={split} trajectories=0/{num_trajectories}",
            flush=True,
        )
        while collected_trajectories < num_trajectories:
            batch_trajectories = min(
                args.num_envs, num_trajectories - collected_trajectories
            )
            env.reset()
            state = env.states.clone()
            action = torch.zeros(
                (args.num_envs, env.action_dim), device=env.torch_device
            )
            action_limits = torch.empty(
                (env.action_dim, 2), dtype=state.dtype, device=env.torch_device
            )
            position_control = (
                hasattr(env.env, "robot_spec")
                and env.env.robot_spec.actuator_mode == "position"
            )
            if position_control:
                controlled_q_indices = [
                    env.env.robot_spec.joint_index[name]
                    for name in env.env.robot_spec.controllable_dofs
                ]
                action_limits = torch.as_tensor(
                    env.action_limits,
                    dtype=state.dtype,
                    device=env.torch_device,
                )
                joint_limits = torch.as_tensor(
                    [
                        env.env.robot_spec.joint_limits[index]
                        for index in controlled_q_indices
                    ],
                    dtype=state.dtype,
                    device=env.torch_device,
                )
                joint_fraction = (
                    (state[:, controlled_q_indices] - joint_limits[:, 0])
                    / (joint_limits[:, 1] - joint_limits[:, 0])
                )
                action = (
                    action_limits[:, 0]
                    + joint_fraction
                    * (action_limits[:, 1] - action_limits[:, 0])
                ).clamp(
                    action_limits[:, 0], action_limits[:, 1]
                )
            states, actions, joint_forces, next_states, valid = [], [], [], [], []
            contacts = []
            contact_features = np.zeros(
                (args.num_envs, contact_config["max_contacts"] * 8), dtype=np.float32
            )
            for step in range(args.horizon):
                was_terminated = env.terminated.clone()
                random_action = (
                    torch.rand(
                        (args.num_envs, env.action_dim),
                        generator=generator,
                        device=env.torch_device,
                    )
                    * 2.0
                    - 1.0
                )
                if position_control:
                    if step % args.action_hold_steps == 0:
                        action = (action + random_action * args.action_scale).clamp(
                            action_limits[:, 0], action_limits[:, 1]
                        )
                else:
                    action = random_action * args.action_scale
                next_state = env.step(action, env_mode="ground-truth").clone()
                if collect_contacts:
                    contact_features = mujoco_warp_features(
                        env.solver_gt,
                        num_envs=args.num_envs,
                        features=contact_config["features"],
                        max_contacts=contact_config["max_contacts"],
                    )
                    contact_tensor = torch.from_numpy(contact_features).to(
                        device=env.torch_device
                    )
                    contact_valid = contact_transition_mask(
                        contact_tensor, contact_config
                    )
                else:
                    contact_valid = torch.ones(
                        args.num_envs, dtype=torch.bool, device=env.torch_device
                    )
                accepted = valid_transition_mask(
                    was_terminated,
                    env.action_saturation_mask,
                    env.terminated,
                ) & contact_valid
                accepted &= (
                    torch.isfinite(action).all(dim=1)
                    & torch.isfinite(state).all(dim=1)
                    & torch.isfinite(next_state).all(dim=1)
                )
                rejected_transitions += int(
                    (~accepted[:batch_trajectories]).sum().item()
                )
                states.append(state[:batch_trajectories].cpu())
                actions.append(action[:batch_trajectories].cpu())
                joint_forces.append(env.joint_f[:batch_trajectories].cpu())
                if collect_contacts:
                    contacts.append(torch.from_numpy(contact_features[:batch_trajectories]))
                next_states.append(next_state[:batch_trajectories].cpu())
                valid.append(accepted[:batch_trajectories].cpu())
                if args.diagnostics and render:
                    env.log_robot_diagnostics(step, action)
                if render:
                    env.render()
                state = next_state

            batches["states"].append(torch.stack(states, dim=1).numpy())
            batches["actions"].append(torch.stack(actions, dim=1).numpy())
            batches["joint_f"].append(torch.stack(joint_forces, dim=1).numpy())
            batches["next_states"].append(torch.stack(next_states, dim=1).numpy())
            batches["valid"].append(torch.stack(valid, dim=1).numpy())
            if collect_contacts:
                batches["self_contact"].append(torch.stack(contacts, dim=1).numpy())
            collected_trajectories += batch_trajectories
            if collected_trajectories >= next_progress or collected_trajectories == num_trajectories:
                print(
                    f"collecting split={split} trajectories="
                    f"{collected_trajectories}/{num_trajectories} "
                    f"valid={int(sum(batch.sum() for batch in batches['valid']))} "
                    f"rejected={rejected_transitions}",
                    flush=True,
                )
                while next_progress <= collected_trajectories:
                    next_progress += args.generation_print_interval

        dataset: dict[str, Any] = {
            name: np.concatenate(values, axis=0) for name, values in batches.items()
        }
        if len(valid_window_starts(dataset["valid"])) == 0:
            raise RuntimeError(
                f"No physically valid {SEQUENCE_LENGTH}-step robot sequences were "
                "collected; increase --horizon, reduce --action-scale, or use more "
                "parallel environments."
            )
        dataset["rejected_transitions"] = rejected_transitions
        if not all(np.isfinite(value).all() for value in dataset.values() if isinstance(value, np.ndarray)):
            raise RuntimeError("Collected robot data contains non-finite values")
        validate_dynamics_signal(dataset, env.dof_qd_per_env)
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
        group.attrs["training_schema_version"] = int(
            args.training_config_data["schema_version"]
        )
        group.attrs["training_inputs"] = json.dumps(
            args.training_config_data["inputs"]["low_dim"]
        )
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
    if not torch.allclose(checkpoint.get("output_mean"), model.output_rms.mean):
        raise RuntimeError("Resume checkpoint output mean does not match the dataset")
    if not torch.allclose(checkpoint.get("output_variance"), model.output_rms.var):
        raise RuntimeError("Resume checkpoint output variance does not match the dataset")

    model.load_state_dict(checkpoint["state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("completed_epochs", 0))


def evaluate_model(model, dataset, device, batch_size, max_windows=None):
    states = torch.from_numpy(dataset["states"]).to(device)
    joint_f = torch.from_numpy(dataset["joint_f"]).to(device)
    targets = torch.from_numpy(dataset["next_states"] - dataset["states"]).to(device)
    starts = valid_window_starts(dataset["valid"]).to(device)
    if len(starts) == 0:
        raise ValueError(f"Dataset has no valid {SEQUENCE_LENGTH}-step sequences")
    if max_windows is not None:
        starts = starts[:max_windows]

    physical_squared_error = 0.0
    normalized_squared_error = 0.0
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
            physical_squared_error += functional.mse_loss(
                prediction, target_windows, reduction="sum"
            ).item()
            normalized_squared_error += functional.mse_loss(
                model.output_rms.normalize(prediction),
                model.output_rms.normalize(target_windows),
                reduction="sum",
            ).item()
            element_count += target_windows.numel()
    return (
        physical_squared_error / element_count,
        normalized_squared_error / element_count,
    )


def checkpoint_payload(model, optimizer, args, state_dim, joint_f_dim, completed_epochs):
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
        "completed_epochs": completed_epochs,
        "output_mean": model.output_rms.mean.detach().cpu(),
        "output_variance": model.output_rms.var.detach().cpu(),
    }
    if "test" in args.splits:
        checkpoint["test_dataset_path"] = os.path.abspath(args.test_dataset_path)
    return checkpoint


def latest_checkpoint_path(checkpoint_path):
    stem, extension = os.path.splitext(checkpoint_path)
    return f"{stem}.latest{extension or '.pt'}"


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

    train_valid = torch.from_numpy(train_dataset["valid"]).to(device=device, dtype=torch.bool)
    valid_train_targets = train_targets[train_valid]
    output_mean = valid_train_targets.mean(dim=0)
    output_variance = valid_train_targets.var(dim=0, unbiased=False)
    del valid_train_targets, train_valid
    model = create_model(state_dim, joint_f_dim, device)
    set_output_statistics(model, output_mean, output_variance)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = resume_training(
        model, optimizer, args, state_dim, joint_f_dim, device
    )
    training_epochs = args.epochs
    if args.target_epochs is not None:
        training_epochs = max(0, args.target_epochs - start_epoch)
    end_epoch = start_epoch + training_epochs
    if training_epochs == 0:
        print(f"checkpoint already reached target epoch={end_epoch}", flush=True)
    generator = torch.Generator(device=device).manual_seed(args.seed + 10)
    writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None
    model.train()
    for epoch in range(start_epoch, end_epoch):
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
        output_rms = model.output_rms
        if output_rms is None:
            raise RuntimeError("Output normalization statistics are not configured")
        normalized_prediction = output_rms.normalize(prediction)
        normalized_target = output_rms.normalize(target_windows)
        loss = functional.mse_loss(normalized_prediction, normalized_target)
        physical_loss = functional.mse_loss(prediction, target_windows)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_number = epoch + 1
        train_loss = float(loss.detach().cpu())
        train_physical_loss = float(physical_loss.detach().cpu())
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if writer is not None:
            writer.add_scalar("loss/train_normalized", train_loss, epoch_number)
            writer.add_scalar("loss/train_physical", train_physical_loss, epoch_number)
            writer.add_scalar("optimizer/learning_rate", learning_rate, epoch_number)

        validation_metrics = None
        if epoch_number % args.validation_interval == 0 or epoch_number == end_epoch:
            model.eval()
            validation_metrics = evaluate_model(
                model,
                validation_dataset,
                device,
                args.batch_size,
                max_windows=args.validation_windows,
            )
            model.train()
            if writer is not None:
                writer.add_scalar(
                    "loss/validation_physical", validation_metrics[0], epoch_number
                )
                writer.add_scalar(
                    "loss/validation_normalized", validation_metrics[1], epoch_number
                )

        if epoch_number % args.print_interval == 0 or validation_metrics is not None:
            validation_text = (
                ""
                if validation_metrics is None
                else (
                    f" validation_normalized_mse={validation_metrics[1]:.6g}"
                    f" validation_physical_mse={validation_metrics[0]:.6g}"
                )
            )
            print(
                f"epoch={epoch_number} learning_rate={learning_rate:.6g} "
                f"train_normalized_mse={train_loss:.6g} "
                f"train_physical_mse={train_physical_loss:.6g}{validation_text}",
                flush=True,
            )
        if epoch_number % args.checkpoint_interval == 0:
            periodic_path = latest_checkpoint_path(args.checkpoint_path)
            os.makedirs(os.path.dirname(os.path.abspath(periodic_path)), exist_ok=True)
            torch.save(
                checkpoint_payload(
                    model, optimizer, args, state_dim, joint_f_dim, epoch_number
                ),
                periodic_path,
            )
            print(f"checkpoint={periodic_path} epoch={epoch_number}", flush=True)
        if args.render and "train" in args.render_splits and args.render_backend == "rerun":
            import rerun as rr

            rr.set_time("training_epoch", sequence=epoch)
            rr.log("robot/training/mse", rr.Scalars(float(loss.detach().cpu())))

    model.eval()
    validation_mse, validation_normalized_mse = evaluate_model(
        model, validation_dataset, device, args.batch_size
    )
    if not np.isfinite(validation_mse):
        raise RuntimeError("Validation produced a non-finite loss")
    if writer is not None:
        writer.add_scalar("loss/validation_full_physical", validation_mse, end_epoch)
        writer.add_scalar(
            "loss/validation_full_normalized", validation_normalized_mse, end_epoch
        )

    checkpoint = checkpoint_payload(
        model,
        optimizer,
        args,
        state_dim,
        joint_f_dim,
        end_epoch,
    )
    if test_dataset is not None:
        test_mse, test_normalized_mse = evaluate_model(
            model, test_dataset, device, args.batch_size
        )
        if not np.isfinite(test_mse):
            raise RuntimeError("Test produced a non-finite loss")
        checkpoint["test_dataset_path"] = os.path.abspath(args.test_dataset_path)
        if writer is not None:
            writer.add_scalar("loss/test_physical", test_mse, end_epoch)
            writer.add_scalar("loss/test_normalized", test_normalized_mse, end_epoch)
    else:
        test_mse = None
    os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint_path)), exist_ok=True)
    torch.save(checkpoint, args.checkpoint_path)
    loaded = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    reloaded_model = create_model(loaded["state_dim"], loaded["joint_f_dim"], device)
    set_output_statistics(
        reloaded_model,
        loaded["output_mean"].to(device),
        loaded["output_variance"].to(device),
    )
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
    if writer is not None:
        writer.flush()
        writer.close()
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
        "--training-config",
        help="Training profile path or robot profile name (defaults to --robot-id).",
    )
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
    parser.add_argument(
        "--num-trajectories",
        type=int,
        help="Trajectories to collect for each selected split; generated in --num-envs batches",
    )
    parser.add_argument("--generation-print-interval", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-pose", action="store_true")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument(
        "--action-hold-steps",
        type=int,
        default=20,
        help="Steps to hold each random-walk target for position-controlled robots",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--target-epochs",
        type=int,
        help="Stop at this total epoch/update count when resuming",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--log-dir",
        help="TensorBoard output directory; disabled when omitted",
    )
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument(
        "--validation-windows",
        type=int,
        default=4096,
        help="Maximum fixed validation windows used for periodic loss monitoring",
    )
    parser.add_argument("--print-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
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
    parser.add_argument(
        "--env-spacing",
        type=float,
        default=0.0,
        help="X/Z spacing between parallel robots; use about 1.0 for 3D visualization",
    )
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
    args.training_config_data = load_training_config(
        args.training_config or args.robot_id
    )
    config_robot = args.training_config_data.get("robot_id")
    if config_robot is not None and config_robot != args.robot_id:
        parser.error("--training-config robot_id must match --robot-id")

    paths = default_dataset_paths(args.robot_id)
    args.train_dataset_path = args.train_dataset_path or args.dataset_path or paths["train"]
    args.validation_dataset_path = args.validation_dataset_path or paths["validation"]
    args.test_dataset_path = args.test_dataset_path or paths["test"]
    args.checkpoint_path = args.checkpoint_path or f"outputs/{args.robot_id}_nerd_model.pt"
    if args.generate_only and args.skip_generation:
        parser.error("--generate-only and --skip-generation cannot be used together")
    if args.num_trajectories is not None and args.num_trajectories < 1:
        parser.error("--num-trajectories must be positive")
    if args.action_hold_steps < 1:
        parser.error("--action-hold-steps must be positive")
    if args.env_spacing < 0.0:
        parser.error("--env-spacing cannot be negative")
    if args.target_epochs is not None and args.target_epochs < 0:
        parser.error("--target-epochs cannot be negative")
    if (
        args.generation_print_interval < 1
        or args.validation_interval < 1
        or args.validation_windows < 1
        or args.print_interval < 1
        or args.checkpoint_interval < 1
    ):
        parser.error("generation, validation, print, and checkpoint intervals must be positive")
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