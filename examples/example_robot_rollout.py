import argparse
import os
import time

import torch

from envs.neural_environment import NeuralEnvironment
from models.models import ModelMixedInput
from utils.running_mean_std import RunningMeanStd
from training.model_factory import build_model


def load_nerd_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("format") == "nerd_sequence_v3":
        model = build_model(
            checkpoint["input_dimensions"],
            checkpoint["output_dim"],
            checkpoint["training_config"],
            device,
        )
        if model.output_rms is not None:
            model.output_rms.mean = checkpoint["output_mean"].to(device)
            model.output_rms.var = checkpoint["output_variance"].to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        checkpoint.setdefault("state_dim", checkpoint["output_dim"])
        checkpoint.setdefault("joint_f_dim", checkpoint["input_dimensions"]["joint_f"])
        checkpoint.setdefault("solver_name", "TransformerNeuralSolver")
        checkpoint.setdefault("prediction_type", "relative")
        checkpoint.setdefault("num_states_history", checkpoint["training_config"]["sequence"]["length"])
        return model, checkpoint
    required = {"state_dim", "joint_f_dim", "input_cfg", "network_cfg", "state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(
            f"NeRD checkpoint {checkpoint_path} is missing fields: {sorted(missing)}"
        )
    input_cfg = {"low_dim": ["states_embedding", "joint_f"]}
    if checkpoint["input_cfg"] != input_cfg:
        raise RuntimeError("Robot rollout only supports states_embedding + joint_f inputs")
    sample = {
        "states_embedding": torch.zeros((1, 1, checkpoint["state_dim"]), device=device),
        "joint_f": torch.zeros((1, 1, checkpoint["joint_f_dim"]), device=device),
    }
    model = ModelMixedInput(
        input_sample=sample,
        output_dim=checkpoint["state_dim"],
        input_cfg=input_cfg,
        network_cfg=checkpoint["network_cfg"],
        device=device,
    )
    if checkpoint["network_cfg"].get("normalize_output", False):
        if "output_mean" not in checkpoint or "output_variance" not in checkpoint:
            raise RuntimeError("Normalized NeRD checkpoint is missing output statistics")
        output_rms = RunningMeanStd(
            shape=tuple(checkpoint["output_mean"].shape), device=device
        )
        output_rms.mean = checkpoint["output_mean"].to(device)
        output_rms.var = checkpoint["output_variance"].to(device)
        output_rms.count = 1.0
        model.set_output_rms(output_rms)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Headless NeRD Newton robot rollout")
    parser.add_argument("--robot-id", default="franka_panda")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nerd-checkpoint",
        help="Robot NeRD checkpoint created by example_robot_nerd_train.py; enables neural dynamics.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Log robot state, limits, controls, and violations to Rerun.",
    )
    parser.add_argument(
        "--random-actions",
        action="store_true",
        help="Use random actions instead of zero actions.",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help="Scale random actions before the environment clamps them.",
    )
    parser.add_argument(
        "--default-pose",
        action="store_true",
        help="Start the robot in its configured default pose instead of a random pose.",
    )
    parser.add_argument(
        "--render-backend",
        choices=("opengl", "rerun"),
        default="opengl",
        help="Viewer backend used with --render; rerun serves the scene to a browser.",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep the viewer process alive after the rollout finishes.",
    )
    parser.add_argument("--grpc-port", type=int, default=9876)
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--browser-host", default="localhost")
    parser.add_argument(
        "--rerun-view",
        choices=("3d", "camera"),
        default="camera",
        help="Rerun view: clean native camera image or experimental 3D scene.",
    )
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    render_config = {}
    if args.render_backend == "rerun":
        from envs.newton_envs import RenderMode

        os.environ.setdefault("MUJOCO_GL", "egl")
        render_config = {
            "render_mode": RenderMode.RERUN,
            "rerun_render_settings": {
                "grpc_port": args.grpc_port,
                "web_port": args.web_port,
                "browser_host": args.browser_host,
                "native_camera": args.rerun_view == "camera",
            },
        }

    neural_model = None
    neural_solver_cfg = None
    env_mode = "ground-truth"
    checkpoint = None
    if args.nerd_checkpoint is not None:
        neural_model, checkpoint = load_nerd_model(args.nerd_checkpoint, device)
        solver_name = checkpoint.get("solver_name", "NeuralSolver")
        neural_solver_cfg = {
            "name": solver_name,
            "prediction_type": checkpoint.get("prediction_type", "relative"),
        }
        if solver_name == "TransformerNeuralSolver":
            neural_solver_cfg["num_states_history"] = checkpoint.get(
                "num_states_history", 10
            )
        env_mode = "neural"

    env = NeuralEnvironment(
        env_name="Robot",
        num_envs=args.num_envs,
        newton_env_cfg={
            "robot_spec": args.robot_id,
            "seed": args.seed,
            "random_reset": not args.default_pose,
            **render_config,
        },
        neural_solver_cfg=neural_solver_cfg,
        neural_model=neural_model,
        default_env_mode=env_mode,
        device=device,
        render=args.render,
    )
    try:
        if checkpoint is not None and (
            checkpoint["state_dim"] != env.state_dim
            or checkpoint["joint_f_dim"] != env.joint_f_dim
        ):
            raise RuntimeError("NeRD checkpoint dimensions do not match this robot")
        env.reset()
        state = env.states
        generator = torch.Generator(device=env.torch_device).manual_seed(args.seed + 1)
        for step in range(args.horizon):
            if args.random_actions:
                actions = (
                    torch.rand(
                        (args.num_envs, env.action_dim),
                        generator=generator,
                        device=env.torch_device,
                    )
                    * 2.0
                    - 1.0
                ) * args.action_scale
            else:
                actions = torch.zeros(
                    (args.num_envs, env.action_dim), device=env.torch_device
                )
            state = env.step(actions, env_mode=env_mode)
            if args.diagnostics:
                env.log_robot_diagnostics(step, actions)
            if args.render:
                env.render()
        if not torch.isfinite(state).all():
            raise RuntimeError("Robot rollout produced non-finite state")
        print(
            f"robot={args.robot_id} mode={env_mode} envs={args.num_envs} horizon={args.horizon} "
            f"state_shape={tuple(state.shape)} action_dim={env.action_dim}"
        )
        if args.keep_open and args.render:
            print("Viewer is running. Press Ctrl+C to exit.", flush=True)
            while True:
                time.sleep(1.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()