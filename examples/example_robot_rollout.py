import argparse
import os
import time

import torch

from envs.neural_environment import NeuralEnvironment


def main():
    parser = argparse.ArgumentParser(description="Headless NeRD Newton robot rollout")
    parser.add_argument("--robot-id", default="franka_panda")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
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

    env = NeuralEnvironment(
        env_name="Robot",
        num_envs=args.num_envs,
        newton_env_cfg={
            "robot_spec": args.robot_id,
            "seed": args.seed,
            "random_reset": not args.default_pose,
            **render_config,
        },
        default_env_mode="ground-truth",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        render=args.render,
    )
    try:
        env.reset()
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
            state = env.step(actions, env_mode="ground-truth")
            if args.diagnostics:
                env.log_robot_diagnostics(step, actions)
            if args.render:
                env.render()
        if not torch.isfinite(state).all():
            raise RuntimeError("Robot rollout produced non-finite state")
        print(
            f"robot={args.robot_id} envs={args.num_envs} horizon={args.horizon} "
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