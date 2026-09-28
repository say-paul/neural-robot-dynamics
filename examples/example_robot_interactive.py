import argparse
import os

import torch

from envs.neural_environment import NeuralEnvironment
from examples.example_robot_rollout import enforce_robot_limits, load_nerd_model


def configure_render(args):
    if not args.render:
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


def actions_for_joint_target(env, target_positions):
    robot_spec = env.env.robot_spec
    target = torch.as_tensor(
        target_positions, dtype=env.states.dtype, device=env.torch_device
    )
    expected_targets = len(robot_spec.controllable_dofs)
    if target.numel() != expected_targets:
        raise ValueError(f"expected {expected_targets} target angles, got {target.numel()}")

    limits = torch.as_tensor(
        [
            robot_spec.joint_limits[robot_spec.joint_index[name]]
            for name in robot_spec.controllable_dofs
        ],
        dtype=target.dtype,
        device=target.device,
    )
    if ((target < limits[:, 0]) | (target > limits[:, 1])).any():
        raise ValueError("target is outside the configured joint limits")
    return (2.0 * (target - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1.0).unsqueeze(0)


def joint_errors(env, target_positions):
    robot_spec = env.env.robot_spec
    indices = [robot_spec.joint_index[name] for name in robot_spec.controllable_dofs]
    target = torch.as_tensor(
        target_positions, dtype=env.states.dtype, device=env.torch_device
    )
    return env.states[0, indices] - target


def print_target_error(env, target_positions, prefix):
    robot_spec = env.env.robot_spec
    errors = joint_errors(env, target_positions).detach().cpu().tolist()
    max_error = max(abs(error) for error in errors)
    print(f"{prefix}: max joint error={max_error:.4f} rad")
    print(
        "joint errors (current - target, rad):",
        " ".join(
            f"{name}={error:+.3f}"
            for name, error in zip(robot_spec.controllable_dofs, errors, strict=True)
        ),
    )
    return max_error


def print_joint_help(env):
    robot_spec = env.env.robot_spec
    print("Enter: go <six joint angles in radians>, home, reset, status, or quit")
    for name in robot_spec.controllable_dofs:
        lower, upper = robot_spec.joint_limits[robot_spec.joint_index[name]]
        print(f"  {name}: [{lower:.3f}, {upper:.3f}]")


def run_target(env, target_positions, args, step_count, actions):
    target_actions = actions_for_joint_target(env, target_positions)
    for target_step in range(args.max_steps):
        if target_step % args.action_hold_steps == 0:
            actions = actions + (target_actions - actions).clamp(
                -args.action_step, args.action_step
            )
        env.step(actions, env_mode="neural")
        enforce_robot_limits(env, env.states)
        if args.diagnostics:
            env.log_robot_diagnostics(step_count, actions)
        if args.render:
            env.render()
        step_count += 1
        error = joint_errors(env, target_positions).abs().max().item()
        if error <= args.tolerance:
            print_target_error(env, target_positions, "target reached")
            return step_count, actions
    print_target_error(env, target_positions, "target time limit reached")
    return step_count, actions


def parse_target(command, num_joints):
    fields = command.split()
    if fields[0].lower() != "go":
        return None
    if len(fields) != num_joints + 1:
        raise ValueError(f"use: go followed by {num_joints} joint angles in radians")
    return [float(value) for value in fields[1:]]


def main():
    parser = argparse.ArgumentParser(
        description="Interactive joint-target SO101 rollout using a trained NeRD model"
    )
    parser.add_argument("--robot-id", default="so101")
    parser.add_argument("--nerd-checkpoint", required=True)
    parser.add_argument(
        "--target",
        type=float,
        nargs="+",
        help="Run one joint target in radians and exit instead of prompting.",
    )
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument(
        "--action-step",
        type=float,
        default=0.01,
        help="Maximum normalized target change per action hold, matching training data.",
    )
    parser.add_argument(
        "--action-hold-steps",
        type=int,
        default=20,
        help="Simulation frames for each action target, matching training data.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--render-backend", choices=("rerun",), default="rerun"
    )
    parser.add_argument("--grpc-port", type=int, default=19878)
    parser.add_argument("--web-port", type=int, default=19092)
    parser.add_argument("--browser-host", default="localhost")
    parser.add_argument("--rerun-view", choices=("3d", "camera"), default="camera")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.tolerance <= 0.0:
        parser.error("--tolerance must be positive")
    if args.action_step <= 0.0:
        parser.error("--action-step must be positive")
    if args.action_hold_steps < 1:
        parser.error("--action-hold-steps must be positive")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    neural_model, checkpoint = load_nerd_model(args.nerd_checkpoint, device)
    solver_cfg = {
        "name": checkpoint.get("solver_name", "TransformerNeuralSolver"),
        "prediction_type": checkpoint.get("prediction_type", "relative"),
    }
    if solver_cfg["name"] == "TransformerNeuralSolver":
        solver_cfg["num_states_history"] = checkpoint.get("num_states_history", 10)

    env = NeuralEnvironment(
        env_name="Robot",
        num_envs=1,
        newton_env_cfg={
            "robot_spec": args.robot_id,
            "seed": args.seed,
            "random_reset": False,
            **configure_render(args),
        },
        neural_solver_cfg=solver_cfg,
        neural_model=neural_model,
        default_env_mode="neural",
        device=device,
        render=args.render,
    )
    try:
        if checkpoint["state_dim"] != env.state_dim or checkpoint["joint_f_dim"] != env.joint_f_dim:
            raise RuntimeError("NeRD checkpoint dimensions do not match this robot")
        env.reset()
        step_count = 0
        actions = actions_for_joint_target(
            env, env.states[0, : env.dof_q_per_env]
        )
        if args.target is not None:
            run_target(env, args.target, args, step_count, actions)
            return

        print_joint_help(env)
        while True:
            try:
                command = input("nerd> ").strip()
            except EOFError:
                break
            if not command:
                continue
            if command.lower() in {"quit", "exit"}:
                break
            if command.lower() == "home":
                target = env.env.robot_spec.default_q
            elif command.lower() == "reset":
                env.reset()
                actions = actions_for_joint_target(
                    env, env.states[0, : env.dof_q_per_env]
                )
                print("reset to the default pose")
                continue
            elif command.lower() == "status":
                positions = env.states[0, : env.dof_q_per_env].detach().cpu().tolist()
                print("joint positions (rad):", " ".join(f"{value:.3f}" for value in positions))
                continue
            else:
                try:
                    target = parse_target(command, env.action_dim)
                    if target is None:
                        print_joint_help(env)
                        continue
                except ValueError as error:
                    print(error)
                    continue
            try:
                step_count, actions = run_target(
                    env, target, args, step_count, actions
                )
            except ValueError as error:
                print(error)
    finally:
        env.close()


if __name__ == "__main__":
    main()