import numpy as np
import torch

from envs.neural_environment import NeuralEnvironment
from examples.example_robot_nerd_train import (
    SEQUENCE_LENGTH,
    create_model,
    model_config,
    valid_transition_mask,
    valid_window_starts,
)
from robot_specs import load_robot_spec


def test_franka_spec_has_explicit_action_mapping():
    spec = load_robot_spec("franka_panda")

    assert spec.dof == 9
    assert spec.action_dim == 7
    assert spec.joint_names[:7] == tuple(f"joint{i}" for i in range(1, 8))
    assert spec.controllable_dofs == spec.joint_names[:7]
    assert spec.damping[0] == 1.0
    assert spec.armature[0] == 0.1


def test_franka_cpu_rollout_is_deterministic():
    def rollout():
        env = NeuralEnvironment(
            env_name="Robot",
            num_envs=2,
            newton_env_cfg={
                "robot_spec": "franka_panda",
                "seed": 2026,
                "random_reset": True,
            },
            default_env_mode="ground-truth",
            device="cpu",
            render=False,
        )
        try:
            env.reset()
            assert env.contact_validity_mask.numel() == 0
            action = torch.zeros((2, env.action_dim), device=env.torch_device)
            states = [env.states.clone()]
            for _ in range(10):
                states.append(env.step(action, env_mode="ground-truth").clone())
            return torch.stack(states)
        finally:
            env.close()

    first = rollout()
    second = rollout()
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_so101_applies_explicit_dynamics_and_position_control():
    env = NeuralEnvironment(
        env_name="Robot",
        num_envs=1,
        newton_env_cfg={"robot_spec": "so101", "random_reset": False},
        default_env_mode="ground-truth",
        device="cpu",
        render=False,
    )
    try:
        model = env.model
        spec = load_robot_spec("so101")
        assert env.env.actuator_mode_code == 1
        assert torch.allclose(
            torch.as_tensor(model.joint_limit_lower.numpy()),
            torch.as_tensor([pair[0] for pair in spec.joint_limits]),
        )
        assert torch.allclose(
            torch.as_tensor(model.joint_limit_upper.numpy()),
            torch.as_tensor([pair[1] for pair in spec.joint_limits]),
        )
        assert torch.allclose(
            torch.as_tensor(model.joint_armature.numpy()),
            torch.as_tensor(spec.armature),
        )
        assert torch.allclose(
            torch.as_tensor(model.joint_target_kd.numpy()),
            torch.as_tensor(spec.damping),
        )
        assert torch.allclose(
            torch.as_tensor(model.joint_target_ke.numpy()),
            torch.as_tensor(spec.position_gains),
        )
        env.reset()
        initial_state = env.states.clone()
        state = env.step(
            torch.full((1, env.action_dim), 0.5, device=env.torch_device),
            env_mode="ground-truth",
        )
        assert (state[:, :spec.dof] - initial_state[:, :spec.dof]).abs().max() > 1e-5
        assert state[:, spec.dof:].abs().max() > 1e-3
    finally:
        env.close()


def test_so101_action_saturation_is_recorded():
    env = NeuralEnvironment(
        env_name="Robot",
        num_envs=1,
        newton_env_cfg={"robot_spec": "so101", "random_reset": False},
        default_env_mode="ground-truth",
        device="cpu",
        render=False,
    )
    try:
        env.reset()
        env.step(torch.full((1, env.action_dim), 2.0), env_mode="ground-truth")
        assert env.action_saturation_mask.tolist() == [True]
        assert env.action_saturation_count.tolist() == [6]
    finally:
        env.close()


def test_dataset_transition_filter_rejects_unphysical_samples():
    accepted = valid_transition_mask(
        was_terminated=torch.tensor([False, False, True, False]),
        action_saturation_mask=torch.tensor([False, True, False, False]),
        is_terminated=torch.tensor([False, False, True, True]),
    )

    assert accepted.tolist() == [True, False, False, False]


def test_transformer_windows_do_not_cross_rejected_transitions():
    valid = torch.tensor(
        [[True] * SEQUENCE_LENGTH + [False] + [True] * SEQUENCE_LENGTH]
    )

    starts = valid_window_starts(valid)

    assert starts.tolist() == [[0, 0], [0, SEQUENCE_LENGTH + 1]]


def test_robot_nerd_model_uses_causal_transformer_history():
    config = model_config()
    model = create_model(state_dim=12, joint_f_dim=6, device="cpu")

    prediction = model(
        {
            "states_embedding": torch.zeros((2, SEQUENCE_LENGTH, 12)),
            "joint_f": torch.zeros((2, SEQUENCE_LENGTH, 6)),
        }
    )

    assert config["transformer"]["block_size"] >= SEQUENCE_LENGTH
    assert model.is_transformer
    assert prediction.shape == (2, SEQUENCE_LENGTH, 12)


def test_so101_random_rollout_is_finite_and_records_limit_termination():
    env = NeuralEnvironment(
        env_name="Robot",
        num_envs=5,
        newton_env_cfg={"robot_spec": "so101", "seed": 42, "random_reset": True},
        default_env_mode="ground-truth",
        device="cpu",
        render=False,
    )
    try:
        spec = load_robot_spec("so101")
        env.reset()
        state = env.states
        generator = torch.Generator(device=env.torch_device).manual_seed(123)
        action_limits = torch.as_tensor(spec.action_limits, device=env.torch_device)
        for _ in range(1000):
            actions = torch.rand(
                (env.num_envs, env.action_dim),
                generator=generator,
                device=env.torch_device,
            )
            actions = actions * (action_limits[:, 1] - action_limits[:, 0]) + action_limits[:, 0]
            state = env.step(actions, env_mode="ground-truth")
        assert torch.isfinite(state).all()
        for env_id, terminated in enumerate(env.terminated.tolist()):
            if terminated:
                assert env.termination_reasons[env_id].startswith(
                    ("joint_limit:", "velocity_limit:")
                )
    finally:
        env.close()


def test_so101_state_and_action_load_into_mujoco_backend():
    env = NeuralEnvironment(
        env_name="Robot",
        num_envs=1,
        newton_env_cfg={"robot_spec": "so101", "random_reset": False},
        default_env_mode="ground-truth",
        device="cpu",
        render=False,
    )
    try:
        env.reset()
        action = torch.zeros((1, env.action_dim), device=env.torch_device)
        env.step(action, env_mode="ground-truth")
        solver = env.solver_gt
        solver.update_mjc_data(solver.mj_data, env.model, env.sim_states)
        solver.apply_mjc_control(env.model, env.sim_states, env.joint_control, solver.mj_data)
        np.testing.assert_allclose(
            solver.mj_data.qpos,
            env.sim_states.joint_q.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        assert np.isfinite(solver.mj_data.ctrl).all()
    finally:
        env.close()