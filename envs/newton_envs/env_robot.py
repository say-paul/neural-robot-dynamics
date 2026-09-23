from __future__ import annotations

import os

import newton
import warp as wp

from envs.newton_envs import Environment, RenderMode, SolverType
from robot_specs import RobotSpec, load_robot_spec


ACTUATOR_TORQUE = 0
ACTUATOR_POSITION = 1
ACTUATOR_VELOCITY = 2


@wp.kernel(enable_backward=False)
def reset_robot_state(
    reset: wp.array(dtype=wp.bool),
    seed: int,
    random_reset: bool,
    dof_q_per_env: int,
    dof_qd_per_env: int,
    q_lower: wp.array(dtype=wp.float32),
    q_upper: wp.array(dtype=wp.float32),
    qd_lower: wp.array(dtype=wp.float32),
    qd_upper: wp.array(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    if reset and not reset[env_id]:
        return
    if not random_reset:
        return
    random_state = wp.rand_init(seed, env_id)
    for index in range(dof_q_per_env):
        joint_q[env_id * dof_q_per_env + index] = wp.randf(
            random_state, q_lower[index], q_upper[index]
        )
    for index in range(dof_qd_per_env):
        joint_qd[env_id * dof_qd_per_env + index] = wp.randf(
            random_state, qd_lower[index], qd_upper[index]
        )


@wp.kernel
def compute_robot_observations(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    dof_q: int,
    dof_qd: int,
    observations: wp.array(dtype=wp.float32, ndim=2),
):
    env_id = wp.tid()
    for index in range(dof_q):
        observations[env_id, index] = joint_q[env_id * dof_q + index]
    for index in range(dof_qd):
        observations[env_id, dof_q + index] = joint_qd[env_id * dof_qd + index]


@wp.kernel
def assign_robot_controls(
    actions: wp.array(dtype=wp.float32, ndim=2),
    action_limits: wp.array(dtype=wp.float32, ndim=2),
    joint_target_limits: wp.array(dtype=wp.float32, ndim=2),
    effort_limits: wp.array(dtype=wp.float32),
    controllable_dofs: wp.array(dtype=wp.int32),
    actuator_mode: int,
    control_full_dim: int,
    joint_f: wp.array(dtype=wp.float32),
    joint_target: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    for index in range(effort_limits.shape[0]):
        action = wp.clamp(
            actions[env_id, index],
            action_limits[index, 0],
            action_limits[index, 1],
        )
        dof = env_id * control_full_dim + controllable_dofs[index]
        joint_f[dof] = 0.0
        joint_target[dof] = 0.0
        if actuator_mode == ACTUATOR_TORQUE:
            joint_f[dof] = action * effort_limits[index]
        elif actuator_mode == ACTUATOR_POSITION:
            action_fraction = (
                (action - action_limits[index, 0])
                / (action_limits[index, 1] - action_limits[index, 0])
            )
            joint_target[dof] = (
                joint_target_limits[index, 0]
                + action_fraction
                * (joint_target_limits[index, 1] - joint_target_limits[index, 0])
            )
        else:
            joint_target[dof] = action


@wp.kernel
def apply_robot_damping(
    joint_qd: wp.array(dtype=wp.float32),
    damping: wp.array(dtype=wp.float32),
    dof_qd_per_env: int,
    joint_f: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    for index in range(dof_qd_per_env):
        offset = env_id * dof_qd_per_env + index
        joint_f[offset] -= damping[index] * joint_qd[offset]


class RobotEnvironment(Environment):
    sim_name = "robot"
    solver_type = SolverType.MUJOCO
    activate_ground_plane = False
    env_offset = (0.0, 0.0, 0.0)
    eval_fk = True

    def __init__(
        self,
        robot_spec: RobotSpec | str | os.PathLike[str] = "franka_panda",
        seed: int = 42,
        random_reset: bool = True,
        **kwargs,
    ):
        self.robot_spec = (
            robot_spec
            if isinstance(robot_spec, RobotSpec)
            else load_robot_spec(robot_spec)
        )
        self.seed = seed
        self.random_reset = random_reset
        self.robot_name = self.robot_spec.robot_id
        self.solver_type = SolverType.MUJOCO
        self.activate_ground_plane = bool(
            self.robot_spec.solver.get("ground_plane", False)
        )
        self.fps = int(self.robot_spec.solver.get("fps", 120))
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps_mujoco = int(self.robot_spec.solver.get("sim_substeps", 4))
        self.gravity = float(self.robot_spec.solver.get("gravity", -9.81))
        if "integrator" in self.robot_spec.solver:
            self.mujoco_settings = dict(self.mujoco_settings)
            self.mujoco_settings["integrator"] = str(
                self.robot_spec.solver["integrator"]
            )
        if "njmax" in self.robot_spec.solver:
            self.mujoco_settings = dict(self.mujoco_settings)
            self.mujoco_settings["njmax"] = int(self.robot_spec.solver["njmax"])
        if "ncon_per_env" in self.robot_spec.solver:
            self.mujoco_settings = dict(self.mujoco_settings)
            self.mujoco_settings["ncon_per_env"] = int(
                self.robot_spec.solver["ncon_per_env"]
            )
        if kwargs.get("render_mode") == RenderMode.RERUN:
            self.mujoco_settings = dict(self.mujoco_settings)
            rerun_settings = dict(kwargs.get("rerun_render_settings", {}))
            rerun_settings["native_model_path"] = self.robot_spec.asset_source
            kwargs["rerun_render_settings"] = rerun_settings
        super().__init__(**kwargs)
        self.controllable_effort_limits_wp = wp.array(
            [
                self.robot_spec.effort_limits[self.robot_spec.joint_index[name]]
                for name in self.robot_spec.controllable_dofs
            ],
            dtype=wp.float32,
            device=self.device,
        )
        self.controllable_joint_limits_wp = wp.array(
            [
                self.robot_spec.joint_limits[self.robot_spec.joint_index[name]]
                for name in self.robot_spec.controllable_dofs
            ],
            dtype=wp.float32,
            device=self.device,
        )
        self.damping_wp = wp.array(
            self.robot_spec.damping,
            dtype=wp.float32,
            device=self.device,
        )
        self.actuator_mode_code = {
            "torque": ACTUATOR_TORQUE,
            "position": ACTUATOR_POSITION,
            "velocity": ACTUATOR_VELOCITY,
        }[self.robot_spec.actuator_mode]

    def create_articulation(self, builder: newton.ModelBuilder):
        if not os.path.exists(self.robot_spec.asset_source):
            raise FileNotFoundError(
                f"Asset for {self.robot_spec.robot_id} not found: "
                f"{self.robot_spec.asset_source}"
            )
        builder.add_mjcf(
            self.robot_spec.asset_source,
            floating=self.robot_spec.base_type == "floating",
            enable_self_collisions=bool(
                self.robot_spec.solver.get("self_collisions", False)
            ),
            force_show_colliders=bool(
                self.robot_spec.solver.get("show_collision_mesh", False)
            ),
            collapse_fixed_joints=True,
            skip_equality_constraints=True,
        )
        imported_joint_names = tuple(builder.joint_key)
        if imported_joint_names != self.robot_spec.joint_names:
            raise ValueError(
                f"{self.robot_spec.robot_id} joint order mismatch: "
                f"asset={imported_joint_names}, spec={self.robot_spec.joint_names}"
            )
        for index in range(self.robot_spec.dof):
            builder.joint_limit_lower[index] = self.robot_spec.joint_limits[index][0]
            builder.joint_limit_upper[index] = self.robot_spec.joint_limits[index][1]
            builder.joint_velocity_limit[index] = self.robot_spec.velocity_limits[index]
            builder.joint_effort_limit[index] = self.robot_spec.effort_limits[index]
            builder.joint_armature[index] = self.robot_spec.armature[index]
            builder.joint_target_kd[index] = self.robot_spec.damping[index]
            if self.robot_spec.actuator_mode == "position":
                builder.joint_dof_mode[index] = newton.JointMode.TARGET_POSITION
                builder.joint_target_ke[index] = self.robot_spec.position_gains[index]
            elif self.robot_spec.actuator_mode == "velocity":
                builder.joint_dof_mode[index] = newton.JointMode.TARGET_VELOCITY
                builder.joint_target_ke[index] = 0.0
            else:
                builder.joint_dof_mode[index] = newton.JointMode.NONE
                builder.joint_target_ke[index] = 0.0
        builder.joint_q[:] = list(self.robot_spec.default_q)
        builder.joint_qd[:] = list(self.robot_spec.default_qd)
        qd_starts = list(builder.joint_qd_start)
        self.controllable_dofs = [
            qd_starts[builder.joint_key.index(name)]
            for name in self.robot_spec.controllable_dofs
        ]
        self.control_gains = [
            self.robot_spec.effort_limits[self.robot_spec.joint_index[name]]
            for name in self.robot_spec.controllable_dofs
        ]
        self.control_limits = list(self.robot_spec.action_limits)

    def assign_control(self, actions, control, state):
        wp.launch(
            assign_robot_controls,
            dim=self.num_envs,
            inputs=[
                actions,
                self.control_limits_wp,
                self.controllable_joint_limits_wp,
                self.controllable_effort_limits_wp,
                self.controllable_dofs_wp,
                self.actuator_mode_code,
                self.joint_f_dim,
            ],
            outputs=[control.joint_f, control.joint_target],
            device=self.device,
        )

    def before_step(self, state, next_state, control, eval_collisions=True):
        if self.actuator_mode_code == ACTUATOR_TORQUE:
            wp.launch(
                apply_robot_damping,
                dim=self.num_envs,
                inputs=[state.joint_qd, self.damping_wp, self.dof_qd_per_env],
                outputs=[control.joint_f],
                device=self.device,
            )

    def compute_observations(
        self,
        state: newton.State,
        control: newton.Control,
        observations: wp.array,
        step: int,
        horizon_length: int,
    ):
        wp.launch(
            compute_robot_observations,
            dim=self.num_envs,
            inputs=[state.joint_q, state.joint_qd, self.dof_q_per_env, self.dof_qd_per_env],
            outputs=[observations],
            device=self.device,
        )

    def reset_envs(self, env_ids: wp.array = None):
        super().reset_envs(env_ids)
        reset_mask = env_ids
        if reset_mask is None:
            reset_mask = wp.ones(self.num_envs, dtype=wp.bool, device=self.device)
        q_lower = wp.array(
            [pair[0] for pair in self.robot_spec.initial_q_ranges],
            dtype=wp.float32,
            device=self.device,
        )
        q_upper = wp.array(
            [pair[1] for pair in self.robot_spec.initial_q_ranges],
            dtype=wp.float32,
            device=self.device,
        )
        qd_lower = wp.array(
            [pair[0] for pair in self.robot_spec.initial_qd_ranges],
            dtype=wp.float32,
            device=self.device,
        )
        qd_upper = wp.array(
            [pair[1] for pair in self.robot_spec.initial_qd_ranges],
            dtype=wp.float32,
            device=self.device,
        )
        wp.launch(
            reset_robot_state,
            dim=self.num_envs,
            inputs=[
                reset_mask,
                self.seed,
                self.random_reset,
                self.dof_q_per_env,
                self.dof_qd_per_env,
                q_lower,
                q_upper,
                qd_lower,
                qd_upper,
            ],
            outputs=[self.state.joint_q, self.state.joint_qd],
            device=self.device,
        )
        self.seed += self.num_envs
        newton.eval_fk(
            model=self.model,
            joint_q=self.state.joint_q,
            joint_qd=self.state.joint_qd,
            state=self.state,
            mask=None,
        )

    def compute_cost_termination(
        self,
        state: newton.State,
        control: newton.Control,
        step: int,
        traj_length: int,
        cost: wp.array,
        terminated: wp.array,
    ):
        return
