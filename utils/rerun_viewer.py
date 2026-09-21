import os
from urllib.parse import quote

import numpy as np
import warp as wp
import newton

from newton._src.viewer.viewer import ViewerBase
from newton._src.viewer.viewer_rerun import ViewerRerun as NewtonViewerRerun


class ViewerRerun(NewtonViewerRerun):
    def __init__(
        self,
        grpc_port: int = 9876,
        web_port: int = 9090,
        browser_host: str = "localhost",
        app_id: str | None = None,
        native_camera: bool = False,
        native_model_path: str | None = None,
    ):
        try:
            import rerun as rr
        except ImportError as exc:
            raise ImportError(
                "Rerun rendering requires rerun-sdk. Install it with: "
                ".venv312/bin/python -m pip install rerun-sdk"
            ) from exc

        self._rr = rr
        self.native_camera = native_camera
        self.native_model_path = native_model_path
        ViewerBase.__init__(self)
        self.server = True
        self.address = f"{browser_host}:{grpc_port}"
        self.launch_viewer = False
        self.app_id = app_id or f"newton-viewer-{os.getpid()}-{web_port}"
        self._running = True
        self._viewer_process = None

        rr.init(self.app_id)
        self.recording = rr.get_global_data_recording()
        if self.recording is None:
            raise RuntimeError("Rerun did not create a global recording stream")
        server_uri = self.recording.serve_grpc(
            grpc_port=grpc_port,
            cors_allow_origin=["*"],
        )
        rr.serve_web_viewer(
            web_port=web_port,
            open_browser=False,
            connect_to=server_uri,
        )

        browser_uri = f"rerun+http://{browser_host}:{grpc_port}/proxy"
        self.web_url = (
            f"http://{browser_host}:{web_port}/?url={quote(browser_uri, safe='')}"
        )
        print("\n" + "=" * 72, flush=True)
        print("RERUN WEB VIEWER URL", flush=True)
        print(self.web_url, flush=True)
        print("=" * 72, flush=True)
        print(f"Rerun data server: {server_uri}", flush=True)

        self._meshes = {}
        self._instances = {}
        self._incoming_xforms = {}
        self._solver = None
        self._mujoco = None
        self._renderer = None
        self._camera = None
        self._render_model = None
        self._render_data = None

    def set_solver(self, solver, width=640, height=480):
        self._solver = solver
        if not self.native_camera:
            return
        try:
            os.environ.setdefault("MUJOCO_GL", "egl")
            import mujoco

            self._mujoco = mujoco
            if self.native_model_path is not None:
                self._render_model = mujoco.MjModel.from_xml_path(self.native_model_path)
                self._render_data = mujoco.MjData(self._render_model)
            else:
                self._render_model = solver.mj_model
                self._render_data = solver.mj_data
            self._render_model.vis.headlight.active = 1
            self._render_model.vis.headlight.ambient[:] = (0.3, 0.3, 0.3)
            self._render_model.vis.headlight.diffuse[:] = (0.7, 0.7, 0.7)
            self._render_model.vis.headlight.specular[:] = (0.9, 0.9, 0.9)
            self._renderer = mujoco.Renderer(
                self._render_model,
                width=width,
                height=height,
            )
            self._camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self._camera)
            self._camera.azimuth = 135.0
            self._camera.elevation = -20.0
        except (AttributeError, ImportError, RuntimeError, ValueError) as exc:
            print(f"Native MuJoCo web rendering unavailable; using mesh view: {exc}")
            self._solver = None
            self._mujoco = None
            self._renderer = None
            self._camera = None
            self._render_model = None
            self._render_data = None

    def set_model(self, model):
        if self._renderer is None or not self.native_camera:
            super().set_model(model)
            return
        if self.model is not None:
            raise RuntimeError("Viewer set_model() can be called only once.")
        self.model = model
        self.device = model.device

    def _should_show_shape(self, flags, is_static):
        if is_static:
            return False
        return super()._should_show_shape(flags, is_static)

    def set_shape_incoming_xform(self, incoming_xforms):
        incoming_np = incoming_xforms.numpy()
        shape_body = self.model.shape_body.numpy()
        shape_type = self.model.shape_type.numpy()
        shape_scale = self.model.shape_scale.numpy()
        shape_thickness = self.model.shape_thickness.numpy()
        shape_is_solid = self.model.shape_is_solid.numpy()
        shape_flags = self.model.shape_flags.numpy()

        grouped_xforms = {}
        for shape_index in range(len(shape_body)):
            geo_scale = tuple(float(value) for value in shape_scale[shape_index])
            geo_hash = self._hash_geometry(
                int(shape_type[shape_index]),
                geo_scale,
                float(shape_thickness[shape_index]),
                bool(shape_is_solid[shape_index]),
                self.model.shape_source[shape_index],
            )
            shape_hash = self._hash_shape(
                geo_hash,
                shape_body[shape_index] == -1,
                shape_flags[shape_index],
            )
            grouped_xforms.setdefault(shape_hash, []).append(incoming_np[shape_index])

        self._incoming_xforms = {
            self._shape_instances[shape_hash].name: np.asarray(values, dtype=np.float32)
            for shape_hash, values in grouped_xforms.items()
            if shape_hash in self._shape_instances
        }

    def log_mesh(
        self,
        name,
        points,
        indices,
        normals=None,
        uvs=None,
        hidden=False,
        backface_culling=True,
    ):
        points_np = np.asarray(points.numpy(), dtype=np.float32)
        indices_np = np.asarray(indices.numpy(), dtype=np.uint32).reshape(-1)
        if len(points_np) == 0 or len(indices_np) % 3:
            return
        indices_np = indices_np.reshape(-1, 3)
        if indices_np.size and int(indices_np.max()) >= len(points_np):
            return

        self._meshes[name] = {
            "points": points_np,
            "indices": indices_np,
            "normals": None,
            "uvs": None,
        }
        if hidden:
            return
        self._rr.log(
            name,
            self._rr.Mesh3D(
                vertex_positions=points_np,
                triangle_indices=indices_np,
            ),
            static=True,
        )

    @staticmethod
    def _rotate_vectors(quaternions, vectors):
        xyz = quaternions[:, :3]
        uv = np.cross(xyz, vectors)
        uuv = np.cross(xyz, uv)
        return vectors + 2.0 * (
            quaternions[:, 3:4] * uv + uuv
        )

    @staticmethod
    def _multiply_quaternions(left, right):
        lx, ly, lz, lw = left.T
        rx, ry, rz, rw = right.T
        return np.column_stack(
            (
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            )
        )

    def _apply_incoming_xforms(self, name, xforms):
        incoming = self._incoming_xforms.get(name)
        if incoming is None or len(incoming) != len(xforms):
            return xforms

        world = xforms.numpy()
        incoming_position = incoming[:, :3]
        incoming_quaternion = incoming[:, 3:7]
        world_position = world[:, :3]
        world_quaternion = world[:, 3:7]
        corrected = np.empty_like(world)
        corrected[:, :3] = incoming_position + self._rotate_vectors(
            incoming_quaternion, world_position
        )
        corrected[:, 3:7] = self._multiply_quaternions(
            incoming_quaternion, world_quaternion
        )
        return wp.array(corrected, dtype=wp.transform, device=xforms.device)

    def log_instances(
        self,
        name,
        mesh,
        xforms,
        scales,
        colors,
        materials,
        hidden=False,
    ):
        if hidden:
            return
        xforms = self._apply_incoming_xforms(name, xforms)
        if xforms is not None:
            xforms_np = xforms.numpy()
            quaternions = xforms_np[:, 3:7]
            norms = np.linalg.norm(quaternions, axis=1)
            invalid = ~np.isfinite(quaternions).all(axis=1) | (norms < 1e-6)
            if np.any(invalid):
                xforms_np = xforms_np.copy()
                xforms_np[invalid, 3:7] = (0.0, 0.0, 0.0, 1.0)
                xforms = wp.array(xforms_np, dtype=wp.transform, device=xforms.device)

        super().log_instances(
            name,
            mesh,
            xforms,
            scales,
            colors,
            materials,
            hidden=hidden,
        )

    def log_state(self, state):
        if self._renderer is None or self._solver is None:
            super().log_state(state)
            return

        if self.native_model_path is not None:
            joint_q = np.asarray(state.joint_q.numpy()).reshape(-1)
            joint_qd = np.asarray(state.joint_qd.numpy()).reshape(-1)
            self._render_data.qpos[:] = joint_q[: self._render_model.nq]
            if self._render_model.nv:
                self._render_data.qvel[:] = joint_qd[: self._render_model.nv]
        else:
            self._solver.update_mjc_data(self._solver.mj_data, self.model, state)
            self._render_data = self._solver.mj_data
        self._mujoco.mj_forward(self._render_model, self._render_data)

        positions = np.asarray(self._render_data.geom_xpos)
        radii = np.asarray(self._render_model.geom_rbound)
        valid = np.isfinite(positions).all(axis=1) & np.isfinite(radii)
        positions = positions[valid]
        radii = radii[valid]
        if len(positions):
            lower = (positions - radii[:, None]).min(axis=0)
            upper = (positions + radii[:, None]).max(axis=0)
            center = 0.5 * (lower + upper)
            radius = max(float(np.linalg.norm(upper - lower)) * 0.5, 0.25)
            self._camera.lookat[:] = center
            self._camera.distance = radius * 2.2

        self._renderer.update_scene(self._render_data, self._camera)
        scene = self._renderer.scene

        frame = np.asarray(self._renderer.render()).copy()
        height, width = frame.shape[:2]
        vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        top = np.array([205.0, 210.0, 220.0], dtype=np.float32)
        bottom = np.array([145.0, 155.0, 170.0], dtype=np.float32)
        background = (top * (1.0 - vertical) + bottom * vertical).astype(np.uint8)
        background = np.broadcast_to(background, (height, width, 3))
        empty = frame.max(axis=2) < 8
        frame[empty] = background[empty]
        self._rr.log("robot/native_camera", self._rr.Image(frame))

    def log_robot_diagnostics(self, step, states, actions, joint_f, robot_spec):
        """Log robot contract signals as Rerun time-series scalars."""
        rr = self._rr
        rr.set_time("step", sequence=int(step))

        states_np = states.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        joint_f_np = joint_f.detach().cpu().numpy()
        q_dim = len(robot_spec.joint_names)
        q = states_np[:, :q_dim]
        qd = states_np[:, q_dim : q_dim + len(robot_spec.joint_names)]
        lower = np.asarray(robot_spec.joint_limits, dtype=np.float32)[:, 0]
        upper = np.asarray(robot_spec.joint_limits, dtype=np.float32)[:, 1]
        action_lower = np.asarray(robot_spec.action_limits, dtype=np.float32)[:, 0]
        action_upper = np.asarray(robot_spec.action_limits, dtype=np.float32)[:, 1]
        saturated = (actions_np < action_lower) | (actions_np > action_upper)
        violations = (q < lower) | (q > upper)

        for env_id in range(q.shape[0]):
            env_path = f"robot/diagnostics/env_{env_id}"
            for joint_id, joint_name in enumerate(robot_spec.joint_names):
                joint_path = f"{env_path}/joints/{joint_name}"
                rr.log(f"{joint_path}/position", rr.Scalars([float(q[env_id, joint_id])]))
                rr.log(f"{joint_path}/velocity", rr.Scalars([float(qd[env_id, joint_id])]))
                rr.log(f"{joint_path}/limit_lower", rr.Scalars([float(lower[joint_id])]))
                rr.log(f"{joint_path}/limit_upper", rr.Scalars([float(upper[joint_id])]))
                rr.log(
                    f"{joint_path}/limit_violation",
                    rr.Scalars([float(violations[env_id, joint_id])]),
                )

            for action_id, joint_name in enumerate(robot_spec.controllable_dofs):
                action_path = f"{env_path}/controls/{joint_name}"
                rr.log(
                    f"{action_path}/requested",
                    rr.Scalars([float(actions_np[env_id, action_id])]),
                )
                rr.log(
                    f"{action_path}/applied_effort",
                    rr.Scalars([float(joint_f_np[env_id, action_id])]),
                )
                rr.log(
                    f"{action_path}/saturated",
                    rr.Scalars([float(saturated[env_id, action_id])]),
                )

        rr.log("robot/diagnostics/summary/limit_violation_count", rr.Scalars([float(violations.sum())]))
        rr.log("robot/diagnostics/summary/saturated_action_count", rr.Scalars([float(saturated.sum())]))
        rr.log("robot/config/actuator_mode", rr.TextLog(robot_spec.actuator_mode))
        for joint_id, joint_name in enumerate(robot_spec.joint_names):
            rr.log(
                f"robot/config/joints/{joint_name}/damping",
                    rr.Scalars([float(robot_spec.damping[joint_id])]),
            )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()