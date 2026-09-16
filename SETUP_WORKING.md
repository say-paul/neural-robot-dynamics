# Working NeRD Newton Setup

This checkout uses the `nerd_newton_dev` branch at commit `476dc67`.

The working environment is:

- venv: `/home/core/neural-robot-dynamics-newton/.venv312`
- Python: 3.12.14
- PyTorch: 2.5.1+cu124
- Warp: 1.12.0
- MuJoCo: 3.13.1
- MuJoCo-Warp: 3.8.1
- Newton: commit `668dfb` (package version 0.1.3)
- GPU: NVIDIA RTX A6000

The branch's unpinned latest Warp/MuJoCo-Warp combination is not compatible
with the older Newton API. Warp 1.12.0 and MuJoCo-Warp 3.8.1 are the tested
pair here.

Robot MJCF imports require the mesh dependencies used by Newton:

```bash
cd /home/core/neural-robot-dynamics-newton
.venv312/bin/python -m pip install trimesh scipy
```

## Generic Robot Setup

Robot support is driven by a specification file in `robot_specs/` and a
compatible MJCF asset. The current Newton robot environment expects the asset
joint order to exactly match the specification. A new robot therefore needs:

- `robot_specs/<robot_id>.yaml`
- an accessible MJCF/XML asset referenced by `asset_source`
- matching `joint_names`, `joint_types`, and `dof`
- `joint_limits`, `velocity_limits`, `effort_limits`, default states, and
  action limits with one entry per joint or controllable joint as appropriate
- an actuator mode (`torque`, `position`, or `velocity`) with suitable damping,
  armature, and position gains

The loader validates array lengths, default positions, controllable joint
names, actuator settings, and positive velocity/effort limits. The importer
also verifies that the joint names and ordering from the MJCF asset match the
YAML specification. URDF files are not currently imported by this robot path.

Available specifications can be selected with `--robot-id`, for example:

```bash
cd /home/core/neural-robot-dynamics-newton
PYTHONPATH="$PWD" .venv312/bin/python examples/example_robot_rollout.py \
    --robot-id so101 --num-envs 1 --horizon 10 --seed 42 --default-pose
```

The Panda fixture uses the default scene from the Panda package at
`/home/core/robotics-rl/modules/body/franka_emika_panda/panda.xml`. It keeps
the arm's seven joints controllable and leaves the two finger joints in the
state contract:

```bash
PYTHONPATH="$PWD" .venv312/bin/python examples/example_robot_rollout.py \
    --robot-id franka_panda --num-envs 1 --horizon 10 --seed 42
```

For another robot, copy the structure of `robot_specs/so101.yaml`, update all
robot-specific values, and verify the asset's imported joint order before
using random actions or a trained policy. A policy must also have the same
action dimension as `controllable_dofs`.

## Rerun Diagnostics and Blueprints

Use the Rerun backend to inspect the robot and log diagnostics:

```bash
PYTHONPATH="$PWD" .venv312/bin/python examples/example_robot_rollout.py \
    --robot-id so101 --num-envs 1 --horizon 1200 --seed 42 --default-pose \
    --render --render-backend rerun --rerun-view camera \
    --grpc-port 19878 --web-port 19092 --browser-host localhost \
    --diagnostics --random-actions --action-scale 2.0 --keep-open
```

The diagnostics logger emits paths for every environment and joint, including
position, velocity, `limit_lower`, `limit_upper`, and limit violation. It also
emits control signals and configured damping values. These values are scalar
series; constant configuration values appear as horizontal lines.

Rerun's automatic layout may show only the currently selected series. The
layout can be controlled explicitly with `rerun.blueprint`: create a
`TimeSeriesView` whose origin is each joint's diagnostic path, then place the
views in a grid. A camera or 3D view can remain in a separate tab:

```python
import rerun as rr
import rerun.blueprint as rrb

joint_views = [
    rrb.TimeSeriesView(
        origin=f"/robot/diagnostics/env_0/joints/{joint_name}",
        contents="$origin/**",
        name=joint_name,
    )
    for joint_name in robot_spec.joint_names
]

rr.send_blueprint(
    rrb.Blueprint(
        rrb.Tabs(
            rrb.Spatial2DView(origin="/robot/native_camera", name="Camera"),
            rrb.Grid(
                contents=joint_views,
                grid_columns=2,
                name="Joint diagnostics",
            ),
            active_tab=0,
        )
    )
)
```

For `--rerun-view 3d`, use `rrb.Spatial3DView` instead of
`rrb.Spatial2DView`. The blueprint should be sent after `rr.init` and before
or during the first diagnostic logging call. For multiple environments, build
one grid per `env_<id>` and place those grids in tabs.

## Headless Panda robot check

Run the generic Phase 1 smoke check with:

```bash
cd /home/core/neural-robot-dynamics-newton
PYTHONPATH="$PWD" .venv312/bin/python examples/example_robot_rollout.py \
    --robot-id franka_panda --num-envs 1 --horizon 10 --seed 42
```

## Headless ground-truth check

```bash
cd /home/core/neural-robot-dynamics-newton
PYTHONPATH="$PWD" .venv312/bin/python - <<'PY'
import torch
from envs.neural_environment import NeuralEnvironment

env = NeuralEnvironment(
    env_name="Cartpole",
    num_envs=1,
    newton_env_cfg={"seed": 1234, "random_reset": True},
    default_env_mode="ground-truth",
    render=False,
)
try:
    env.reset()
    action = torch.zeros((env.num_envs, env.action_dim), device=env.torch_device)
    for _ in range(5):
        state = env.step(action)
    assert torch.isfinite(state).all()
    print(state.shape, env.action_dim)
finally:
    env.close()
PY
```

The pretrained Cartpole checkpoint is present at:

```text
pretrained_models/NeRD_models/Cartpole/model/nn/model.pt
```

The stock example opens a viewer and needs X11/Xvfb. The same neural solver was
validated headlessly with this checkpoint for five finite steps.
