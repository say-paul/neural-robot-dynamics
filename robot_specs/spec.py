from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


def _tuple_of_strings(values: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings")
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise ValueError(f"{field_name} cannot contain empty names")
    return result


def _tuple_of_pairs(values: Any, field_name: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of [lower, upper] pairs")
    result = tuple((float(pair[0]), float(pair[1])) for pair in values)
    if any(lower > upper for lower, upper in result):
        raise ValueError(f"{field_name} contains an inverted range")
    return result


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    asset_source: str
    base_type: str
    dof: int
    joint_names: tuple[str, ...]
    joint_types: tuple[str, ...]
    controllable_dofs: tuple[str, ...]
    joint_limits: tuple[tuple[float, float], ...]
    velocity_limits: tuple[float, ...]
    effort_limits: tuple[float, ...]
    default_q: tuple[float, ...]
    default_qd: tuple[float, ...]
    action_limits: tuple[tuple[float, float], ...]
    actuator_mode: str
    damping: tuple[float, ...]
    armature: tuple[float, ...]
    position_gains: tuple[float, ...] | None = None
    initial_q_ranges: tuple[tuple[float, float], ...] | None = None
    initial_qd_ranges: tuple[tuple[float, float], ...] | None = None
    end_effector: Mapping[str, Any] | None = None
    gripper: Mapping[str, Any] | None = None
    contact_groups: Mapping[str, Any] = field(default_factory=dict)
    solver: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.dof != len(self.joint_names):
            raise ValueError(f"dof={self.dof} does not match joint_names")
        if len(set(self.joint_names)) != self.dof:
            raise ValueError("joint_names must be unique")
        if len(self.joint_types) != self.dof:
            raise ValueError("joint_types must match dof")
        for field_name, values in (
            ("joint_limits", self.joint_limits),
            ("velocity_limits", self.velocity_limits),
            ("effort_limits", self.effort_limits),
            ("default_q", self.default_q),
            ("default_qd", self.default_qd),
            ("damping", self.damping),
            ("armature", self.armature),
        ):
            if len(values) != self.dof:
                raise ValueError(f"{field_name} must match dof")
        if self.position_gains is None:
            object.__setattr__(self, "position_gains", tuple(0.0 for _ in range(self.dof)))
        elif len(self.position_gains) != self.dof:
            raise ValueError("position_gains must match dof")
        if self.initial_q_ranges is not None and len(self.initial_q_ranges) != self.dof:
            raise ValueError("initial_q_ranges must match dof")
        if self.initial_qd_ranges is not None and len(self.initial_qd_ranges) != self.dof:
            raise ValueError("initial_qd_ranges must match dof")
        if not self.controllable_dofs:
            raise ValueError("controllable_dofs cannot be empty")
        unknown_dofs = set(self.controllable_dofs) - set(self.joint_names)
        if unknown_dofs:
            raise ValueError(f"controllable_dofs contains unknown joints: {sorted(unknown_dofs)}")
        if len(self.action_limits) != len(self.controllable_dofs):
            raise ValueError("action_limits must match controllable_dofs")
        if self.base_type not in {"fixed", "floating"}:
            raise ValueError("base_type must be 'fixed' or 'floating'")
        if self.actuator_mode not in {"torque", "position", "velocity"}:
            raise ValueError("actuator_mode must be torque, position, or velocity")
        if self.actuator_mode == "position" and any(value <= 0 for value in self.position_gains):
            raise ValueError("position_gains must be positive for position actuators")
        for index, (lower, upper) in enumerate(self.joint_limits):
            if not lower <= self.default_q[index] <= upper:
                raise ValueError(f"default_q[{index}] is outside joint_limits")
        if any(value <= 0 for value in self.velocity_limits):
            raise ValueError("velocity_limits must be positive")
        if any(value <= 0 for value in self.effort_limits):
            raise ValueError("effort_limits must be positive")

    @property
    def action_dim(self) -> int:
        return len(self.controllable_dofs)

    @property
    def joint_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.joint_names)}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, source: Path | None = None) -> RobotSpec:
        required = {
            "robot_id",
            "asset_source",
            "base_type",
            "dof",
            "joint_names",
            "joint_types",
            "controllable_dofs",
            "joint_limits",
            "velocity_limits",
            "effort_limits",
            "default_q",
            "default_qd",
            "action_limits",
            "actuator_mode",
        }
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Robot spec is missing fields: {sorted(missing)}")

        asset_source = os.path.expandvars(os.path.expanduser(str(data["asset_source"])))
        asset_path = Path(asset_source)
        if source is not None and not asset_path.is_absolute():
            asset_path = source.parent / asset_path

        joint_limits = _tuple_of_pairs(data["joint_limits"], "joint_limits")
        initial_q_ranges = data.get("initial_q_ranges")
        initial_qd_ranges = data.get("initial_qd_ranges")
        return cls(
            robot_id=str(data["robot_id"]),
            asset_source=str(asset_path),
            base_type=str(data["base_type"]),
            dof=int(data["dof"]),
            joint_names=_tuple_of_strings(data["joint_names"], "joint_names"),
            joint_types=_tuple_of_strings(data["joint_types"], "joint_types"),
            controllable_dofs=_tuple_of_strings(data["controllable_dofs"], "controllable_dofs"),
            joint_limits=joint_limits,
            velocity_limits=tuple(float(value) for value in data["velocity_limits"]),
            effort_limits=tuple(float(value) for value in data["effort_limits"]),
            default_q=tuple(float(value) for value in data["default_q"]),
            default_qd=tuple(float(value) for value in data["default_qd"]),
            action_limits=_tuple_of_pairs(data["action_limits"], "action_limits"),
            actuator_mode=str(data["actuator_mode"]),
            damping=tuple(float(value) for value in data.get("damping", [0.0] * int(data["dof"]))),
            armature=tuple(float(value) for value in data.get("armature", [0.01] * int(data["dof"]))),
            position_gains=tuple(
                float(value) for value in data.get("position_gains", [0.0] * int(data["dof"]))
            ),
            initial_q_ranges=(
                _tuple_of_pairs(initial_q_ranges, "initial_q_ranges")
                if initial_q_ranges is not None
                else joint_limits
            ),
            initial_qd_ranges=(
                _tuple_of_pairs(initial_qd_ranges, "initial_qd_ranges")
                if initial_qd_ranges is not None
                else tuple((-value, value) for value in data["velocity_limits"])
            ),
            end_effector=data.get("end_effector"),
            gripper=data.get("gripper"),
            contact_groups=data.get("contact_groups", {}),
            solver=data.get("solver", {}),
        )


def load_robot_spec(path_or_id: str | os.PathLike[str]) -> RobotSpec:
    requested = Path(path_or_id)
    if requested.exists():
        path = requested
    else:
        path = Path(__file__).parent / f"{path_or_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Robot spec not found: {path_or_id}")
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, Mapping):
        raise ValueError(f"Robot spec must contain a mapping: {path}")
    return RobotSpec.from_mapping(data, source=path)