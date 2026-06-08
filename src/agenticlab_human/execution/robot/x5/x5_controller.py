"""X5 controller contract, mock implementation, and real X5 adapter."""

from __future__ import annotations

import copy
import importlib
import math
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from agenticlab_human.execution.robot.x5.contracts import (
    ArmState,
    ComponentHealth,
    GetStateCommand,
    MoveJointsCommand,
    RobotCommand,
    RobotState,
    StopCommand,
)


@runtime_checkable
class X5Controller(Protocol):
    def initialize(self) -> None:
        """Connect and prepare robot resources."""

    def execute(self, command: RobotCommand) -> None:
        """Execute one validated low-level command."""

    def get_state(self) -> RobotState:
        """Read current robot state."""

    def health(self) -> ComponentHealth:
        """Return current controller readiness."""

    def shutdown(self) -> None:
        """Stop motion and release resources."""


class MockX5Controller:
    """Immediate, in-memory execution model for both X5 arms."""

    def __init__(
        self,
        arms: Sequence[str] = ("left", "right"),
        initial_joints_rad: Sequence[float] | None = None,
    ) -> None:
        initial = list(initial_joints_rad or [0.0] * 7)
        if len(initial) != 7:
            raise ValueError("initial_joints_rad must contain 7 values")
        if not arms:
            raise ValueError("at least one mock arm is required")

        self._initialized = False
        self._lock = threading.Lock()
        self._arms = {
            str(arm): ArmState(
                connected=False,
                moving=False,
                joints_rad=initial.copy(),
                tcp_pose_xyzw=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                gripper_position=1.0,
            )
            for arm in arms
        }

    def initialize(self) -> None:
        with self._lock:
            self._initialized = True
            for state in self._arms.values():
                state.connected = True

    def execute(self, command: RobotCommand) -> None:
        with self._lock:
            self._require_initialized()
            if isinstance(command, GetStateCommand):
                self._require_target(command.arm)
                return
            if isinstance(command, MoveJointsCommand):
                state = self._require_arm(command.arm)
                state.moving = True
                state.joints_rad = [float(value) for value in command.joints_rad]
                state.moving = False
                return
            if isinstance(command, StopCommand):
                for state in self._selected_states(command.arm):
                    state.moving = False
                return
            raise ValueError(f"unsupported robot command: {type(command).__name__}")

    def get_state(self) -> RobotState:
        with self._lock:
            self._require_initialized()
            arms = {name: copy.deepcopy(state) for name, state in self._arms.items()}
        return RobotState(arms=arms, timestamp_ns=time.time_ns())

    def health(self) -> ComponentHealth:
        with self._lock:
            initialized = self._initialized
        return ComponentHealth(
            ready=initialized,
            backend="mock",
            detail="ready" if initialized else "not initialized",
        )

    def shutdown(self) -> None:
        with self._lock:
            for state in self._arms.values():
                state.moving = False
                state.connected = False
            self._initialized = False

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("mock X5 controller is not initialized")

    def _require_arm(self, arm: str) -> ArmState:
        try:
            return self._arms[arm]
        except KeyError as exc:
            raise ValueError(f"mock arm is not configured: {arm}") from exc

    def _require_target(self, arm: str) -> None:
        if arm != "all":
            self._require_arm(arm)

    def _selected_states(self, arm: str) -> list[ArmState]:
        if arm == "all":
            return list(self._arms.values())
        return [self._require_arm(arm)]


class RealX5Controller:
    """Minimal xapi-backed X5 controller for discrete low-speed commands.

    Public HTTP contracts stay in radians/meters/quaternions. The xapi-specific
    degrees and SDK objects are converted here on the server PC.
    """

    DEFAULT_JOINT_LIMITS_DEG = (
        (-165.0, 165.0),
        (-28.0, 90.0),
        (-153.0, 153.0),
        (-28.0, 103.0),
        (-165.0, 165.0),
        (-28.0, 98.0),
        (-170.0, 170.0),
    )
    DEFAULT_HEAD_JOINTS_DEG = (0.0, 28.0)

    def __init__(
        self,
        arm_configs: Mapping[str, Mapping[str, Any]],
        *,
        mode: int = 100,
        remote: bool = False,
        enable_servo_on_start: bool = True,
        default_speed_ratio: float = 0.05,
        max_command_speed_ratio: float = 0.1,
        move_timeout_ms: int = 60_000,
        max_joint_delta_deg: float = 5.0,
        joint_limits_deg: Sequence[Sequence[float]] | None = None,
        stop_on_shutdown: bool = False,
        x5_api: Any | None = None,
    ) -> None:
        if not arm_configs:
            raise ValueError("at least one X5 arm config is required")

        self._arm_configs = {
            str(arm): dict(config)
            for arm, config in arm_configs.items()
        }
        for arm, config in self._arm_configs.items():
            if not config.get("robot_ip"):
                raise ValueError(f"missing robot_ip for X5 arm: {arm}")
        self._home_joints_deg = {
            arm: _first_seven_floats(config.get("home_joints_deg"))
            for arm, config in self._arm_configs.items()
            if config.get("home_joints_deg") is not None
        }
        self._head_joints_deg = {
            arm: _two_floats(config.get("head_joints_deg", self.DEFAULT_HEAD_JOINTS_DEG))
            for arm, config in self._arm_configs.items()
        }

        self._mode = int(mode)
        self._remote = bool(remote)
        self._enable_servo_on_start = bool(enable_servo_on_start)
        self._default_speed_ratio = float(default_speed_ratio)
        self._max_command_speed_ratio = float(max_command_speed_ratio)
        self._move_timeout_ms = int(move_timeout_ms)
        self._max_joint_delta_deg = float(max_joint_delta_deg)
        self._joint_limits_deg = _normalize_joint_limits(
            joint_limits_deg or self.DEFAULT_JOINT_LIMITS_DEG
        )
        self._stop_on_shutdown = bool(stop_on_shutdown)

        if self._default_speed_ratio <= 0.0:
            raise ValueError("default_speed_ratio must be positive")
        if not (0.0 < self._max_command_speed_ratio <= 1.0):
            raise ValueError("max_command_speed_ratio must be in (0, 1]")
        if self._default_speed_ratio > self._max_command_speed_ratio:
            raise ValueError("default_speed_ratio cannot exceed max_command_speed_ratio")
        if self._max_joint_delta_deg <= 0.0:
            raise ValueError("max_joint_delta_deg must be positive")

        self._lock = threading.RLock()
        self._x5 = x5_api
        self._handles: dict[str, Any] = {}
        self._versions: dict[str, str] = {}
        self._last_error = ""
        self._initialized = False

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return

            x5 = self._load_x5()
            self._try_call(getattr(x5, "enable_debug_output", None), 0)

            try:
                for arm, config in self._arm_configs.items():
                    ip = str(config["robot_ip"])
                    handle = x5.connect(ip)
                    if handle is None or handle == -1:
                        raise RuntimeError(
                            f"xapi connect failed for {arm} arm at {ip}: {handle}"
                        )
                    self._handles[arm] = handle

                    version = self._try_call(getattr(x5, "get_version", None), handle)
                    if version is not None:
                        self._versions[arm] = str(version)

                    self._set_mode(handle)
                self._initialized = True
                self._last_error = ""
            except Exception as exc:
                self._last_error = str(exc)
                self._disconnect_all()
                raise

    def execute(self, command: RobotCommand) -> None:
        with self._lock:
            self._require_initialized()
            if isinstance(command, GetStateCommand):
                self._require_target(command.arm)
                return
            if isinstance(command, StopCommand):
                self._stop_target(command.arm)
                return
            if isinstance(command, MoveJointsCommand):
                self._move_joints(command)
                return
            raise ValueError(f"unsupported robot command: {type(command).__name__}")

    def get_state(self) -> RobotState:
        with self._lock:
            self._require_initialized()
            arms = {
                arm: self._read_arm_state(arm, handle)
                for arm, handle in self._handles.items()
            }
        return RobotState(arms=arms, timestamp_ns=time.time_ns())

    def health(self) -> ComponentHealth:
        with self._lock:
            ready = self._initialized and len(self._handles) == len(self._arm_configs)
            if ready:
                arm_details = ", ".join(
                    f"{arm}@{self._arm_configs[arm]['robot_ip']}"
                    for arm in sorted(self._handles)
                )
                detail = f"ready: {arm_details}"
                if self._versions:
                    detail += f"; versions={self._versions}"
            else:
                detail = self._last_error or "not initialized"
        return ComponentHealth(ready=ready, backend="x5", detail=detail)

    def shutdown(self) -> None:
        with self._lock:
            if self._stop_on_shutdown:
                self._stop_target("all", suppress_errors=True)
            self._disconnect_all()
            self._initialized = False

    def _load_x5(self) -> Any:
        if self._x5 is None:
            self._x5 = importlib.import_module("xapi.api")
        return self._x5

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("real X5 controller is not initialized")

    def _require_target(self, arm: str) -> None:
        if arm == "all":
            return
        self._require_handle(arm)

    def _require_handle(self, arm: str) -> Any:
        try:
            return self._handles[arm]
        except KeyError as exc:
            raise ValueError(f"X5 arm is not configured or connected: {arm}") from exc

    def _set_mode(self, handle: Any) -> None:
        x5 = self._load_x5()
        set_system_mode = getattr(x5, "set_system_mode", None)
        if set_system_mode is not None:
            set_system_mode(handle, self._mode)

        if self._enable_servo_on_start:
            self._set_servo_enabled(handle, True)

        if self._remote:
            set_remote = getattr(x5, "set_remote", None)
            if set_remote is None:
                raise RuntimeError("xapi.set_remote is unavailable")
            set_remote(handle, True)

    def _set_servo_enabled(self, handle: Any, enabled: bool) -> None:
        x5 = self._load_x5()
        enable_servo = getattr(x5, "enable_servo", None) or getattr(x5, "servo_enable", None)
        if enable_servo is None:
            raise RuntimeError("xapi servo enable function is unavailable")
        enable_servo(handle, bool(enabled))

    def _read_arm_state(self, arm: str, handle: Any) -> ArmState:
        x5 = self._load_x5()
        joints_deg = _joint_to_degrees(x5.get_cjoint(handle))
        tcp_pose_xyzw = self._read_tcp_pose_xyzw(handle)
        moving = False
        system_state = self._try_call(getattr(x5, "get_system_state", None), handle)
        if system_state is not None:
            moving = _state_is_moving(system_state)

        return ArmState(
            connected=True,
            moving=moving,
            joints_rad=[math.radians(value) for value in joints_deg],
            tcp_pose_xyzw=tcp_pose_xyzw,
            gripper_position=None,
        )

    def _read_tcp_pose_xyzw(self, handle: Any) -> list[float]:
        x5 = self._load_x5()
        get_cpoint = getattr(x5, "get_cpoint", None)
        if get_cpoint is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        point = self._try_call(get_cpoint, handle)
        if point is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        try:
            pose = getattr(point, "pose", point)
            x_mm = _read_value(pose, "x", 0)
            y_mm = _read_value(pose, "y", 1)
            z_mm = _read_value(pose, "z", 2)
            a_deg = _read_value(pose, "a", 3, default=0.0)
            b_deg = _read_value(pose, "b", 4, default=0.0)
            c_deg = _read_value(pose, "c", 5, default=0.0)
        except (TypeError, ValueError):
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        qx, qy, qz, qw = _euler_xyz_deg_to_quat_xyzw(a_deg, b_deg, c_deg)
        return [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0, qx, qy, qz, qw]

    def _stop_target(self, arm: str, *, suppress_errors: bool = False) -> None:
        handles = self._selected_handles(arm)
        errors: list[str] = []
        for selected_arm, handle in handles:
            try:
                self._stop_handle(handle)
            except Exception as exc:
                if not suppress_errors:
                    errors.append(f"{selected_arm}: {exc}")
        if errors:
            raise RuntimeError("failed to stop X5 arm(s): " + "; ".join(errors))

    def _stop_handle(self, handle: Any) -> None:
        x5 = self._load_x5()
        x5.stop(handle)
        self._wait_cmd_send_done(handle)
        abort = getattr(x5, "abort", None)
        if abort is not None:
            abort(handle)
            self._wait_cmd_send_done(handle)
        self._wait_move_done(handle)

    def _move_joints(self, command: MoveJointsCommand) -> None:
        if command.speed_ratio > self._max_command_speed_ratio:
            raise ValueError(
                f"speed_ratio {command.speed_ratio:.3f} exceeds configured "
                f"max_command_speed_ratio {self._max_command_speed_ratio:.3f}"
            )

        handle = self._require_handle(command.arm)
        target_deg = [math.degrees(value) for value in command.joints_rad]
        current_deg = _joint_to_degrees(self._load_x5().get_cjoint(handle))
        self._check_joint_target(command.arm, current_deg, target_deg)

        target = self._make_joint(command.arm, target_deg)
        add_data = self._make_mov_point_add(command.speed_ratio)
        x5 = self._load_x5()

        try:
            if add_data is not None:
                x5.movj(handle, target, add_data)
            else:
                x5.movj(handle, target)
        except TypeError:
            x5.movj(handle, target)

        self._wait_cmd_send_done(handle)
        if command.wait:
            self._wait_move_done(handle)

    def _check_joint_target(
        self,
        arm: str,
        current_deg: Sequence[float],
        target_deg: Sequence[float],
    ) -> None:
        for index, (value, (lower, upper)) in enumerate(
            zip(target_deg, self._joint_limits_deg, strict=True),
            start=1,
        ):
            if value < lower or value > upper:
                raise ValueError(
                    f"{arm} joint {index} target {value:.3f} deg is outside "
                    f"configured limit [{lower:.3f}, {upper:.3f}] deg"
                )

        deltas = [
            abs(target - current)
            for target, current in zip(target_deg, current_deg, strict=True)
        ]
        max_delta = max(deltas)
        if max_delta > self._max_joint_delta_deg:
            raise ValueError(
                f"{arm} move_joints max delta {max_delta:.3f} deg exceeds "
                f"configured max_joint_delta_deg {self._max_joint_delta_deg:.3f}"
            )

    def _make_joint(self, arm: str, joints_deg: Sequence[float]) -> Any:
        x5 = self._load_x5()
        joint_cls = getattr(x5, "Joint", None)
        if joint_cls is None:
            return list(joints_deg) + self._head_joints_deg[arm]

        # xapi Joint carries the 7 arm axes plus external axes. The last two
        # values are the head joints on the current X5 setup; keep them stable
        # across arm-only move_joints commands instead of implicitly zeroing them.
        values9 = list(joints_deg) + self._head_joints_deg[arm]
        try:
            return joint_cls(*values9)
        except TypeError:
            try:
                return joint_cls(*joints_deg)
            except TypeError:
                joint = joint_cls()
                for name, value in zip(
                    ("j1", "j2", "j3", "j4", "j5", "j6", "e1"),
                    joints_deg,
                    strict=True,
                ):
                    setattr(joint, name, float(value))
                return joint

    def _make_mov_point_add(self, speed_ratio: float) -> Any | None:
        mov_point_add_cls = getattr(self._load_x5(), "MovPointAdd", None)
        if mov_point_add_cls is None:
            return None

        speed_percent = max(1, min(100, int(round(speed_ratio * 100.0))))
        try:
            return mov_point_add_cls(vel=speed_percent, acc=speed_percent)
        except TypeError:
            add = mov_point_add_cls()
            self._try_setattr(add, "vel", speed_percent)
            self._try_setattr(add, "acc", speed_percent)
            return add

    def _wait_cmd_send_done(self, handle: Any) -> None:
        wait_cmd_send_done = getattr(self._load_x5(), "wait_cmd_send_done", None)
        if wait_cmd_send_done is not None:
            wait_cmd_send_done(handle)

    def _wait_move_done(self, handle: Any) -> None:
        wait_move_done = getattr(self._load_x5(), "wait_move_done", None)
        if wait_move_done is None:
            return
        try:
            wait_move_done(handle, self._move_timeout_ms)
        except TypeError:
            wait_move_done(handle)

    def _selected_handles(self, arm: str) -> list[tuple[str, Any]]:
        if arm == "all":
            return list(self._handles.items())
        return [(arm, self._require_handle(arm))]

    def _disconnect_all(self) -> None:
        x5 = self._load_x5()
        disconnect = getattr(x5, "disconnect", None)
        if disconnect is not None:
            for handle in list(self._handles.values()):
                self._try_call(disconnect, handle)
        self._handles.clear()
        self._versions.clear()

    @staticmethod
    def _try_setattr(obj: Any, name: str, value: Any) -> None:
        try:
            setattr(obj, name, value)
        except Exception:
            pass

    @staticmethod
    def _try_call(func: Any | None, *args: Any) -> Any | None:
        if func is None:
            return None
        try:
            return func(*args)
        except Exception:
            return None


def _normalize_joint_limits(
    joint_limits_deg: Sequence[Sequence[float]],
) -> list[tuple[float, float]]:
    limits = [tuple(float(value) for value in limit) for limit in joint_limits_deg]
    if len(limits) != 7 or any(len(limit) != 2 for limit in limits):
        raise ValueError("joint_limits_deg must contain 7 [lower, upper] pairs")
    for index, (lower, upper) in enumerate(limits, start=1):
        if lower >= upper:
            raise ValueError(f"invalid joint limit for joint {index}: {lower} >= {upper}")
    return [(lower, upper) for lower, upper in limits]


def _first_seven_floats(values: Any) -> list[float]:
    if values is None:
        raise ValueError("expected 7 joint values, got None")
    converted = [float(value) for value in values]
    if len(converted) < 7:
        raise ValueError("expected at least 7 joint values")
    return converted[:7]


def _two_floats(values: Any) -> list[float]:
    if values is None:
        raise ValueError("expected 2 head joint values, got None")
    converted = [float(value) for value in values]
    if len(converted) != 2:
        raise ValueError("head_joints_deg must contain exactly 2 values")
    return converted


def _joint_to_degrees(joint: Any) -> list[float]:
    if isinstance(joint, Mapping):
        names = ("j1", "j2", "j3", "j4", "j5", "j6")
        values = [float(joint[name]) for name in names]
        values.append(float(joint.get("e1", joint.get("j7"))))
        return values

    if isinstance(joint, Sequence) and not isinstance(joint, (str, bytes)):
        values = [float(value) for value in list(joint)[:7]]
        if len(values) == 7:
            return values

    names = ("j1", "j2", "j3", "j4", "j5", "j6")
    values = [float(getattr(joint, name)) for name in names]
    if hasattr(joint, "e1"):
        values.append(float(getattr(joint, "e1")))
    else:
        values.append(float(getattr(joint, "j7")))
    return values


def _read_value(obj: Any, attr_name: str, index: int, *, default: float | None = None) -> float:
    if isinstance(obj, Mapping) and attr_name in obj:
        return float(obj[attr_name])
    if hasattr(obj, attr_name):
        return float(getattr(obj, attr_name))
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)) and len(obj) > index:
        return float(obj[index])
    if default is not None:
        return default
    raise ValueError(f"missing value: {attr_name}")


def _state_is_moving(system_state: Any) -> bool:
    for attr in ("moving", "is_moving", "in_motion"):
        if hasattr(system_state, attr):
            return bool(getattr(system_state, attr))
    if isinstance(system_state, Mapping):
        for key in ("moving", "is_moving", "in_motion"):
            if key in system_state:
                return bool(system_state[key])
    return False


def _euler_xyz_deg_to_quat_xyzw(a_deg: float, b_deg: float, c_deg: float) -> tuple[float, float, float, float]:
    roll = math.radians(a_deg)
    pitch = math.radians(b_deg)
    yaw = math.radians(c_deg)

    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw
