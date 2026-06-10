"""X5 controller contract, mock implementation, and real X5 adapter."""

from __future__ import annotations

import copy
import importlib
import logging
import math
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from agenticlab_human.execution.robot.x5.contracts import (
    ArmState,
    ComponentHealth,
    GetStateCommand,
    MoveJPointCommand,
    MoveJointsCommand,
    MoveLPointCommand,
    RobotCommand,
    RobotState,
    StopCommand,
)

logger = logging.getLogger("uvicorn.error")


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
            command_type = command.type
            if command_type == "get_state":
                self._require_target(command.arm)
                return
            if command_type == "move_joints":
                state = self._require_arm(command.arm)
                state.moving = True
                state.joints_rad = [float(value) for value in command.joints_rad]
                state.moving = False
                return
            if command_type in ("movej_point", "movel_point"):
                state = self._require_arm(command.arm)
                state.moving = True
                xyz = [float(value) for value in command.tcp_pose_xyz_rotvec[:3]]
                quaternion = _rotvec_to_quat_xyzw(command.tcp_pose_xyz_rotvec[3:])
                state.tcp_pose_xyzw = xyz + list(quaternion)
                state.moving = False
                return
            if command_type == "stop":
                for state in self._selected_states(command.arm):
                    state.moving = False
                return
            raise ValueError(
                f"unsupported robot command type={command_type!r} "
                f"class={type(command).__module__}.{type(command).__name__}"
            )

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
        max_movej_point_translation_m: float = 0.5,
        max_movej_point_rotation_deg: float = 180.0,
        max_movel_point_translation_m: float = 0.1,
        max_movel_point_rotation_deg: float = 30.0,
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
        self._requested_tool_frame_nos = {
            arm: _tool_frame_no(config)
            for arm, config in self._arm_configs.items()
        }
        self._requested_tool_frame_poses = {
            arm: _tool_frame_pose(config)
            for arm, config in self._arm_configs.items()
        }

        self._mode = int(mode)
        self._remote = bool(remote)
        self._enable_servo_on_start = bool(enable_servo_on_start)
        self._default_speed_ratio = float(default_speed_ratio)
        self._max_command_speed_ratio = float(max_command_speed_ratio)
        self._move_timeout_ms = int(move_timeout_ms)
        self._max_joint_delta_deg = float(max_joint_delta_deg)
        self._max_movej_point_translation_m = float(max_movej_point_translation_m)
        self._max_movej_point_rotation_deg = float(max_movej_point_rotation_deg)
        self._max_movel_point_translation_m = float(max_movel_point_translation_m)
        self._max_movel_point_rotation_deg = float(max_movel_point_rotation_deg)
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
        cartesian_limits = {
            "max_movej_point_translation_m": self._max_movej_point_translation_m,
            "max_movej_point_rotation_deg": self._max_movej_point_rotation_deg,
            "max_movel_point_translation_m": self._max_movel_point_translation_m,
            "max_movel_point_rotation_deg": self._max_movel_point_rotation_deg,
        }
        for name, value in cartesian_limits.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

        self._lock = threading.RLock()
        self._x5 = x5_api
        self._handles: dict[str, Any] = {}
        self._versions: dict[str, str] = {}
        self._active_tool_frame_nos: dict[str, int] = {}
        self._tool_frame_poses_xyzw: dict[str, list[float]] = {}
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
                    self._initialize_tool_frame(arm, handle)
                self._initialized = True
                self._last_error = ""
            except Exception as exc:
                self._last_error = str(exc)
                self._disconnect_all()
                raise

    def execute(self, command: RobotCommand) -> None:
        with self._lock:
            self._require_initialized()
            command_type = command.type
            logger.info(
                "X5 controller dispatch: type=%s class=%s.%s payload=%s",
                command_type,
                type(command).__module__,
                type(command).__name__,
                command.model_dump(mode="json"),
            )
            if command_type == "get_state":
                self._require_target(command.arm)
                return
            if command_type == "stop":
                self._stop_target(command.arm)
                return
            if command_type == "move_joints":
                self._move_joints(command)
                return
            if command_type == "movej_point":
                self._move_point(command, motion="movj")
                return
            if command_type == "movel_point":
                self._move_point(command, motion="movl")
                return
            raise ValueError(
                f"unsupported robot command type={command_type!r} "
                f"class={type(command).__module__}.{type(command).__name__}"
            )

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
                if self._active_tool_frame_nos:
                    detail += f"; tool_frames={self._active_tool_frame_nos}"
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

    def _initialize_tool_frame(self, arm: str, handle: Any) -> None:
        x5 = self._load_x5()
        requested_tf_no = self._requested_tool_frame_nos[arm]
        get_tf = getattr(x5, "get_tf", None)
        get_tfno = getattr(x5, "get_tfno", None)

        if requested_tf_no is not None:
            if get_tf is None:
                raise RuntimeError("xapi.get_tf is unavailable")
            requested_pose = self._requested_tool_frame_poses[arm]
            if requested_pose is not None:
                if requested_tf_no == 0:
                    raise ValueError("TF0 cannot be overwritten with a tool frame offset")
                set_tf = getattr(x5, "set_tf", None)
                if set_tf is None:
                    raise RuntimeError("xapi.set_tf is unavailable")
                set_tf(
                    handle,
                    requested_tf_no,
                    self._make_tool_frame_pose(requested_pose),
                )

            tool_pose = get_tf(handle, requested_tf_no)
            tool_pose_xyzw = _pose_to_xyzw(tool_pose)
            if requested_pose is not None:
                _validate_tool_frame_pose(
                    arm,
                    requested_tf_no,
                    requested_pose,
                    tool_pose_xyzw,
                )
            self._tool_frame_poses_xyzw[arm] = tool_pose_xyzw

            set_tfno = getattr(x5, "set_tfno", None)
            if set_tfno is None:
                raise RuntimeError("xapi.set_tfno is unavailable")
            set_tfno(handle, requested_tf_no)

            if get_tfno is None:
                raise RuntimeError("xapi.get_tfno is unavailable")
            active_tf_no = int(get_tfno(handle))
            if active_tf_no != requested_tf_no:
                raise RuntimeError(
                    f"{arm} X5 active TF is {active_tf_no}, expected {requested_tf_no}"
                )
            self._active_tool_frame_nos[arm] = active_tf_no
            return

        if get_tfno is None:
            return
        active_tf_no = int(get_tfno(handle))
        self._active_tool_frame_nos[arm] = active_tf_no
        if get_tf is not None:
            self._tool_frame_poses_xyzw[arm] = _pose_to_xyzw(
                get_tf(handle, active_tf_no)
            )

    def _make_tool_frame_pose(self, pose_config: Sequence[float]) -> Any:
        pose_cls = getattr(self._load_x5(), "Pose", None)
        if pose_cls is None:
            raise RuntimeError("xapi.Pose is unavailable")

        x_m, y_m, z_m, roll_deg, pitch_deg, yaw_deg = pose_config
        values = (
            x_m * 1000.0,
            y_m * 1000.0,
            z_m * 1000.0,
            roll_deg,
            pitch_deg,
            yaw_deg,
            0.0,
            0.0,
            0.0,
        )
        try:
            return pose_cls(*values)
        except TypeError:
            return pose_cls(
                x=values[0],
                y=values[1],
                z=values[2],
                a=values[3],
                b=values[4],
                c=values[5],
            )

    def _read_arm_state(self, arm: str, handle: Any) -> ArmState:
        x5 = self._load_x5()
        joints_deg = _joint_to_degrees(x5.get_cjoint(handle))
        self._refresh_tool_frame_state(arm, handle)
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
            tool_frame_no=self._active_tool_frame_nos.get(arm),
            tool_frame_pose_xyzw=self._tool_frame_poses_xyzw.get(arm),
            gripper_position=None,
        )

    def _refresh_tool_frame_state(self, arm: str, handle: Any) -> None:
        x5 = self._load_x5()
        get_tfno = getattr(x5, "get_tfno", None)
        if get_tfno is None:
            return

        active_tf_no = int(get_tfno(handle))
        previous_tf_no = self._active_tool_frame_nos.get(arm)
        self._active_tool_frame_nos[arm] = active_tf_no
        if previous_tf_no == active_tf_no and arm in self._tool_frame_poses_xyzw:
            return

        get_tf = getattr(x5, "get_tf", None)
        if get_tf is not None:
            self._tool_frame_poses_xyzw[arm] = _pose_to_xyzw(
                get_tf(handle, active_tf_no)
            )

    def _read_tcp_pose_xyzw(self, handle: Any) -> list[float]:
        x5 = self._load_x5()
        get_wpoint = getattr(x5, "get_wpoint", None)
        if get_wpoint is not None:
            return _pose_to_xyzw(get_wpoint(handle))

        get_cpoint = getattr(x5, "get_cpoint", None)
        if get_cpoint is None:
            raise RuntimeError("xapi.get_wpoint and xapi.get_cpoint are unavailable")
        return _pose_to_xyzw(get_cpoint(handle))

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
        self._check_speed_ratio(command.speed_ratio)

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

    def _move_point(
        self,
        command: MoveJPointCommand | MoveLPointCommand,
        *,
        motion: str,
    ) -> None:
        self._check_speed_ratio(command.speed_ratio)
        handle = self._require_handle(command.arm)
        pose6d = _six_finite_floats(command.tcp_pose_xyz_rotvec)
        self._check_cartesian_target(command.arm, handle, pose6d, motion=motion)
        target = self._make_point(command.arm, handle, pose6d, motion=motion)
        add_data = self._make_mov_point_add(command.speed_ratio)
        motion_func = getattr(self._load_x5(), motion, None)
        if motion_func is None:
            raise RuntimeError(f"xapi.{motion} is unavailable")

        try:
            if add_data is not None:
                motion_func(handle, target, add_data)
            else:
                motion_func(handle, target)
        except TypeError:
            motion_func(handle, target)

        self._wait_cmd_send_done(handle)
        if command.wait:
            self._wait_move_done(handle)

    def _check_speed_ratio(self, speed_ratio: float) -> None:
        if speed_ratio > self._max_command_speed_ratio:
            raise ValueError(
                f"speed_ratio {speed_ratio:.3f} exceeds configured "
                f"max_command_speed_ratio {self._max_command_speed_ratio:.3f}"
            )

    def _check_cartesian_target(
        self,
        arm: str,
        handle: Any,
        target_pose6d: Sequence[float],
        *,
        motion: str,
    ) -> None:
        current_pose = self._read_tcp_pose_xyzw(handle)
        translation_m = math.sqrt(
            sum(
                (target - current) ** 2
                for target, current in zip(
                    target_pose6d[:3],
                    current_pose[:3],
                    strict=True,
                )
            )
        )
        target_quaternion = _rotvec_to_quat_xyzw(target_pose6d[3:])
        rotation_deg = _quaternion_distance_deg(
            current_pose[3:7],
            target_quaternion,
        )

        if motion == "movj":
            max_translation_m = self._max_movej_point_translation_m
            max_rotation_deg = self._max_movej_point_rotation_deg
        elif motion == "movl":
            max_translation_m = self._max_movel_point_translation_m
            max_rotation_deg = self._max_movel_point_rotation_deg
        else:
            raise ValueError(f"unsupported point motion: {motion}")

        if translation_m > max_translation_m:
            raise ValueError(
                f"{arm} {motion}(Point) translation {translation_m:.4f} m exceeds "
                f"configured limit {max_translation_m:.4f} m"
            )
        if rotation_deg > max_rotation_deg:
            raise ValueError(
                f"{arm} {motion}(Point) rotation {rotation_deg:.3f} deg exceeds "
                f"configured limit {max_rotation_deg:.3f} deg"
            )

    def _make_point(
        self,
        arm: str,
        handle: Any,
        tcp_pose_xyz_rotvec: Sequence[float],
        *,
        motion: str = "point",
    ) -> Any:
        x5 = self._load_x5()
        point_cls = getattr(x5, "Point", None)
        if point_cls is None:
            raise RuntimeError("xapi.Point is unavailable")

        reference = self._read_reference_point(handle)
        current_joint = x5.get_cjoint(handle)
        cfg = _point_cfg(reference)
        tf_no = self._active_tool_frame_nos.get(
            arm,
            _point_frame_no(reference, "tf", default=0),
        )
        e1 = _joint_axis(current_joint, "e1", fallback="j7")
        e2 = _joint_axis(
            current_joint,
            "e2",
            default=self._head_joints_deg[arm][0],
        )
        e3 = _joint_axis(
            current_joint,
            "e3",
            default=self._head_joints_deg[arm][1],
        )
        a_deg, b_deg, c_deg = _rotvec_to_euler_xyz_deg(
            tcp_pose_xyz_rotvec[3:]
        )
        values = (
            float(tcp_pose_xyz_rotvec[0]) * 1000.0,
            float(tcp_pose_xyz_rotvec[1]) * 1000.0,
            float(tcp_pose_xyz_rotvec[2]) * 1000.0,
            a_deg,
            b_deg,
            c_deg,
            e1,
            e2,
            e3,
        )
        logger.info(
            "X5 %s Point constructor: pose_xyzabc_r1e2e3=%s uf=%d tf=%d cfg=%s",
            motion,
            values,
            0,
            tf_no,
            cfg,
        )
        try:
            return point_cls(values, 0, tf_no, cfg)
        except TypeError:
            pose_cls = getattr(x5, "Pose", None)
            if pose_cls is None:
                raise RuntimeError("xapi.Pose is unavailable")
            pose = pose_cls(*values)
            return point_cls(pose=pose, uf=0, tf=tf_no, cfg=cfg)

    def _read_reference_point(self, handle: Any) -> Any:
        x5 = self._load_x5()
        get_wpoint = getattr(x5, "get_wpoint", None)
        if get_wpoint is not None:
            point = get_wpoint(handle)
            if point is not None and point is not False:
                return point
        get_cpoint = getattr(x5, "get_cpoint", None)
        if get_cpoint is None:
            raise RuntimeError("xapi.get_wpoint and xapi.get_cpoint are unavailable")
        point = get_cpoint(handle)
        if point is None or point is False:
            raise RuntimeError("xapi returned an invalid reference Point")
        return point

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
        self._active_tool_frame_nos.clear()
        self._tool_frame_poses_xyzw.clear()

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


def _six_finite_floats(values: Any) -> list[float]:
    converted = [float(value) for value in values]
    if len(converted) != 6:
        raise ValueError("tcp_pose_xyz_rotvec must contain exactly 6 values")
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("tcp_pose_xyz_rotvec values must be finite")
    return converted


def _point_frame_no(point: Any, name: str, *, default: int) -> int:
    value = getattr(point, name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _point_cfg(point: Any) -> tuple[int, ...]:
    cfg = getattr(point, "cfg", (0, 0, 0, 1))
    try:
        return tuple(int(value) for value in cfg)
    except (TypeError, ValueError):
        return (0, 0, 0, 1)


def _joint_axis(
    joint: Any,
    name: str,
    *,
    fallback: str | None = None,
    default: float | None = None,
) -> float:
    if isinstance(joint, Mapping):
        if name in joint:
            return float(joint[name])
        if fallback and fallback in joint:
            return float(joint[fallback])
    else:
        if hasattr(joint, name):
            return float(getattr(joint, name))
        if fallback and hasattr(joint, fallback):
            return float(getattr(joint, fallback))
    if default is not None:
        return float(default)
    raise ValueError(f"current joint is missing axis {name}")


def _tool_frame_no(config: Mapping[str, Any]) -> int | None:
    tool_frame_config = config.get("tool_frame", {})
    value = config.get("tf_no")
    if value is None and isinstance(tool_frame_config, Mapping):
        value = tool_frame_config.get("tf_no")
    if value is None:
        return None

    tf_no = int(value)
    if tf_no < 0 or tf_no > 16:
        raise ValueError(f"tf_no must be in [0, 16], got {tf_no}")
    return tf_no


def _tool_frame_pose(config: Mapping[str, Any]) -> list[float] | None:
    tool_frame_config = config.get("tool_frame", {})
    if not isinstance(tool_frame_config, Mapping):
        raise ValueError("tool_frame must be a mapping")

    position = tool_frame_config.get("position_m")
    orientation = tool_frame_config.get("rpy_deg")
    if position is None and orientation is None:
        return None
    if position is None:
        position = [0.0, 0.0, 0.0]
    if orientation is None:
        orientation = [0.0, 0.0, 0.0]

    position_values = [float(value) for value in position]
    orientation_values = [float(value) for value in orientation]
    if len(position_values) != 3:
        raise ValueError("tool_frame.position_m must contain exactly 3 values")
    if len(orientation_values) != 3:
        raise ValueError("tool_frame.rpy_deg must contain exactly 3 values")

    values = position_values + orientation_values
    if not all(math.isfinite(value) for value in values):
        raise ValueError("tool_frame pose values must be finite")
    return values


def _validate_tool_frame_pose(
    arm: str,
    tf_no: int,
    requested_pose: Sequence[float],
    actual_pose_xyzw: Sequence[float],
) -> None:
    requested_position = requested_pose[:3]
    actual_position = actual_pose_xyzw[:3]
    max_position_error_m = max(
        abs(requested - actual)
        for requested, actual in zip(
            requested_position,
            actual_position,
            strict=True,
        )
    )

    requested_quaternion = _euler_xyz_deg_to_quat_xyzw(*requested_pose[3:6])
    actual_quaternion = actual_pose_xyzw[3:7]
    quaternion_dot = abs(
        sum(
            requested * actual
            for requested, actual in zip(
                requested_quaternion,
                actual_quaternion,
                strict=True,
            )
        )
    )
    if max_position_error_m > 0.001 or quaternion_dot < 0.99999:
        raise RuntimeError(
            f"{arm} TF{tf_no} readback does not match configured pose: "
            f"requested={list(requested_pose)}, actual_xyzw={list(actual_pose_xyzw)}"
        )


def _pose_to_xyzw(point_or_pose: Any) -> list[float]:
    if point_or_pose is None or point_or_pose is False:
        raise ValueError("xapi returned an invalid pose")

    pose = getattr(point_or_pose, "pose", point_or_pose)
    x_mm = _read_value(pose, "x", 0)
    y_mm = _read_value(pose, "y", 1)
    z_mm = _read_value(pose, "z", 2)
    a_deg = _read_value(pose, "a", 3, default=0.0)
    b_deg = _read_value(pose, "b", 4, default=0.0)
    c_deg = _read_value(pose, "c", 5, default=0.0)

    qx, qy, qz, qw = _euler_xyz_deg_to_quat_xyzw(a_deg, b_deg, c_deg)
    return [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0, qx, qy, qz, qw]


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


def _rotvec_to_quat_xyzw(
    rotvec_rad: Sequence[float],
) -> tuple[float, float, float, float]:
    rx, ry, rz = (float(value) for value in rotvec_rad)
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-12:
        return 0.0, 0.0, 0.0, 1.0

    scale = math.sin(theta / 2.0) / theta
    return rx * scale, ry * scale, rz * scale, math.cos(theta / 2.0)


def _rotvec_to_euler_xyz_deg(rotvec_rad: Sequence[float]) -> tuple[float, float, float]:
    """Convert axis-angle radians to X5 Euler XYZ degrees."""

    rx, ry, rz = (float(value) for value in rotvec_rad)
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-12:
        return 0.0, 0.0, 0.0

    x = rx / theta
    y = ry / theta
    z = rz / theta
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    one_minus_cos = 1.0 - cos_theta
    rotation = (
        (
            cos_theta + x * x * one_minus_cos,
            x * y * one_minus_cos - z * sin_theta,
            x * z * one_minus_cos + y * sin_theta,
        ),
        (
            y * x * one_minus_cos + z * sin_theta,
            cos_theta + y * y * one_minus_cos,
            y * z * one_minus_cos - x * sin_theta,
        ),
        (
            z * x * one_minus_cos - y * sin_theta,
            z * y * one_minus_cos + x * sin_theta,
            cos_theta + z * z * one_minus_cos,
        ),
    )

    horizontal = math.hypot(rotation[0][0], rotation[1][0])
    if horizontal > 1e-9:
        a_rad = math.atan2(rotation[2][1], rotation[2][2])
        b_rad = math.atan2(-rotation[2][0], horizontal)
        c_rad = math.atan2(rotation[1][0], rotation[0][0])
    else:
        a_rad = math.atan2(-rotation[1][2], rotation[1][1])
        b_rad = math.atan2(-rotation[2][0], horizontal)
        c_rad = 0.0

    return math.degrees(a_rad), math.degrees(b_rad), math.degrees(c_rad)


def _quaternion_distance_deg(
    first_xyzw: Sequence[float],
    second_xyzw: Sequence[float],
) -> float:
    first = [float(value) for value in first_xyzw]
    second = [float(value) for value in second_xyzw]
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm < 1e-12 or second_norm < 1e-12:
        raise ValueError("orientation quaternion must be non-zero")
    dot = abs(
        sum(
            left / first_norm * (right / second_norm)
            for left, right in zip(first, second, strict=True)
        )
    )
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))
