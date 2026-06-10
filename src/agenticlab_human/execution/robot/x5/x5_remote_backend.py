"""AgenticLab ActionBackend for executing pick trajectories through X5 HTTP."""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
import yaml

from agenticlab_human.execution.action_backend import ActionResult
from agenticlab_human.execution.robot.x5.client import (
    X5HTTPClient,
    tcp_pose_xyzw_to_xyz_rotvec,
)
from agenticlab_human.perception.backend.grasp_backend import GraspCandidate
from agenticlab_human.perception.backend.perception_backend import BBox


DEFAULT_ROBOT_CONFIG = "configs/robot/x5_config.yaml"
DEFAULT_CAMERA_CONFIG = "configs/perception/camera_config.yaml"
PickExecutionStage = Literal["home", "approach", "grasp"]
PlaceExecutionStage = Literal["retreat", "home", "preplace", "place"]


T_GRASP_EE = np.eye(4, dtype=float)
T_GRASP_EE[:3, :3] = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=float,
)
T_GRASP_TCP = T_GRASP_EE.copy()


class RemoteX5ActionBackend:
    """Plan or execute staged X5 pick and place trajectories over HTTP."""

    def __init__(
        self,
        robot_config_path: str = DEFAULT_ROBOT_CONFIG,
        camera_config_path: str = DEFAULT_CAMERA_CONFIG,
        *,
        server_url: str | None = None,
        arm: str | None = None,
        camera_name: str | None = None,
        execute: bool = False,
        execute_until: PickExecutionStage | None = None,
        place_execute_until: PlaceExecutionStage | None = None,
        client: Any | None = None,
    ) -> None:
        self.robot_config_path = robot_config_path
        self.camera_config_path = camera_config_path
        self.robot_config = _load_yaml(robot_config_path)
        self.camera_config = _load_yaml(camera_config_path)
        backend_config = self.robot_config.get("action_backend", {})
        robot_config = self.robot_config.get("robot", {})

        self.server_url = str(
            server_url or backend_config.get("server_url", "http://127.0.0.1:8000")
        ).rstrip("/")
        self.arm = str(arm or backend_config.get("arm", "left"))
        self.camera_name = str(
            camera_name or backend_config.get("camera_name", "Gemini335")
        )
        self.execute = bool(execute)
        self.execute_until = _execution_stage(
            execute_until or backend_config.get("execute_until", "grasp")
        )
        self.place_execute_until = _place_execution_stage(
            place_execute_until
            or backend_config.get("place_execute_until", "place")
        )
        self.approach_distance_m = _positive_float(
            backend_config.get("approach_distance_m", 0.05),
            "action_backend.approach_distance_m",
        )
        self.home_before_pick = bool(backend_config.get("home_before_pick", True))
        max_command_speed_ratio = _positive_float(
            robot_config.get("max_command_speed_ratio", 0.1),
            "robot.max_command_speed_ratio",
        )
        self.home_speed_ratio = _speed_ratio(
            backend_config.get("home_speed_ratio", 0.05),
            "action_backend.home_speed_ratio",
            max_command_speed_ratio,
        )
        self.home_max_step_deg = _positive_float(
            backend_config.get("home_max_step_deg", 4.0),
            "action_backend.home_max_step_deg",
        )
        max_joint_delta_deg = _positive_float(
            robot_config.get("max_joint_delta_deg", 5.0),
            "robot.max_joint_delta_deg",
        )
        if self.home_max_step_deg > max_joint_delta_deg:
            raise ValueError(
                "action_backend.home_max_step_deg cannot exceed "
                "robot.max_joint_delta_deg"
            )
        self.approach_speed_ratio = _speed_ratio(
            backend_config.get("approach_speed_ratio", 0.03),
            "action_backend.approach_speed_ratio",
            max_command_speed_ratio,
        )
        self.grasp_speed_ratio = _speed_ratio(
            backend_config.get("grasp_speed_ratio", 0.02),
            "action_backend.grasp_speed_ratio",
            max_command_speed_ratio,
        )
        self.retreat_speed_ratio = _speed_ratio(
            backend_config.get("retreat_speed_ratio", self.grasp_speed_ratio),
            "action_backend.retreat_speed_ratio",
            max_command_speed_ratio,
        )
        self.place_approach_speed_ratio = _speed_ratio(
            backend_config.get("place_approach_speed_ratio", 0.03),
            "action_backend.place_approach_speed_ratio",
            max_command_speed_ratio,
        )
        self.place_speed_ratio = _speed_ratio(
            backend_config.get("place_speed_ratio", 0.02),
            "action_backend.place_speed_ratio",
            max_command_speed_ratio,
        )
        self.place_approach_offset_x_m = _finite_float(
            backend_config.get("place_approach_offset_x_m", -0.05),
            "action_backend.place_approach_offset_x_m",
        )
        max_movel_translation_m = _positive_float(
            robot_config.get("max_movel_point_translation_m", 0.1),
            "robot.max_movel_point_translation_m",
        )
        if abs(self.place_approach_offset_x_m) > max_movel_translation_m:
            raise ValueError(
                "absolute action_backend.place_approach_offset_x_m cannot exceed "
                "robot.max_movel_point_translation_m"
            )
        configured_home_rotvec = backend_config.get("home_tcp_rotvec")
        self.configured_home_tcp_rotvec = (
            _three_finite_floats(
                configured_home_rotvec,
                "action_backend.home_tcp_rotvec",
            )
            if configured_home_rotvec is not None
            else None
        )
        self.request_timeout_s = _positive_float(
            backend_config.get("request_timeout_s", 90.0),
            "action_backend.request_timeout_s",
        )

        arm_config = robot_config.get(self.arm, {})
        home_values = arm_config.get(
            "home_joints_deg",
            robot_config.get("home_joints_deg"),
        )
        if home_values is None:
            raise ValueError(f"home_joints_deg is not configured for X5 arm '{self.arm}'")
        self.home_joints_deg = _seven_finite_floats(
            home_values,
            f"robot.{self.arm}.home_joints_deg",
        )
        self.joint_limits_deg = _joint_limits(robot_config.get("joint_limits_deg"))

        camera_config = self.camera_config.get(self.camera_name)
        if not camera_config:
            raise ValueError(f"Camera config not found: {self.camera_name}")
        handeye = camera_config.get("handeye_calibration", {})
        self.T_world_camera = np.eye(4, dtype=float)
        self.T_world_camera[:3, :3] = np.asarray(handeye["rotation"], dtype=float)
        self.T_world_camera[:3, 3] = np.asarray(handeye["translation"], dtype=float)
        self.T_world_camera = _coerce_se3(self.T_world_camera, "T_world_camera")

        self._client = client
        self._owns_client = client is None
        self._initialized = False
        self._last_pick_plan: dict[str, np.ndarray] | None = None
        self._grasp_reached = False
        self._last_home_pose_xyz_rotvec: list[float] | None = None
        if self.execute_until == "home" and not self.home_before_pick:
            raise ValueError(
                "execute_until='home' requires action_backend.home_before_pick=true"
            )

    def initialize(self) -> None:
        if self._initialized:
            return
        try:
            if self.execute and self._client is None:
                self._client = X5HTTPClient(
                    self.server_url,
                    timeout_s=self.request_timeout_s,
                )
            if self.execute:
                health = self._client.health()
                if not health.robot.ready:
                    raise RuntimeError(
                        f"X5 server robot is not ready: {health.robot.detail}"
                    )
        except Exception:
            if self._owns_client and self._client is not None:
                self._client.close()
                self._client = None
            raise
        self._initialized = True

    def shutdown(self, move_home: bool = False) -> None:
        try:
            if self.execute and move_home and self._client is not None:
                self.move_home()
        finally:
            if self._owns_client and self._client is not None:
                self._client.close()
                self._client = None
            self._initialized = False

    def pick(
        self,
        object_name: str,
        from_name: Optional[str] = None,
        grasp_candidates: Optional[Sequence[GraspCandidate]] = None,
        object_bbox: Optional[BBox] = None,
        object_pose: Any = None,
    ) -> ActionResult:
        grasp = _select_grasp(grasp_candidates)
        if grasp is None:
            return ActionResult(
                success=False,
                action_name="pick",
                error=f"No grasp candidate available for {object_name}.",
                metadata={"object": object_name, "from": from_name},
            )
        if grasp.metadata.get("frame") not in (None, "camera"):
            return ActionResult(
                success=False,
                action_name="pick",
                error=(
                    "RemoteX5ActionBackend expects grasp poses in the camera frame, "
                    f"got {grasp.metadata.get('frame')!r}."
                ),
                metadata={"object": object_name, "from": from_name},
            )

        try:
            plan = build_world_tcp_pick_poses(
                self.T_world_camera,
                grasp.pose,
                approach_distance_m=self.approach_distance_m,
            )
        except ValueError as exc:
            return ActionResult(
                success=False,
                action_name="pick",
                error=str(exc),
                metadata={"object": object_name, "from": from_name},
            )

        self._last_pick_plan = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in plan.items()
        }
        self._grasp_reached = False
        metadata: dict[str, Any] = {
            "object": object_name,
            "from": from_name,
            "execute": self.execute,
            "execute_until": self.execute_until,
            "arm": self.arm,
            "camera_name": self.camera_name,
            "server_url": self.server_url,
            "grasp_score": grasp.score,
            "grasp_width": grasp.metadata.get("width"),
            "object_has_bbox": object_bbox is not None,
            "object_has_pose": object_pose is not None,
            "approach_distance_m": self.approach_distance_m,
            **plan,
        }
        if not self.execute:
            return ActionResult(
                success=True,
                action_name="pick",
                message=f"X5 dry-run pick trajectory generated for {object_name}.",
                metadata=metadata,
            )
        if not self._initialized or self._client is None:
            return ActionResult(
                success=False,
                action_name="pick",
                error="RemoteX5ActionBackend is not initialized.",
                metadata=metadata,
            )

        completed_steps: list[dict[str, Any]] = []
        failed_step = "home"
        try:
            if self.home_before_pick:
                completed_steps.extend(self._move_home_joints())
            if self.execute_until == "home":
                metadata["completed_steps"] = completed_steps
                return _pick_motion_success(object_name, "home", metadata)

            failed_step = "approach"
            approach_result = self._client.movej_point(
                self.arm,
                plan["approach_pose_xyz_rotvec"].tolist(),
                speed_ratio=self.approach_speed_ratio,
                wait=True,
                request_id=self._request_id("approach"),
            )
            _require_success(approach_result, "movej_point approach")
            completed_steps.append(_response_summary("approach", approach_result))
            if self.execute_until == "approach":
                metadata["completed_steps"] = completed_steps
                return _pick_motion_success(object_name, "approach", metadata)

            failed_step = "grasp"
            grasp_result = self._client.movel_point(
                self.arm,
                plan["grasp_pose_xyz_rotvec"].tolist(),
                speed_ratio=self.grasp_speed_ratio,
                wait=True,
                request_id=self._request_id("grasp"),
            )
            _require_success(grasp_result, "movel_point grasp")
            completed_steps.append(_response_summary("grasp", grasp_result))
            self._grasp_reached = True
            metadata["completed_steps"] = completed_steps
            return _pick_motion_success(object_name, "grasp", metadata)
        except Exception as exc:
            metadata["completed_steps"] = completed_steps
            metadata["failed_step"] = failed_step
            stop_error = self._try_stop()
            if stop_error:
                metadata["stop_error"] = stop_error
            return ActionResult(
                success=False,
                action_name="pick",
                error=f"X5 pick failed during {failed_step}: {exc}",
                metadata=metadata,
            )

    def move_home(self) -> ActionResult:
        metadata = {
            "execute": self.execute,
            "arm": self.arm,
            "home_joints_deg": self.home_joints_deg,
        }
        if not self.execute:
            return ActionResult(
                success=True,
                action_name="move-home",
                message="X5 dry-run move home.",
                metadata=metadata,
            )
        if not self._initialized or self._client is None:
            return ActionResult(
                success=False,
                action_name="move-home",
                error="RemoteX5ActionBackend is not initialized.",
                metadata=metadata,
            )
        try:
            metadata["completed_steps"] = self._move_home_joints()
            return ActionResult(
                success=True,
                action_name="move-home",
                message=f"X5 {self.arm} arm moved home.",
                metadata=metadata,
            )
        except Exception as exc:
            stop_error = self._try_stop()
            if stop_error:
                metadata["stop_error"] = stop_error
            return ActionResult(
                success=False,
                action_name="move-home",
                error=f"X5 move home failed: {exc}",
                metadata=metadata,
            )

    def get_eef_pose(self) -> Any:
        if not self.execute or self._client is None:
            return None
        return self._client.get_state(self.arm).state_after.arms[self.arm].tcp_pose_xyzw

    def place_on_object(
        self,
        object_name: str,
        target_name: str,
        target_bbox: Optional[BBox] = None,
        target_pose: Any = None,
    ) -> ActionResult:
        return self._place(
            "place-on-object",
            object_name,
            target_name,
            target_bbox=target_bbox,
            target_pose=target_pose,
        )

    def place_on_surface(
        self,
        object_name: str,
        surface_name: str,
        target_bbox: Optional[BBox] = None,
        target_pose: Any = None,
    ) -> ActionResult:
        return self._place(
            "place-on-surface",
            object_name,
            surface_name,
            target_bbox=target_bbox,
            target_pose=target_pose,
        )

    def place_in_container(
        self,
        object_name: str,
        container_name: str,
        target_bbox: Optional[BBox] = None,
        target_pose: Any = None,
    ) -> ActionResult:
        return self._place(
            "place-in-container",
            object_name,
            container_name,
            target_bbox=target_bbox,
            target_pose=target_pose,
        )

    def _place(
        self,
        action_name: str,
        object_name: str,
        target_name: str,
        *,
        target_bbox: Optional[BBox],
        target_pose: Any,
    ) -> ActionResult:
        try:
            target_xyz = _target_xyz(target_pose)
        except ValueError as exc:
            return ActionResult(
                success=False,
                action_name=action_name,
                error=str(exc),
                metadata={"object": object_name, "target": target_name},
            )
        if self._last_pick_plan is None:
            return ActionResult(
                success=False,
                action_name=action_name,
                error="X5 place requires a preceding pick plan for grasp retreat.",
                metadata={"object": object_name, "target": target_name},
            )

        retreat_pose = self._last_pick_plan["approach_pose_xyz_rotvec"].copy()
        home_rotvec = (
            self._last_home_pose_xyz_rotvec[3:]
            if self._last_home_pose_xyz_rotvec is not None
            else self.configured_home_tcp_rotvec
        )
        plan = (
            build_world_tcp_place_poses(
                target_xyz,
                home_rotvec,
                approach_offset_x_m=self.place_approach_offset_x_m,
            )
            if home_rotvec is not None
            else None
        )
        metadata: dict[str, Any] = {
            "object": object_name,
            "target": target_name,
            "execute": self.execute,
            "execute_until": self.place_execute_until,
            "arm": self.arm,
            "server_url": self.server_url,
            "target_has_bbox": target_bbox is not None,
            "target_pose_xyz": target_xyz,
            "target_orientation_ignored": True,
            "place_approach_offset_x_m": self.place_approach_offset_x_m,
            "retreat_pose_xyz_rotvec": retreat_pose,
        }
        if plan is not None:
            metadata.update(plan)

        if not self.execute:
            if plan is None:
                return ActionResult(
                    success=False,
                    action_name=action_name,
                    error=(
                        "Dry-run place requires action_backend.home_tcp_rotvec "
                        "because the runtime home TCP state is unavailable."
                    ),
                    metadata=metadata,
                )
            return ActionResult(
                success=True,
                action_name=action_name,
                message=(
                    f"X5 dry-run place trajectory generated for {object_name} "
                    f"to {target_name}."
                ),
                metadata=metadata,
            )
        if not self._initialized or self._client is None:
            return ActionResult(
                success=False,
                action_name=action_name,
                error="RemoteX5ActionBackend is not initialized.",
                metadata=metadata,
            )
        if not self._grasp_reached:
            return ActionResult(
                success=False,
                action_name=action_name,
                error="X5 place requires pick execution to complete through grasp.",
                metadata=metadata,
            )

        completed_steps: list[dict[str, Any]] = []
        failed_step: PlaceExecutionStage = "retreat"
        try:
            retreat_result = self._client.movel_point(
                self.arm,
                retreat_pose.tolist(),
                speed_ratio=self.retreat_speed_ratio,
                wait=True,
                request_id=self._request_id("retreat", action="place"),
            )
            _require_success(retreat_result, "movel_point grasp retreat")
            completed_steps.append(_response_summary("retreat", retreat_result))
            if self.place_execute_until == "retreat":
                metadata["completed_steps"] = completed_steps
                return _place_motion_success(
                    action_name, object_name, target_name, "retreat", metadata
                )

            failed_step = "home"
            completed_steps.extend(self._move_home_joints(action="place"))
            if self.place_execute_until == "home":
                metadata["completed_steps"] = completed_steps
                return _place_motion_success(
                    action_name, object_name, target_name, "home", metadata
                )

            if self._last_home_pose_xyz_rotvec is None:
                raise RuntimeError("home TCP pose was not returned by the X5 server")
            plan = build_world_tcp_place_poses(
                target_xyz,
                self._last_home_pose_xyz_rotvec[3:],
                approach_offset_x_m=self.place_approach_offset_x_m,
            )
            metadata.update(plan)
            metadata["home_pose_xyz_rotvec"] = self._last_home_pose_xyz_rotvec

            failed_step = "preplace"
            preplace_result = self._client.movej_point(
                self.arm,
                plan["preplace_pose_xyz_rotvec"].tolist(),
                speed_ratio=self.place_approach_speed_ratio,
                wait=True,
                request_id=self._request_id("preplace", action="place"),
            )
            _require_success(preplace_result, "movej_point pre-place")
            completed_steps.append(_response_summary("preplace", preplace_result))
            if self.place_execute_until == "preplace":
                metadata["completed_steps"] = completed_steps
                return _place_motion_success(
                    action_name, object_name, target_name, "preplace", metadata
                )

            failed_step = "place"
            place_result = self._client.movel_point(
                self.arm,
                plan["place_pose_xyz_rotvec"].tolist(),
                speed_ratio=self.place_speed_ratio,
                wait=True,
                request_id=self._request_id("place", action="place"),
            )
            _require_success(place_result, "movel_point place")
            completed_steps.append(_response_summary("place", place_result))
            metadata["completed_steps"] = completed_steps
            return _place_motion_success(
                action_name, object_name, target_name, "place", metadata
            )
        except Exception as exc:
            metadata["completed_steps"] = completed_steps
            metadata["failed_step"] = failed_step
            stop_error = self._try_stop()
            if stop_error:
                metadata["stop_error"] = stop_error
            return ActionResult(
                success=False,
                action_name=action_name,
                error=f"X5 place failed during {failed_step}: {exc}",
                metadata=metadata,
            )

    def _move_home_joints(self, *, action: str = "pick") -> list[dict[str, Any]]:
        state_result = self._client.get_state(self.arm)
        _require_success(state_result, "get_state before home")
        start_rad = state_result.state_after.arms[self.arm].joints_rad
        start_deg = [math.degrees(float(value)) for value in start_rad]
        self._check_home_limits()
        max_delta = max(
            abs(target - current)
            for current, target in zip(
                start_deg,
                self.home_joints_deg,
                strict=True,
            )
        )
        step_count = max(1, math.ceil(max_delta / self.home_max_step_deg))
        completed_steps: list[dict[str, Any]] = []
        for step_index in range(1, step_count + 1):
            fraction = step_index / step_count
            waypoint_deg = [
                current + (target - current) * fraction
                for current, target in zip(
                    start_deg,
                    self.home_joints_deg,
                    strict=True,
                )
            ]
            response = self._client.move_joints(
                self.arm,
                [math.radians(value) for value in waypoint_deg],
                speed_ratio=self.home_speed_ratio,
                wait=True,
                request_id=self._request_id(
                    f"home-{step_index:03d}",
                    action=action,
                ),
            )
            _require_success(response, f"home waypoint {step_index}/{step_count}")
            summary = _response_summary("home", response)
            summary["waypoint_index"] = step_index
            summary["waypoint_count"] = step_count
            summary["joints_deg"] = waypoint_deg
            completed_steps.append(summary)
        final_arm_state = response.state_after.arms[self.arm]
        self._last_home_pose_xyz_rotvec = tcp_pose_xyzw_to_xyz_rotvec(
            final_arm_state.tcp_pose_xyzw
        )
        return completed_steps

    def _check_home_limits(self) -> None:
        if self.joint_limits_deg is None:
            return
        for index, (value, (lower, upper)) in enumerate(
            zip(self.home_joints_deg, self.joint_limits_deg, strict=True),
            start=1,
        ):
            if value < lower or value > upper:
                raise ValueError(
                    f"home joint {index} target {value:.3f} deg is outside "
                    f"configured limit [{lower:.3f}, {upper:.3f}] deg"
                )

    def _try_stop(self) -> str | None:
        try:
            self._client.stop(self.arm)
        except Exception as exc:
            return str(exc)
        return None

    def _request_id(self, step: str, *, action: str = "pick") -> str:
        return f"x5-{action}-{step}-{uuid.uuid4().hex[:12]}"


def build_world_tcp_pick_poses(
    T_world_camera: Any,
    T_camera_grasp: Any,
    *,
    approach_distance_m: float,
    T_grasp_tcp: Any = T_GRASP_TCP,
) -> dict[str, np.ndarray]:
    """Convert an AnyGrasp pose into world-frame approach and TCP targets."""

    world_camera = _coerce_se3(T_world_camera, "T_world_camera")
    camera_grasp = _coerce_se3(T_camera_grasp, "T_camera_grasp")
    grasp_tcp = _coerce_se3(T_grasp_tcp, "T_grasp_tcp")
    approach_distance = float(approach_distance_m)
    if not math.isfinite(approach_distance) or approach_distance <= 0.0:
        raise ValueError("approach_distance_m must be a positive finite value")

    camera_approach = camera_grasp.copy()
    camera_approach[:3, 3] = (
        camera_grasp[:3, 3]
        + camera_grasp[:3, :3] @ np.array([-approach_distance, 0.0, 0.0])
    )
    world_tcp_grasp = world_camera @ camera_grasp @ grasp_tcp
    world_tcp_approach = world_camera @ camera_approach @ grasp_tcp
    return {
        "T_world_tcp_grasp": world_tcp_grasp,
        "T_world_tcp_approach": world_tcp_approach,
        "grasp_pose_xyz_rotvec": se3_to_xyz_rotvec(world_tcp_grasp),
        "approach_pose_xyz_rotvec": se3_to_xyz_rotvec(world_tcp_approach),
    }


def build_world_tcp_place_poses(
    target_pose: Any,
    home_tcp_rotvec: Any,
    *,
    approach_offset_x_m: float,
) -> dict[str, np.ndarray]:
    """Build world-frame pre-place and place poses using the home TCP orientation."""

    target_xyz = _target_xyz(target_pose)
    home_rotvec = np.asarray(
        _three_finite_floats(home_tcp_rotvec, "home_tcp_rotvec"),
        dtype=float,
    )
    offset_x = _finite_float(approach_offset_x_m, "approach_offset_x_m")

    place_pose = np.concatenate([target_xyz, home_rotvec])
    preplace_pose = place_pose.copy()
    preplace_pose[0] += offset_x
    return {
        "preplace_pose_xyz_rotvec": preplace_pose,
        "place_pose_xyz_rotvec": place_pose,
    }


def se3_to_xyz_rotvec(transform: Any) -> np.ndarray:
    """Return [x, y, z, rx, ry, rz] in meters and rotation-vector radians."""

    matrix = _coerce_se3(transform, "transform")
    return np.concatenate(
        [matrix[:3, 3], _rotation_matrix_to_rotvec(matrix[:3, :3])]
    )


def _coerce_se3(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must be a homogeneous SE3 transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix


def _rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    cos_theta = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3, dtype=float)
    if math.pi - theta < 1e-6:
        axis = np.sqrt(np.maximum(np.diag(rotation) + 1.0, 0.0) / 2.0)
        axis[0] = math.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
        axis[1] = math.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
        axis[2] = math.copysign(axis[2], rotation[1, 0] - rotation[0, 1])
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return np.zeros(3, dtype=float)
        return axis / norm * theta
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    ) / (2.0 * math.sin(theta))
    return axis * theta


def _load_yaml(path: str) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path_obj.read_text()) or {}


def _select_grasp(
    candidates: Optional[Sequence[GraspCandidate]],
) -> Optional[GraspCandidate]:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            candidate.score if candidate.score is not None else float("-inf")
        ),
    )


def _execution_stage(value: Any) -> PickExecutionStage:
    stage = str(value)
    if stage not in {"home", "approach", "grasp"}:
        raise ValueError(
            "action_backend.execute_until must be one of: home, approach, grasp"
        )
    return stage


def _place_execution_stage(value: Any) -> PlaceExecutionStage:
    stage = str(value)
    if stage not in {"retreat", "home", "preplace", "place"}:
        raise ValueError(
            "action_backend.place_execute_until must be one of: "
            "retreat, home, preplace, place"
        )
    return stage


def _positive_float(value: Any, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return converted


def _finite_float(value: Any, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite value")
    return converted


def _speed_ratio(value: Any, name: str, maximum: float) -> float:
    converted = _positive_float(value, name)
    if converted > maximum:
        raise ValueError(
            f"{name} {converted:.3f} exceeds robot.max_command_speed_ratio "
            f"{maximum:.3f}"
        )
    return converted


def _three_finite_floats(values: Any, name: str) -> list[float]:
    converted = [float(value) for value in values]
    if len(converted) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain finite values")
    return converted


def _seven_finite_floats(values: Any, name: str) -> list[float]:
    converted = [float(value) for value in values]
    if len(converted) < 7:
        raise ValueError(f"{name} must contain at least 7 values")
    converted = converted[:7]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain finite values")
    return converted


def _joint_limits(values: Any) -> list[tuple[float, float]] | None:
    if values is None:
        return None
    limits = [tuple(float(value) for value in pair) for pair in values]
    if len(limits) != 7 or any(len(pair) != 2 for pair in limits):
        raise ValueError("robot.joint_limits_deg must contain 7 [lower, upper] pairs")
    if any(lower >= upper for lower, upper in limits):
        raise ValueError("each robot joint lower limit must be less than its upper limit")
    return limits


def _target_xyz(target_pose: Any) -> np.ndarray:
    if target_pose is None:
        raise ValueError("X5 place requires a world-frame target_pose")
    values = np.asarray(target_pose, dtype=float)
    if values.shape == (4, 4):
        xyz = values[:3, 3]
    elif values.ndim == 1 and values.size >= 3:
        xyz = values[:3]
    else:
        raise ValueError(
            "X5 place target_pose must be a pose vector with at least 3 values "
            "or a 4x4 world-frame transform"
        )
    if not np.all(np.isfinite(xyz)):
        raise ValueError("X5 place target position must contain finite values")
    return np.asarray(xyz, dtype=float)


def _require_success(response: Any, step: str) -> None:
    if not bool(getattr(response, "success", False)):
        error = getattr(response, "error", None) or "server rejected command"
        raise RuntimeError(f"{step}: {error}")


def _response_summary(step: str, response: Any) -> dict[str, Any]:
    return {
        "step": step,
        "request_id": getattr(response, "request_id", None),
        "duration_ms": getattr(response, "duration_ms", None),
    }


def _pick_motion_success(
    object_name: str,
    completed_stage: PickExecutionStage,
    metadata: dict[str, Any],
) -> ActionResult:
    return ActionResult(
        success=True,
        action_name="pick",
        message=(
            f"X5 pick trajectory for {object_name} completed through "
            f"{completed_stage}; gripper actuation is not included yet."
        ),
        metadata=metadata,
    )


def _place_motion_success(
    action_name: str,
    object_name: str,
    target_name: str,
    completed_stage: PlaceExecutionStage,
    metadata: dict[str, Any],
) -> ActionResult:
    return ActionResult(
        success=True,
        action_name=action_name,
        message=(
            f"X5 place trajectory for {object_name} to {target_name} completed "
            f"through {completed_stage}; gripper release is not included yet."
        ),
        metadata=metadata,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute one X5 home -> approach -> grasp trajectory."
    )
    parser.add_argument("--robot-config", default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--camera-config", default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--camera-name")
    parser.add_argument("--server-url")
    parser.add_argument("--arm", choices=("left", "right"))
    parser.add_argument(
        "--translation",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="AnyGrasp translation in the camera frame, in meters.",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        nargs=9,
        required=True,
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
        help="AnyGrasp 3x3 rotation matrix in row-major order.",
    )
    parser.add_argument("--object-name", default="validation-object")
    parser.add_argument("--score", type=float, default=1.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send commands to the X5 server. Default is planning-only.",
    )
    parser.add_argument(
        "--execute-until",
        choices=("home", "approach", "grasp"),
        help="Override action_backend.execute_until for staged validation.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    T_camera_grasp = np.eye(4, dtype=float)
    T_camera_grasp[:3, :3] = np.asarray(args.rotation, dtype=float).reshape(3, 3)
    T_camera_grasp[:3, 3] = np.asarray(args.translation, dtype=float)
    grasp = GraspCandidate(
        pose=T_camera_grasp,
        score=args.score,
        object_name=args.object_name,
        metadata={"frame": "camera", "source": "x5-validation-cli"},
    )
    backend = RemoteX5ActionBackend(
        robot_config_path=args.robot_config,
        camera_config_path=args.camera_config,
        server_url=args.server_url,
        arm=args.arm,
        camera_name=args.camera_name,
        execute=args.execute,
        execute_until=args.execute_until,
    )
    backend.initialize()
    try:
        result = backend.pick(
            object_name=args.object_name,
            grasp_candidates=[grasp],
        )
    finally:
        backend.shutdown()
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
