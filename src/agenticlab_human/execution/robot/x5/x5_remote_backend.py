"""Production X5 pick-and-place backend over HTTP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import yaml

from agenticlab_human.execution.action_backend import ActionResult
from agenticlab_human.execution.robot.x5.client import X5HTTPClient
from agenticlab_human.execution.robot.x5.conversion import (
    coerce_se3,
    degrees_to_radians,
    radians_to_degrees,
    se3_to_xyz_rotvec,
)
from agenticlab_human.perception.backend.grasp_backend import GraspCandidate
from agenticlab_human.perception.backend.perception_backend import BBox


DEFAULT_ROBOT_CONFIG = "configs/robot/x5_config.yaml"
DEFAULT_CAMERA_CONFIG = "configs/perception/camera_config.yaml"

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


@dataclass(frozen=True)
class _BackendConfig:
    server_url: str
    arm: str
    camera_name: str
    request_timeout_s: float
    approach_distance_m: float
    home_speed_ratio: float
    home_max_step_deg: float
    approach_speed_ratio: float
    grasp_speed_ratio: float
    retreat_speed_ratio: float
    place_approach_speed_ratio: float
    place_speed_ratio: float
    place_approach_offset_x_m: float
    default_place_orientation_rotvec: np.ndarray
    home_joints_deg: list[float]
    check_gripper_joints_deg: list[float]
    T_world_camera: np.ndarray


class RemoteX5ActionBackend:
    """Execute the production X5 pick-and-place motion sequence."""

    def __init__(
        self,
        robot_config_path: str = DEFAULT_ROBOT_CONFIG,
        camera_config_path: str = DEFAULT_CAMERA_CONFIG,
        *,
        server_url: str | None = None,
        arm: str | None = None,
        camera_name: str | None = None,
        place_approach_offset_x_m: float | None = None,
        default_place_orientation_rotvec: Sequence[float] | None = None,
        client: Any | None = None,
    ) -> None:
        config = _load_backend_config(
            robot_config_path,
            camera_config_path,
            server_url=server_url,
            arm=arm,
            camera_name=camera_name,
            place_approach_offset_x_m=place_approach_offset_x_m,
            default_place_orientation_rotvec=default_place_orientation_rotvec,
        )
        self.config = config
        self.server_url = config.server_url
        self.arm = config.arm
        self.camera_name = config.camera_name
        self.T_world_camera = config.T_world_camera
        self.default_place_orientation_rotvec = (
            config.default_place_orientation_rotvec
        )

        self._client = client
        self._owns_client = client is None
        self._initialized = False
        self._holding_object = False
        self._retreat_pose: np.ndarray | None = None

    def initialize(self) -> None:
        if self._initialized:
            return
        if self._client is None:
            self._client = X5HTTPClient(
                self.server_url,
                timeout_s=self.config.request_timeout_s,
            )
        try:
            health = self._client.health()
            if not health.robot.ready:
                raise RuntimeError(
                    f"X5 server robot is not ready: {health.robot.detail}"
                )
            _require_success(
                self._client.open_gripper(arm=self.arm, wait=True),
                "open gripper during initialize",
            )
        except Exception:
            if self._owns_client:
                self._client.close()
                self._client = None
            raise
        self._holding_object = False
        self._initialized = True

    def shutdown(self, move_home: bool = False) -> None:
        try:
            if move_home and self._initialized:
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
        """home -> approach -> grasp -> close gripper -> retreat -> check gripper."""

        self._require_initialized()
        grasp = _select_grasp(grasp_candidates)
        if grasp is None:
            return _failure("pick", f"No grasp candidate available for {object_name}.")
        if grasp.metadata.get("frame") not in (None, "camera"):
            return _failure(
                "pick",
                f"X5 expects a camera-frame grasp, got {grasp.metadata.get('frame')!r}.",
            )

        plan = build_world_tcp_pick_poses(
            self.T_world_camera,
            grasp.pose,
            approach_distance_m=self.config.approach_distance_m,
        )
        retreat_pose = plan["approach_pose_xyz_rotvec"].copy()
        completed_steps: list[dict[str, Any]] = []
        metadata = {
            "object": object_name,
            "from": from_name,
            "grasp_score": grasp.score,
            "grasp_width": grasp.metadata.get("width"),
            "retreat_pose_xyz_rotvec": retreat_pose,
            **plan,
        }

        try:
            completed_steps.extend(
                self._move_joint_target(self.config.home_joints_deg, "home")
            )
            completed_steps.append(
                self._movej(
                    plan["approach_pose_xyz_rotvec"],
                    self.config.approach_speed_ratio,
                    "approach",
                )
            )
            completed_steps.append(
                self._movel(
                    plan["grasp_pose_xyz_rotvec"],
                    self.config.grasp_speed_ratio,
                    "grasp",
                )
            )
            completed_steps.append(self._close_gripper())
            completed_steps.append(
                self._movel(
                    retreat_pose,
                    self.config.retreat_speed_ratio,
                    "retreat",
                )
            )
            completed_steps.extend(
                self._move_joint_target(
                    self.config.check_gripper_joints_deg,
                    "check_gripper",
                )
            )
        except Exception as exc:
            self._stop()
            metadata["completed_steps"] = completed_steps
            return _failure("pick", str(exc), metadata)

        self._retreat_pose = retreat_pose
        self._holding_object = True
        metadata["completed_steps"] = completed_steps
        return ActionResult(
            success=True,
            action_name="pick",
            message=(
                f"X5 picked {object_name}, closed the gripper, retreated, "
                "and checked the gripper."
            ),
            metadata=metadata,
        )

    def place(
        self,
        object_name: str,
        target_name: str,
        target_bbox: Optional[BBox] = None,
        target_pose: Any = None,
    ) -> ActionResult:
        """preplace -> place -> open -> preplace -> home."""

        self._require_initialized()
        if not self._holding_object:
            return _failure("place", "X5 place requires a successful preceding pick.")

        plan = build_world_tcp_place_poses(
            target_pose,
            self.config.default_place_orientation_rotvec,
            approach_offset_x_m=self.config.place_approach_offset_x_m,
        )
        completed_steps: list[dict[str, Any]] = []
        metadata = {
            "object": object_name,
            "target": target_name,
            "retreat_pose_xyz_rotvec": self._retreat_pose,
            **plan,
        }

        try:
            completed_steps.append(
                self._movej(
                    plan["preplace_pose_xyz_rotvec"],
                    self.config.place_approach_speed_ratio,
                    "preplace",
                )
            )
            completed_steps.append(
                self._movel(
                    plan["place_pose_xyz_rotvec"],
                    self.config.place_speed_ratio,
                    "place",
                )
            )
            completed_steps.append(self._open_gripper())
            self._holding_object = False
            completed_steps.append(
                self._movel(
                    plan["preplace_pose_xyz_rotvec"],
                    self.config.place_approach_speed_ratio,
                    "preplace",
                )
            )
            completed_steps.extend(
                self._move_joint_target(self.config.home_joints_deg, "home")
            )
        except Exception as exc:
            self._stop()
            metadata["completed_steps"] = completed_steps
            return _failure("place", str(exc), metadata)

        self._retreat_pose = None
        metadata["completed_steps"] = completed_steps
        return ActionResult(
            success=True,
            action_name="place",
            message=f"X5 placed {object_name} at {target_name} and returned home.",
            metadata=metadata,
        )

    def move_home(self) -> ActionResult:
        self._require_initialized()
        try:
            completed_steps = self._move_joint_target(
                self.config.home_joints_deg,
                "home",
            )
        except Exception as exc:
            self._stop()
            return _failure("move-home", str(exc))
        return ActionResult(
            success=True,
            action_name="move-home",
            message=f"X5 {self.arm} arm moved home.",
            metadata={"completed_steps": completed_steps},
        )

    def get_eef_pose(self) -> Any:
        self._require_initialized()
        response = self._client.get_state(self.arm)
        _require_success(response, "get X5 state")
        return response.state_after.arms[self.arm].tcp_pose_xyzw

    def _move_joint_target(
        self,
        target_joints_deg: Sequence[float],
        step: str,
    ) -> list[dict[str, Any]]:
        state = self._client.get_state(self.arm)
        _require_success(state, f"get state before {step}")
        start_deg = radians_to_degrees(
            state.state_after.arms[self.arm].joints_rad
        )
        step_count = max(
            1,
            math.ceil(
                max(
                    abs(target - current)
                    for current, target in zip(
                        start_deg,
                        target_joints_deg,
                        strict=True,
                    )
                )
                / self.config.home_max_step_deg
            ),
        )

        completed = []
        for index in range(1, step_count + 1):
            fraction = index / step_count
            waypoint_deg = [
                current + (target - current) * fraction
                for current, target in zip(
                    start_deg,
                    target_joints_deg,
                    strict=True,
                )
            ]
            response = self._client.move_joints(
                self.arm,
                degrees_to_radians(waypoint_deg),
                speed_ratio=self.config.home_speed_ratio,
                wait=True,
            )
            _require_success(response, f"{step} waypoint {index}/{step_count}")
            summary = _response_summary(step, response)
            summary.update(
                waypoint_index=index,
                waypoint_count=step_count,
                joints_deg=waypoint_deg,
            )
            completed.append(summary)
        return completed

    def _movej(
        self,
        pose: Sequence[float],
        speed_ratio: float,
        step: str,
    ) -> dict[str, Any]:
        response = self._client.movej_point(
            self.arm,
            np.asarray(pose, dtype=float).tolist(),
            speed_ratio=speed_ratio,
            wait=True,
        )
        _require_success(response, f"movej_point {step}")
        return _response_summary(step, response)

    def _movel(
        self,
        pose: Sequence[float],
        speed_ratio: float,
        step: str,
    ) -> dict[str, Any]:
        response = self._client.movel_point(
            self.arm,
            np.asarray(pose, dtype=float).tolist(),
            speed_ratio=speed_ratio,
            wait=True,
        )
        _require_success(response, f"movel_point {step}")
        return _response_summary(step, response)

    def _close_gripper(self) -> dict[str, Any]:
        response = self._client.close_gripper(arm=self.arm, wait=True)
        _require_success(response, "close gripper")
        return _response_summary("close_gripper", response)

    def _open_gripper(self) -> dict[str, Any]:
        response = self._client.open_gripper(arm=self.arm, wait=True)
        _require_success(response, "open gripper")
        return _response_summary("open_gripper", response)

    def _require_initialized(self) -> None:
        if not self._initialized or self._client is None:
            raise RuntimeError("RemoteX5ActionBackend is not initialized")

    def _stop(self) -> None:
        try:
            self._client.stop(self.arm)
        except Exception:
            pass


def build_world_tcp_pick_poses(
    T_world_camera: Any,
    T_camera_grasp: Any,
    *,
    approach_distance_m: float,
    T_grasp_tcp: Any = T_GRASP_TCP,
) -> dict[str, np.ndarray]:
    """Convert a camera-frame grasp into world-frame approach and TCP poses."""

    T_world_camera = coerce_se3(T_world_camera, "T_world_camera")
    T_camera_grasp = coerce_se3(T_camera_grasp, "T_camera_grasp")
    T_grasp_tcp = coerce_se3(T_grasp_tcp, "T_grasp_tcp")
    approach_distance = float(approach_distance_m)
    if approach_distance <= 0.0:
        raise ValueError("approach_distance_m must be positive")

    T_camera_approach = T_camera_grasp.copy()
    T_camera_approach[:3, 3] += (
        T_camera_grasp[:3, :3]
        @ np.array([-approach_distance, 0.0, 0.0])
    )
    T_world_tcp_grasp = T_world_camera @ T_camera_grasp @ T_grasp_tcp
    T_world_tcp_approach = T_world_camera @ T_camera_approach @ T_grasp_tcp
    return {
        "T_world_tcp_grasp": T_world_tcp_grasp,
        "T_world_tcp_approach": T_world_tcp_approach,
        "grasp_pose_xyz_rotvec": se3_to_xyz_rotvec(T_world_tcp_grasp),
        "approach_pose_xyz_rotvec": se3_to_xyz_rotvec(T_world_tcp_approach),
    }


def build_world_tcp_place_poses(
    target_pose: Any,
    default_place_orientation_rotvec: Any,
    *,
    approach_offset_x_m: float,
) -> dict[str, np.ndarray]:
    """Build world-frame preplace and place poses with fixed orientation."""

    place_pose = np.concatenate(
        [
            _target_xyz(target_pose),
            _vector(
                default_place_orientation_rotvec,
                3,
                "default_place_orientation_rotvec",
            ),
        ]
    )
    preplace_pose = place_pose.copy()
    preplace_pose[0] += float(approach_offset_x_m)
    return {
        "preplace_pose_xyz_rotvec": preplace_pose,
        "place_pose_xyz_rotvec": place_pose,
    }


def _load_backend_config(
    robot_config_path: str,
    camera_config_path: str,
    *,
    server_url: str | None,
    arm: str | None,
    camera_name: str | None,
    place_approach_offset_x_m: float | None,
    default_place_orientation_rotvec: Sequence[float] | None,
) -> _BackendConfig:
    robot_document = _load_yaml(robot_config_path)
    camera_document = _load_yaml(camera_config_path)
    motion = robot_document.get("action_backend", {})
    robot = robot_document["robot"]

    arm_name = str(arm or motion.get("arm", "left"))
    camera_name = str(camera_name or motion.get("camera_name", "Gemini335"))
    arm_config = robot[arm_name]
    handeye = camera_document[camera_name]["handeye_calibration"]

    T_world_camera = np.eye(4, dtype=float)
    T_world_camera[:3, :3] = np.asarray(handeye["rotation"], dtype=float)
    T_world_camera[:3, 3] = np.asarray(handeye["translation"], dtype=float)

    orientation = (
        default_place_orientation_rotvec
        if default_place_orientation_rotvec is not None
        else motion["default_place_orientation_rotvec"]
    )
    place_offset = (
        place_approach_offset_x_m
        if place_approach_offset_x_m is not None
        else motion.get("place_approach_offset_x_m", -0.05)
    )
    return _BackendConfig(
        server_url=str(
            server_url or motion.get("server_url", "http://127.0.0.1:8000")
        ).rstrip("/"),
        arm=arm_name,
        camera_name=camera_name,
        request_timeout_s=float(motion.get("request_timeout_s", 90.0)),
        approach_distance_m=float(motion.get("approach_distance_m", 0.05)),
        home_speed_ratio=float(motion.get("home_speed_ratio", 0.05)),
        home_max_step_deg=float(motion.get("home_max_step_deg", 4.0)),
        approach_speed_ratio=float(motion.get("approach_speed_ratio", 0.03)),
        grasp_speed_ratio=float(motion.get("grasp_speed_ratio", 0.02)),
        retreat_speed_ratio=float(motion.get("retreat_speed_ratio", 0.02)),
        place_approach_speed_ratio=float(
            motion.get("place_approach_speed_ratio", 0.03)
        ),
        place_speed_ratio=float(motion.get("place_speed_ratio", 0.02)),
        place_approach_offset_x_m=float(place_offset),
        default_place_orientation_rotvec=_vector(
            orientation,
            3,
            "default_place_orientation_rotvec",
        ),
        home_joints_deg=_vector(
            arm_config["home_joints_deg"],
            7,
            f"robot.{arm_name}.home_joints_deg",
        ).tolist(),
        check_gripper_joints_deg=_vector(
            arm_config["check_gripper_joints_deg"],
            7,
            f"robot.{arm_name}.check_gripper_joints_deg",
        ).tolist(),
        T_world_camera=coerce_se3(T_world_camera, "T_world_camera"),
    )


def _load_yaml(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def _vector(values: Any, size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {size} finite values")
    return vector


def _target_xyz(target_pose: Any) -> np.ndarray:
    values = np.asarray(target_pose, dtype=float)
    if values.shape == (4, 4):
        return _vector(values[:3, 3], 3, "target_pose")
    if values.ndim == 1 and values.size >= 3:
        return _vector(values[:3], 3, "target_pose")
    raise ValueError("target_pose must contain world XYZ or be a 4x4 transform")


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


def _failure(
    action_name: str,
    error: str,
    metadata: dict[str, Any] | None = None,
) -> ActionResult:
    return ActionResult(
        success=False,
        action_name=action_name,
        error=error,
        metadata=metadata or {},
    )
