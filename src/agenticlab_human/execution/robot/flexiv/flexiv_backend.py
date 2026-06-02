"""Flexiv implementation of the semantic ActionBackend contract.

Default mode is planning-only (`execute=False`): the backend validates data
flow, converts a camera-frame grasp into Flexiv base-frame TCP poses, and
returns those poses in ActionResult metadata without connecting to hardware.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import yaml

from agenticlab_human.execution.action_backend import ActionResult
from agenticlab_human.perception.backend.grasp_backend import GraspCandidate
from agenticlab_human.perception.backend.perception_backend import BBox


DEFAULT_ROBOT_CONFIG = "configs/robot/flexiv_config.yaml"
DEFAULT_CAMERA_CONFIG = "configs/perception/camera_config.yaml"


class FlexivActionBackend:
    """Flexiv backend that can dry-run pose conversion or execute on hardware."""

    def __init__(
        self,
        robot_config_path: str = DEFAULT_ROBOT_CONFIG,
        camera_config_path: str = DEFAULT_CAMERA_CONFIG,
        camera_name: str = "FemtoBolt",
        execute: bool = False,
    ) -> None:
        self.robot_config_path = robot_config_path
        self.camera_config_path = camera_config_path
        self.camera_name = camera_name
        self.execute = execute
        self.robot_config = _load_yaml(robot_config_path)
        self.camera_config = _load_yaml(camera_config_path)
        self.action_config = self.robot_config.get("ActionWrapper", {})
        self.flexiv_config = self.robot_config.get("Flexiv", {})
        self.robot_speed = self.flexiv_config.get("max_cartesian_speed", 0.1)
        self.robot_acceleration = self.flexiv_config.get("max_cartesian_acceleration", 0.1)
        self.approach_distance = self.action_config.get("approach_distance", 0.1)
        self.controller = None
        self.gripper = None

        camera_cfg = self.camera_config.get(camera_name)
        if not camera_cfg:
            raise ValueError(f"Camera config not found: {camera_name}")
        handeye = camera_cfg.get("handeye_calibration", {})
        self.T_base_camera = np.eye(4, dtype=float)
        self.T_base_camera[:3, :3] = np.asarray(handeye["rotation"], dtype=float)
        self.T_base_camera[:3, 3] = np.asarray(handeye["translation"], dtype=float)

    def initialize(self) -> None:
        if not self.execute:
            return
        from agenticlab_human.execution.robot.flexiv.flexiv_controller import FlexivController
        from agenticlab_human.execution.robot.flexiv.flexiv_gripper_controller import (
            FlexivGripperController,
        )

        self.controller = FlexivController(self.robot_config)
        if not self.controller.connect():
            raise ConnectionError("Failed to connect to Flexiv robot.")
        self.gripper = FlexivGripperController(self.robot_config, self.controller.robot)
        self.gripper.connect()

    def shutdown(self, move_home: bool = False) -> None:
        if not self.execute:
            return
        if move_home and self.controller:
            self.controller.move_to_home()
        if self.gripper:
            self.gripper.disconnect()
            self.gripper = None
        if self.controller:
            self.controller.disconnect()
            self.controller = None

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

        try:
            plan = self._plan_pick_poses(grasp)
        except ValueError as exc:
            return ActionResult(
                success=False,
                action_name="pick",
                error=str(exc),
                metadata={"object": object_name, "from": from_name},
            )

        metadata = {
            "object": object_name,
            "from": from_name,
            "execute": self.execute,
            "camera_name": self.camera_name,
            "grasp_score": grasp.score,
            "grasp_width": grasp.metadata.get("width"),
            "object_has_bbox": object_bbox is not None,
            "object_has_pose": object_pose is not None,
            **plan,
        }

        if not self.execute:
            return ActionResult(
                success=True,
                action_name="pick",
                message=f"Flexiv dry-run pick plan generated for {object_name}.",
                metadata=metadata,
            )

        if self.controller is None or self.gripper is None:
            return ActionResult(
                success=False,
                action_name="pick",
                error="Flexiv backend is not initialized.",
                metadata=metadata,
            )

        width = grasp.metadata.get("width")
        open_width = _safe_open_width(width, self.robot_config.get("Gripper", {}))
        self.gripper.open_gripper(width=open_width)
        if not self.controller.moveptp(plan["approach_pose6d"], zone_radius="Z50"):
            return _motion_failed("pick", "moveptp approach failed", metadata)
        if not self.controller.movel(
            plan["grasp_pose6d"],
            speed=self.robot_speed,
            acceleration=self.robot_acceleration,
        ):
            return _motion_failed("pick", "movel grasp failed", metadata)
        self.gripper.close_gripper()
        if not self.controller.movel(
            plan["approach_pose6d"],
            speed=self.robot_speed,
            acceleration=self.robot_acceleration,
        ):
            return _motion_failed("pick", "movel retreat failed", metadata)
        return ActionResult(
            success=True,
            action_name="pick",
            message=f"Flexiv executed pick for {object_name}.",
            metadata=metadata,
        )

    def place_on_object(
        self,
        object_name: str,
        target_name: str,
        target_bbox: Optional[BBox] = None,
        target_pose: Any = None,
    ) -> ActionResult:
        return ActionResult(
            success=False,
            action_name="place-on-object",
            error="Flexiv place_on_object is not implemented in this one-action test backend.",
            metadata={
                "object": object_name,
                "target": target_name,
                "execute": self.execute,
                "has_target_bbox": target_bbox is not None,
                "has_target_pose": target_pose is not None,
            },
        )

    def place_on_surface(
        self,
        object_name: str,
        surface_name: str,
        target_bbox: Optional[BBox] = None,
        target_pose: Any = None,
    ) -> ActionResult:
        return ActionResult(
            success=False,
            action_name="place-on-surface",
            error="Flexiv place_on_surface is not implemented in this one-action test backend.",
            metadata={"object": object_name, "surface": surface_name, "execute": self.execute},
        )

    def place_in_container(
        self,
        object_name: str,
        container_name: str,
        target_bbox: Optional[BBox] = None,
        target_pose: Any = None,
    ) -> ActionResult:
        return ActionResult(
            success=False,
            action_name="place-in-container",
            error="Flexiv place_in_container is not implemented in this one-action test backend.",
            metadata={"object": object_name, "container": container_name, "execute": self.execute},
        )

    def move_home(self) -> ActionResult:
        if not self.execute:
            return ActionResult(
                success=True,
                action_name="move-home",
                message="Flexiv dry-run move home.",
                metadata={"execute": False},
            )
        if self.controller and self.controller.move_to_home():
            return ActionResult(success=True, action_name="move-home", message="Flexiv moved home.")
        return ActionResult(
            success=False,
            action_name="move-home",
            error="Flexiv move_home failed.",
            metadata={"execute": True},
        )

    def get_eef_pose(self) -> Any:
        if self.controller:
            return self.controller.get_tcp_pose()
        return None

    def _plan_pick_poses(self, grasp: GraspCandidate) -> dict[str, Any]:
        T_cam_grasp = _coerce_se3(grasp.pose)
        T_cam_approach = T_cam_grasp.copy()
        T_cam_approach[:3, 3] = (
            T_cam_grasp[:3, 3]
            + T_cam_grasp[:3, :3] @ np.array([-self.approach_distance, 0.0, 0.0])
        )
        T_base_ee_grasp = self._camera_grasp_to_base_ee(T_cam_grasp)
        T_base_ee_approach = self._camera_grasp_to_base_ee(T_cam_approach)
        return {
            "T_cam_grasp": T_cam_grasp,
            "T_base_camera": self.T_base_camera,
            "T_base_ee_grasp": T_base_ee_grasp,
            "T_base_ee_approach": T_base_ee_approach,
            "grasp_pose6d": _se3_to_pose6d(T_base_ee_grasp),
            "approach_pose6d": _se3_to_pose6d(T_base_ee_approach),
            "approach_distance": self.approach_distance,
        }

    def _camera_grasp_to_base_ee(self, T_cam_grasp: np.ndarray) -> np.ndarray:
        T_base_grasp = self.T_base_camera @ T_cam_grasp
        R_eg = np.array(
            [
                [0, 0, -1],
                [0, 1, 0],
                [1, 0, 0],
            ],
            dtype=float,
        )
        T_grasp_ee = np.eye(4, dtype=float)
        T_grasp_ee[:3, :3] = R_eg.T
        return T_base_grasp @ T_grasp_ee


def _load_yaml(path: str) -> dict:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path_obj.read_text()) or {}


def _select_grasp(candidates: Optional[Sequence[GraspCandidate]]) -> Optional[GraspCandidate]:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda candidate: candidate.score if candidate.score is not None else 0.0,
        reverse=True,
    )[0]


def _coerce_se3(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == (4, 4):
        return arr
    raise ValueError(f"Expected grasp pose as 4x4 SE3 matrix, got shape {arr.shape}.")


def _se3_to_pose6d(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[:3, 3], _rotation_matrix_to_rotvec(T[:3, :3])])


def _rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3, dtype=float)
    if math.pi - theta < 1e-6:
        axis = np.sqrt(np.maximum(np.diag(R) + 1.0, 0.0) / 2.0)
        axis[0] = math.copysign(axis[0], R[2, 1] - R[1, 2])
        axis[1] = math.copysign(axis[1], R[0, 2] - R[2, 0])
        axis[2] = math.copysign(axis[2], R[1, 0] - R[0, 1])
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return np.zeros(3, dtype=float)
        return axis / norm * theta
    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=float,
    ) / (2.0 * math.sin(theta))
    return axis * theta


def _safe_open_width(width: Any, gripper_cfg: dict) -> Optional[float]:
    if width is None:
        return None
    open_width = float(gripper_cfg.get("open_width", 0.09))
    close_width = float(gripper_cfg.get("close_width", 0.005))
    return float(np.clip(float(width) + 0.01, close_width, open_width))


def _motion_failed(action_name: str, error: str, metadata: dict[str, Any]) -> ActionResult:
    return ActionResult(
        success=False,
        action_name=action_name,
        error=error,
        metadata=metadata,
    )
