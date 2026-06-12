"""Wire contracts shared by the X5 HTTP client and server."""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Dict, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator


API_VERSION = "v1"
RGBD_MEDIA_TYPE = "application/x-npz"

ArmName = Literal["left", "right"]
ArmTarget = Literal["left", "right", "all"]


class CameraIntrinsics(BaseModel):
    """Pinhole intrinsics for the aligned RGB-D frame."""

    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    width: int = Field(gt=0)
    height: int = Field(gt=0)


@dataclass(frozen=True)
class RGBDFrame:
    """One aligned RGB-D capture in the camera frame."""

    rgb: np.ndarray
    depth_mm: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp_ns: int
    frame_id: str
    color_timestamp_ns: Optional[int] = None
    depth_timestamp_ns: Optional[int] = None
    depth_unit: str = "mm"

    def validate(self) -> None:
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb must be an HxWx3 uint8 array")
        if self.depth_mm.dtype != np.float32 or self.depth_mm.ndim != 2:
            raise ValueError("depth_mm must be an HxW float32 array")
        if self.rgb.shape[:2] != self.depth_mm.shape:
            raise ValueError("rgb and depth_mm must have matching height and width")
        height, width = self.depth_mm.shape
        if (width, height) != (self.intrinsics.width, self.intrinsics.height):
            raise ValueError("frame shape does not match camera intrinsics")
        if self.timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be positive")
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        if self.depth_unit != "mm":
            raise ValueError("depth_unit must be 'mm'")
        timestamps = (self.color_timestamp_ns, self.depth_timestamp_ns)
        if any(value is not None and value <= 0 for value in timestamps):
            raise ValueError("color/depth timestamps must be positive when provided")

    @property
    def sync_delta_ms(self) -> Optional[float]:
        if self.color_timestamp_ns is None or self.depth_timestamp_ns is None:
            return None
        return abs(self.color_timestamp_ns - self.depth_timestamp_ns) / 1_000_000.0


class ComponentHealth(BaseModel):
    ready: bool
    backend: str
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api_version: str = API_VERSION
    camera: ComponentHealth
    robot: ComponentHealth


class ArmState(BaseModel):
    connected: bool
    moving: bool
    joints_rad: list[float] = Field(min_length=7, max_length=7)
    tcp_pose_xyzw: list[float] = Field(min_length=7, max_length=7)
    tool_frame_no: Optional[int] = Field(default=None, ge=0, le=16)
    tool_frame_pose_xyzw: Optional[list[float]] = Field(
        default=None,
        min_length=7,
        max_length=7,
    )
    gripper_position: Optional[float] = None


class GripperState(BaseModel):
    connected: bool
    moving: bool
    position: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    raw_position: Optional[int] = Field(default=None, ge=0, le=1000)
    grip_status: Optional[int] = Field(default=None, ge=0, le=3)


class RobotState(BaseModel):
    arms: Dict[str, ArmState]
    gripper: Optional[GripperState] = None
    timestamp_ns: int


class GetStateCommand(BaseModel):
    type: Literal["get_state"] = "get_state"
    arm: ArmTarget = "all"


class MoveJointsCommand(BaseModel):
    type: Literal["move_joints"] = "move_joints"
    arm: ArmName
    joints_rad: list[float] = Field(min_length=7, max_length=7)
    speed_ratio: float = Field(default=0.1, gt=0.0, le=1.0)
    wait: bool = True


class CartesianPointCommand(BaseModel):
    """World-frame TCP target in meters and rotation-vector radians."""

    arm: ArmName
    tcp_pose_xyz_rotvec: list[float] = Field(min_length=6, max_length=6)
    speed_ratio: float = Field(default=0.05, gt=0.0, le=1.0)
    wait: bool = True

    @field_validator("tcp_pose_xyz_rotvec")
    @classmethod
    def validate_finite_pose(cls, values: list[float]) -> list[float]:
        converted = [float(value) for value in values]
        if not all(np.isfinite(value) for value in converted):
            raise ValueError("tcp_pose_xyz_rotvec values must be finite")
        return converted


class MoveJPointCommand(CartesianPointCommand):
    type: Literal["movej_point"] = "movej_point"


class MoveLPointCommand(CartesianPointCommand):
    type: Literal["movel_point"] = "movel_point"


class StopCommand(BaseModel):
    type: Literal["stop"] = "stop"
    arm: ArmTarget = "all"


class SetGripperCommand(BaseModel):
    """Set the single gripper position: 0.0 closed, 1.0 fully open."""

    type: Literal["set_gripper"] = "set_gripper"
    position: float = Field(ge=0.0, le=1.0)
    wait: bool = True


RobotCommand = Annotated[
    Union[
        GetStateCommand,
        MoveJointsCommand,
        MoveJPointCommand,
        MoveLPointCommand,
        SetGripperCommand,
        StopCommand,
    ],
    Field(discriminator="type"),
]


class RobotCommandRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    command: RobotCommand


class RobotCommandResponse(BaseModel):
    request_id: str
    success: bool
    accepted_command: Dict[str, Any]
    state_before: RobotState
    state_after: RobotState
    server_timestamp_ns: int
    duration_ms: float
    error: Optional[str] = None


def encode_rgbd_frame(frame: RGBDFrame) -> bytes:
    """Serialize an RGB-D frame without converting image arrays to JSON."""

    frame.validate()
    intrinsics = np.asarray(
        [
            frame.intrinsics.fx,
            frame.intrinsics.fy,
            frame.intrinsics.cx,
            frame.intrinsics.cy,
            frame.intrinsics.width,
            frame.intrinsics.height,
        ],
        dtype=np.float64,
    )
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        wire_version=np.asarray(1, dtype=np.uint8),
        rgb=frame.rgb,
        depth_mm=frame.depth_mm,
        intrinsics=intrinsics,
        timestamp_ns=np.asarray(frame.timestamp_ns, dtype=np.int64),
        frame_id=np.asarray(frame.frame_id),
        color_timestamp_ns=np.asarray(frame.color_timestamp_ns or -1, dtype=np.int64),
        depth_timestamp_ns=np.asarray(frame.depth_timestamp_ns or -1, dtype=np.int64),
        depth_unit=np.asarray(frame.depth_unit),
    )
    return buffer.getvalue()


def decode_rgbd_frame(payload: bytes) -> RGBDFrame:
    """Deserialize and validate an RGB-D NPZ response."""

    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            if int(data["wire_version"].item()) != 1:
                raise ValueError("unsupported RGB-D wire version")
            intrinsics_values = np.asarray(data["intrinsics"], dtype=np.float64).tolist()
            color_timestamp_ns = (
                int(data["color_timestamp_ns"].item())
                if "color_timestamp_ns" in data
                else -1
            )
            depth_timestamp_ns = (
                int(data["depth_timestamp_ns"].item())
                if "depth_timestamp_ns" in data
                else -1
            )
            frame = RGBDFrame(
                rgb=np.asarray(data["rgb"], dtype=np.uint8),
                depth_mm=np.asarray(data["depth_mm"], dtype=np.float32),
                intrinsics=CameraIntrinsics(
                    fx=intrinsics_values[0],
                    fy=intrinsics_values[1],
                    cx=intrinsics_values[2],
                    cy=intrinsics_values[3],
                    width=int(intrinsics_values[4]),
                    height=int(intrinsics_values[5]),
                ),
                timestamp_ns=int(data["timestamp_ns"].item()),
                frame_id=str(data["frame_id"].item()),
                color_timestamp_ns=color_timestamp_ns if color_timestamp_ns > 0 else None,
                depth_timestamp_ns=depth_timestamp_ns if depth_timestamp_ns > 0 else None,
                depth_unit=str(data["depth_unit"].item()) if "depth_unit" in data else "mm",
            )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid RGB-D payload: {exc}") from exc

    frame.validate()
    return frame
