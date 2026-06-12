"""HTTP contracts shared by the GraspNet client and server."""

from __future__ import annotations

import math
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


API_VERSION = "v1"
POSE_FRAME = "camera"


class GraspPredictRequest(BaseModel):
    """Request grasp candidates from RGB-D files on the shared local PC."""

    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    rgb_path: str
    depth_path: str
    workspace_mask_path: str | None = None
    bbox_xyxy: list[float] | None = Field(default=None, min_length=4, max_length=4)
    object_label: str = "object"
    depth_unit: Literal["mm"] = "mm"
    mask_offset_px: int = Field(default=10, ge=0, le=500)
    max_grasps: int = Field(default=20, ge=1, le=500)
    score_threshold: float = Field(default=0.0, ge=0.0)
    collision_detection: bool = True
    nms: bool = True

    @field_validator("bbox_xyxy")
    @classmethod
    def validate_bbox(cls, values: list[float] | None) -> list[float] | None:
        if values is None:
            return None
        converted = [float(value) for value in values]
        if not all(math.isfinite(value) for value in converted):
            raise ValueError("bbox_xyxy values must be finite")
        x1, y1, x2, y2 = converted
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must satisfy x2 > x1 and y2 > y1")
        return converted

class GraspPoseCandidate(BaseModel):
    """One camera-frame parallel-jaw grasp candidate."""

    pose_4x4: list[list[float]]
    score: float
    width: float
    height: float
    depth: float
    object_label: str
    image_xy: list[float] | None = Field(default=None, min_length=2, max_length=2)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("pose_4x4")
    @classmethod
    def validate_pose(cls, rows: list[list[float]]) -> list[list[float]]:
        if len(rows) != 4 or any(len(row) != 4 for row in rows):
            raise ValueError("pose_4x4 must be a 4x4 matrix")
        converted = [[float(value) for value in row] for row in rows]
        if not all(math.isfinite(value) for row in converted for value in row):
            raise ValueError("pose_4x4 values must be finite")
        expected_last_row = [0.0, 0.0, 0.0, 1.0]
        if any(abs(value - expected) > 1e-6 for value, expected in zip(converted[3], expected_last_row)):
            raise ValueError("pose_4x4 must be a homogeneous transform")
        return converted


class GraspPredictResponse(BaseModel):
    request_id: str
    success: bool
    api_version: str = API_VERSION
    pose_frame: Literal["camera"] = POSE_FRAME
    object_label: str
    grasps: list[GraspPoseCandidate]
    num_grasps: int = Field(ge=0)
    duration_ms: float = Field(ge=0.0)
    input_summary: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class GraspHealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api_version: str = API_VERSION
    backend: str
    model_loaded: bool
    device: str
    detail: str = ""
