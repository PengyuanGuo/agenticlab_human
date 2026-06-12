"""Geometry helpers for estimating an X5 place target from aligned RGB-D."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from agenticlab_human.execution.robot.x5.contracts import CameraIntrinsics


@dataclass(frozen=True)
class PlaceTargetResult:
    """A selected place pixel represented in camera and world frames."""

    pixel_xy: tuple[int, int]
    depth_mm: float
    valid_depth_count: int
    depth_patch_xyxy: tuple[int, int, int, int]
    p_camera: np.ndarray
    p_world_target: np.ndarray
    p_world_place: np.ndarray
    place_offset_world_x_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixel_xy": list(self.pixel_xy),
            "depth_mm": self.depth_mm,
            "valid_depth_count": self.valid_depth_count,
            "depth_patch_xyxy": list(self.depth_patch_xyxy),
            "p_camera": self.p_camera.tolist(),
            "p_world_target": self.p_world_target.tolist(),
            "place_offset_world_x_m": self.place_offset_world_x_m,
            "p_world_place": self.p_world_place.tolist(),
        }


def estimate_place_target(
    *,
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    pixel_xy: Sequence[float],
    T_world_camera: Any,
    place_offset_world_x_m: float,
    depth_patch_px: int = 9,
) -> PlaceTargetResult:
    """Estimate a world-frame place point from a target pixel."""

    depth_value_mm, valid_count, patch_xyxy, pixel = median_depth_at_pixel(
        depth_mm,
        pixel_xy,
        patch_size=depth_patch_px,
    )
    p_camera = deproject_pixel_to_camera(
        pixel,
        depth_value_mm,
        intrinsics,
    )
    p_world_target = transform_camera_point_to_world(
        p_camera,
        T_world_camera,
    )
    offset_x = _finite_float(
        place_offset_world_x_m,
        "place_offset_world_x_m",
    )
    p_world_place = p_world_target.copy()
    p_world_place[0] += offset_x
    return PlaceTargetResult(
        pixel_xy=pixel,
        depth_mm=depth_value_mm,
        valid_depth_count=valid_count,
        depth_patch_xyxy=patch_xyxy,
        p_camera=p_camera,
        p_world_target=p_world_target,
        p_world_place=p_world_place,
        place_offset_world_x_m=offset_x,
    )


def median_depth_at_pixel(
    depth_mm: np.ndarray,
    pixel_xy: Sequence[float],
    *,
    patch_size: int = 9,
) -> tuple[float, int, tuple[int, int, int, int], tuple[int, int]]:
    """Return median valid depth around a pixel and the clipped patch bounds."""

    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError(f"depth_mm must be a 2D array, got shape {depth.shape}")
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("depth_patch_px must be a positive odd integer")
    if len(pixel_xy) != 2:
        raise ValueError("pixel_xy must contain exactly 2 values")

    u = int(round(float(pixel_xy[0])))
    v = int(round(float(pixel_xy[1])))
    height, width = depth.shape
    if not 0 <= u < width or not 0 <= v < height:
        raise ValueError(
            f"place pixel {(u, v)} is outside depth image {width}x{height}"
        )

    radius = patch_size // 2
    x1 = max(0, u - radius)
    y1 = max(0, v - radius)
    x2 = min(width, u + radius + 1)
    y2 = min(height, v + radius + 1)
    patch = np.asarray(depth[y1:y2, x1:x2], dtype=float)
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if not valid.size:
        raise ValueError(
            f"no valid depth around place pixel {(u, v)} "
            f"within patch {(x1, y1, x2, y2)}"
        )
    return (
        float(np.median(valid)),
        int(valid.size),
        (x1, y1, x2, y2),
        (u, v),
    )


def deproject_pixel_to_camera(
    pixel_xy: Sequence[float],
    depth_mm: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Convert one aligned RGB pixel and depth into camera-frame XYZ meters."""

    if len(pixel_xy) != 2:
        raise ValueError("pixel_xy must contain exactly 2 values")
    u = float(pixel_xy[0])
    v = float(pixel_xy[1])
    z = _finite_float(depth_mm, "depth_mm") / 1000.0
    if z <= 0.0:
        raise ValueError("depth_mm must be positive")
    x = (u - float(intrinsics.cx)) * z / float(intrinsics.fx)
    y = (v - float(intrinsics.cy)) * z / float(intrinsics.fy)
    point = np.asarray([x, y, z], dtype=float)
    if not np.all(np.isfinite(point)):
        raise ValueError("deprojected camera point must contain finite values")
    return point


def transform_camera_point_to_world(
    p_camera: Sequence[float],
    T_world_camera: Any,
) -> np.ndarray:
    """Transform camera-frame XYZ into world-frame XYZ."""

    point = np.asarray(p_camera, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("p_camera must contain exactly 3 finite values")
    transform = _coerce_se3(T_world_camera, "T_world_camera")
    point_h = np.concatenate([point, [1.0]])
    p_world = (transform @ point_h)[:3]
    if not np.all(np.isfinite(p_world)):
        raise ValueError("p_world must contain finite values")
    return p_world


def _coerce_se3(value: Any, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must contain finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must be a homogeneous transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return transform


def _finite_float(value: Any, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted
