"""Unit and pose conversions shared by the X5 client and server."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def degrees_to_radians(values: Sequence[float]) -> list[float]:
    return [math.radians(float(value)) for value in values]


def radians_to_degrees(values: Sequence[float]) -> list[float]:
    return [math.degrees(float(value)) for value in values]


def euler_xyz_deg_to_quat_xyzw(
    a_deg: float,
    b_deg: float,
    c_deg: float,
) -> tuple[float, float, float, float]:
    roll, pitch, yaw = degrees_to_radians([a_deg, b_deg, c_deg])
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rotvec_to_quat_xyzw(
    rotvec_rad: Sequence[float],
) -> tuple[float, float, float, float]:
    rx, ry, rz = _vector(rotvec_rad, 3, "rotvec_rad")
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    scale = math.sin(theta / 2.0) / theta
    return rx * scale, ry * scale, rz * scale, math.cos(theta / 2.0)


def rotvec_to_euler_xyz_deg(
    rotvec_rad: Sequence[float],
) -> tuple[float, float, float]:
    rotation = _rotvec_to_rotation_matrix(rotvec_rad)
    horizontal = math.hypot(rotation[0, 0], rotation[1, 0])
    if horizontal > 1e-9:
        a_rad = math.atan2(rotation[2, 1], rotation[2, 2])
        b_rad = math.atan2(-rotation[2, 0], horizontal)
        c_rad = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        a_rad = math.atan2(-rotation[1, 2], rotation[1, 1])
        b_rad = math.atan2(-rotation[2, 0], horizontal)
        c_rad = 0.0
    return tuple(radians_to_degrees([a_rad, b_rad, c_rad]))


def tcp_pose_xyzw_to_xyz_rotvec(
    tcp_pose_xyzw: Sequence[float],
) -> list[float]:
    values = _vector(tcp_pose_xyzw, 7, "tcp_pose_xyzw")
    quaternion = _normalized_quaternion(
        values[3:],
        "tcp_pose_xyzw quaternion",
    )
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    qw = min(1.0, max(-1.0, float(quaternion[3])))
    angle = 2.0 * math.acos(qw)
    sin_half_angle = math.sqrt(max(0.0, 1.0 - qw * qw))
    if sin_half_angle < 1e-9:
        rotvec = np.zeros(3, dtype=float)
    else:
        rotvec = quaternion[:3] * (angle / sin_half_angle)
    return values[:3].tolist() + rotvec.tolist()


def rotation_matrix_to_rotvec(rotation: Any) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    cos_theta = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3, dtype=float)
    if math.pi - theta < 1e-6:
        axis = np.sqrt(np.maximum(np.diag(matrix) + 1.0, 0.0) / 2.0)
        axis[0] = math.copysign(axis[0], matrix[2, 1] - matrix[1, 2])
        axis[1] = math.copysign(axis[1], matrix[0, 2] - matrix[2, 0])
        axis[2] = math.copysign(axis[2], matrix[1, 0] - matrix[0, 1])
        norm = np.linalg.norm(axis)
        return np.zeros(3, dtype=float) if norm < 1e-9 else axis / norm * theta
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=float,
    ) / (2.0 * math.sin(theta))
    return axis * theta


def coerce_se3(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must be a homogeneous SE3 transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix


def se3_to_xyz_rotvec(transform: Any) -> np.ndarray:
    matrix = coerce_se3(transform, "transform")
    return np.concatenate(
        [matrix[:3, 3], rotation_matrix_to_rotvec(matrix[:3, :3])]
    )


def xapi_pose_to_xyzw(point_or_pose: Any) -> list[float]:
    """Convert an xapi millimeter/Euler pose to meters/quaternion."""

    if point_or_pose is None or point_or_pose is False:
        raise ValueError("xapi returned an invalid pose")

    pose = getattr(point_or_pose, "pose", point_or_pose)
    x_mm = _read_value(pose, "x", 0)
    y_mm = _read_value(pose, "y", 1)
    z_mm = _read_value(pose, "z", 2)
    a_deg = _read_value(pose, "a", 3, default=0.0)
    b_deg = _read_value(pose, "b", 4, default=0.0)
    c_deg = _read_value(pose, "c", 5, default=0.0)
    quaternion = euler_xyz_deg_to_quat_xyzw(a_deg, b_deg, c_deg)
    return [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0, *quaternion]


def _vector(values: Any, size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {size} finite values")
    return vector


def _normalized_quaternion(values: Any, name: str) -> np.ndarray:
    quaternion = _vector(values, 4, name)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return quaternion / norm


def _rotvec_to_rotation_matrix(rotvec_rad: Sequence[float]) -> np.ndarray:
    rx, ry, rz = _vector(rotvec_rad, 3, "rotvec_rad")
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-12:
        return np.eye(3, dtype=float)
    x, y, z = rx / theta, ry / theta, rz / theta
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )
    return (
        np.eye(3, dtype=float)
        + math.sin(theta) * skew
        + (1.0 - math.cos(theta)) * (skew @ skew)
    )


def _read_value(
    obj: Any,
    attr_name: str,
    index: int,
    *,
    default: float | None = None,
) -> float:
    if isinstance(obj, Mapping) and attr_name in obj:
        return float(obj[attr_name])
    if hasattr(obj, attr_name):
        return float(getattr(obj, attr_name))
    if (
        isinstance(obj, Sequence)
        and not isinstance(obj, (str, bytes))
        and len(obj) > index
    ):
        return float(obj[index])
    if default is not None:
        return default
    raise ValueError(f"missing value: {attr_name}")
