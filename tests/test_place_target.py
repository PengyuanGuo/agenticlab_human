import numpy as np
import pytest

from agenticlab_human.execution.place_target import (
    estimate_place_target,
    median_depth_at_pixel,
)
from agenticlab_human.execution.robot.x5.contracts import CameraIntrinsics


INTRINSICS = CameraIntrinsics(
    fx=100.0,
    fy=100.0,
    cx=2.0,
    cy=2.0,
    width=5,
    height=5,
)


def test_estimate_place_target_uses_patch_median_transform_and_world_x_offset():
    depth_mm = np.full((5, 5), 1000.0, dtype=np.float32)
    depth_mm[2, 2] = 0.0
    depth_mm[1, 1] = np.nan
    T_world_camera = np.eye(4)
    T_world_camera[:3, 3] = [0.5, -0.2, 0.1]

    result = estimate_place_target(
        depth_mm=depth_mm,
        intrinsics=INTRINSICS,
        pixel_xy=(3, 2),
        T_world_camera=T_world_camera,
        place_offset_world_x_m=0.05,
        depth_patch_px=3,
    )

    assert result.depth_mm == 1000.0
    np.testing.assert_allclose(result.p_camera, [0.01, 0.0, 1.0])
    np.testing.assert_allclose(result.p_world_target, [0.51, -0.2, 1.1])
    np.testing.assert_allclose(result.p_world_place, [0.56, -0.2, 1.1])


def test_median_depth_rejects_patch_without_valid_depth():
    depth_mm = np.zeros((5, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="no valid depth"):
        median_depth_at_pixel(depth_mm, (2, 2), patch_size=3)


def test_median_depth_requires_odd_patch_size():
    depth_mm = np.ones((5, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="positive odd"):
        median_depth_at_pixel(depth_mm, (2, 2), patch_size=4)
