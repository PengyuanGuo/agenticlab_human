import numpy as np
import pytest


pytest.importorskip("MinkowskiEngine")
pytest.importorskip("pointnet2._ext")

from graspnetAPI import GraspGroup  # noqa: E402
from agenticlab_human.perception.grasping.graspnet_wrapper import GraspNetWrapper  # noqa: E402


def _grasp(score: float, rotation: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray([score, 0.08, 0.02, 0.03]),
            rotation.reshape(-1),
            np.asarray([0.1, 0.2, 0.3, -1.0]),
        )
    )


def _wrapper() -> GraspNetWrapper:
    wrapper = object.__new__(GraspNetWrapper)
    wrapper.cfg = {"angle_threshold": 30}
    wrapper.base_from_camera_rotation = np.eye(3)
    return wrapper


def test_approach_angle_uses_grasp_x_axis_in_robot_base():
    assert GraspNetWrapper._approach_angle_deg(np.eye(3), np.eye(3)) == pytest.approx(0)

    rotate_z_90 = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert GraspNetWrapper._approach_angle_deg(
        rotate_z_90,
        np.eye(3),
    ) == pytest.approx(90)


def test_filter_grasps_keeps_score_order_within_angle_range():
    rotate_z_45 = np.asarray(
        [
            [np.sqrt(0.5), -np.sqrt(0.5), 0.0],
            [np.sqrt(0.5), np.sqrt(0.5), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    grasps = GraspGroup(
        np.asarray(
            [
                _grasp(0.9, rotate_z_45),
                _grasp(0.8, np.eye(3)),
                _grasp(0.7, np.eye(3)),
            ]
        )
    )

    filtered = _wrapper().filter_grasps(grasps, angle_range=(0, 30))

    assert filtered.scores.tolist() == pytest.approx([0.8, 0.7])


def test_filter_grasps_falls_back_to_closest_angle():
    rotate_z_45 = np.asarray(
        [
            [np.sqrt(0.5), -np.sqrt(0.5), 0.0],
            [np.sqrt(0.5), np.sqrt(0.5), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotate_z_90 = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    grasps = GraspGroup(
        np.asarray(
            [
                _grasp(0.9, rotate_z_90),
                _grasp(0.8, rotate_z_45),
            ]
        )
    )

    filtered = _wrapper().filter_grasps(grasps, angle_range=(0, 30))

    assert len(filtered) == 1
    assert filtered.scores[0] == pytest.approx(0.8)
