from types import SimpleNamespace

import numpy as np
import pytest

from agenticlab_human.perception.backend.grasp_backend import GraspCandidate
from agenticlab_human.execution.robot.x5.x5_remote_backend import (
    RemoteX5ActionBackend,
    T_GRASP_EE,
    T_GRASP_TCP,
    build_world_tcp_pick_poses,
    build_world_tcp_place_poses,
)

ROBOT_CONFIG = "configs/robot/x5_config.yaml"
CAMERA_CONFIG = "configs/perception/camera_config.yaml"


class FakeX5HTTPClient:
    def __init__(self, *, fail_step=None):
        self.fail_step = fail_step
        self.calls = []
        self.home_joints_rad = np.radians([-24, 10, -53, 102, 101, 80, -18]).tolist()
        self.joints_rad = self.home_joints_rad.copy()
        self.tcp_pose_xyzw = [0.2965, -0.4514, 0.3033, 0.0, 0.0, 0.0, 1.0]

    def health(self):
        self.calls.append(("health",))
        return SimpleNamespace(
            robot=SimpleNamespace(ready=True, detail="ready"),
        )

    def get_state(self, arm):
        self.calls.append(("get_state", arm))
        return self._response()

    def move_joints(
        self,
        arm,
        joints_rad,
        *,
        speed_ratio,
        wait,
        request_id=None,
    ):
        self.calls.append(
            ("move_joints", arm, list(joints_rad), speed_ratio, wait, request_id)
        )
        if self.fail_step == "home":
            return self._response(success=False, error="home rejected")
        self.joints_rad = list(joints_rad)
        return self._response(request_id=request_id)

    def movej_point(
        self,
        arm,
        pose6d,
        *,
        speed_ratio,
        wait,
        request_id=None,
    ):
        self.calls.append(
            ("movej_point", arm, list(pose6d), speed_ratio, wait, request_id)
        )
        if self.fail_step == "approach" and "approach" in (request_id or ""):
            return self._response(success=False, error="approach rejected")
        if self.fail_step == "preplace" and "preplace" in (request_id or ""):
            return self._response(success=False, error="pre-place rejected")
        return self._response(request_id=request_id)

    def movel_point(
        self,
        arm,
        pose6d,
        *,
        speed_ratio,
        wait,
        request_id=None,
    ):
        self.calls.append(
            ("movel_point", arm, list(pose6d), speed_ratio, wait, request_id)
        )
        if self.fail_step == "grasp" and "grasp" in (request_id or ""):
            return self._response(success=False, error="grasp rejected")
        if self.fail_step == "retreat" and "retreat" in (request_id or ""):
            return self._response(success=False, error="retreat rejected")
        if self.fail_step == "place" and "place-place" in (request_id or ""):
            return self._response(success=False, error="place rejected")
        return self._response(request_id=request_id)

    def stop(self, arm):
        self.calls.append(("stop", arm))
        return self._response()

    def close(self):
        self.calls.append(("close",))

    def _response(self, *, success=True, error=None, request_id="fake-request"):
        arm_state = SimpleNamespace(
            joints_rad=self.joints_rad.copy(),
            tcp_pose_xyzw=self.tcp_pose_xyzw.copy(),
        )
        return SimpleNamespace(
            success=success,
            error=error,
            request_id=request_id,
            duration_ms=1.0,
            state_after=SimpleNamespace(arms={"left": arm_state}),
        )


def _known_grasp():
    pose = np.eye(4)
    pose[:3, :3] = np.array(
        [
            [-0.10528455, -0.9855515, 0.13267802],
            [0.61262065, -0.16937618, -0.77201533],
            [0.78333336, 0.0, 0.6216019],
        ]
    )
    pose[:3, 3] = [-0.22220398, 0.35045108, 0.893]
    return GraspCandidate(
        pose=pose,
        score=0.95,
        object_name="number-block",
        metadata={"frame": "camera"},
    )


def test_grasp_to_tcp_transform_matches_x5_tool_axis_convention():
    expected_rotation = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(T_GRASP_EE[:3, :3], expected_rotation)
    np.testing.assert_array_equal(T_GRASP_TCP, T_GRASP_EE)


def test_pick_pose_rejects_non_orthonormal_rotation():
    invalid_grasp = np.eye(4)
    invalid_grasp[0, 0] = 2.0

    with pytest.raises(ValueError, match="orthonormal"):
        build_world_tcp_pick_poses(
            np.eye(4),
            invalid_grasp,
            approach_distance_m=0.05,
        )


def test_build_world_tcp_pick_poses_applies_approach_and_grasp_tcp_transform():
    T_world_camera = np.eye(4)
    T_world_camera[:3, 3] = [0.5, 0.1, 0.2]
    T_camera_grasp = np.eye(4)
    T_camera_grasp[:3, 3] = [0.2, 0.0, 0.3]

    plan = build_world_tcp_pick_poses(
        T_world_camera,
        T_camera_grasp,
        approach_distance_m=0.1,
    )

    np.testing.assert_allclose(
        plan["T_world_tcp_grasp"][:3, 3],
        [0.7, 0.1, 0.5],
    )
    np.testing.assert_allclose(
        plan["T_world_tcp_approach"][:3, 3],
        [0.6, 0.1, 0.5],
    )
    np.testing.assert_allclose(
        plan["T_world_tcp_grasp"][:3, :3],
        T_GRASP_TCP[:3, :3],
    )


def test_build_world_tcp_place_poses_offsets_only_world_x_and_uses_home_rotvec():
    plan = build_world_tcp_place_poses(
        [0.4, -0.2, 0.3, 9.0, 8.0, 7.0],
        [0.1, 0.2, 0.3],
        approach_offset_x_m=-0.05,
    )

    np.testing.assert_allclose(
        plan["preplace_pose_xyz_rotvec"],
        [0.35, -0.2, 0.3, 0.1, 0.2, 0.3],
    )
    np.testing.assert_allclose(
        plan["place_pose_xyz_rotvec"],
        [0.4, -0.2, 0.3, 0.1, 0.2, 0.3],
    )


def test_gemini335_known_anygrasp_pose_produces_expected_world_tcp_pose():
    T_world_camera = np.eye(4)
    T_world_camera[:3, :3] = np.array(
        [
            [0.00491955, 0.8216197, 0.5700148],
            [-0.0104614, 0.57003279, -0.82155534],
            [-0.99993318, -0.00192147, 0.01139959],
        ]
    )
    T_world_camera[:3, 3] = [-0.13925853, -0.12331877, -0.24238386]

    T_camera_grasp = np.eye(4)
    T_camera_grasp[:3, :3] = np.array(
        [
            [-0.10528455, -0.9855515, 0.13267802],
            [0.61262065, -0.16937618, -0.77201533],
            [0.78333336, 0.0, 0.6216019],
        ]
    )
    T_camera_grasp[:3, 3] = [-0.22220398, 0.35045108, 0.893]

    plan = build_world_tcp_pick_poses(
        T_world_camera,
        T_camera_grasp,
        approach_distance_m=0.05,
    )

    np.testing.assert_allclose(
        plan["grasp_pose_xyz_rotvec"],
        [
            0.65660905,
            -0.65487452,
            -0.01068828,
            1.31268428,
            0.84693738,
            1.12498065,
        ],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        plan["approach_pose_xyz_rotvec"],
        [
            0.60914231,
            -0.64021270,
            -0.01633978,
            1.31268428,
            0.84693738,
            1.12498065,
        ],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        plan["T_world_tcp_grasp"][:3, :3].T
        @ plan["T_world_tcp_grasp"][:3, :3],
        np.eye(3),
        atol=1e-7,
    )
    approach_to_grasp = (
        plan["T_world_tcp_grasp"][:3, 3]
        - plan["T_world_tcp_approach"][:3, 3]
    )
    np.testing.assert_allclose(
        approach_to_grasp,
        0.05 * plan["T_world_tcp_grasp"][:3, 2],
        atol=1e-7,
    )


def test_remote_x5_backend_dry_run_returns_home_approach_grasp_plan():
    backend = RemoteX5ActionBackend(
        ROBOT_CONFIG,
        CAMERA_CONFIG,
        execute=False,
    )

    backend.initialize()
    result = backend.pick("number-block", grasp_candidates=[_known_grasp()])
    backend.shutdown()

    assert result.success is True
    assert result.metadata["execute"] is False
    assert result.metadata["execute_until"] == "grasp"
    assert result.metadata["approach_pose_xyz_rotvec"].shape == (6,)
    assert result.metadata["grasp_pose_xyz_rotvec"].shape == (6,)


def test_remote_x5_backend_executes_home_approach_grasp_in_order():
    client = FakeX5HTTPClient()
    backend = RemoteX5ActionBackend(
        ROBOT_CONFIG,
        CAMERA_CONFIG,
        execute=True,
        execute_until="grasp",
        client=client,
    )

    backend.initialize()
    result = backend.pick("number-block", grasp_candidates=[_known_grasp()])
    backend.shutdown()

    assert result.success is True
    names = [call[0] for call in client.calls]
    assert names == [
        "health",
        "get_state",
        "move_joints",
        "movej_point",
        "movel_point",
    ]
    movej_pose = next(call[2] for call in client.calls if call[0] == "movej_point")
    movel_pose = next(call[2] for call in client.calls if call[0] == "movel_point")
    np.testing.assert_allclose(
        movej_pose,
        result.metadata["approach_pose_xyz_rotvec"],
    )
    np.testing.assert_allclose(
        movel_pose,
        result.metadata["grasp_pose_xyz_rotvec"],
    )


def test_remote_x5_backend_splits_home_into_server_safe_joint_steps():
    client = FakeX5HTTPClient()
    client.joints_rad[0] = np.radians(-12.0)
    backend = RemoteX5ActionBackend(
        ROBOT_CONFIG,
        CAMERA_CONFIG,
        execute=True,
        execute_until="home",
        client=client,
    )

    backend.initialize()
    result = backend.pick("number-block", grasp_candidates=[_known_grasp()])

    assert result.success is True
    home_calls = [call for call in client.calls if call[0] == "move_joints"]
    assert len(home_calls) == 3
    home_j1_deg = [np.degrees(call[2][0]) for call in home_calls]
    np.testing.assert_allclose(home_j1_deg, [-16.0, -20.0, -24.0])
    assert "movej_point" not in [call[0] for call in client.calls]


def test_remote_x5_backend_can_stop_validation_after_approach():
    client = FakeX5HTTPClient()
    backend = RemoteX5ActionBackend(
        ROBOT_CONFIG,
        CAMERA_CONFIG,
        execute=True,
        execute_until="approach",
        client=client,
    )

    backend.initialize()
    result = backend.pick("number-block", grasp_candidates=[_known_grasp()])

    assert result.success is True
    assert result.metadata["execute_until"] == "approach"
    assert "movej_point" in [call[0] for call in client.calls]
    assert "movel_point" not in [call[0] for call in client.calls]


def test_remote_x5_backend_stops_robot_when_grasp_motion_fails():
    client = FakeX5HTTPClient(fail_step="grasp")
    backend = RemoteX5ActionBackend(
        ROBOT_CONFIG,
        CAMERA_CONFIG,
        execute=True,
        client=client,
    )

    backend.initialize()
    result = backend.pick("number-block", grasp_candidates=[_known_grasp()])

    assert result.success is False
    assert result.metadata["failed_step"] == "grasp"
    assert [call[0] for call in client.calls][-1] == "stop"


def test_remote_x5_backend_dry_run_builds_place_from_home_orientation():
    backend = RemoteX5ActionBackend(
        ROBOT_CONFIG,
        CAMERA_CONFIG,
        execute=False,
    )

    backend.initialize()
    pick_result = backend.pick("number-block", grasp_candidates=[_known_grasp()])
    result = backend.place_on_surface(
        "number-block",
        "target-surface",
        target_pose=[0.4, -0.2, 0.3, 9.0, 8.0, 7.0],
    )
    backend.shutdown()

    assert pick_result.success is True
    assert result.success is True
    np.testing.assert_allclose(
        result.metadata["preplace_pose_xyz_rotvec"],
        [0.35, -0.2, 0.3, 1.2172784, 1.2123690, 1.2159012],
    )
    np.testing.assert_allclose(
        result.metadata["place_pose_xyz_rotvec"],
        [0.4, -0.2, 0.3, 1.2172784, 1.2123690, 1.2159012],
    )
    assert result.metadata["target_orientation_ignored"] is True


def test_remote_x5_backend_executes_retreat_home_preplace_place_in_order():
    client = FakeX5HTTPClient()
    backend = RemoteX5ActionBackend(
        ROBOT_CONFIG,
        CAMERA_CONFIG,
        execute=True,
        execute_until="grasp",
        place_execute_until="place",
        client=client,
    )

    backend.initialize()
    pick_result = backend.pick("number-block", grasp_candidates=[_known_grasp()])
    result = backend.place_on_surface(
        "number-block",
        "target-surface",
        target_pose=[0.4, -0.2, 0.3, 9.0, 8.0, 7.0],
    )

    assert pick_result.success is True
    assert result.success is True
    place_calls = client.calls[5:]
    assert [call[0] for call in place_calls] == [
        "movel_point",
        "get_state",
        "move_joints",
        "movej_point",
        "movel_point",
    ]
    np.testing.assert_allclose(
        place_calls[0][2],
        pick_result.metadata["approach_pose_xyz_rotvec"],
    )
    np.testing.assert_allclose(
        place_calls[-2][2],
        [0.35, -0.2, 0.3, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        place_calls[-1][2],
        [0.4, -0.2, 0.3, 0.0, 0.0, 0.0],
    )


def test_remote_x5_backend_rejects_place_without_preceding_pick():
    backend = RemoteX5ActionBackend(
        ROBOT_CONFIG,
        CAMERA_CONFIG,
        execute=False,
    )

    result = backend.place_in_container(
        "number-block",
        "container",
        target_pose=[0.4, -0.2, 0.3],
    )

    assert result.success is False
    assert "preceding pick plan" in result.error
