import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from agenticlab_human.execution.robot.x5.x5_remote_backend import (
    RemoteX5ActionBackend,
    T_GRASP_EE,
    T_GRASP_TCP,
    build_world_tcp_pick_poses,
    build_world_tcp_place_poses,
)
from agenticlab_human.perception.backend.grasp_backend import GraspCandidate


ROBOT_CONFIG = "configs/robot/x5_config.yaml"
CAMERA_CONFIG = "configs/perception/camera_config.yaml"
_ROBOT_DOCUMENT = yaml.safe_load(Path(ROBOT_CONFIG).read_text())
HOME_JOINTS_DEG = _ROBOT_DOCUMENT["robot"]["left"]["home_joints_deg"]
CHECK_GRIPPER_JOINTS_DEG = _ROBOT_DOCUMENT["robot"]["left"]["check_gripper_joints_deg"]


class FakeX5HTTPClient:
    def __init__(self, *, fail_step=None):
        self.fail_step = fail_step
        self.calls = []
        self.joints_rad = np.radians(HOME_JOINTS_DEG).tolist()
        self.tcp_pose_xyzw = [0.2965, -0.4514, 0.3033, 0.0, 0.0, 0.0, 1.0]
        self.open_count = 0
        self.response_count = 0

    def health(self):
        self.calls.append(("health",))
        return SimpleNamespace(robot=SimpleNamespace(ready=True, detail="ready"))

    def get_state(self, arm):
        self.calls.append(("get_state", arm))
        return self._response()

    def move_joints(self, arm, joints_rad, *, speed_ratio, wait):
        self.calls.append(
            ("move_joints", arm, list(joints_rad), speed_ratio, wait)
        )
        self.joints_rad = list(joints_rad)
        return self._response()

    def movej_point(self, arm, pose6d, *, speed_ratio, wait):
        self.calls.append(
            ("movej_point", arm, list(pose6d), speed_ratio, wait)
        )
        return self._response()

    def movel_point(self, arm, pose6d, *, speed_ratio, wait):
        self.calls.append(
            ("movel_point", arm, list(pose6d), speed_ratio, wait)
        )
        if self.fail_step == "movel":
            return self._response(success=False, error="movel rejected")
        return self._response()

    def close_gripper(self, *, wait, arm="left"):
        self.calls.append(("close_gripper", arm, wait))
        if self.fail_step == "close_gripper":
            return self._response(success=False, error="close rejected")
        return self._response()

    def open_gripper(self, *, wait, arm="left"):
        self.open_count += 1
        self.calls.append(("open_gripper", arm, wait))
        if self.fail_step == "initialize_open" and self.open_count == 1:
            return self._response(success=False, error="initial open rejected")
        if self.fail_step == "place_open" and self.open_count == 2:
            return self._response(success=False, error="place open rejected")
        return self._response()

    def stop(self, arm):
        self.calls.append(("stop", arm))
        return self._response()

    def close(self):
        self.calls.append(("close",))

    def _response(self, *, success=True, error=None):
        self.response_count += 1
        arm_state = SimpleNamespace(
            joints_rad=self.joints_rad.copy(),
            tcp_pose_xyzw=self.tcp_pose_xyzw.copy(),
        )
        return SimpleNamespace(
            success=success,
            error=error,
            request_id=f"fake-{self.response_count}",
            duration_ms=1.0,
            state_after=SimpleNamespace(
                arms={
                    "left": arm_state,
                    "right": arm_state,
                },
            ),
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


def _backend(client):
    return RemoteX5ActionBackend(
        ROBOT_CONFIG,
        CAMERA_CONFIG,
        client=client,
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


def test_build_world_tcp_pick_poses_applies_approach_and_tcp_transform():
    T_world_camera = np.eye(4)
    T_world_camera[:3, 3] = [0.5, 0.1, 0.2]
    T_camera_grasp = np.eye(4)
    T_camera_grasp[:3, 3] = [0.2, 0.0, 0.3]

    plan = build_world_tcp_pick_poses(
        T_world_camera,
        T_camera_grasp,
        approach_distance_m=0.1,
    )

    np.testing.assert_allclose(plan["T_world_tcp_grasp"][:3, 3], [0.7, 0.1, 0.5])
    np.testing.assert_allclose(
        plan["T_world_tcp_approach"][:3, 3],
        [0.6, 0.1, 0.5],
    )
    np.testing.assert_allclose(
        plan["T_world_tcp_grasp"][:3, :3],
        T_GRASP_TCP[:3, :3],
    )


def test_build_world_tcp_place_poses_offsets_only_world_x():
    plan = build_world_tcp_place_poses(
        [0.4, -0.2, 0.3],
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


def test_initialize_opens_gripper():
    client = FakeX5HTTPClient()
    backend = _backend(client)

    backend.initialize()

    assert [call[0] for call in client.calls] == ["health", "open_gripper"]
    assert client.calls[-1] == ("open_gripper", "left", True)


def test_right_backend_routes_gripper_commands_to_right_arm():
    client = FakeX5HTTPClient()
    backend = RemoteX5ActionBackend(
        ROBOT_CONFIG,
        CAMERA_CONFIG,
        arm="right",
        client=client,
    )

    backend.initialize()
    result = backend.pick("number-block", grasp_candidates=[_known_grasp()])

    assert result.success is True
    gripper_calls = [call for call in client.calls if call[0].endswith("_gripper")]
    assert gripper_calls == [
        ("open_gripper", "right", True),
        ("close_gripper", "right", True),
    ]


def test_initialize_fails_when_gripper_cannot_open():
    client = FakeX5HTTPClient(fail_step="initialize_open")
    backend = _backend(client)

    with pytest.raises(RuntimeError, match="open gripper during initialize"):
        backend.initialize()


def test_pick_executes_home_approach_grasp_close_retreat_check_in_order():
    client = FakeX5HTTPClient()
    backend = _backend(client)
    backend.initialize()

    result = backend.pick("number-block", grasp_candidates=[_known_grasp()])

    assert result.success is True
    steps = [step["step"] for step in result.metadata["completed_steps"]]
    assert steps[:5] == [
        "home",
        "approach",
        "grasp",
        "close_gripper",
        "retreat",
    ]
    assert steps[-1] == "check_gripper"
    motion_names = [call[0] for call in client.calls[2:]]
    assert motion_names[:6] == [
        "get_state",
        "move_joints",
        "movej_point",
        "movel_point",
        "close_gripper",
        "movel_point",
    ]
    check_calls = [
        call
        for call in client.calls
        if call[0] == "move_joints"
        and np.allclose(np.degrees(call[2]), CHECK_GRIPPER_JOINTS_DEG)
    ]
    assert check_calls


def test_joint_target_is_split_into_server_safe_steps():
    client = FakeX5HTTPClient()
    client.joints_rad[0] = np.radians(120.0)
    backend = _backend(client)
    backend.initialize()

    result = backend.move_home()

    assert result.success is True
    home_calls = [call for call in client.calls if call[0] == "move_joints"]
    assert len(home_calls) == math.ceil(
        abs(HOME_JOINTS_DEG[0] - 120.0) / backend.config.home_max_step_deg
    )
    previous = 120.0
    for call in home_calls:
        current = np.degrees(call[2][0])
        assert abs(current - previous) <= backend.config.home_max_step_deg
        previous = current
    np.testing.assert_allclose(np.degrees(home_calls[-1][2]), HOME_JOINTS_DEG)


def test_pick_failure_stops_robot():
    client = FakeX5HTTPClient(fail_step="close_gripper")
    backend = _backend(client)
    backend.initialize()

    result = backend.pick("number-block", grasp_candidates=[_known_grasp()])

    assert result.success is False
    assert [call[0] for call in client.calls][-2:] == ["close_gripper", "stop"]


def test_place_executes_preplace_place_open_home():
    client = FakeX5HTTPClient()
    backend = _backend(client)
    backend.initialize()
    pick_result = backend.pick(
        "number-block",
        grasp_candidates=[_known_grasp()],
    )

    result = backend.place(
        "number-block",
        "target-surface",
        target_pose=[0.4, -0.2, 0.3],
    )

    assert pick_result.success is True
    assert result.success is True
    assert [step["step"] for step in pick_result.metadata["completed_steps"]][-1] == (
        "check_gripper"
    )
    steps = [step["step"] for step in result.metadata["completed_steps"]]
    assert steps[0] == "preplace"
    assert "retreat" not in steps
    assert "check_gripper" not in steps
    assert steps.index("preplace") < steps.index("place")
    assert steps.index("place") < steps.index("open_gripper")
    assert steps.index("open_gripper") < steps.index("home")
    assert steps[-1] == "home"
    assert "release_retreat" not in steps

    check_calls = [
        call
        for call in client.calls
        if call[0] == "move_joints"
        and np.allclose(np.degrees(call[2]), CHECK_GRIPPER_JOINTS_DEG)
    ]
    home_calls = [
        call
        for call in client.calls
        if call[0] == "move_joints"
        and np.allclose(np.degrees(call[2]), HOME_JOINTS_DEG)
    ]
    assert check_calls
    assert len(home_calls) >= 2


def test_place_open_failure_stops_before_home():
    client = FakeX5HTTPClient(fail_step="place_open")
    backend = _backend(client)
    backend.initialize()
    assert backend.pick(
        "number-block",
        grasp_candidates=[_known_grasp()],
    ).success

    result = backend.place(
        "number-block",
        "target-surface",
        target_pose=[0.4, -0.2, 0.3],
    )

    assert result.success is False
    assert [call[0] for call in client.calls][-2:] == ["open_gripper", "stop"]


def test_place_requires_successful_pick():
    client = FakeX5HTTPClient()
    backend = _backend(client)
    backend.initialize()

    result = backend.place(
        "number-block",
        "container",
        target_pose=[0.4, -0.2, 0.3],
    )

    assert result.success is False
    assert "preceding pick" in result.error
