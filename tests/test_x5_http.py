import numpy as np
import pytest
from fastapi.testclient import TestClient

from agenticlab_human.execution.robot.x5.camera import MockRGBDCamera, OrbbecRGBDCamera
from agenticlab_human.execution.robot.x5.client import (
    X5HTTPClient,
    save_rgbd_frame,
    tcp_pose_xyzw_to_xyz_rotvec,
)
from agenticlab_human.execution.robot.x5.contracts import CameraIntrinsics
from agenticlab_human.execution.robot.x5.server import create_app
from agenticlab_human.execution.robot.x5.x5_controller import MockX5Controller


def _build_app():
    return create_app(
        camera=MockRGBDCamera(width=64, height=48, depth_mm=900.0),
        controller=MockX5Controller(),
    )


def test_health_reports_ready_mock_components():
    with TestClient(_build_app()) as transport:
        client = X5HTTPClient("http://testserver", timeout_s=None, session=transport)

        health = client.health()

    assert health.status == "ok"
    assert health.camera.ready is True
    assert health.camera.backend == "mock"
    assert health.robot.ready is True
    assert health.robot.backend == "mock"


def test_capture_rgbd_round_trip_preserves_arrays_and_metadata():
    with TestClient(_build_app()) as transport:
        client = X5HTTPClient("http://testserver", timeout_s=None, session=transport)

        frame = client.capture_rgbd()

    assert frame.rgb.shape == (48, 64, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.depth_mm.shape == (48, 64)
    assert frame.depth_mm.dtype == np.float32
    assert frame.intrinsics.width == 64
    assert frame.intrinsics.height == 48
    assert frame.frame_id == "mock-000001"
    assert frame.timestamp_ns > 0
    assert frame.depth_unit == "mm"
    assert frame.sync_delta_ms == 0.0


def test_move_joints_returns_accepted_command_and_updated_state():
    target = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    with TestClient(_build_app()) as transport:
        client = X5HTTPClient("http://testserver", timeout_s=None, session=transport)

        result = client.move_joints(
            "left",
            target,
            speed_ratio=0.2,
            request_id="test-move-left",
        )

    assert result.success is True
    assert result.request_id == "test-move-left"
    assert result.accepted_command["type"] == "move_joints"
    assert result.state_before.arms["left"].joints_rad == [0.0] * 7
    assert result.state_after.arms["left"].joints_rad == target
    assert result.state_after.arms["right"].joints_rad == [0.0] * 7


def test_cartesian_point_commands_update_mock_tcp_pose():
    movej_target = [0.2, -0.1, 0.4, 0.0, 0.0, np.pi / 2.0]
    movel_target = [0.2, -0.1, 0.35, 0.0, 0.0, np.pi / 2.0]
    with TestClient(_build_app()) as transport:
        client = X5HTTPClient("http://testserver", timeout_s=None, session=transport)

        movej_result = client.movej_point(
            "left",
            movej_target,
            speed_ratio=0.05,
            request_id="test-movej-point",
        )
        movel_result = client.movel_point(
            "left",
            movel_target,
            speed_ratio=0.03,
            request_id="test-movel-point",
        )

    assert movej_result.success is True
    assert movej_result.accepted_command["type"] == "movej_point"
    assert movej_result.state_after.arms["left"].tcp_pose_xyzw[:3] == movej_target[:3]
    assert movej_result.state_after.arms["left"].tcp_pose_xyzw[5:] == pytest.approx(
        [np.sqrt(0.5), np.sqrt(0.5)]
    )
    assert movel_result.success is True
    assert movel_result.accepted_command["type"] == "movel_point"
    assert movel_result.state_after.arms["left"].tcp_pose_xyzw[:3] == movel_target[:3]


def test_invalid_joint_count_is_rejected_by_contract():
    with TestClient(_build_app()) as transport:
        response = transport.post(
            "/v1/robot/command",
            json={
                "request_id": "bad-joints",
                "command": {
                    "type": "move_joints",
                    "arm": "left",
                    "joints_rad": [0.0, 0.0],
                },
            },
        )

    assert response.status_code == 422


def test_invalid_cartesian_pose_is_rejected_by_contract():
    with TestClient(_build_app()) as transport:
        response = transport.post(
            "/v1/robot/command",
            json={
                "request_id": "bad-point",
                "command": {
                    "type": "movel_point",
                    "arm": "left",
                    "tcp_pose_xyz_rotvec": [0.1, 0.2, 0.3],
                },
            },
        )

    assert response.status_code == 422


def test_client_preserves_robot_command_error_from_http_400():
    class RejectingController(MockX5Controller):
        def execute(self, command):
            if command.type == "movej_point":
                raise ValueError("X5 rejected Cartesian target")
            return super().execute(command)

    app = create_app(
        camera=MockRGBDCamera(),
        controller=RejectingController(),
    )
    with TestClient(app) as transport:
        client = X5HTTPClient("http://testserver", timeout_s=None, session=transport)

        result = client.movej_point(
            "left",
            [0.2, -0.1, 0.4, 0.0, 0.0, 0.0],
        )

    assert result.success is False
    assert result.error == "X5 rejected Cartesian target"


def test_tcp_state_quaternion_converts_to_cartesian_command_rotvec():
    pose6d = tcp_pose_xyzw_to_xyz_rotvec(
        [0.1, 0.2, 0.3, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]
    )

    assert pose6d[:3] == [0.1, 0.2, 0.3]
    assert pose6d[3:] == pytest.approx([0.0, 0.0, np.pi / 2.0])


def test_orbbec_adapter_converts_camera_capture_to_wire_frame():
    class FakeIntrinsics:
        fx = 600.0
        fy = 601.0
        cx = 320.0
        cy = 240.0
        width = 640
        height = 480

    class FakeCapture:
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_mm = np.full((480, 640), 750.0, dtype=np.float32)
        intrinsics = FakeIntrinsics()
        color_timestamp_ns = 1_000_000_000
        depth_timestamp_ns = 1_001_000_000
        timestamp_ns = depth_timestamp_ns
        frame_index = 42

    class FakeCamera:
        def __init__(self, *args, **kwargs):
            self.destroyed = False

        def capture_with_metadata(self):
            return FakeCapture()

        def destroy(self):
            self.destroyed = True

    camera = OrbbecRGBDCamera(camera_factory=FakeCamera)
    camera.initialize()

    frame = camera.capture()

    assert camera.health().ready is True
    assert frame.frame_id == "orbbec-0000000042"
    assert frame.intrinsics == CameraIntrinsics(
        fx=600.0,
        fy=601.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )
    assert frame.sync_delta_ms == 1.0
    camera.shutdown()


def test_save_rgbd_frame_writes_lossless_depth_and_metadata(tmp_path):
    camera = MockRGBDCamera(width=8, height=6)
    camera.initialize()
    frame = camera.capture()

    paths = save_rgbd_frame(frame, tmp_path)

    assert all(path.exists() for path in paths.values())
    saved_depth = np.load(paths["depth_npy"])
    np.testing.assert_array_equal(saved_depth, frame.depth_mm)
