import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agenticlab_human.execution.robot.x5.camera import OrbbecRGBDCamera
from agenticlab_human.execution.robot.x5.client import (
    X5HTTPClient,
    save_rgbd_frame,
    tcp_pose_xyzw_to_xyz_rotvec,
)
from agenticlab_human.execution.robot.x5.contracts import (
    CameraIntrinsics,
    ComponentHealth,
    RGBDFrame,
)
from agenticlab_human.execution.robot.x5.gripper_controller import (
    MockGripperService,
    MultiGripperService,
)
from agenticlab_human.execution.robot.x5.mock_controller import MockX5Controller
from agenticlab_human.execution.robot.x5.server import create_app


class MockRGBDCamera:
    """Deterministic RGB-D source used only by HTTP transport tests."""

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        depth_mm: float = 800.0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.base_depth_mm = float(depth_mm)
        self._initialized = False
        self._frame_index = 0
        self._lock = threading.Lock()

    def initialize(self) -> None:
        with self._lock:
            self._initialized = True

    def capture(self) -> RGBDFrame:
        with self._lock:
            if not self._initialized:
                raise RuntimeError("mock camera is not initialized")
            self._frame_index += 1
            frame_index = self._frame_index

        x = np.linspace(0, 255, self.width, dtype=np.uint8)
        y = np.linspace(0, 255, self.height, dtype=np.uint8)
        x_grid = np.broadcast_to(x, (self.height, self.width))
        y_grid = np.broadcast_to(y[:, None], (self.height, self.width))
        rgb = np.stack(
            [x_grid, y_grid, np.full_like(x_grid, frame_index % 256)],
            axis=-1,
        )
        depth = np.broadcast_to(
            np.linspace(0.0, 100.0, self.width, dtype=np.float32),
            (self.height, self.width),
        ).copy()
        depth += np.float32(self.base_depth_mm + frame_index)

        timestamp_ns = time.time_ns()
        return RGBDFrame(
            rgb=rgb,
            depth_mm=depth,
            intrinsics=CameraIntrinsics(
                fx=float(self.width),
                fy=float(self.width),
                cx=(self.width - 1) / 2.0,
                cy=(self.height - 1) / 2.0,
                width=self.width,
                height=self.height,
            ),
            timestamp_ns=timestamp_ns,
            frame_id=f"mock-{frame_index:06d}",
            color_timestamp_ns=timestamp_ns,
            depth_timestamp_ns=timestamp_ns,
        )

    def health(self) -> ComponentHealth:
        with self._lock:
            initialized = self._initialized
        return ComponentHealth(
            ready=initialized,
            backend="mock",
            detail="ready" if initialized else "not initialized",
        )

    def shutdown(self) -> None:
        with self._lock:
            self._initialized = False


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
            torso_joints_deg=[15.0],
            speed_ratio=0.2,
            request_id="test-move-left",
        )

    assert result.success is True
    assert result.request_id == "test-move-left"
    assert result.accepted_command["type"] == "move_joints"
    assert result.accepted_command["torso_joints_deg"] == [15.0]
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


def test_check_ik_point_returns_metadata_without_motion():
    target = [0.2, -0.1, 0.4, 0.0, 0.0, np.pi / 2.0]
    with TestClient(_build_app()) as transport:
        client = X5HTTPClient("http://testserver", timeout_s=None, session=transport)

        result = client.check_ik_point(
            "left",
            target,
            inverse_type=0,
            seed_joints_rad=[0.0] * 7,
            request_id="test-check-ik",
        )

    assert result.success is True
    assert result.request_id == "test-check-ik"
    assert result.accepted_command["type"] == "check_ik_point"
    assert result.accepted_command["inverse_type"] == 0
    assert result.metadata["ik_joints_rad"] == [0.0] * 7
    assert result.metadata["ik_joints_deg"] == [0.0] * 7
    assert result.state_after.arms["left"].tcp_pose_xyzw[:3] == [0.0, 0.0, 0.0]


def test_single_gripper_command_round_trip_updates_robot_level_state():
    with TestClient(_build_app()) as transport:
        client = X5HTTPClient("http://testserver", timeout_s=None, session=transport)

        close_result = client.close_gripper(request_id="test-close-gripper")
        init_result = client.init_gripper(request_id="test-init-gripper")
        open_result = client.open_gripper(request_id="test-open-gripper")

    assert close_result.success is True
    assert close_result.accepted_command == {
        "type": "set_gripper",
        "arm": "left",
        "position": 0.0,
        "wait": True,
    }
    assert close_result.state_after.gripper.position == 0.0
    assert close_result.state_after.gripper.raw_position == 0
    assert init_result.success is True
    assert init_result.accepted_command == {
        "type": "init_gripper",
        "arm": "left",
    }
    assert init_result.state_after.gripper.position == 1.0
    assert init_result.state_after.gripper.raw_position == 1000
    assert open_result.success is True
    assert open_result.state_after.gripper.position == 1.0
    assert open_result.state_after.gripper.raw_position == 1000


def test_right_gripper_command_routes_to_right_gripper_state():
    gripper = MultiGripperService(
        {
            "left": MockGripperService(),
            "right": MockGripperService(),
        },
    )
    app = create_app(
        camera=MockRGBDCamera(width=64, height=48),
        controller=MockX5Controller(),
        gripper=gripper,
    )
    with TestClient(app) as transport:
        client = X5HTTPClient("http://testserver", timeout_s=None, session=transport)

        close_result = client.close_gripper(
            arm="right",
            request_id="test-close-right-gripper",
        )
        init_result = client.init_gripper(
            arm="right",
            request_id="test-init-right-gripper",
        )

    assert close_result.success is True
    assert close_result.accepted_command == {
        "type": "set_gripper",
        "arm": "right",
        "position": 0.0,
        "wait": True,
    }
    assert close_result.state_after.gripper.position == 1.0
    assert close_result.state_after.grippers["left"].position == 1.0
    assert close_result.state_after.grippers["right"].position == 0.0
    assert init_result.success is True
    assert init_result.accepted_command == {
        "type": "init_gripper",
        "arm": "right",
    }
    assert init_result.state_after.grippers["left"].position == 1.0
    assert init_result.state_after.grippers["right"].position == 1.0


def test_invalid_gripper_position_is_rejected_by_contract():
    with TestClient(_build_app()) as transport:
        response = transport.post(
            "/v1/robot/command",
            json={
                "request_id": "bad-gripper-position",
                "command": {
                    "type": "set_gripper",
                    "position": 1.1,
                },
            },
        )

    assert response.status_code == 422


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


def test_client_preserves_unexpected_robot_command_exception_as_json_error():
    class CrashingController(MockX5Controller):
        def execute(self, command):
            if command.type == "movej_point":
                raise AttributeError("xapi movj rejected target point")
            return super().execute(command)

    app = create_app(
        camera=MockRGBDCamera(),
        controller=CrashingController(),
    )
    with TestClient(app) as transport:
        client = X5HTTPClient("http://testserver", timeout_s=None, session=transport)

        result = client.movej_point(
            "left",
            [0.2, -0.1, 0.4, 0.0, 0.0, 0.0],
        )

    assert result.success is False
    assert result.error == "xapi movj rejected target point"
    assert result.metadata["error_type"] == "AttributeError"


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
