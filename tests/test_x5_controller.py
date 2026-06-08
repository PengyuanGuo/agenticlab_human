import math
from types import SimpleNamespace

import pytest

from agenticlab_human.execution.robot.x5.contracts import MoveJointsCommand, StopCommand
from agenticlab_human.execution.robot.x5.x5_controller import RealX5Controller


class FakeJoint:
    def __init__(self, *values):
        names = ("j1", "j2", "j3", "j4", "j5", "j6", "e1", "e2", "e3")
        for name, value in zip(names, values):
            setattr(self, name, float(value))
        for name in names:
            if not hasattr(self, name):
                setattr(self, name, 0.0)


class FakeMovPointAdd:
    def __init__(self, vel=0, acc=0, **kwargs):
        self.vel = vel
        self.acc = acc
        for name, value in kwargs.items():
            setattr(self, name, value)


class FakeX5API:
    Joint = FakeJoint
    MovPointAdd = FakeMovPointAdd

    def __init__(self):
        self.calls = []
        self.current_joint = FakeJoint(-24, 10, -53, 102, 101, 80, -18, 0, 28)

    def enable_debug_output(self, enabled):
        self.calls.append(("enable_debug_output", enabled))

    def connect(self, ip):
        self.calls.append(("connect", ip))
        return 0

    def disconnect(self, handle):
        self.calls.append(("disconnect", handle))

    def get_version(self, handle):
        self.calls.append(("get_version", handle))
        return "fake-x5"

    def set_system_mode(self, handle, mode):
        self.calls.append(("set_system_mode", handle, mode))

    def enable_servo(self, handle, enabled):
        self.calls.append(("enable_servo", handle, enabled))

    def get_cjoint(self, handle):
        self.calls.append(("get_cjoint", handle))
        return self.current_joint

    def get_cpoint(self, handle):
        self.calls.append(("get_cpoint", handle))
        return SimpleNamespace(
            pose=SimpleNamespace(x=100.0, y=200.0, z=300.0, a=0.0, b=0.0, c=90.0)
        )

    def get_system_state(self, handle):
        self.calls.append(("get_system_state", handle))
        return SimpleNamespace(moving=False)

    def movj(self, handle, target, add_data=None):
        self.calls.append(("movj", handle, target, add_data))
        self.current_joint = target

    def wait_cmd_send_done(self, handle):
        self.calls.append(("wait_cmd_send_done", handle))

    def wait_move_done(self, handle, timeout_ms=None):
        self.calls.append(("wait_move_done", handle, timeout_ms))

    def stop(self, handle):
        self.calls.append(("stop", handle))

    def abort(self, handle):
        self.calls.append(("abort", handle))


def _build_controller(fake_x5):
    controller = RealX5Controller(
        {
            "left": {
                "robot_ip": "192.168.1.7",
                "home_joints_deg": [-24, 10, -53, 102, 101, 80, -18],
                "head_joints_deg": [0, 28],
            }
        },
        x5_api=fake_x5,
        default_speed_ratio=0.05,
        max_command_speed_ratio=0.1,
        max_joint_delta_deg=5.0,
    )
    controller.initialize()
    return controller


def test_real_x5_controller_reads_state_in_public_units():
    fake_x5 = FakeX5API()
    controller = _build_controller(fake_x5)

    state = controller.get_state()

    assert state.arms["left"].connected is True
    assert state.arms["left"].moving is False
    assert state.arms["left"].joints_rad == pytest.approx(
        [math.radians(value) for value in [-24, 10, -53, 102, 101, 80, -18]]
    )
    assert state.arms["left"].tcp_pose_xyzw[:3] == pytest.approx([0.1, 0.2, 0.3])
    assert state.arms["left"].tcp_pose_xyzw[6] == pytest.approx(math.sqrt(0.5))
    assert ("set_system_mode", 0, 100) in fake_x5.calls
    assert ("enable_servo", 0, True) in fake_x5.calls


def test_real_x5_controller_stop_uses_xapi_stop_abort_sequence():
    fake_x5 = FakeX5API()
    controller = _build_controller(fake_x5)

    controller.execute(StopCommand(arm="left"))

    sequence = [call[0] for call in fake_x5.calls]
    stop_index = sequence.index("stop")
    assert sequence[stop_index : stop_index + 5] == [
        "stop",
        "wait_cmd_send_done",
        "abort",
        "wait_cmd_send_done",
        "wait_move_done",
    ]


def test_real_x5_controller_moves_small_joint_delta_with_movpointadd():
    fake_x5 = FakeX5API()
    controller = _build_controller(fake_x5)
    target_deg = [-23.0, 10.0, -53.0, 102.0, 101.0, 80.0, -18.0]

    controller.execute(
        MoveJointsCommand(
            arm="left",
            joints_rad=[math.radians(value) for value in target_deg],
            speed_ratio=0.05,
            wait=True,
        )
    )

    movj_call = next(call for call in fake_x5.calls if call[0] == "movj")
    _, handle, target_joint, add_data = movj_call
    assert handle == 0
    assert target_joint.j1 == pytest.approx(-23.0)
    assert target_joint.e1 == pytest.approx(-18.0)
    assert target_joint.e2 == pytest.approx(0.0)
    assert target_joint.e3 == pytest.approx(28.0)
    assert add_data.vel == 5
    assert add_data.acc == 5
    assert ("wait_move_done", 0, 60000) in fake_x5.calls


def test_real_x5_controller_rejects_large_joint_delta_before_movj():
    fake_x5 = FakeX5API()
    controller = _build_controller(fake_x5)
    target_deg = [-10.0, 10.0, -53.0, 102.0, 101.0, 80.0, -18.0]

    with pytest.raises(ValueError, match="max delta"):
        controller.execute(
            MoveJointsCommand(
                arm="left",
                joints_rad=[math.radians(value) for value in target_deg],
                speed_ratio=0.05,
            )
        )

    assert "movj" not in [call[0] for call in fake_x5.calls]
