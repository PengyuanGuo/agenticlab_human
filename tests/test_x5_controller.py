import math
from types import SimpleNamespace

import pytest

from agenticlab_human.execution.robot.x5.contracts import (
    MoveJPointCommand,
    MoveJointsCommand,
    MoveLPointCommand,
    SetGripperCommand,
    StopCommand,
)
from agenticlab_human.execution.robot.x5.x5_controller import (
    RealX5Controller,
    _euler_xyz_deg_to_quat_xyzw,
    _rotvec_to_euler_xyz_deg,
    _rotvec_to_quat_xyzw,
)


class FakeJoint:
    def __init__(self, *values):
        names = ("j1", "j2", "j3", "j4", "j5", "j6", "e1", "e2", "e3")
        for name, value in zip(names, values):
            setattr(self, name, float(value))
        for name in names:
            if not hasattr(self, name):
                setattr(self, name, 0.0)


class FakePose:
    def __init__(
        self,
        x=0.0,
        y=0.0,
        z=0.0,
        a=0.0,
        b=0.0,
        c=0.0,
        e1=0.0,
        e2=0.0,
        e3=0.0,
    ):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.e1 = float(e1)
        self.e2 = float(e2)
        self.e3 = float(e3)


class FakePoint:
    def __init__(self, pose, uf=0, tf=0, cfg=(0, 0, 0, 1)):
        self.pose = pose if isinstance(pose, FakePose) else FakePose(*pose)
        self.uf = int(uf)
        self.tf = int(tf)
        self.cfg = tuple(cfg)


class FakeMovPointAdd:
    def __init__(self, vel=0, acc=0, **kwargs):
        self.vel = vel
        self.acc = acc
        for name, value in kwargs.items():
            setattr(self, name, value)


class FakeX5API:
    Joint = FakeJoint
    Pose = FakePose
    Point = FakePoint
    MovPointAdd = FakeMovPointAdd

    def __init__(self):
        self.calls = []
        self.current_joint = FakeJoint(-24, 10, -53, 102, 101, 80, -18, 0, 28)
        self.current_point = FakePoint(
            (100, 200, 300, 0, 0, 90, -18, 0, 28),
            0,
            1,
            (1, 0, -1, 1),
        )
        self.active_tf_no = 0
        self.tool_frames = {1: FakePose()}

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

    def get_tf(self, handle, index):
        self.calls.append(("get_tf", handle, index))
        return self.tool_frames[index]

    def set_tf(self, handle, index, pose):
        self.calls.append(("set_tf", handle, index, pose))
        self.tool_frames[index] = pose

    def set_tfno(self, handle, index):
        self.calls.append(("set_tfno", handle, index))
        self.active_tf_no = index

    def get_tfno(self, handle):
        self.calls.append(("get_tfno", handle))
        return self.active_tf_no

    def get_wpoint(self, handle):
        self.calls.append(("get_wpoint", handle))
        return self.current_point

    def get_cpoint(self, handle):
        self.calls.append(("get_cpoint", handle))
        return SimpleNamespace(
            pose=SimpleNamespace(x=1.0, y=2.0, z=3.0, a=0.0, b=0.0, c=0.0)
        )

    def get_system_state(self, handle):
        self.calls.append(("get_system_state", handle))
        return SimpleNamespace(moving=False)

    def movj(self, handle, target, add_data=None):
        self.calls.append(("movj", handle, target, add_data))
        if isinstance(target, FakeJoint):
            self.current_joint = target
        else:
            self.current_point = target

    def movl(self, handle, target, add_data=None):
        self.calls.append(("movl", handle, target, add_data))
        self.current_point = target

    def wait_cmd_send_done(self, handle):
        self.calls.append(("wait_cmd_send_done", handle))

    def wait_move_done(self, handle, timeout_ms=None):
        self.calls.append(("wait_move_done", handle, timeout_ms))

    def stop(self, handle):
        self.calls.append(("stop", handle))

    def abort(self, handle):
        self.calls.append(("abort", handle))


class FakeGripper:
    def __init__(self):
        self.calls = []
        self.init_status = 0
        self.grip_status = 1
        self.position = 1000

    def get_init_status(self):
        self.calls.append(("get_init_status",))
        return self.init_status

    def init_gripper(self, *, timeout_s, poll_interval_s):
        self.calls.append(("init_gripper", timeout_s, poll_interval_s))
        self.init_status = 1

    def set_force(self, force):
        self.calls.append(("set_force", force))

    def set_position(self, position):
        self.calls.append(("set_position", position))
        self.position = position
        self.grip_status = 1

    def get_grip_status(self):
        self.calls.append(("get_grip_status",))
        return self.grip_status

    def get_current_position(self):
        self.calls.append(("get_current_position",))
        return self.position


def _build_controller(fake_x5, *, gripper=None):
    controller = RealX5Controller(
        {
            "left": {
                "robot_ip": "192.168.1.7",
                "tool_frame": {
                    "tf_no": 1,
                    "position_m": [0.0, 0.0, 0.16],
                    "rpy_deg": [0.0, 0.0, 0.0],
                },
                "home_joints_deg": [-24, 10, -53, 102, 101, 80, -18],
                "head_joints_deg": [0, 28],
            }
        },
        x5_api=fake_x5,
        default_speed_ratio=0.05,
        max_command_speed_ratio=0.1,
        max_joint_delta_deg=5.0,
        gripper_config={
            "enabled": gripper is not None,
            "force": 80,
            "closed_position": 0,
            "open_position": 1000,
            "init_timeout_s": 3.0,
            "move_timeout_s": 2.0,
            "poll_interval_s": 0.01,
        },
        gripper_controller=gripper,
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
    assert state.arms["left"].tool_frame_no == 1
    assert state.arms["left"].tool_frame_pose_xyzw == pytest.approx(
        [0.0, 0.0, 0.16, 0.0, 0.0, 0.0, 1.0]
    )
    assert ("set_system_mode", 0, 100) in fake_x5.calls
    assert ("enable_servo", 0, True) in fake_x5.calls
    assert ("get_wpoint", 0) in fake_x5.calls
    assert ("get_cpoint", 0) not in fake_x5.calls

    set_tf_call = next(call for call in fake_x5.calls if call[0] == "set_tf")
    _, handle, tf_no, pose = set_tf_call
    assert handle == 0
    assert tf_no == 1
    assert [pose.x, pose.y, pose.z, pose.a, pose.b, pose.c] == pytest.approx(
        [0.0, 0.0, 160.0, 0.0, 0.0, 0.0]
    )

    calls = [call[0] for call in fake_x5.calls]
    assert calls.index("set_tf") < calls.index("get_tf")
    assert calls.index("get_tf") < calls.index("set_tfno") < calls.index("get_tfno")


def test_real_x5_controller_initializes_and_moves_single_gripper():
    fake_x5 = FakeX5API()
    gripper = FakeGripper()
    controller = _build_controller(fake_x5, gripper=gripper)

    controller.execute(SetGripperCommand(position=0.0, wait=True))
    closed_state = controller.get_state().gripper
    controller.execute(SetGripperCommand(position=1.0, wait=True))
    open_state = controller.get_state().gripper

    assert ("init_gripper", 3.0, 0.01) in gripper.calls
    assert ("set_force", 80) in gripper.calls
    assert ("set_position", 0) in gripper.calls
    assert closed_state.connected is True
    assert closed_state.position == 0.0
    assert closed_state.raw_position == 0
    assert ("set_position", 1000) in gripper.calls
    assert open_state.position == 1.0
    assert open_state.raw_position == 1000
    assert "gripper=ready" in controller.health().detail


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


def test_real_x5_controller_builds_world_tcp_point_and_preserves_redundancy():
    fake_x5 = FakeX5API()
    controller = _build_controller(fake_x5)

    controller.execute(
        MoveJPointCommand(
            arm="left",
            tcp_pose_xyz_rotvec=[0.1, 0.2, 0.305, 0.0, 0.0, math.pi / 2.0],
            speed_ratio=0.05,
            wait=True,
        )
    )

    movj_call = next(
        call
        for call in fake_x5.calls
        if call[0] == "movj" and isinstance(call[2], FakePoint)
    )
    _, handle, target, add_data = movj_call
    assert handle == 0
    assert target.uf == 0
    assert target.tf == 1
    assert target.cfg == (1, 0, -1, 1)
    assert [
        target.pose.x,
        target.pose.y,
        target.pose.z,
        target.pose.a,
        target.pose.b,
        target.pose.c,
    ] == pytest.approx([100.0, 200.0, 305.0, 0.0, 0.0, 90.0])
    assert target.pose.e1 == pytest.approx(-18.0)
    assert target.pose.e2 == pytest.approx(0.0)
    assert target.pose.e3 == pytest.approx(28.0)
    assert add_data.vel == 5
    assert add_data.acc == 5


def test_real_x5_controller_executes_small_linear_point_move():
    fake_x5 = FakeX5API()
    controller = _build_controller(fake_x5)

    controller.execute(
        MoveLPointCommand(
            arm="left",
            tcp_pose_xyz_rotvec=[0.105, 0.2, 0.3, 0.0, 0.0, math.pi / 2.0],
            speed_ratio=0.03,
            wait=True,
        )
    )

    movl_call = next(call for call in fake_x5.calls if call[0] == "movl")
    _, handle, target, add_data = movl_call
    assert handle == 0
    assert target.pose.x == pytest.approx(105.0)
    assert target.pose.c == pytest.approx(90.0)
    assert target.pose.e1 == pytest.approx(-18.0)
    assert target.pose.e2 == pytest.approx(0.0)
    assert target.pose.e3 == pytest.approx(28.0)
    assert add_data.vel == 3
    assert ("wait_move_done", 0, 60000) in fake_x5.calls


def test_real_x5_controller_rejects_movel_beyond_linear_limit():
    fake_x5 = FakeX5API()
    controller = _build_controller(fake_x5)

    with pytest.raises(ValueError, match=r"movl\(Point\) translation"):
        controller.execute(
            MoveLPointCommand(
                arm="left",
                tcp_pose_xyz_rotvec=[0.25, 0.2, 0.3, 0.0, 0.0, math.pi / 2.0],
            )
        )

    assert "movl" not in [call[0] for call in fake_x5.calls]


def test_rotvec_to_x5_euler_xyz_preserves_nontrivial_orientation():
    rotvec = [0.3, -0.2, 0.4]
    euler_deg = _rotvec_to_euler_xyz_deg(rotvec)
    expected_quaternion = _rotvec_to_quat_xyzw(rotvec)
    actual_quaternion = _euler_xyz_deg_to_quat_xyzw(*euler_deg)
    dot = abs(
        sum(
            expected * actual
            for expected, actual in zip(
                expected_quaternion,
                actual_quaternion,
                strict=True,
            )
        )
    )

    assert dot == pytest.approx(1.0, abs=1e-12)


def test_known_anygrasp_world_pose_builds_expected_x5_point_without_motion():
    fake_x5 = FakeX5API()
    controller = _build_controller(fake_x5)

    point = controller._make_point(
        "left",
        0,
        [
            0.65660905,
            -0.65487452,
            -0.01068828,
            1.31268428,
            0.84693738,
            1.12498065,
        ],
    )

    assert [point.pose.x, point.pose.y, point.pose.z] == pytest.approx(
        [656.60905, -654.87452, -10.68828]
    )
    assert [point.pose.a, point.pose.b, point.pose.c] == pytest.approx(
        [83.4592067, -7.12877146, 73.6500058],
        abs=1e-6,
    )
    assert point.uf == 0
    assert point.tf == 1
    assert point.cfg == (1, 0, -1, 1)
    assert [point.pose.e1, point.pose.e2, point.pose.e3] == pytest.approx(
        [-18.0, 0.0, 28.0]
    )
    assert "movj" not in [call[0] for call in fake_x5.calls]
    assert "movl" not in [call[0] for call in fake_x5.calls]
