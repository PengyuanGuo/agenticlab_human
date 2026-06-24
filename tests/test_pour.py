from types import SimpleNamespace

import numpy as np

from agenticlab_human.execution.pour import (
    PourConfig,
    _execute_gripper_step,
    build_pour_steps,
    move_joint_target,
)


class FakePourClient:
    def __init__(self):
        self.joints_rad = np.radians([74.0, 32.0, 0.0, 99.0, -100.0, 77.0, 46.0]).tolist()
        self.calls = []
        self.response_count = 0

    def get_state(self, arm):
        self.calls.append(("get_state", arm))
        arm_state = SimpleNamespace(joints_rad=self.joints_rad.copy())
        return SimpleNamespace(
            success=True,
            error=None,
            state_after=SimpleNamespace(arms={arm: arm_state}),
        )

    def move_joints(
        self,
        arm,
        joints_rad,
        *,
        torso_joints_deg,
        speed_ratio,
        wait,
    ):
        self.response_count += 1
        self.calls.append(
            (
                "move_joints",
                arm,
                list(joints_rad),
                list(torso_joints_deg),
                speed_ratio,
                wait,
            )
        )
        self.joints_rad = list(joints_rad)
        return SimpleNamespace(
            success=True,
            error=None,
            request_id=f"pour-{self.response_count}",
            duration_ms=1.0,
        )

    def close_gripper(self, *, arm, wait):
        self.response_count += 1
        self.calls.append(("close_gripper", arm, wait))
        return SimpleNamespace(
            success=True,
            error=None,
            request_id=f"pour-{self.response_count}",
            duration_ms=1.0,
        )

    def open_gripper(self, *, arm, wait):
        self.response_count += 1
        self.calls.append(("open_gripper", arm, wait))
        return SimpleNamespace(
            success=True,
            error=None,
            request_id=f"pour-{self.response_count}",
            duration_ms=1.0,
        )


def _config():
    return PourConfig(
        server_url="http://testserver",
        arm="right",
        speed_ratio=0.5,
        max_joint_delta_deg=45.0,
        home_joints_deg=[74.0, 32.0, 0.0, 99.0, -100.0, 77.0, 46.0],
        pre_grasp_joints_deg=[75.0, 70.0, 0.0, 70.0, -99.0, 84.0, 146.0],
        grasp_joints_deg=[65.0, 74.0, 0.0, 59.0, -105.0, 76.0, 141.0],
        pour_joints_deg=[75.0, 75.0, 0.0, 67.0, -111.0, 100.0, 149.0],
        home_torso_deg=[0.0],
        pre_grasp_torso_deg=[8.0],
        pour_torso_deg=[15.0],
    )


def test_build_pour_steps_matches_fixed_right_arm_sequence():
    steps = build_pour_steps(_config())

    assert [step.name for step in steps] == [
        "home",
        "pre_grasp",
        "grasp",
        "close_gripper",
        "pre_grasp",
        "pour",
        "pre_grasp_home_torso",
        "grasp_release",
        "open_gripper",
        "pre_grasp",
        "home",
    ]
    assert len(steps) == 11
    assert steps[1].torso_joints_deg == [8.0]
    assert steps[5].torso_joints_deg == [15.0]
    assert steps[10].torso_joints_deg == [0.0]
    assert steps[8].gripper == "open"


def test_move_joint_target_splits_large_right_arm_delta_and_passes_torso():
    client = FakePourClient()
    cfg = _config()

    completed = move_joint_target(
        client,
        arm=cfg.arm,
        target_joints_deg=cfg.pre_grasp_joints_deg,
        target_torso_deg=cfg.pour_torso_deg,
        current_torso_deg=cfg.home_torso_deg,
        speed_ratio=cfg.speed_ratio,
        max_joint_delta_deg=cfg.max_joint_delta_deg,
        step_name="pre_grasp",
    )

    move_calls = [call for call in client.calls if call[0] == "move_joints"]
    assert len(move_calls) == 3
    assert all(call[1] == "right" for call in move_calls)
    assert move_calls[-1][3] == [15.0]
    np.testing.assert_allclose(
        np.degrees(move_calls[-1][2]),
        cfg.pre_grasp_joints_deg,
    )
    assert [item["waypoint_index"] for item in completed] == [1, 2, 3]


def test_gripper_step_routes_to_configured_right_arm():
    client = FakePourClient()
    step = build_pour_steps(_config())[3]

    result = _execute_gripper_step(
        client,
        step,
        index=4,
        arm="right",
        control_gripper=True,
    )

    assert result.success is True
    assert client.calls[-1] == ("close_gripper", "right", True)
    assert result.metadata["arm"] == "right"
