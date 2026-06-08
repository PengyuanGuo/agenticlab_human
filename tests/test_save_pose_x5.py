import csv
import math
from types import SimpleNamespace

import pytest

from agenticlab_human.perception.calibration.save_pose_x5 import (
    CSV_HEADER,
    move_to_joint_target_deg,
    parse_joint_target_deg,
    save_pose_to_csv,
)


class FakeX5Client:
    def __init__(self):
        self.joints_rad = [0.0] * 7
        self.targets_rad = []

    def _state(self):
        arm_state = SimpleNamespace(
            joints_rad=self.joints_rad.copy(),
            tcp_pose_xyzw=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            moving=False,
        )
        return SimpleNamespace(arms={"left": arm_state})

    def get_state(self, arm):
        return SimpleNamespace(state_after=self._state())

    def move_joints(self, arm, joints_rad, *, speed_ratio, wait):
        self.joints_rad = list(joints_rad)
        self.targets_rad.append(list(joints_rad))
        return SimpleNamespace(
            success=True,
            error=None,
            state_after=self._state(),
        )


def test_save_pose_to_csv_matches_flexiv_format(tmp_path):
    output_path = tmp_path / "poses.csv"
    pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]

    save_pose_to_csv(pose, output_path)
    save_pose_to_csv(pose, output_path)

    with output_path.open(newline="") as csvfile:
        rows = list(csv.reader(csvfile))
    assert rows[0] == CSV_HEADER
    assert len(rows) == 3
    assert [float(value) for value in rows[1]] == pose


def test_parse_joint_target_deg_accepts_spaces_and_commas():
    assert parse_joint_target_deg("1, 2, 3, 4, 5, 6, 7") == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]

    with pytest.raises(ValueError, match="exactly 7"):
        parse_joint_target_deg("1 2 3")


def test_move_to_joint_target_deg_splits_large_delta_into_small_steps():
    client = FakeX5Client()

    state = move_to_joint_target_deg(
        client,
        "left",
        [9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        speed_ratio=0.05,
        max_step_deg=4.0,
    )

    assert len(client.targets_rad) == 3
    assert [math.degrees(target[0]) for target in client.targets_rad] == pytest.approx(
        [3.0, 6.0, 9.0]
    )
    assert math.degrees(state.joints_rad[0]) == pytest.approx(9.0)


def test_move_to_joint_target_deg_rejects_final_limit_violation_before_motion():
    client = FakeX5Client()

    with pytest.raises(ValueError, match="joint 1 target"):
        move_to_joint_target_deg(
            client,
            "left",
            [11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            speed_ratio=0.05,
            max_step_deg=4.0,
            joint_limits_deg=[[-10.0, 10.0]] * 7,
        )

    assert client.targets_rad == []
