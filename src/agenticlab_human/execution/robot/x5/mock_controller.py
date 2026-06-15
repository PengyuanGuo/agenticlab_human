"""In-memory X5 arm controller used by HTTP and pipeline tests."""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Sequence

from agenticlab_human.execution.robot.x5.contracts import (
    ArmState,
    ComponentHealth,
    RobotCommand,
    RobotState,
)
from agenticlab_human.execution.robot.x5.conversion import rotvec_to_quat_xyzw


class MockX5Controller:
    """Immediate execution model for the configured X5 arms."""

    def __init__(
        self,
        arms: Sequence[str] = ("left", "right"),
        initial_joints_rad: Sequence[float] | None = None,
    ) -> None:
        initial = list(initial_joints_rad or [0.0] * 7)
        if len(initial) != 7:
            raise ValueError("initial_joints_rad must contain 7 values")
        if not arms:
            raise ValueError("at least one mock arm is required")

        self._initialized = False
        self._lock = threading.Lock()
        self._arms = {
            str(arm): ArmState(
                connected=False,
                moving=False,
                joints_rad=initial.copy(),
                tcp_pose_xyzw=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            )
            for arm in arms
        }

    def initialize(self) -> None:
        with self._lock:
            self._initialized = True
            for state in self._arms.values():
                state.connected = True

    def execute(self, command: RobotCommand) -> None:
        with self._lock:
            self._require_initialized()
            if command.type == "get_state":
                self._require_target(command.arm)
                return
            if command.type == "move_joints":
                state = self._require_arm(command.arm)
                state.moving = True
                state.joints_rad = [float(value) for value in command.joints_rad]
                state.moving = False
                return
            if command.type in ("movej_point", "movel_point"):
                state = self._require_arm(command.arm)
                state.moving = True
                xyz = [float(value) for value in command.tcp_pose_xyz_rotvec[:3]]
                quaternion = rotvec_to_quat_xyzw(
                    command.tcp_pose_xyz_rotvec[3:]
                )
                state.tcp_pose_xyzw = xyz + list(quaternion)
                state.moving = False
                return
            if command.type == "stop":
                for state in self._selected_states(command.arm):
                    state.moving = False
                return
            raise ValueError(f"unsupported X5 arm command: {command.type}")

    def get_state(self) -> RobotState:
        with self._lock:
            self._require_initialized()
            arms = {
                name: copy.deepcopy(state)
                for name, state in self._arms.items()
            }
        return RobotState(arms=arms, timestamp_ns=time.time_ns())

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
            for state in self._arms.values():
                state.moving = False
                state.connected = False
            self._initialized = False

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("mock X5 controller is not initialized")

    def _require_arm(self, arm: str) -> ArmState:
        try:
            return self._arms[arm]
        except KeyError as exc:
            raise ValueError(f"mock arm is not configured: {arm}") from exc

    def _require_target(self, arm: str) -> None:
        if arm != "all":
            self._require_arm(arm)

    def _selected_states(self, arm: str) -> list[ArmState]:
        if arm == "all":
            return list(self._arms.values())
        return [self._require_arm(arm)]
