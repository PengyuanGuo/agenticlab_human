"""Right-arm X5 pour sequence using joint targets only."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from agenticlab_human.execution.action_backend import ActionResult, ExecutionReport
from agenticlab_human.execution.robot.x5.client import X5HTTPClient
from agenticlab_human.execution.robot.x5.conversion import degrees_to_radians


DEFAULT_ROBOT_CONFIG = "configs/robot/x5_config.yaml"
DEFAULT_ARM = "right"


@dataclass(frozen=True)
class PourConfig:
    server_url: str
    arm: str
    speed_ratio: float
    max_joint_delta_deg: float
    home_joints_deg: list[float]
    pre_grasp_joints_deg: list[float]
    grasp_joints_deg: list[float]
    pour_joints_deg: list[float]
    home_torso_deg: list[float]
    pre_grasp_torso_deg: list[float]
    pour_torso_deg: list[float]


@dataclass(frozen=True)
class PourStep:
    name: str
    joints_deg: list[float] | None = None
    torso_joints_deg: list[float] | None = None
    gripper: str | None = None


def execute_pour(
    *,
    config_path: str = DEFAULT_ROBOT_CONFIG,
    arm: str = DEFAULT_ARM,
    server_url: str | None = None,
    speed_ratio: float | None = None,
    control_gripper: bool = False,
) -> ExecutionReport:
    """Execute the fixed right-arm pour sequence."""

    config = load_pour_config(
        config_path=config_path,
        arm=arm,
        server_url=server_url,
        speed_ratio=speed_ratio,
    )
    steps = build_pour_steps(config)
    results: list[ActionResult] = []
    failed_action_id: int | None = None
    error: str | None = None

    with X5HTTPClient(config.server_url, timeout_s=120.0) as client:
        try:
            health = client.health()
            if not health.robot.ready:
                raise RuntimeError(f"X5 robot is not ready: {health.robot.detail}")

            current_torso_deg = config.home_torso_deg
            for index, step in enumerate(steps, start=1):
                if step.gripper:
                    result = _execute_gripper_step(
                        client,
                        step,
                        index=index,
                        arm=config.arm,
                        control_gripper=control_gripper,
                    )
                else:
                    completed = move_joint_target(
                        client,
                        arm=config.arm,
                        target_joints_deg=step.joints_deg or [],
                        target_torso_deg=step.torso_joints_deg
                        or config.home_torso_deg,
                        current_torso_deg=current_torso_deg,
                        speed_ratio=config.speed_ratio,
                        max_joint_delta_deg=config.max_joint_delta_deg,
                        step_name=step.name,
                    )
                    current_torso_deg = step.torso_joints_deg or config.home_torso_deg
                    result = ActionResult(
                        success=True,
                        action_name="move_joints",
                        message=f"Completed pour step {index}: {step.name}.",
                        metadata={
                            "step_index": index,
                            "step": step.name,
                            "arm": config.arm,
                            "joints_deg": step.joints_deg,
                            "torso_joints_deg": current_torso_deg,
                            "completed_steps": completed,
                        },
                    )
                results.append(result)
                if not result.success:
                    failed_action_id = index
                    error = result.error
                    break
        except Exception as exc:
            failed_action_id = failed_action_id or len(results) + 1
            error = str(exc)
            results.append(
                ActionResult(
                    success=False,
                    action_name="pour",
                    error=error,
                    metadata={
                        "step_index": failed_action_id,
                        "arm": config.arm,
                    },
                )
            )
            try:
                client.stop(config.arm)
            except Exception:
                pass

    success = failed_action_id is None and all(result.success for result in results)
    return ExecutionReport(
        success=success,
        task=f"{config.arm}-arm-pour",
        total_actions=len(steps),
        results=results,
        failed_action_id=None if success else failed_action_id,
        failed_action_name=None if success else results[-1].action_name,
        error=None if success else error,
        metadata={
            "config_path": config_path,
            "server_url": config.server_url,
            "arm": config.arm,
            "control_gripper": control_gripper,
        },
    )


def load_pour_config(
    *,
    config_path: str = DEFAULT_ROBOT_CONFIG,
    arm: str = DEFAULT_ARM,
    server_url: str | None = None,
    speed_ratio: float | None = None,
) -> PourConfig:
    document = yaml.safe_load(Path(config_path).read_text()) or {}
    robot = document["robot"]
    action_backend = document.get("action_backend", {})
    arm_config = robot[arm]
    return PourConfig(
        server_url=str(
            server_url
            or action_backend.get("server_url")
            or "http://127.0.0.1:8000"
        ).rstrip("/"),
        arm=arm,
        speed_ratio=float(
            speed_ratio
            if speed_ratio is not None
            else action_backend.get("home_speed_ratio", 0.1)
        ),
        max_joint_delta_deg=float(robot.get("max_joint_delta_deg", 45.0)),
        home_joints_deg=_required_vector(
            arm_config,
            "home_joints_deg",
            7,
            f"robot.{arm}.home_joints_deg",
        ),
        pre_grasp_joints_deg=_required_vector(
            arm_config,
            "pre_grasp_joints_deg",
            7,
            f"robot.{arm}.pre_grasp_joints_deg",
        ),
        grasp_joints_deg=_required_vector(
            arm_config,
            "grasp_joints_deg",
            7,
            f"robot.{arm}.grasp_joints_deg",
        ),
        pour_joints_deg=_required_vector(
            arm_config,
            "pour_joints_deg",
            7,
            f"robot.{arm}.pour_joints_deg",
        ),
        home_torso_deg=_required_vector(
            arm_config,
            "home_torso_deg",
            (1, 2),
            f"robot.{arm}.home_torso_deg",
        ),
        pre_grasp_torso_deg=_required_vector(
            arm_config,
            "pre_grasp_torso_deg",
            (1, 2),
            f"robot.{arm}.pre_grasp_torso_deg",
        ),
        pour_torso_deg=_required_vector(
            arm_config,
            "pour_torso_deg",
            (1, 2),
            f"robot.{arm}.pour_torso_deg",
        ),
    )


def build_pour_steps(config: PourConfig) -> list[PourStep]:
    home_torso = config.home_torso_deg
    pre_grasp_torso = config.pre_grasp_torso_deg
    return [
        PourStep("home", config.home_joints_deg, home_torso),
        PourStep("pre_grasp", config.pre_grasp_joints_deg, pre_grasp_torso),
        PourStep("grasp", config.grasp_joints_deg, pre_grasp_torso),
        PourStep("close_gripper", gripper="close"),
        PourStep("pre_grasp", config.pre_grasp_joints_deg, pre_grasp_torso),
        PourStep("pour", config.pour_joints_deg, config.pour_torso_deg),
        PourStep("pre_grasp_home_torso", config.pre_grasp_joints_deg, pre_grasp_torso),
        PourStep("grasp_release", config.grasp_joints_deg, pre_grasp_torso),
        PourStep("open_gripper", gripper="open"),
        PourStep("pre_grasp", config.pre_grasp_joints_deg, pre_grasp_torso),
        PourStep("home", config.home_joints_deg, home_torso),
    ]


def move_joint_target(
    client: X5HTTPClient,
    *,
    arm: str,
    target_joints_deg: Sequence[float],
    target_torso_deg: Sequence[float],
    current_torso_deg: Sequence[float],
    speed_ratio: float,
    max_joint_delta_deg: float,
    step_name: str,
) -> list[dict[str, Any]]:
    target = _finite_vector(target_joints_deg, 7, "target_joints_deg")
    target_torso = _finite_vector(target_torso_deg, (1, 2), "target_torso_deg")
    current_torso = _finite_vector(current_torso_deg, (1, 2), "current_torso_deg")
    if target_torso.shape != current_torso.shape:
        raise ValueError("target_torso_deg and current_torso_deg must have the same length")

    state = client.get_state(arm)
    if not state.success:
        raise RuntimeError(state.error or f"get state failed for {arm}")
    current = np.degrees(state.state_after.arms[arm].joints_rad)
    max_delta = max(
        abs(target_value - current_value)
        for target_value, current_value in zip(target, current, strict=True)
    )
    step_count = max(1, math.ceil(max_delta / float(max_joint_delta_deg)))

    completed = []
    for index in range(1, step_count + 1):
        fraction = index / step_count
        waypoint = current + (target - current) * fraction
        torso_waypoint = current_torso + (target_torso - current_torso) * fraction
        response = client.move_joints(
            arm,
            degrees_to_radians(waypoint.tolist()),
            torso_joints_deg=torso_waypoint.tolist(),
            speed_ratio=float(speed_ratio),
            wait=True,
        )
        if not response.success:
            raise RuntimeError(response.error or f"{step_name} waypoint failed")
        completed.append(
            {
                "step": step_name,
                "request_id": response.request_id,
                "duration_ms": response.duration_ms,
                "waypoint_index": index,
                "waypoint_count": step_count,
                "joints_deg": waypoint.tolist(),
                "torso_joints_deg": torso_waypoint.tolist(),
            }
        )
    return completed


def _execute_gripper_step(
    client: X5HTTPClient,
    step: PourStep,
    *,
    index: int,
    arm: str,
    control_gripper: bool,
) -> ActionResult:
    if not control_gripper:
        return ActionResult(
            success=True,
            action_name=step.name,
            message=f"Skipped {step.name}; control gripper manually.",
            metadata={
                "step_index": index,
                "step": step.name,
                "arm": arm,
                "skipped": True,
                "manual_gripper": True,
            },
        )
    if step.gripper == "close":
        response = client.close_gripper(arm=arm, wait=True)
    elif step.gripper == "open":
        response = client.open_gripper(arm=arm, wait=True)
    else:
        return ActionResult(
            success=False,
            action_name=step.name,
            error=f"Unsupported gripper step: {step.gripper}",
            metadata={"step_index": index, "step": step.name},
        )
    return ActionResult(
        success=response.success,
        action_name=step.name,
        error=response.error,
        metadata={
            "step_index": index,
            "step": step.name,
            "arm": arm,
            "request_id": response.request_id,
            "duration_ms": response.duration_ms,
        },
    )


def _required_vector(
    mapping: dict[str, Any],
    key: str,
    size: int | tuple[int, ...],
    name: str,
) -> list[float]:
    if key not in mapping:
        raise ValueError(f"{name} is required")
    return _finite_vector(mapping[key], size, name).tolist()


def _finite_vector(
    values: Any,
    size: int | tuple[int, ...],
    name: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    allowed_sizes = (size,) if isinstance(size, int) else size
    if vector.ndim != 1 or vector.size not in allowed_sizes or not np.all(np.isfinite(vector)):
        expected = (
            str(size)
            if isinstance(size, int)
            else " or ".join(str(item) for item in size)
        )
        raise ValueError(f"{name} must contain {expected} finite values")
    return vector


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the fixed right-arm X5 pour sequence.",
    )
    parser.add_argument("--config", default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--server-url")
    parser.add_argument("--speed-ratio", type=float)
    parser.add_argument(
        "--control-gripper",
        action="store_true",
        help="Use the arm-scoped HTTP gripper service. Default is manual.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        required=True,
        help="Required confirmation that this command sends robot motion.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = execute_pour(
        config_path=args.config,
        arm=args.arm,
        server_url=args.server_url,
        speed_ratio=args.speed_ratio,
        control_gripper=args.control_gripper,
    )
    print(json.dumps(report.to_dict(), indent=2, default=_json_default))
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
