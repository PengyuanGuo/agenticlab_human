"""Looped X5 action-sequence executor with pick/place/pour dispatch."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agenticlab_human.core.action_sequence import Action, ActionSequence
from agenticlab_human.execution.action_backend import ActionResult, ExecutionReport
from agenticlab_human.execution.pipeline import (
    DEFAULT_PIPELINE_CONFIG,
    ExecutionRuntime,
    create_x5_execution_runtime,
    execute_action_sequence,
)
from agenticlab_human.execution.pour import (
    DEFAULT_ARM as DEFAULT_POUR_ARM,
    DEFAULT_ROBOT_CONFIG,
    execute_pour,
)


@dataclass(frozen=True)
class PourExecutionDefaults:
    config_path: str
    server_url: str | None
    arm: str
    speed_ratio: float | None
    control_gripper: bool


PickPlaceExecutor = Callable[[ExecutionRuntime, ActionSequence], ExecutionReport]
PourExecutor = Callable[..., ExecutionReport]
RuntimeFactory = Callable[..., ExecutionRuntime]


def execute_action_sequence_loop(
    runtime: ExecutionRuntime,
    sequence: ActionSequence,
    *,
    max_loops: int = 1,
    pour_defaults: PourExecutionDefaults | None = None,
    pick_place_executor: PickPlaceExecutor = execute_action_sequence,
    pour_executor: PourExecutor = execute_pour,
) -> ExecutionReport:
    """Execute an ActionSequence repeatedly, adding support for pour actions."""

    loop_count = _positive_int(max_loops, "max_loops")
    pour_defaults = pour_defaults or PourExecutionDefaults(
        config_path=DEFAULT_ROBOT_CONFIG,
        server_url=None,
        arm=DEFAULT_POUR_ARM,
        speed_ratio=None,
        control_gripper=False,
    )
    results: list[ActionResult] = []
    action_count = len(sequence.actions)
    failed_result: ActionResult | None = None
    completed_loops = 0

    for loop_index in range(1, loop_count + 1):
        for action_index, original_action in enumerate(sequence.actions, start=1):
            action = _renumber_action(
                original_action,
                loop_index=loop_index,
                action_index=action_index,
                action_count=action_count,
            )
            if action.name == "pour":
                result = _execute_pour_action(
                    action,
                    loop_index=loop_index,
                    original_action_id=original_action.id,
                    pour_defaults=pour_defaults,
                    pour_executor=pour_executor,
                )
            else:
                result = _execute_pick_place_action(
                    runtime,
                    action,
                    loop_index=loop_index,
                    original_action_id=original_action.id,
                    sequence=sequence,
                    pick_place_executor=pick_place_executor,
                )

            results.append(result)
            if not result.success:
                failed_result = result
                break

        if failed_result is not None:
            break
        completed_loops = loop_index

    success = failed_result is None and completed_loops == loop_count
    return ExecutionReport(
        success=success,
        task=f"{sequence.task or 'action-sequence'}-loop",
        total_actions=action_count * loop_count,
        results=results,
        prepared=True,
        failed_action_id=(
            None if failed_result is None else failed_result.metadata.get("action_id")
        ),
        failed_action_name=None if failed_result is None else failed_result.action_name,
        error=None if failed_result is None else failed_result.error,
        metadata={
            "task_description": sequence.task_description,
            "goal_conditions": sequence.goal_conditions,
            "action_count_per_loop": action_count,
            "max_loops": loop_count,
            "completed_loops": completed_loops,
            "pour": {
                "config_path": pour_defaults.config_path,
                "server_url": pour_defaults.server_url,
                "arm": pour_defaults.arm,
                "speed_ratio": pour_defaults.speed_ratio,
                "control_gripper": pour_defaults.control_gripper,
            },
        },
    )


def execute_action_sequence_loop_plan(
    *,
    plan_path: str,
    config_path: str = DEFAULT_PIPELINE_CONFIG,
    max_loops: int = 1,
    pour_config_path: str | None = None,
    pour_arm: str = DEFAULT_POUR_ARM,
    pour_speed_ratio: float | None = None,
    control_pour_gripper: bool = False,
    runtime_factory: RuntimeFactory = create_x5_execution_runtime,
    pick_place_executor: PickPlaceExecutor = execute_action_sequence,
    pour_executor: PourExecutor = execute_pour,
) -> ExecutionReport:
    """Load action_sequence.json and execute pick/place/pour for N loops."""

    try:
        sequence = ActionSequence.load(plan_path)
        pipeline_defaults = _load_pipeline_defaults(config_path)
        pour_defaults = PourExecutionDefaults(
            config_path=pour_config_path or pipeline_defaults["robot_config_path"],
            server_url=pipeline_defaults["server_url"],
            arm=pour_arm,
            speed_ratio=pour_speed_ratio,
            control_gripper=control_pour_gripper,
        )
        runtime = runtime_factory(config_path=config_path)
    except Exception as exc:
        return ExecutionReport(
            success=False,
            task=Path(plan_path).stem,
            total_actions=0,
            failed_action_name="loop-config",
            error=str(exc),
            metadata={
                "config_path": config_path,
                "plan_path": plan_path,
                "max_loops": max_loops,
            },
        )

    try:
        with runtime:
            report = execute_action_sequence_loop(
                runtime,
                sequence,
                max_loops=max_loops,
                pour_defaults=pour_defaults,
                pick_place_executor=pick_place_executor,
                pour_executor=pour_executor,
            )
    except Exception as exc:
        report = ExecutionReport(
            success=False,
            task=sequence.task,
            total_actions=len(sequence.actions) * max(1, int(max_loops)),
            failed_action_name="loop-initialize",
            error=str(exc),
            metadata={
                "run_dir": str(runtime.run_dir),
                "config_path": config_path,
                "plan_path": plan_path,
                "max_loops": max_loops,
            },
        )

    report.metadata.setdefault("run_dir", str(runtime.run_dir))
    report.metadata.setdefault("config_path", config_path)
    report.metadata.setdefault("plan_path", plan_path)
    _write_json(runtime.run_dir / "action_sequence.json", sequence.to_dict())
    _write_json(runtime.run_dir / "execution_report.json", report.to_dict())
    return report


def _execute_pick_place_action(
    runtime: ExecutionRuntime,
    action: Action,
    *,
    loop_index: int,
    original_action_id: int,
    sequence: ActionSequence,
    pick_place_executor: PickPlaceExecutor,
) -> ActionResult:
    single_action_sequence = ActionSequence(
        task=sequence.task,
        task_description=sequence.task_description,
        actions=[action],
        goal_conditions=sequence.goal_conditions,
    )
    report = pick_place_executor(runtime, single_action_sequence)
    if report.results:
        result = report.results[-1]
    else:
        result = ActionResult(
            success=False,
            action_name=action.name,
            error=report.error or "Action produced no result.",
            metadata={"action_id": action.id, "action_args": dict(action.args)},
        )

    result.metadata.setdefault("action_id", action.id)
    result.metadata.setdefault("action_args", dict(action.args))
    result.metadata["loop_index"] = loop_index
    result.metadata["original_action_id"] = original_action_id
    return result


def _execute_pour_action(
    action: Action,
    *,
    loop_index: int,
    original_action_id: int,
    pour_defaults: PourExecutionDefaults,
    pour_executor: PourExecutor,
) -> ActionResult:
    args = dict(action.args)
    config_path = str(
        args.get("config_path")
        or args.get("robot_config")
        or args.get("config")
        or pour_defaults.config_path
    )
    arm = str(args.get("arm") or pour_defaults.arm)
    server_url = args.get("server_url") or pour_defaults.server_url
    speed_ratio = _optional_float(
        args.get("speed_ratio"),
        pour_defaults.speed_ratio,
        "speed_ratio",
    )
    control_gripper = _optional_bool(
        args.get("control_gripper"),
        pour_defaults.control_gripper,
        "control_gripper",
    )

    report = pour_executor(
        config_path=config_path,
        arm=arm,
        server_url=server_url,
        speed_ratio=speed_ratio,
        control_gripper=control_gripper,
    )
    return ActionResult(
        success=report.success,
        action_name="pour",
        message="Completed pour action." if report.success else "",
        error=report.error,
        metadata={
            "action_id": action.id,
            "original_action_id": original_action_id,
            "loop_index": loop_index,
            "action_args": args,
            "pddl_str": action.pddl_str,
            "pour_report": report.to_dict(),
            "pour_config_path": config_path,
            "pour_arm": arm,
            "pour_server_url": server_url,
            "pour_speed_ratio": speed_ratio,
            "control_gripper": control_gripper,
        },
    )


def _renumber_action(
    action: Action,
    *,
    loop_index: int,
    action_index: int,
    action_count: int,
) -> Action:
    execution_id = (loop_index - 1) * action_count + action_index
    return Action(
        id=execution_id,
        name=action.name,
        args=dict(action.args),
        pddl_str=action.pddl_str,
    )


def _load_pipeline_defaults(config_path: str) -> dict[str, str | None]:
    document = yaml.safe_load(Path(config_path).read_text()) or {}
    pipeline_config = document.get("pipeline", {})
    return {
        "robot_config_path": str(
            pipeline_config.get("robot_config", DEFAULT_ROBOT_CONFIG)
        ),
        "server_url": (
            None
            if pipeline_config.get("x5_server_url") is None
            else str(pipeline_config["x5_server_url"])
        ),
    }


def _positive_int(value: Any, name: str) -> int:
    converted = int(value)
    if converted < 1:
        raise ValueError(f"{name} must be >= 1")
    return converted


def _optional_float(value: Any, default: float | None, name: str) -> float | None:
    if value is None:
        return default
    return float(value)


def _optional_bool(value: Any, default: bool, name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default)
        + "\n"
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute a looped X5 pick/place/pour action sequence.",
    )
    parser.add_argument(
        "--plan",
        required=True,
        help="Path to action_sequence.json.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_PIPELINE_CONFIG,
        dest="config_path",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=1,
        help="Number of complete sequence loops to execute.",
    )
    parser.add_argument("--pour-config", dest="pour_config_path")
    parser.add_argument("--pour-arm", default=DEFAULT_POUR_ARM)
    parser.add_argument("--pour-speed-ratio", type=float)
    parser.add_argument(
        "--control-pour-gripper",
        action="store_true",
        help="Use the configured pour-arm gripper service. Default is manual.",
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
    report = execute_action_sequence_loop_plan(
        plan_path=args.plan,
        config_path=args.config_path,
        max_loops=args.max_loops,
        pour_config_path=args.pour_config_path,
        pour_arm=args.pour_arm,
        pour_speed_ratio=args.pour_speed_ratio,
        control_pour_gripper=args.control_pour_gripper,
    )
    print(json.dumps(report.to_dict(), indent=2, default=_json_default))
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
