"""Semantic action executor.

This module consumes ActionSequence and ActionBackend only. It intentionally
does not import ActionChecker, YOLO, AnyGrasp, or robot SDKs.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from agenticlab_human.core.action_sequence import Action as SequenceAction
from agenticlab_human.core.action_sequence import ActionSequence
from agenticlab_human.execution.action_backend import (
    ActionBackend,
    ActionResult,
    DryRunActionBackend,
    ExecutionReport,
)
from agenticlab_human.execution.execution_context import ExecutionContext, PrepareReport
from agenticlab_human.perception.backend.grasp_backend import GraspCandidate


class ActionExecutor:
    """Execute semantic actions against a robot-independent backend."""

    def __init__(
        self,
        backend: ActionBackend,
        context: Optional[ExecutionContext] = None,
        strict_cache_validation: bool = False,
    ) -> None:
        self.backend = backend
        self.context = context or ExecutionContext()
        self.strict_cache_validation = strict_cache_validation
        self.prepare_report: Optional[PrepareReport] = None

    def prepare(self, action_sequence: ActionSequence) -> PrepareReport:
        self.prepare_report = self.context.prepare_for_sequence(action_sequence)
        return self.prepare_report

    def execute_sequence(
        self,
        action_sequence: ActionSequence,
        *,
        move_home_on_shutdown: bool = False,
    ) -> ExecutionReport:
        results = []
        self.backend.initialize()
        try:
            for action in action_sequence.actions:
                result = self.execute_action(action)
                results.append(result)
                if not result.success:
                    return ExecutionReport(
                        success=False,
                        task=action_sequence.task,
                        total_actions=len(action_sequence.actions),
                        results=results,
                        prepared=bool(self.prepare_report and self.prepare_report.prepared),
                        failed_action_id=action.id,
                        failed_action_name=action.name,
                        error=result.error,
                        metadata=self._report_metadata(),
                    )
            return ExecutionReport(
                success=True,
                task=action_sequence.task,
                total_actions=len(action_sequence.actions),
                results=results,
                prepared=bool(self.prepare_report and self.prepare_report.prepared),
                metadata=self._report_metadata(),
            )
        finally:
            self.backend.shutdown(move_home=move_home_on_shutdown)

    def execute_action(self, action: SequenceAction) -> ActionResult:
        if action.name == "pick":
            return self._do_pick(action)
        if action.name == "place-on-object":
            return self._do_place_on_object(action)
        if action.name in {"place-on-surface", "place-on-table"}:
            return self._do_place_on_surface(action)
        if action.name == "place-in-container":
            return self._do_place_in_container(action)
        if action.name in {"move-home", "move_to_home"}:
            return self.backend.move_home()
        return ActionResult(
            success=False,
            action_name=action.name,
            error=f"Unsupported action type: {action.name}",
            metadata={"action_id": action.id, "args": action.args},
        )

    def _do_pick(self, action: SequenceAction) -> ActionResult:
        object_name = action.args.get("object")
        if not object_name:
            return self._missing_arg(action, "object")

        if self.context.is_stale(object_name) and not self.context.refresh_object(object_name):
            if self.strict_cache_validation:
                return ActionResult(
                    success=False,
                    action_name=action.name,
                    error=f"Cached affordances for {object_name} are stale and refresh failed.",
                    metadata={"action_id": action.id, "object": object_name},
                )

        result = self.backend.pick(
            object_name=object_name,
            from_name=action.args.get("from"),
            grasp_candidates=self.context.get_grasps(object_name),
            object_bbox=self.context.get_bbox(object_name),
            object_pose=self.context.get_object_pose(object_name),
        )
        if result.success:
            self.context.mark_stale(object_name)
        result.metadata.setdefault("action_id", action.id)
        return result

    # TODO: remove redundant place action types after implemented whole pipeline on real robot
    def _do_place_on_object(self, action: SequenceAction) -> ActionResult:
        object_name = action.args.get("object")
        target_name = action.args.get("target")
        if not object_name:
            return self._missing_arg(action, "object")
        if not target_name:
            return self._missing_arg(action, "target")

        stale_target = self.context.is_stale(target_name)
        if stale_target and not self.context.refresh_object(target_name) and self.strict_cache_validation:
            return ActionResult(
                success=False,
                action_name=action.name,
                error=f"Target {target_name} is stale and refresh failed.",
                metadata={"action_id": action.id, "target": target_name},
            )

        result = self.backend.place_on_object(
            object_name=object_name,
            target_name=target_name,
            target_bbox=self.context.get_bbox(target_name),
            target_pose=self.context.get_object_pose(target_name),
        )
        if stale_target:
            result.metadata.setdefault("warnings", []).append(
                f"Target {target_name} was marked stale before placement."
            )
        if result.success:
            self.context.mark_object_location(object_name, target_name)
        result.metadata.setdefault("action_id", action.id)
        return result

    def _do_place_on_surface(self, action: SequenceAction) -> ActionResult:
        object_name = action.args.get("object")
        surface_name = action.args.get("surface") or action.args.get("target") or action.args.get("location")
        if not object_name:
            return self._missing_arg(action, "object")
        if not surface_name:
            return self._missing_arg(action, "surface")

        result = self.backend.place_on_surface(
            object_name=object_name,
            surface_name=surface_name,
            target_bbox=self.context.get_bbox(surface_name),
            target_pose=self.context.get_object_pose(surface_name),
        )
        if result.success:
            self.context.mark_object_location(object_name, surface_name)
        result.metadata.setdefault("action_id", action.id)
        return result

    def _do_place_in_container(self, action: SequenceAction) -> ActionResult:
        object_name = action.args.get("object")
        container_name = action.args.get("container") or action.args.get("target")
        if not object_name:
            return self._missing_arg(action, "object")
        if not container_name:
            return self._missing_arg(action, "container")

        result = self.backend.place_in_container(
            object_name=object_name,
            container_name=container_name,
            target_bbox=self.context.get_bbox(container_name),
            target_pose=self.context.get_object_pose(container_name),
        )
        if result.success:
            self.context.mark_object_location(object_name, container_name)
        result.metadata.setdefault("action_id", action.id)
        return result

    def _missing_arg(self, action: SequenceAction, arg_name: str) -> ActionResult:
        return ActionResult(
            success=False,
            action_name=action.name,
            error=f"Action {action.id} ({action.name}) is missing required arg: {arg_name}",
            metadata={"action_id": action.id, "args": action.args},
        )

    def _report_metadata(self) -> dict:
        metadata = {"cache": self.context.describe_cache()}
        if self.prepare_report:
            metadata["prepare"] = self.prepare_report.to_dict()
        return metadata


def _build_backend(args: argparse.Namespace) -> ActionBackend:
    if args.backend == "dry-run":
        return DryRunActionBackend()
    if args.backend == "flexiv":
        from agenticlab_human.execution.robot.flexiv.flexiv_backend import FlexivActionBackend

        return FlexivActionBackend(
            robot_config_path=args.robot_config,
            camera_config_path=args.camera_config,
            camera_name=args.camera_name,
            execute=args.execute,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def _inject_test_grasp_camera_pose(context: ExecutionContext, action_sequence: ActionSequence) -> None:
    pick_objects = [
        action.args["object"]
        for action in action_sequence.actions
        if action.name == "pick" and action.args.get("object")
    ]
    if not pick_objects:
        raise ValueError("Cannot inject test grasp: ActionSequence has no pick action.")

    T_cam_grasp = [
        [-0.15578863, 0.90573424, 0.39417678, 0.04381273],
        [-0.33291125, -0.423846, 0.84233284, -0.07053141],
        [0.93, 0.0, 0.3675595, 0.536],
        [0.0, 0.0, 0.0, 1.0],
    ]
    object_name = pick_objects[0]
    context.grasps[object_name] = [
        GraspCandidate(
            pose=T_cam_grasp,
            score=1.0,
            object_name=object_name,
            metadata={
                "width": 0.08437179774045944,
                "source": "builtin-test-grasp-camera-pose",
                "frame": "camera",
            },
        )
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute or dry-run an ActionSequence.")
    parser.add_argument("--plan", required=True, help="Session directory, task_plan.json, or action_sequence.json.")
    parser.add_argument(
        "--backend",
        default="dry-run",
        choices=["dry-run", "flexiv"],
        help="Execution backend. Defaults to dry-run.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually connect to the selected robot backend and execute motion. Default is no execution.",
    )
    parser.add_argument(
        "--robot-config",
        default="configs/robot/flexiv_config.yaml",
        help="Robot backend config path.",
    )
    parser.add_argument(
        "--camera-config",
        default="configs/perception/camera_config.yaml",
        help="Camera config path with hand-eye calibration.",
    )
    parser.add_argument(
        "--camera-name",
        default="FemtoBolt",
        help="Camera name in --camera-config used for hand-eye conversion.",
    )
    parser.add_argument(
        "--test-grasp-camera-pose",
        action="store_true",
        help="Inject the built-in camera-frame test grasp instead of using perception/grasp backends.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip ExecutionContext.prepare_for_sequence().",
    )
    parser.add_argument(
        "--strict-cache-validation",
        action="store_true",
        help="Fail when stale cached affordances cannot be refreshed.",
    )
    parser.add_argument(
        "--move-home-on-shutdown",
        action="store_true",
        help="Ask backend to move home during shutdown.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    action_sequence = ActionSequence.load(args.plan)
    executor = ActionExecutor(
        backend=_build_backend(args),
        strict_cache_validation=args.strict_cache_validation,
    )
    if not args.skip_prepare:
        executor.prepare(action_sequence)
    if args.test_grasp_camera_pose:
        _inject_test_grasp_camera_pose(executor.context, action_sequence)
    report = executor.execute_sequence(
        action_sequence,
        move_home_on_shutdown=args.move_home_on_shutdown,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
