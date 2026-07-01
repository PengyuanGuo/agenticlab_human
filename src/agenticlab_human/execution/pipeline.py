"""No-planning X5 pick-and-place execution pipeline."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from PIL import Image

from agenticlab_human.core.action_sequence import Action, ActionSequence
from agenticlab_human.execution.action import ActionExecutor
from agenticlab_human.execution.action_backend import ActionResult, ExecutionReport
from agenticlab_human.execution.execution_context import ExecutionContext
from agenticlab_human.execution.pipeline_types import SceneSnapshot
from agenticlab_human.execution.place_target import estimate_place_target
from agenticlab_human.execution.pour import move_joint_target
from agenticlab_human.execution.robot.x5.client import (
    X5HTTPClient,
    save_rgbd_frame,
)
from agenticlab_human.execution.robot.x5.conversion import degrees_to_radians
from agenticlab_human.execution.robot.x5.grasp_feasibility import (
    GraspFeasibilityConfig,
    select_feasible_grasps,
)
from agenticlab_human.execution.robot.x5.x5_remote_backend import (
    RemoteX5ActionBackend,
)
from agenticlab_human.perception.backend.perception_backend import (
    BBox,
    DetectionResult,
)
from agenticlab_human.perception.detection.yolo_detector import YOLODETECTOR
from agenticlab_human.perception.grasping.grasp_backend import GraspCandidate
from agenticlab_human.perception.grasping.http_backend import GraspNetHTTPBackend


DEFAULT_PIPELINE_CONFIG = "configs/execution/x5_pipeline.yaml"


@dataclass(frozen=True)
class PlaceSupportArmMotion:
    """Pipeline-level right-arm posture motion around left-arm place."""

    arm: str
    home_joints_deg: list[float]
    home_torso_deg: list[float]
    place_torso_deg: list[float]
    speed_ratio: float
    max_joint_delta_deg: float


@dataclass(frozen=True)
class PickGripVerification:
    """Pipeline-level grip-status check after each single pick attempt."""

    arm: str
    expected_grip_status: int
    max_retries: int


@dataclass(frozen=True)
class _PickScene:
    snapshot: SceneSnapshot
    detection: DetectionResult
    bbox: BBox


@dataclass(frozen=True)
class _GraspPlanningResult:
    grasps: list[GraspCandidate]
    scene: _PickScene


class ExecutionRuntime:
    """State retained across one X5 pick-and-place session."""

    def __init__(
        self,
        *,
        x5_client: X5HTTPClient,
        detector: Any,
        grasp_backend: Any,
        action_backend: RemoteX5ActionBackend,
        run_dir: Path,
        place_depth_patch_px: int,
        place_offset_world_x_m: float,
        place_support_motion: PlaceSupportArmMotion | None = None,
        pick_grip_verification: PickGripVerification | None = None,
        grasp_feasibility_config: GraspFeasibilityConfig | None = None,
        grasp_replan_attempts: int = 0,
    ) -> None:
        self.x5_client = x5_client
        self.detector = detector
        self.grasp_backend = grasp_backend
        self.action_backend = action_backend
        self.run_dir = Path(run_dir)
        self.place_depth_patch_px = int(place_depth_patch_px)
        self.place_offset_world_x_m = float(place_offset_world_x_m)
        self.place_support_motion = place_support_motion
        self.grasp_feasibility_config = grasp_feasibility_config
        self.grasp_replan_attempts = int(grasp_replan_attempts)
        self.place_support_current_torso_deg = (
            list(place_support_motion.home_torso_deg)
            if place_support_motion is not None
            else None
        )
        self.pick_grip_verification = pick_grip_verification
        self.T_world_camera = np.asarray(
            action_backend.T_world_camera,
            dtype=float,
        )
        self.context = ExecutionContext()
        self.executor = ActionExecutor(
            backend=action_backend,
            context=self.context,
            strict_cache_validation=True,
        )
        self.held_object: str | None = None
        self._next_action_id = 1
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            health = self.x5_client.health()
            if not health.camera.ready:
                raise RuntimeError(
                    f"X5 camera is not ready: {health.camera.detail}"
                )
            if not health.robot.ready:
                raise RuntimeError(
                    f"X5 robot is not ready: {health.robot.detail}"
                )
            self.grasp_backend.initialize()
            self.action_backend.initialize()
        except Exception:
            self.shutdown()
            raise
        self._initialized = True

    def shutdown(self) -> None:
        try:
            self.action_backend.shutdown()
        finally:
            try:
                self.grasp_backend.shutdown()
            finally:
                self.x5_client.close()
                self._initialized = False

    def next_action_id(self) -> int:
        action_id = self._next_action_id
        self._next_action_id += 1
        return action_id

    def claim_action_id(self, action_id: int | None = None) -> int:
        if action_id is None:
            return self.next_action_id()
        claimed = int(action_id)
        if claimed <= 0:
            raise ValueError("action_id must be positive")
        self._next_action_id = max(self._next_action_id, claimed + 1)
        return claimed

    def action_stage_dir(self, action_id: int, action_name: str) -> Path:
        stage_name = f"action_{action_id:03d}_{_path_slug(action_name)}"
        return self.run_dir / stage_name

    def require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("ExecutionRuntime is not initialized")

    def __enter__(self) -> "ExecutionRuntime":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()


def capture_scene_from_x5_server(
    client: X5HTTPClient,
    save_dir: str | Path,
    *,
    stem: str | None = None,
) -> SceneSnapshot:
    """Capture one aligned frame, save it once, and return its paths."""

    frame = client.capture_rgbd()
    paths = save_rgbd_frame(frame, save_dir, stem=stem)
    return SceneSnapshot(
        frame=frame,
        rgb_path=paths["rgb"],
        depth_path=paths["depth_npy"],
        metadata_path=paths["metadata"],
    )


def execute_pick(
    runtime: ExecutionRuntime,
    object_name: str,
    *,
    action_id: int | None = None,
    from_name: str | None = None,
    pddl_str: str | None = None,
) -> ActionResult:
    """Capture, detect, request a grasp, and execute one pick action."""

    claimed_action_id: int | None = None
    try:
        runtime.require_initialized()
        claimed_action_id = runtime.claim_action_id(action_id)
        stage_dir = runtime.action_stage_dir(claimed_action_id, "pick")
        scene = _capture_pick_scene(runtime, object_name, stage_dir)

        plan_result = _plan_feasible_grasps_with_retries(
            runtime,
            object_name,
            initial_scene=scene,
            stage_dir=stage_dir,
        )
        scene = plan_result.scene
        bbox = scene.bbox
        grasps = plan_result.grasps

        runtime.context.bboxes[object_name] = [bbox]
        runtime.context.grasps[object_name] = grasps
        runtime.context.object_states.setdefault(object_name, {})["stale"] = False
        runtime.context.stale_objects.discard(object_name)
        action_args = {"object": object_name}
        if from_name:
            action_args["from"] = from_name
        action = Action(
            id=claimed_action_id,
            name="pick",
            args=action_args,
            pddl_str=pddl_str,
        )
        result = runtime.executor.execute_action(action)
        result.metadata.setdefault("scene", scene.snapshot.to_dict())
        result.metadata.setdefault("stage_dir", str(stage_dir))
        if result.success:
            runtime.held_object = object_name
        return result
    except Exception as exc:
        return _failed_action(
            "pick",
            object_name,
            str(exc),
            action_id=claimed_action_id or action_id,
        )


def _init_gripper_before_pick_retry(
    runtime: ExecutionRuntime,
    arm: str,
    *,
    previous_grip_status: int | None,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        response = runtime.x5_client.init_gripper(arm=arm)
    except Exception as exc:
        error = str(exc)
        return (
            [
                {
                    "step": "init_gripper_before_pick_retry",
                    "arm": arm,
                    "previous_grip_status": previous_grip_status,
                    "success": False,
                    "error": error,
                    "request_id": None,
                    "duration_ms": None,
                }
            ],
            error,
        )

    success = bool(getattr(response, "success", False))
    error = getattr(response, "error", None)
    reset_steps = [
        {
            "step": "init_gripper_before_pick_retry",
            "arm": arm,
            "previous_grip_status": previous_grip_status,
            "success": success,
            "error": error,
            "request_id": getattr(response, "request_id", None),
            "duration_ms": getattr(response, "duration_ms", None),
        }
    ]
    if success:
        return reset_steps, None
    return reset_steps, error or f"init {arm} gripper before pick retry failed"


def _check_pick_grip_status(
    runtime: ExecutionRuntime,
    verification: PickGripVerification,
) -> dict[str, Any]:
    try:
        response = runtime.x5_client.get_state(verification.arm)
    except Exception as exc:
        return {
            "success": False,
            "retryable": False,
            "arm": verification.arm,
            "expected_grip_status": verification.expected_grip_status,
            "grip_status": None,
            "error": f"get_state failed for {verification.arm}: {exc}",
        }

    if not response.success:
        return {
            "success": False,
            "retryable": False,
            "arm": verification.arm,
            "expected_grip_status": verification.expected_grip_status,
            "grip_status": None,
            "error": response.error or f"get_state failed for {verification.arm}",
        }

    grip_status = _grip_status_from_state_response(response, verification.arm)
    if grip_status is None:
        return {
            "success": False,
            "retryable": False,
            "arm": verification.arm,
            "expected_grip_status": verification.expected_grip_status,
            "grip_status": None,
            "error": f"grip_status is unavailable for {verification.arm}",
        }

    success = int(grip_status) == verification.expected_grip_status
    return {
        "success": success,
        "retryable": not success,
        "arm": verification.arm,
        "expected_grip_status": verification.expected_grip_status,
        "grip_status": int(grip_status),
        "error": (
            None
            if success
            else (
                f"grip_status={int(grip_status)}, "
                f"expected {verification.expected_grip_status}"
            )
        ),
    }


def _grip_status_from_state_response(response: Any, arm: str) -> int | None:
    state_after = _field_value(response, "state_after")
    grippers = _field_value(state_after, "grippers")
    if grippers:
        gripper_state = (
            grippers.get(arm)
            if hasattr(grippers, "get")
            else None
        )
        status = _field_value(gripper_state, "grip_status")
        if status is not None:
            return int(status)

    gripper = _field_value(state_after, "gripper")
    status = _field_value(gripper, "grip_status")
    return None if status is None else int(status)


def _field_value(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _execute_pick_with_grip_retry(
    runtime: ExecutionRuntime,
    object_name: str,
    *,
    action_id: int | None = None,
    from_name: str | None = None,
    pddl_str: str | None = None,
) -> ActionResult:
    verification = runtime.pick_grip_verification
    max_retries = 0 if verification is None else verification.max_retries
    attempts: list[dict[str, Any]] = []
    result: ActionResult | None = None
    public_action_id = action_id
    previous_grip_status: int | None = None

    for attempt_index in range(max_retries + 1):
        retry_reset_steps: list[dict[str, Any]] = []
        if attempt_index > 0 and verification is not None:
            retry_reset_steps, reset_error = _init_gripper_before_pick_retry(
                runtime,
                verification.arm,
                previous_grip_status=previous_grip_status,
            )
            if reset_error:
                failure = _failed_action(
                    "pick",
                    object_name,
                    reset_error,
                    action_id=public_action_id,
                )
                failure.metadata["pick_attempts"] = attempts
                failure.metadata["retry_reset_steps"] = retry_reset_steps
                return failure

        result = execute_pick(
            runtime,
            object_name,
            action_id=action_id if attempt_index == 0 else None,
            from_name=from_name,
            pddl_str=pddl_str,
        )
        if public_action_id is None:
            public_action_id = result.metadata.get("action_id")
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "stage_dir": result.metadata.get("stage_dir"),
                "success": result.success,
                "error": result.error,
            }
        )
        if retry_reset_steps:
            attempts[-1]["retry_reset_steps"] = retry_reset_steps
        if not result.success:
            result.metadata["action_id"] = public_action_id
            result.metadata.setdefault("pick_attempts", attempts)
            return result

        if verification is None:
            result.metadata["action_id"] = public_action_id
            result.metadata.setdefault("pick_attempts", attempts)
            return result

        grip_check = _check_pick_grip_status(runtime, verification)
        result.metadata["gripper_verification"] = grip_check
        attempts[-1]["grip_status"] = grip_check.get("grip_status")
        attempts[-1]["verified"] = grip_check["success"]
        attempts[-1]["retryable"] = grip_check["retryable"]
        previous_grip_status = grip_check.get("grip_status")
        if grip_check["success"]:
            result.metadata["action_id"] = public_action_id
            result.metadata.setdefault("pick_attempts", attempts)
            return result

        runtime.held_object = None
        if not grip_check["retryable"] or attempt_index >= max_retries:
            result.success = False
            result.error = (
                "pick grip verification failed: "
                f"{grip_check['error']}"
            )
            result.metadata["action_id"] = public_action_id
            result.metadata.setdefault("pick_attempts", attempts)
            return result

    raise RuntimeError("unreachable pick retry state")


def execute_place(
    runtime: ExecutionRuntime,
    object_name: str,
    target_name: str,
    *,
    action_id: int | None = None,
    pddl_str: str | None = None,
) -> ActionResult:
    """Capture, estimate a place point, and execute one place action."""

    claimed_action_id: int | None = None
    try:
        runtime.require_initialized()
        claimed_action_id = runtime.claim_action_id(action_id)
        if runtime.held_object != object_name:
            raise RuntimeError(
                f"place requires held_object={object_name!r}, "
                f"got {runtime.held_object!r}"
            )

        stage_dir = runtime.action_stage_dir(claimed_action_id, "place")
        right_place_pre_steps = _move_place_support_arm(
            runtime,
            step_name="right_place_pre_pose",
            target_torso_deg=(
                None
                if runtime.place_support_motion is None
                else runtime.place_support_motion.place_torso_deg
            ),
        )
        snapshot = capture_scene_from_x5_server(runtime.x5_client, stage_dir)
        rgb_image = Image.fromarray(snapshot.frame.rgb, mode="RGB")
        detection = runtime.detector.detect(rgb_image, [target_name])
        selected_detection, bbox = _select_detection(detection, target_name)
        selected_detection.set_random_obj_point(margin_ratio=0.25)
        target_pixel = selected_detection.get_object_center()
        if target_pixel is None:
            raise RuntimeError(f"no place pixel available for {target_name}")
        _save_detection(runtime.detector, rgb_image, selected_detection, stage_dir)

        place_target = estimate_place_target(
            depth_mm=snapshot.frame.depth_mm,
            intrinsics=snapshot.frame.intrinsics,
            pixel_xy=target_pixel,
            T_world_camera=runtime.T_world_camera,
            place_offset_world_x_m=runtime.place_offset_world_x_m,
            depth_patch_px=runtime.place_depth_patch_px,
        )
        _write_json(stage_dir / "target_pose.json", place_target.to_dict())

        runtime.context.bboxes[target_name] = [bbox]
        target_state = runtime.context.object_states.setdefault(target_name, {})
        target_state["pose"] = place_target.p_world_place
        target_state["stale"] = False
        runtime.context.stale_objects.discard(target_name)
        action = Action(
            id=claimed_action_id,
            name="place",
            args={"object": object_name, "target": target_name},
            pddl_str=pddl_str,
        )
        result = runtime.executor.execute_action(action)
        result.metadata.setdefault("scene", snapshot.to_dict())
        result.metadata.setdefault("place_target", place_target.to_dict())
        result.metadata.setdefault("stage_dir", str(stage_dir))
        if right_place_pre_steps is not None:
            result.metadata.setdefault(
                "right_arm_place_pre_steps",
                right_place_pre_steps,
            )
        completed_steps = result.metadata.get("completed_steps", [])
        released = any(
            step.get("step") == "open_gripper"
            for step in completed_steps
            if isinstance(step, dict)
        )
        if result.success:
            try:
                right_home_steps = _move_place_support_arm(
                    runtime,
                    step_name="right_home_after_place",
                    target_torso_deg=(
                        None
                        if runtime.place_support_motion is None
                        else runtime.place_support_motion.home_torso_deg
                    ),
                )
                if right_home_steps is not None:
                    result.metadata.setdefault(
                        "right_arm_home_steps",
                        right_home_steps,
                    )
            except Exception as exc:
                result.success = False
                result.error = f"right arm home after place failed: {exc}"
                result.metadata["right_arm_home_error"] = str(exc)
        if result.success or released:
            runtime.held_object = None
        return result
    except Exception as exc:
        return _failed_action(
            "place",
            object_name,
            str(exc),
            target=target_name,
            action_id=claimed_action_id or action_id,
        )


def _move_place_support_arm(
    runtime: ExecutionRuntime,
    *,
    step_name: str,
    target_torso_deg: Sequence[float] | None,
) -> list[dict[str, Any]] | None:
    motion = runtime.place_support_motion
    if motion is None or target_torso_deg is None:
        return None
    current_torso_deg = (
        runtime.place_support_current_torso_deg
        if runtime.place_support_current_torso_deg is not None
        else motion.home_torso_deg
    )
    completed = move_joint_target(
        runtime.x5_client,
        arm=motion.arm,
        target_joints_deg=motion.home_joints_deg,
        target_torso_deg=target_torso_deg,
        current_torso_deg=current_torso_deg,
        speed_ratio=motion.speed_ratio,
        max_joint_delta_deg=motion.max_joint_delta_deg,
        step_name=step_name,
    )
    runtime.place_support_current_torso_deg = list(target_torso_deg)
    return completed


def execute_action_sequence(
    runtime: ExecutionRuntime,
    sequence: ActionSequence,
) -> ExecutionReport:
    """Execute a planner ActionSequence using per-action perception stages."""

    results: list[ActionResult] = []
    for action in sequence.actions:
        if action.name == "pick":
            object_name = action.args.get("object")
            if not object_name:
                result = _missing_sequence_arg(action, "object")
            else:
                result = _execute_pick_with_grip_retry(
                    runtime,
                    object_name,
                    action_id=action.id,
                    from_name=action.args.get("from"),
                    pddl_str=action.pddl_str,
                )
        elif action.name == "place":
            object_name = action.args.get("object")
            target_name = action.args.get("target")
            if not object_name:
                result = _missing_sequence_arg(action, "object")
            elif not target_name:
                result = _missing_sequence_arg(action, "target")
            else:
                result = execute_place(
                    runtime,
                    object_name,
                    target_name,
                    action_id=action.id,
                    pddl_str=action.pddl_str,
                )
        else:
            result = ActionResult(
                success=False,
                action_name=action.name,
                error=f"Unsupported action type: {action.name}",
                metadata={"action_id": action.id, "args": action.args},
            )

        result.metadata.setdefault("action_id", action.id)
        result.metadata.setdefault("action_args", dict(action.args))
        if action.pddl_str:
            result.metadata.setdefault("pddl_str", action.pddl_str)
        results.append(result)
        if not result.success:
            break

    failed_result = next((result for result in results if not result.success), None)
    failed_action_id = None
    if failed_result is not None:
        failed_action_id = failed_result.metadata.get("action_id")
    return ExecutionReport(
        success=failed_result is None and len(results) == len(sequence.actions),
        task=sequence.task,
        total_actions=len(sequence.actions),
        results=results,
        prepared=True,
        failed_action_id=failed_action_id,
        failed_action_name=(
            None if failed_result is None else failed_result.action_name
        ),
        error=None if failed_result is None else failed_result.error,
        metadata={
            "task_description": sequence.task_description,
            "goal_conditions": sequence.goal_conditions,
        },
    )


def execute_action_sequence_plan(
    *,
    plan_path: str,
    config_path: str = DEFAULT_PIPELINE_CONFIG,
) -> ExecutionReport:
    """Load action_sequence.json and execute all supported pick/place actions."""

    try:
        sequence = ActionSequence.load(plan_path)
        runtime = create_x5_execution_runtime(config_path=config_path)
    except Exception as exc:
        return ExecutionReport(
            success=False,
            task=Path(plan_path).stem,
            total_actions=0,
            failed_action_name="sequence-config",
            error=str(exc),
            metadata={"config_path": config_path, "plan_path": plan_path},
        )

    try:
        with runtime:
            report = execute_action_sequence(runtime, sequence)
    except Exception as exc:
        report = ExecutionReport(
            success=False,
            task=sequence.task,
            total_actions=len(sequence.actions),
            failed_action_name="sequence-initialize",
            error=str(exc),
            metadata={
                "run_dir": str(runtime.run_dir),
                "config_path": config_path,
                "plan_path": plan_path,
            },
        )

    report.metadata.setdefault("run_dir", str(runtime.run_dir))
    report.metadata.setdefault("config_path", config_path)
    report.metadata.setdefault("plan_path", plan_path)
    _write_json(runtime.run_dir / "action_sequence.json", sequence.to_dict())
    _write_json(runtime.run_dir / "execution_report.json", report.to_dict())
    return report


def execute_pipeline(
    *,
    object_name: str,
    target_name: str,
    config_path: str = DEFAULT_PIPELINE_CONFIG,
) -> ExecutionReport:
    """Execute one explicit pick followed by one explicit place."""

    try:
        runtime = create_x5_execution_runtime(
            config_path=config_path,
        )
    except Exception as exc:
        return ExecutionReport(
            success=False,
            task=f"pick-{object_name}-place-{target_name}",
            total_actions=2,
            failed_action_name="pipeline-config",
            error=str(exc),
            metadata={"config_path": config_path},
        )
    results: list[ActionResult] = []
    report: ExecutionReport
    try:
        with runtime:
            pick_result = _execute_pick_with_grip_retry(runtime, object_name)
            results.append(pick_result)
            if not pick_result.success:
                report = _execution_report(
                    object_name,
                    target_name,
                    results,
                    failed_action_id=pick_result.metadata.get("action_id", 1),
                )
            else:
                place_result = execute_place(runtime, object_name, target_name)
                results.append(place_result)
                report = _execution_report(
                    object_name,
                    target_name,
                    results,
                    failed_action_id=(
                        None
                        if place_result.success
                        else place_result.metadata.get("action_id", 2)
                    ),
                )
    except Exception as exc:
        report = ExecutionReport(
            success=False,
            task=f"pick-{object_name}-place-{target_name}",
            total_actions=2,
            results=results,
            failed_action_name="pipeline-initialize",
            error=str(exc),
            metadata={"run_dir": str(runtime.run_dir)},
        )

    _write_json(runtime.run_dir / "execution_report.json", report.to_dict())
    return report


def create_x5_execution_runtime(
    *,
    config_path: str = DEFAULT_PIPELINE_CONFIG,
    x5_client: X5HTTPClient | None = None,
    detector: Any | None = None,
    grasp_backend: Any | None = None,
    action_backend: RemoteX5ActionBackend | None = None,
) -> ExecutionRuntime:
    """Build the production runtime while allowing explicit test doubles."""

    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text()) or {}
    pipeline_config = config.get("pipeline", {})
    detector_config = config.get("detector", {})
    grasp_config = config.get("grasp", {})
    x5_pick_config = config.get("x5_pick", {})
    x5_place_config = config.get("x5_place", {})
    detector_type = str(detector_config.get("type", "yolo"))
    if detector_type != "yolo":
        raise ValueError("detector.type must be 'yolo' for the X5 pipeline")
    robot_config_path = str(
        pipeline_config.get(
            "robot_config",
            "configs/robot/x5_config.yaml",
        )
    )
    camera_config_path = str(
        pipeline_config.get(
            "camera_config",
            "configs/perception/camera_config.yaml",
        )
    )
    robot_config = yaml.safe_load(Path(robot_config_path).read_text()) or {}

    place_offset = pipeline_config.get("place_offset_world_x_m")
    if place_offset is None:
        raise ValueError(
            "pipeline.place_offset_world_x_m must be calibrated before execution"
        )
    place_offset = _finite_float(
        place_offset,
        "pipeline.place_offset_world_x_m",
    )

    output_root = Path(pipeline_config.get("output_dir", "output/execution"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    server_url = str(
        pipeline_config.get("x5_server_url", "http://127.0.0.1:8000")
    )
    request_timeout_s = float(
        pipeline_config.get("request_timeout_s", 120.0)
    )
    runtime_x5_client = x5_client or X5HTTPClient(
        server_url,
        timeout_s=request_timeout_s,
    )
    runtime_detector = detector or YOLODETECTOR(
        model_path=_required_string(detector_config, "model_path", "detector"),
        confidence=float(detector_config.get("confidence", 0.25)),
        image_size=int(detector_config.get("image_size", 960)),
        output_dir=str(run_dir / "detector"),
        session_name="pipeline",
    )
    runtime_grasp_backend = grasp_backend or GraspNetHTTPBackend(
        str(
            pipeline_config.get(
                "grasp_server_url",
                "http://127.0.0.1:8010",
            )
        ),
        timeout_s=request_timeout_s,
        mask_offset_px=int(grasp_config.get("mask_offset_px", 10)),
        max_grasps=int(grasp_config.get("max_grasps", 20)),
        score_threshold=float(grasp_config.get("score_threshold", 0.0)),
        collision_detection=bool(
            grasp_config.get("collision_detection", True)
        ),
        nms=bool(grasp_config.get("nms", True)),
    )
    runtime_action_backend = action_backend or RemoteX5ActionBackend(
        robot_config_path=robot_config_path,
        camera_config_path=camera_config_path,
        server_url=server_url,
        arm=pipeline_config.get("arm"),
        camera_name=pipeline_config.get("camera_name"),
        place_approach_offset_x_m=x5_place_config.get(
            "place_approach_offset_x_m"
        ),
        default_place_orientation_rotvec=x5_place_config.get(
            "default_place_orientation_rotvec"
        ),
        client=runtime_x5_client,
    )

    runtime = ExecutionRuntime(
        x5_client=runtime_x5_client,
        detector=runtime_detector,
        grasp_backend=runtime_grasp_backend,
        action_backend=runtime_action_backend,
        run_dir=run_dir,
        place_depth_patch_px=int(
            pipeline_config.get("place_depth_patch_px", 9)
        ),
        place_offset_world_x_m=place_offset,
        place_support_motion=_load_place_support_motion(
            robot_config,
            x5_place_config,
        ),
        pick_grip_verification=_load_pick_grip_verification(
            x5_pick_config,
            default_arm=str(pipeline_config.get("arm", "left")),
        ),
        grasp_feasibility_config=_load_grasp_feasibility_config(
            robot_config,
            x5_pick_config,
        ),
        grasp_replan_attempts=_load_grasp_replan_attempts(x5_pick_config),
    )
    _write_json(
        run_dir / "run.json",
        {
            "config_path": str(config_file),
            "x5_server_url": server_url,
            "grasp_server_url": pipeline_config.get("grasp_server_url"),
            "camera_name": runtime_action_backend.camera_name,
            "place_offset_world_x_m": place_offset,
            "pick_grip_verification": (
                None
                if runtime.pick_grip_verification is None
                else {
                    "arm": runtime.pick_grip_verification.arm,
                    "expected_grip_status": (
                        runtime.pick_grip_verification.expected_grip_status
                    ),
                    "max_retries": runtime.pick_grip_verification.max_retries,
                }
            ),
            "default_place_orientation_rotvec": (
                runtime_action_backend.default_place_orientation_rotvec.tolist()
            ),
            "grasp_feasibility": (
                None
                if runtime.grasp_feasibility_config is None
                else {
                    "soft_limit_margin_deg": (
                        runtime.grasp_feasibility_config.soft_limit_margin_deg
                    ),
                    "movel_sample_step_m": (
                        runtime.grasp_feasibility_config.movel_sample_step_m
                    ),
                    "inverse_type": runtime.grasp_feasibility_config.inverse_type,
                    "max_candidates": runtime.grasp_feasibility_config.max_candidates,
                }
            ),
            "grasp_replan_attempts": runtime.grasp_replan_attempts,
        },
    )
    return runtime


def _load_place_support_motion(
    robot_config: dict[str, Any],
    x5_place_config: dict[str, Any],
) -> PlaceSupportArmMotion | None:
    if x5_place_config.get("support_arm_enabled", True) is False:
        return None

    robot = robot_config["robot"]
    action_backend = robot_config.get("action_backend", {})
    arm = str(x5_place_config.get("support_arm", "right"))
    arm_config = robot[arm]
    speed_ratio = _finite_float(
        x5_place_config.get(
            "support_arm_speed_ratio",
            action_backend.get("home_speed_ratio", 0.1),
        ),
        "x5_place.support_arm_speed_ratio",
    )
    max_joint_delta_deg = _finite_float(
        x5_place_config.get(
            "support_arm_max_joint_delta_deg",
            robot.get(
                "max_joint_delta_deg",
                action_backend.get("home_max_step_deg", 45.0),
            ),
        ),
        "x5_place.support_arm_max_joint_delta_deg",
    )
    if speed_ratio <= 0.0:
        raise ValueError("x5_place.support_arm_speed_ratio must be positive")
    if max_joint_delta_deg <= 0.0:
        raise ValueError(
            "x5_place.support_arm_max_joint_delta_deg must be positive"
        )

    return PlaceSupportArmMotion(
        arm=arm,
        home_joints_deg=_required_vector(
            arm_config,
            "home_joints_deg",
            7,
            f"robot.{arm}.home_joints_deg",
        ),
        home_torso_deg=_required_vector(
            arm_config,
            "home_torso_deg",
            (1, 2),
            f"robot.{arm}.home_torso_deg",
        ),
        place_torso_deg=_required_vector(
            arm_config,
            "place_torso_deg",
            (1, 2),
            f"robot.{arm}.place_torso_deg",
        ),
        speed_ratio=speed_ratio,
        max_joint_delta_deg=max_joint_delta_deg,
    )


def _load_pick_grip_verification(
    x5_pick_config: dict[str, Any],
    *,
    default_arm: str,
) -> PickGripVerification | None:
    if x5_pick_config.get("verify_grip_status", True) is False:
        return None

    expected_status = int(x5_pick_config.get("expected_grip_status", 2))
    if expected_status < 0 or expected_status > 3:
        raise ValueError("x5_pick.expected_grip_status must be in [0, 3]")

    max_retries = int(x5_pick_config.get("max_retries", 2))
    if max_retries < 0:
        raise ValueError("x5_pick.max_retries must be non-negative")

    return PickGripVerification(
        arm=str(x5_pick_config.get("gripper_arm", default_arm)),
        expected_grip_status=expected_status,
        max_retries=max_retries,
    )


def _load_grasp_feasibility_config(
    robot_config: dict[str, Any],
    x5_pick_config: dict[str, Any],
) -> GraspFeasibilityConfig | None:
    if x5_pick_config.get("check_grasp_feasibility", True) is False:
        return None

    robot = robot_config["robot"]
    sample_step_m = _finite_float(
        x5_pick_config.get("grasp_movel_sample_step_m", 0.01),
        "x5_pick.grasp_movel_sample_step_m",
    )
    if sample_step_m <= 0.0:
        raise ValueError("x5_pick.grasp_movel_sample_step_m must be positive")

    inverse_type = int(x5_pick_config.get("grasp_ik_inverse_type", 0))
    if inverse_type not in (0, 1, 2):
        raise ValueError("x5_pick.grasp_ik_inverse_type must be 0, 1, or 2")

    max_candidates = x5_pick_config.get("grasp_feasibility_max_candidates")
    if max_candidates is not None:
        max_candidates = int(max_candidates)
        if max_candidates <= 0:
            raise ValueError(
                "x5_pick.grasp_feasibility_max_candidates must be positive"
            )

    return GraspFeasibilityConfig(
        joint_limits_deg=_required_vector_pairs(
            robot,
            "joint_limits_deg",
            "robot.joint_limits_deg",
        ),
        soft_limit_margin_deg=_finite_float(
            x5_pick_config.get("grasp_soft_limit_margin_deg", 3.0),
            "x5_pick.grasp_soft_limit_margin_deg",
        ),
        movel_sample_step_m=sample_step_m,
        inverse_type=inverse_type,
        max_candidates=max_candidates,
    )


def _load_grasp_replan_attempts(x5_pick_config: dict[str, Any]) -> int:
    attempts = int(x5_pick_config.get("grasp_replan_attempts", 0))
    if attempts < 0:
        raise ValueError("x5_pick.grasp_replan_attempts must be non-negative")
    return attempts


def _capture_pick_scene(
    runtime: ExecutionRuntime,
    object_name: str,
    stage_dir: Path,
) -> _PickScene:
    snapshot = capture_scene_from_x5_server(
        runtime.x5_client,
        stage_dir,
        stem="pick_scene",
    )
    rgb_image = Image.fromarray(snapshot.frame.rgb, mode="RGB")
    detection = runtime.detector.detect(rgb_image, [object_name])
    selected_detection, bbox = _select_detection(detection, object_name)
    _save_detection(runtime.detector, rgb_image, selected_detection, stage_dir)
    return _PickScene(
        snapshot=snapshot,
        detection=selected_detection,
        bbox=bbox,
    )


def _plan_feasible_grasps_with_retries(
    runtime: ExecutionRuntime,
    object_name: str,
    *,
    initial_scene: _PickScene,
    stage_dir: Path,
) -> _GraspPlanningResult:
    max_attempts = runtime.grasp_replan_attempts + 1
    attempts: list[dict[str, Any]] = []
    last_error = f"no grasp candidate available for {object_name}"
    scene = initial_scene

    for attempt_index in range(1, max_attempts + 1):
        attempt_prefix = f"grasp_attempt_{attempt_index:03d}"
        if attempt_index > 1:
            try:
                scene = _capture_pick_scene(runtime, object_name, stage_dir)
            except Exception as exc:
                last_error = str(exc)
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "success": False,
                        "error": last_error,
                        "raw_count": 0,
                        "feasible_count": 0,
                    }
                )
                _write_json(stage_dir / "grasp_replan_attempts.json", attempts)
                if attempt_index < max_attempts:
                    continue
                raise

        _write_pick_scene_attempt(stage_dir, attempt_prefix, scene)
        try:
            raw_grasps = runtime.grasp_backend.plan_for_object(
                rgb_path=scene.snapshot.rgb_path,
                depth_path=scene.snapshot.depth_path,
                bbox=scene.bbox,
                object_name=object_name,
            )
        except Exception as exc:
            last_error = str(exc)
            attempts.append(
                {
                    "attempt": attempt_index,
                    "success": False,
                    "error": last_error,
                    "raw_count": 0,
                    "feasible_count": 0,
                    **_pick_scene_attempt_summary(scene),
                }
            )
            _write_json(stage_dir / "grasp_replan_attempts.json", attempts)
            if attempt_index < max_attempts:
                continue
            raise

        _write_grasp_candidates(
            stage_dir,
            raw_grasps,
            base_name="grasp_candidates",
            attempt_prefix=f"{attempt_prefix}_candidates",
        )
        if not raw_grasps:
            last_error = f"no grasp candidate available for {object_name}"
            attempts.append(
                {
                    "attempt": attempt_index,
                    "success": False,
                    "error": last_error,
                    "raw_count": 0,
                    "feasible_count": 0,
                    **_pick_scene_attempt_summary(scene),
                }
            )
            _write_json(stage_dir / "grasp_replan_attempts.json", attempts)
            if attempt_index < max_attempts:
                continue
            raise RuntimeError(last_error)

        if runtime.grasp_feasibility_config is None:
            attempts.append(
                {
                    "attempt": attempt_index,
                    "success": True,
                    "error": None,
                    "raw_count": len(raw_grasps),
                    "feasible_count": len(raw_grasps),
                    **_pick_scene_attempt_summary(scene),
                }
            )
            _write_json(stage_dir / "grasp_replan_attempts.json", attempts)
            return _GraspPlanningResult(grasps=raw_grasps, scene=scene)

        selection = select_feasible_grasps(
            candidates=raw_grasps,
            ik_client=runtime.x5_client,
            arm=runtime.action_backend.arm,
            T_world_camera=runtime.T_world_camera,
            approach_distance_m=runtime.action_backend.config.approach_distance_m,
            config=runtime.grasp_feasibility_config,
            seed_joints_rad=degrees_to_radians(
                runtime.action_backend.config.home_joints_deg
            ),
        )
        _write_json(stage_dir / "grasp_feasibility.json", selection.to_dict())
        _write_json(
            stage_dir / f"{attempt_prefix}_feasibility.json",
            selection.to_dict(),
        )
        feasible_grasps = selection.feasible_candidates
        _write_grasp_candidates(
            stage_dir,
            feasible_grasps,
            base_name="grasp_candidates_feasible",
            attempt_prefix=f"{attempt_prefix}_candidates_feasible",
        )
        last_error = f"no IK-feasible grasp candidate available for {object_name}"
        attempts.append(
            {
                "attempt": attempt_index,
                "success": bool(feasible_grasps),
                "error": None if feasible_grasps else last_error,
                "raw_count": len(raw_grasps),
                "feasible_count": len(feasible_grasps),
                **_pick_scene_attempt_summary(scene),
            }
        )
        _write_json(stage_dir / "grasp_replan_attempts.json", attempts)
        if feasible_grasps:
            return _GraspPlanningResult(grasps=feasible_grasps, scene=scene)

    raise RuntimeError(last_error)


def _write_pick_scene_attempt(
    stage_dir: Path,
    attempt_prefix: str,
    scene: _PickScene,
) -> None:
    _write_json(
        stage_dir / f"{attempt_prefix}_scene.json",
        {
            **scene.snapshot.to_dict(),
            "bbox": _bbox_xyxy(scene.bbox),
        },
    )
    _write_json(
        stage_dir / f"{attempt_prefix}_detection.json",
        scene.detection.output_to_json,
    )


def _pick_scene_attempt_summary(scene: _PickScene) -> dict[str, Any]:
    return {
        "frame_id": scene.snapshot.frame.frame_id,
        "rgb_path": str(scene.snapshot.rgb_path),
        "depth_path": str(scene.snapshot.depth_path),
        "bbox": _bbox_xyxy(scene.bbox),
    }


def _bbox_xyxy(bbox: BBox) -> list[float]:
    return [float(value) for value in bbox.xyxy]


def _write_grasp_candidates(
    stage_dir: Path,
    candidates: Sequence[GraspCandidate],
    *,
    base_name: str,
    attempt_prefix: str,
) -> None:
    payload = [_grasp_to_dict(candidate) for candidate in candidates]
    _write_json(stage_dir / f"{base_name}.json", payload)
    _write_json(stage_dir / f"{attempt_prefix}.json", payload)


def _select_detection(
    detection: DetectionResult,
    object_name: str,
) -> tuple[DetectionResult, BBox]:
    if not detection.success or not detection.objects:
        reason = (detection.summary or {}).get("reason", "no detection")
        raise RuntimeError(f"detection failed for {object_name}: {reason}")
    matches = [
        item
        for item in detection.objects
        if str(item.get("label", "")).lower() == object_name.lower()
    ]
    if not matches:
        raise RuntimeError(f"detection returned no bbox for {object_name}")
    selected = dict(max(matches, key=lambda item: float(item.get("score", 0.0))))
    selected_detection = DetectionResult(
        success=True,
        objects=[selected],
        image_shape=detection.image_shape,
        raw_output=detection.raw_output,
        summary=detection.summary,
    )
    bbox = selected_detection.to_bboxes()[str(selected["label"])][0]
    return selected_detection, bbox


def _save_detection(
    detector: Any,
    image: Image.Image,
    detection: DetectionResult,
    stage_dir: Path,
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_json(stage_dir / "detection.json", detection.output_to_json)
    if hasattr(detector, "save_detection"):
        detector.save_detection(
            image,
            detection,
            str(stage_dir / "detection_visualization"),
        )


def _grasp_to_dict(candidate: GraspCandidate) -> dict[str, Any]:
    pose = (
        candidate.pose.tolist()
        if hasattr(candidate.pose, "tolist")
        else candidate.pose
    )
    return {
        "pose": pose,
        "score": candidate.score,
        "image_xy": list(candidate.image_xy) if candidate.image_xy else None,
        "object_name": candidate.object_name,
        "metadata": candidate.metadata,
    }


def _failed_action(
    action_name: str,
    object_name: str,
    error: str,
    *,
    target: str | None = None,
    action_id: int | None = None,
) -> ActionResult:
    metadata: dict[str, Any] = {"object": object_name}
    if target is not None:
        metadata["target"] = target
    if action_id is not None:
        metadata["action_id"] = action_id
    return ActionResult(
        success=False,
        action_name=action_name,
        error=error,
        metadata=metadata,
    )


def _missing_sequence_arg(action: Action, arg_name: str) -> ActionResult:
    return ActionResult(
        success=False,
        action_name=action.name,
        error=f"Action {action.id} ({action.name}) is missing required arg: {arg_name}",
        metadata={"action_id": action.id, "args": action.args},
    )


def _execution_report(
    object_name: str,
    target_name: str,
    results: Sequence[ActionResult],
    *,
    failed_action_id: int | None,
) -> ExecutionReport:
    success = len(results) == 2 and all(result.success for result in results)
    failed_result = next(
        (result for result in results if not result.success),
        None,
    )
    return ExecutionReport(
        success=success,
        task=f"pick-{object_name}-place-{target_name}",
        total_actions=2,
        results=list(results),
        prepared=True,
        failed_action_id=None if success else failed_action_id,
        failed_action_name=(
            None if failed_result is None else failed_result.action_name
        ),
        error=None if failed_result is None else failed_result.error,
    )


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


def _required_string(config: dict[str, Any], key: str, section: str) -> str:
    value = config.get(key)
    if not value:
        raise ValueError(f"{section}.{key} is required")
    return str(value)


def _finite_float(value: Any, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _required_vector(
    mapping: dict[str, Any],
    key: str,
    size: int | tuple[int, ...],
    name: str,
) -> list[float]:
    if key not in mapping:
        raise ValueError(f"{name} is required")
    return _finite_vector(mapping[key], size, name).tolist()


def _required_vector_pairs(
    mapping: dict[str, Any],
    key: str,
    name: str,
) -> list[list[float]]:
    if key not in mapping:
        raise ValueError(f"{name} is required")
    values = np.asarray(mapping[key], dtype=float)
    if values.shape != (7, 2) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain 7 finite [lower, upper] pairs")
    if np.any(values[:, 0] >= values[:, 1]):
        raise ValueError(f"{name} lower bounds must be less than upper bounds")
    return values.tolist()


def _finite_vector(
    values: Any,
    size: int | tuple[int, ...],
    name: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    allowed_shapes = (
        {(size,)}
        if isinstance(size, int)
        else {(candidate,) for candidate in size}
    )
    if vector.shape not in allowed_shapes or not np.all(np.isfinite(vector)):
        expected = (
            str(size)
            if isinstance(size, int)
            else " or ".join(str(candidate) for candidate in size)
        )
        raise ValueError(f"{name} must contain {expected} finite values")
    return vector


def _path_slug(value: str) -> str:
    slug = "".join(
        character.lower() if character.isalnum() else "-"
        for character in str(value)
    ).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "action"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute an explicit X5 pick-and-place pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    pipeline_parser = subparsers.add_parser("pipeline")
    pipeline_parser.add_argument("--object", required=True, dest="object_name")
    pipeline_parser.add_argument("--target", required=True, dest="target_name")
    pipeline_parser.add_argument(
        "--config",
        default=DEFAULT_PIPELINE_CONFIG,
        dest="config_path",
    )
    pipeline_parser.add_argument(
        "--execute",
        action="store_true",
        required=True,
        help="Required confirmation that this command sends robot motion.",
    )

    sequence_parser = subparsers.add_parser("sequence")
    sequence_parser.add_argument(
        "--plan",
        required=True,
        help="Path to action_sequence.json.",
    )
    sequence_parser.add_argument(
        "--config",
        default=DEFAULT_PIPELINE_CONFIG,
        dest="config_path",
    )
    sequence_parser.add_argument(
        "--execute",
        action="store_true",
        required=True,
        help="Required confirmation that this command sends robot motion.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "pipeline":
        report = execute_pipeline(
            object_name=args.object_name,
            target_name=args.target_name,
            config_path=args.config_path,
        )
    elif args.command == "sequence":
        report = execute_action_sequence_plan(
            plan_path=args.plan,
            config_path=args.config_path,
        )
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(report.to_dict(), indent=2, default=_json_default))
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
