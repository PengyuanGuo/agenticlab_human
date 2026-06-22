"""No-planning X5 pick-and-place execution pipeline."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from PIL import Image

from agenticlab_human.core.action_sequence import Action
from agenticlab_human.execution.action import ActionExecutor
from agenticlab_human.execution.action_backend import ActionResult, ExecutionReport
from agenticlab_human.execution.execution_context import ExecutionContext
from agenticlab_human.execution.pipeline_types import SceneSnapshot
from agenticlab_human.execution.place_target import estimate_place_target
from agenticlab_human.execution.robot.x5.client import (
    X5HTTPClient,
    save_rgbd_frame,
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
    ) -> None:
        self.x5_client = x5_client
        self.detector = detector
        self.grasp_backend = grasp_backend
        self.action_backend = action_backend
        self.run_dir = Path(run_dir)
        self.place_depth_patch_px = int(place_depth_patch_px)
        self.place_offset_world_x_m = float(place_offset_world_x_m)
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
) -> SceneSnapshot:
    """Capture one aligned frame, save it once, and return its paths."""

    frame = client.capture_rgbd()
    paths = save_rgbd_frame(frame, save_dir)
    return SceneSnapshot(
        frame=frame,
        rgb_path=paths["rgb"],
        depth_path=paths["depth_npy"],
        metadata_path=paths["metadata"],
    )


def execute_pick(
    runtime: ExecutionRuntime,
    object_name: str,
) -> ActionResult:
    """Capture, detect, request a grasp, and execute one pick action."""

    try:
        runtime.require_initialized()
        stage_dir = runtime.run_dir / "pick"
        snapshot = capture_scene_from_x5_server(runtime.x5_client, stage_dir)
        rgb_image = Image.fromarray(snapshot.frame.rgb, mode="RGB")
        detection = runtime.detector.detect(rgb_image, [object_name])
        selected_detection, bbox = _select_detection(detection, object_name)
        _save_detection(runtime.detector, rgb_image, selected_detection, stage_dir)

        grasps = runtime.grasp_backend.plan_for_object(
            rgb_path=snapshot.rgb_path,
            depth_path=snapshot.depth_path,
            bbox=bbox,
            object_name=object_name,
        )
        if not grasps:
            raise RuntimeError(f"no grasp candidate available for {object_name}")
        _write_json(
            stage_dir / "grasp_candidates.json",
            [_grasp_to_dict(candidate) for candidate in grasps],
        )

        runtime.context.bboxes[object_name] = [bbox]
        runtime.context.grasps[object_name] = grasps
        runtime.context.object_states.setdefault(object_name, {})["stale"] = False
        runtime.context.stale_objects.discard(object_name)
        action = Action(
            id=runtime.next_action_id(),
            name="pick",
            args={"object": object_name},
        )
        result = runtime.executor.execute_action(action)
        result.metadata.setdefault("scene", snapshot.to_dict())
        if result.success:
            runtime.held_object = object_name
        return result
    except Exception as exc:
        return _failed_action("pick", object_name, str(exc))


def execute_place(
    runtime: ExecutionRuntime,
    object_name: str,
    target_name: str,
) -> ActionResult:
    """Capture, estimate a place point, and execute one place action."""

    try:
        runtime.require_initialized()
        if runtime.held_object != object_name:
            raise RuntimeError(
                f"place requires held_object={object_name!r}, "
                f"got {runtime.held_object!r}"
            )

        stage_dir = runtime.run_dir / "place"
        snapshot = capture_scene_from_x5_server(runtime.x5_client, stage_dir)
        rgb_image = Image.fromarray(snapshot.frame.rgb, mode="RGB")
        detection = runtime.detector.detect(rgb_image, [target_name])
        selected_detection, bbox = _select_detection(detection, target_name)
        selected_detection.set_random_obj_point(margin_ratio=0.2)
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
            id=runtime.next_action_id(),
            name="place",
            args={"object": object_name, "target": target_name},
        )
        result = runtime.executor.execute_action(action)
        result.metadata.setdefault("scene", snapshot.to_dict())
        result.metadata.setdefault("place_target", place_target.to_dict())
        completed_steps = result.metadata.get("completed_steps", [])
        released = any(
            step.get("step") == "open_gripper"
            for step in completed_steps
            if isinstance(step, dict)
        )
        if result.success or released:
            runtime.held_object = None
        return result
    except Exception as exc:
        return _failed_action(
            "place",
            object_name,
            str(exc),
            target=target_name,
        )


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
            pick_result = execute_pick(runtime, object_name)
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
    x5_place_config = config.get("x5_place", {})
    detector_type = str(detector_config.get("type", "yolo"))
    if detector_type != "yolo":
        raise ValueError("detector.type must be 'yolo' for the X5 pipeline")

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
        robot_config_path=str(
            pipeline_config.get(
                "robot_config",
                "configs/robot/x5_config.yaml",
            )
        ),
        camera_config_path=str(
            pipeline_config.get(
                "camera_config",
                "configs/perception/camera_config.yaml",
            )
        ),
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
    )
    _write_json(
        run_dir / "run.json",
        {
            "config_path": str(config_file),
            "x5_server_url": server_url,
            "grasp_server_url": pipeline_config.get("grasp_server_url"),
            "camera_name": runtime_action_backend.camera_name,
            "place_offset_world_x_m": place_offset,
            "default_place_orientation_rotvec": (
                runtime_action_backend.default_place_orientation_rotvec.tolist()
            ),
        },
    )
    return runtime


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
) -> ActionResult:
    metadata: dict[str, Any] = {"object": object_name}
    if target is not None:
        metadata["target"] = target
    return ActionResult(
        success=False,
        action_name=action_name,
        error=error,
        metadata=metadata,
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = execute_pipeline(
        object_name=args.object_name,
        target_name=args.target_name,
        config_path=args.config_path,
    )
    print(json.dumps(report.to_dict(), indent=2, default=_json_default))
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
