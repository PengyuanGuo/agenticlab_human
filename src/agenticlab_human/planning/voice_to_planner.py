"""Voice-to-planner adapter.

This module intentionally stops at planning. It converts a spoken task into a
persisted planner session and an ActionSequence, but it does not call the robot
executor.

Current scene sources:
  - StaticImageSceneProvider(image_path)
  - CameraSceneProvider(which_cam), backed by cam_capture.CameraCapture
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Protocol

import yaml

from agenticlab_human.core.action_sequence import ActionSequence

if TYPE_CHECKING:
    from PIL import Image
    from agenticlab_human.planning.task_parser import TaskParser, TaskPlan

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/planning/task_parser_config.yaml"
DEFAULT_SPEECH_MODULE_PATH = "/home/agenticlab/Project/speech_to_text_module"


class SceneProvider(Protocol):
    """Provides the scene image required by TaskParser.generate_plan()."""

    def capture(self) -> "Image.Image":
        """Return the current scene image."""


class StaticImageSceneProvider:
    """Loads a fixed image from disk."""

    def __init__(self, image_path: str):
        self.image_path = Path(image_path).expanduser()

    def capture(self) -> "Image.Image":
        from PIL import Image

        if not self.image_path.exists():
            raise FileNotFoundError(f"Scene image does not exist: {self.image_path}")
        return Image.open(self.image_path).convert("RGB")


class CameraSceneProvider:
    """Capture a planner image from the camera module.

    If capture_fn is provided, it is used directly. Otherwise this provider
    lazily opens agenticlab_human.perception.camera.cam_capture.CameraCapture.
    """

    def __init__(
        self,
        which_cam: str = "Orbbec",
        capture_fn: Optional[Callable[[], "Image.Image"]] = None,
    ):
        self.which_cam = which_cam
        self.capture_fn = capture_fn
        self._camera = None

    def capture(self) -> "Image.Image":
        from PIL import Image

        if self.capture_fn is not None:
            image = self.capture_fn()
            if not isinstance(image, Image.Image):
                raise TypeError("camera capture_fn must return a PIL.Image.Image")
            return image.convert("RGB")

        if self._camera is None:
            from agenticlab_human.perception.camera.cam_capture import CameraCapture

            self._camera = CameraCapture(self.which_cam)

        image = self._camera.capture_pil()
        if not isinstance(image, Image.Image):
            raise TypeError("CameraCapture.capture_pil() must return a PIL.Image.Image")
        return image.convert("RGB")

    def destroy(self) -> None:
        if self._camera is not None:
            self._camera.destroy()
            self._camera = None


@dataclass
class VoiceToPlannerResult:
    task_text: str
    session_dir: str
    task_plan: "TaskPlan"
    action_sequence: ActionSequence


class VoiceToPlanner:
    """Synchronous adapter: listen once, capture one scene image, plan once."""

    def __init__(
        self,
        task_parser: "TaskParser",
        scene_provider: SceneProvider,
        speech_service=None,
    ):
        self.task_parser = task_parser
        self.scene_provider = scene_provider
        self.speech_service = speech_service

    def listen_and_plan(self) -> VoiceToPlannerResult:
        if self.speech_service is None:
            raise ValueError("speech_service is required for listen_and_plan().")

        task_text = self.speech_service.listen().strip()
        if not task_text:
            raise ValueError("SpeechInputService returned empty task text.")
        return self.plan_text(task_text)

    def plan_text(self, task_text: str) -> VoiceToPlannerResult:
        task_text = task_text.strip()
        if not task_text:
            raise ValueError("task_text must not be empty.")

        scene_image = self.scene_provider.capture()
        task_plan = self.task_parser.generate_plan(task_text, scene_image)
        action_sequence = task_plan.to_action_sequence()

        return VoiceToPlannerResult(
            task_text=task_text,
            session_dir=self.task_parser.session_output_dir,
            task_plan=task_plan,
            action_sequence=action_sequence,
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path_value: str, project_root: Path) -> str:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(project_root / path)


def _normalize_task_parser_paths(cfg: dict, project_root: Path) -> dict:
    task_cfg = cfg.get("TaskParser", cfg)
    for key in (
        "prompt_path",
        "prompt_use_pddl_path",
        "pddl_domain_path",
        "example_nl_path",
        "example_pddl_path",
        "llm_config_path",
        "output_dir",
    ):
        value = task_cfg.get(key)
        if value:
            task_cfg[key] = _resolve_path(value, project_root)

    pddl_cfg = task_cfg.get("pddl_cfg", {})
    fd_path = pddl_cfg.get("fast_downward_path")
    if fd_path:
        pddl_cfg["fast_downward_path"] = _resolve_path(fd_path, project_root)

    return cfg


def load_task_parser(config_path: str, project_root: Optional[str] = None) -> TaskParser:
    from agenticlab_human.planning.task_parser import TaskParser

    root = Path(project_root).expanduser().resolve() if project_root else _repo_root()
    cfg_path = Path(config_path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = _normalize_task_parser_paths(cfg, root)
    return TaskParser(cfg)


def load_speech_service(speech_module_path: str, device: str = "cpu"):
    module_path = Path(speech_module_path).expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"speech_to_text_module path not found: {module_path}")

    # Support both package import and direct module import.
    for path in (module_path.parent, module_path):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    try:
        from speech_to_text_module import SpeechInputService
    except ImportError:
        from speech_input_service import SpeechInputService

    return SpeechInputService(device=device)


def build_scene_provider(
    scene_source: str,
    image_path: Optional[str],
    camera_name: str = "Orbbec",
) -> SceneProvider:
    if scene_source == "image":
        if not image_path:
            raise ValueError("--image-path is required when --scene-source=image")
        return StaticImageSceneProvider(image_path)
    if scene_source == "camera":
        return CameraSceneProvider(which_cam=camera_name)
    raise ValueError(f"Unknown scene source: {scene_source}")


def _print_result(result: VoiceToPlannerResult, print_actions: bool = True) -> None:
    print("\nVoice-to-planner completed")
    print(f"Task text: {result.task_text}")
    print(f"Planner session: {result.session_dir}")
    print(f"Action count: {len(result.action_sequence)}")
    if print_actions:
        for action in result.action_sequence.actions:
            print(f"  {action.id}. {action.name} {action.args}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Listen for a task, capture/load a scene image, and run TaskParser."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--scene-source", choices=["image", "camera"], default="image")
    parser.add_argument("--image-path", required=False)
    parser.add_argument(
        "--camera-name",
        default="Orbbec",
        choices=["Orbbec", "FemtoBolt", "Gemini305"],
        help="Camera backend name used when --scene-source=camera.",
    )
    parser.add_argument("--speech-module-path", default=DEFAULT_SPEECH_MODULE_PATH)
    parser.add_argument("--speech-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--task-text",
        default=None,
        help="Bypass STT and plan this text directly. Useful for adapter tests.",
    )
    parser.add_argument(
        "--no-print-actions",
        action="store_true",
        help="Only print task/session summary.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    scene_provider = build_scene_provider(
        args.scene_source,
        args.image_path,
        camera_name=args.camera_name,
    )
    task_parser = load_task_parser(args.config, args.project_root)

    speech_service = None
    try:
        if args.task_text:
            adapter = VoiceToPlanner(task_parser, scene_provider)
            result = adapter.plan_text(args.task_text)
        else:
            speech_service = load_speech_service(
                args.speech_module_path,
                device=args.speech_device,
            )
            adapter = VoiceToPlanner(task_parser, scene_provider, speech_service)
            result = adapter.listen_and_plan()

        _print_result(result, print_actions=not args.no_print_actions)
    finally:
        if hasattr(scene_provider, "destroy"):
            scene_provider.destroy()
        if speech_service is not None and hasattr(speech_service, "shutdown"):
            speech_service.shutdown()


if __name__ == "__main__":
    main()
