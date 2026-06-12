"""Inference backend boundary for the GraspNet HTTP service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import yaml

DEFAULT_GRASPNET_CONFIG = str(
    Path(__file__).resolve().parents[4] / "configs" / "perception" / "graspnet_config.yaml"
)
DEFAULT_CAMERA_CONFIG = str(
    Path(__file__).resolve().parents[4] / "configs" / "perception" / "camera_config.yaml"
)
DEFAULT_CAMERA_NAME = "Gemini335"


@dataclass(frozen=True)
class RawGraspCandidate:
    pose_4x4: np.ndarray
    score: float
    width: float
    height: float
    depth: float
    image_xy: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GraspInferenceBackend(Protocol):
    name: str
    device_name: str

    def initialize(self) -> None:
        """Load model resources."""

    def predict(
        self,
        *,
        rgb: np.ndarray,
        depth_mm: np.ndarray,
        workspace_mask: np.ndarray,
        max_grasps: int,
        score_threshold: float,
        collision_detection: bool,
        nms: bool,
    ) -> list[RawGraspCandidate]:
        """Return camera-frame grasp candidates."""

    def shutdown(self) -> None:
        """Release model resources."""

    def health(self) -> tuple[bool, str]:
        """Return readiness and detail."""


class GraspNetInferenceBackend:
    """Load the copied GraspNet implementation only inside the server env."""

    name = "graspnet"

    def __init__(
        self,
        *,
        config_path: str = DEFAULT_GRASPNET_CONFIG,
        camera_config_path: str = DEFAULT_CAMERA_CONFIG,
        camera_name: str = DEFAULT_CAMERA_NAME,
        checkpoint_path: str | None = None,
        device: str = "cuda:0",
    ) -> None:
        self.config_path = str(config_path)
        self.camera_config_path = str(camera_config_path)
        self.camera_name = str(camera_name)
        self.checkpoint_path = checkpoint_path
        self.device_name = str(device)
        self._wrapper: Any = None
        self._last_error = ""

    def initialize(self) -> None:
        if self._wrapper is not None:
            return
        try:
            config = yaml.safe_load(Path(self.config_path).read_text()) or {}
            camera_config = yaml.safe_load(Path(self.camera_config_path).read_text()) or {}
            from agenticlab_human.perception.grasping.graspnet_wrapper import GraspNetWrapper

            self._wrapper = GraspNetWrapper(
                config=config,
                camera_config=camera_config,
                camera_name=self.camera_name,
                checkpoint_path=self.checkpoint_path,
                device=self.device_name,
            )
            self._last_error = ""
        except Exception as exc:
            self._last_error = str(exc)
            raise

    def predict(
        self,
        *,
        rgb: np.ndarray,
        depth_mm: np.ndarray,
        workspace_mask: np.ndarray,
        max_grasps: int,
        score_threshold: float,
        collision_detection: bool,
        nms: bool,
    ) -> list[RawGraspCandidate]:
        if self._wrapper is None:
            raise RuntimeError("GraspNet backend is not initialized")
        try:
            candidates = self._wrapper.predict_candidates(
                color_data=rgb,
                depth_mm=depth_mm,
                workspace_mask=workspace_mask,
                max_grasps=max_grasps,
                score_threshold=score_threshold,
                collision_detection=collision_detection,
                nms=nms,
            )
            self._last_error = ""
            return [RawGraspCandidate(**candidate) for candidate in candidates]
        except Exception as exc:
            self._last_error = str(exc)
            raise

    def shutdown(self) -> None:
        wrapper, self._wrapper = self._wrapper, None
        if wrapper is not None:
            wrapper.close()

    def health(self) -> tuple[bool, str]:
        ready = self._wrapper is not None and not self._last_error
        return ready, self._last_error or ("ready" if ready else "not initialized")


class MockGraspInferenceBackend:
    """Deterministic backend for HTTP contract tests."""

    name = "mock"
    device_name = "cpu"
    camera_name = "mock"

    def __init__(self) -> None:
        self._initialized = False

    def initialize(self) -> None:
        self._initialized = True

    def predict(
        self,
        *,
        rgb: np.ndarray,
        depth_mm: np.ndarray,
        workspace_mask: np.ndarray,
        max_grasps: int,
        score_threshold: float,
        collision_detection: bool,
        nms: bool,
    ) -> list[RawGraspCandidate]:
        if not self._initialized:
            raise RuntimeError("mock grasp backend is not initialized")
        ys, xs = np.nonzero(workspace_mask & (depth_mm > 0))
        if not len(xs):
            return []
        u = float(np.mean(xs))
        v = float(np.mean(ys))
        z = float(np.median(depth_mm[ys, xs]) / 1000.0)
        height, width = depth_mm.shape
        x = (u - (width - 1) / 2.0) * z / width
        y = (v - (height - 1) / 2.0) * z / width
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [x, y, z]
        score = max(float(score_threshold), 0.9)
        return [
            RawGraspCandidate(
                pose_4x4=pose,
                score=score,
                width=0.05,
                height=0.02,
                depth=0.02,
                image_xy=(u, v),
                metadata={"collision_detection": collision_detection, "nms": nms},
            )
        ][:max_grasps]

    def shutdown(self) -> None:
        self._initialized = False

    def health(self) -> tuple[bool, str]:
        return self._initialized, "ready" if self._initialized else "not initialized"
