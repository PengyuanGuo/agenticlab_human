"""RGB-D camera contract and Orbbec implementation."""

from __future__ import annotations

from typing import Any, Callable, Protocol, Tuple, runtime_checkable

from agenticlab_human.execution.robot.x5.contracts import (
    CameraIntrinsics,
    ComponentHealth,
    RGBDFrame,
)


@runtime_checkable
class RGBDCamera(Protocol):
    def initialize(self) -> None:
        """Open camera resources."""

    def capture(self) -> RGBDFrame:
        """Capture one aligned RGB-D frame."""

    def health(self) -> ComponentHealth:
        """Return current camera readiness."""

    def shutdown(self) -> None:
        """Release camera resources."""


class OrbbecRGBDCamera:
    """Adapt the AgenticLab Orbbec capture API to the X5 HTTP contract."""

    def __init__(
        self,
        *,
        which_cam: str = "Orbbec",
        color_size: Tuple[int, int] = (1280, 720),
        color_fps: int = 30,
        timeout_ms: int = 1000,
        max_capture_attempts: int = 30,
        max_sync_delta_ms: float = 20.0,
        camera_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.which_cam = which_cam
        self.color_size = tuple(color_size)
        self.color_fps = int(color_fps)
        self.timeout_ms = int(timeout_ms)
        self.max_capture_attempts = int(max_capture_attempts)
        self.max_sync_delta_ms = float(max_sync_delta_ms)
        self._camera_factory = camera_factory
        self._camera: Any = None
        self._last_error = ""

    def initialize(self) -> None:
        if self._camera is not None:
            return
        try:
            camera_factory = self._camera_factory
            if camera_factory is None:
                from agenticlab_human.perception.camera.cam_capture import CameraCapture

                camera_factory = CameraCapture
            self._camera = camera_factory(
                self.which_cam,
                color_size=self.color_size,
                color_fps=self.color_fps,
                timeout_ms=self.timeout_ms,
                max_capture_attempts=self.max_capture_attempts,
                max_sync_delta_ms=self.max_sync_delta_ms,
            )
            self._last_error = ""
        except Exception as exc:
            self._last_error = str(exc)
            raise

    def capture(self) -> RGBDFrame:
        if self._camera is None:
            raise RuntimeError("Orbbec camera is not initialized")
        try:
            capture = self._camera.capture_with_metadata()
            intrinsics = capture.intrinsics
            frame = RGBDFrame(
                rgb=capture.rgb,
                depth_mm=capture.depth_mm,
                intrinsics=CameraIntrinsics(
                    fx=intrinsics.fx,
                    fy=intrinsics.fy,
                    cx=intrinsics.cx,
                    cy=intrinsics.cy,
                    width=intrinsics.width,
                    height=intrinsics.height,
                ),
                timestamp_ns=capture.timestamp_ns,
                frame_id=f"orbbec-{capture.frame_index:010d}",
                color_timestamp_ns=capture.color_timestamp_ns,
                depth_timestamp_ns=capture.depth_timestamp_ns,
                depth_unit="mm",
            )
            frame.validate()
            self._last_error = ""
            return frame
        except Exception as exc:
            self._last_error = str(exc)
            raise

    def health(self) -> ComponentHealth:
        return ComponentHealth(
            ready=self._camera is not None and not self._last_error,
            backend="orbbec",
            detail=self._last_error or ("ready" if self._camera is not None else "not initialized"),
        )

    def shutdown(self) -> None:
        camera, self._camera = self._camera, None
        if camera is not None:
            camera.destroy()
