"""
Orbbec camera capture utilities.

This module mirrors the spirit of `cam_capture.py` but uses `pyorbbecsdk`.
It provides a small, reusable capture class that keeps the pipeline open.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBError,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBSensorType,
    OBStreamType,
    Pipeline,
)

from agenticlab_human.perception.camera.orbbec_utils import frame_to_bgr_image


@dataclass(frozen=True)
class RuntimeCameraIntrinsics:
    """Intrinsics reported by the active aligned color stream."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class RGBDCapture:
    """One aligned RGB-D capture with runtime calibration and timestamps."""

    rgb: np.ndarray
    depth_mm: np.ndarray
    intrinsics: RuntimeCameraIntrinsics
    color_timestamp_ns: int
    depth_timestamp_ns: int
    frame_index: int

    @property
    def timestamp_ns(self) -> int:
        return max(self.color_timestamp_ns, self.depth_timestamp_ns)

    @property
    def sync_delta_ms(self) -> float:
        return abs(self.color_timestamp_ns - self.depth_timestamp_ns) / 1_000_000.0


class OrbbecCameraCapture:
    """Keep an Orbbec pipeline open and capture aligned RGB+Depth frames.

    Returns:
    - color_image: RGB uint8 image (H, W, 3)
    - depth_image: float32 depth in millimeters (H, W)
    """

    def __init__(
        self,
        color_size: Tuple[int, int] = (1280, 720),
        color_fps: int = 30,
        align_to: OBStreamType = OBStreamType.COLOR_STREAM,
        enable_frame_sync: bool = True,
        timeout_ms: int = 1000,
        max_capture_attempts: int = 30,
        max_sync_delta_ms: float = 20.0,
    ):
        self._timeout_ms = int(timeout_ms)
        self._max_capture_attempts = int(max_capture_attempts)
        self._max_sync_delta_ms = float(max_sync_delta_ms)
        if self._max_capture_attempts <= 0:
            raise ValueError("max_capture_attempts must be positive")
        if self._max_sync_delta_ms < 0:
            raise ValueError("max_sync_delta_ms must be non-negative")

        self.pipeline = Pipeline()
        self.config = Config()

        # --- Stream configuration (same pattern as examples/sync_align.py) ---
        try:
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if color_profiles is None:
                raise RuntimeError("No COLOR sensor profiles available")

            w, h = color_size
            try:
                color_profile = color_profiles.get_video_stream_profile(w, h, OBFormat.RGB, color_fps)
            except OBError:
                # Fallback to default if exact match not supported on this device.
                color_profile = color_profiles.get_default_video_stream_profile()
            self.config.enable_stream(color_profile)

            depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if depth_profiles is None:
                raise RuntimeError("No DEPTH sensor profiles available")
            depth_profile = depth_profiles.get_default_video_stream_profile()
            self.config.enable_stream(depth_profile)

            # Wait until a full frameset (color+depth) is ready.
            self.config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        except Exception:
            # Keep the exception message useful to the caller.
            raise

        if enable_frame_sync:
            try:
                self.pipeline.enable_frame_sync()
            except Exception:
                # Some devices/SDK builds might not support it; alignment still works without.
                pass

        self.pipeline.start(self.config)

        # D2C alignment is the most common: depth aligned to the RGB coordinate system.
        self.align_filter = AlignFilter(align_to_stream=align_to)

    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture one aligned RGB + depth pair."""
        capture = self.capture_with_metadata()
        return capture.rgb, capture.depth_mm

    def capture_with_metadata(self) -> RGBDCapture:
        """Capture aligned RGB-D plus active-profile intrinsics and timestamps."""

        last_issue = "no complete aligned RGB-D frame received"
        for _ in range(self._max_capture_attempts):
            frames = self.pipeline.wait_for_frames(self._timeout_ms)
            if not frames:
                continue

            frames = self.align_filter.process(frames)
            if not frames:
                continue

            frames = frames.as_frame_set()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_bgr = frame_to_bgr_image(color_frame)
            if color_bgr is None:
                last_issue = "failed to decode Orbbec color frame"
                continue

            color_video = color_frame.as_video_frame()
            depth_video = depth_frame.as_video_frame()
            try:
                raw_depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    (depth_video.get_height(), depth_video.get_width())
                )
            except ValueError:
                last_issue = "depth buffer size does not match depth frame dimensions"
                continue

            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            depth_scale_mm = np.float32(depth_frame.get_depth_scale())
            depth_mm = raw_depth.astype(np.float32) * depth_scale_mm

            if color_rgb.dtype != np.uint8 or color_rgb.ndim != 3 or color_rgb.shape[2] != 3:
                raise RuntimeError(f"unexpected RGB frame format: {color_rgb.shape}, {color_rgb.dtype}")
            if depth_mm.dtype != np.float32 or depth_mm.ndim != 2:
                raise RuntimeError(f"unexpected depth frame format: {depth_mm.shape}, {depth_mm.dtype}")
            if color_rgb.shape[:2] != depth_mm.shape:
                raise RuntimeError(
                    "aligned RGB/depth shape mismatch: "
                    f"rgb={color_rgb.shape[:2]}, depth={depth_mm.shape}"
                )

            color_profile = color_video.get_stream_profile()
            color_intrinsic = color_profile.get_intrinsic()
            intrinsics = RuntimeCameraIntrinsics(
                fx=float(color_intrinsic.fx),
                fy=float(color_intrinsic.fy),
                cx=float(color_intrinsic.cx),
                cy=float(color_intrinsic.cy),
                width=int(color_intrinsic.width),
                height=int(color_intrinsic.height),
            )
            height, width = depth_mm.shape
            if (intrinsics.width, intrinsics.height) != (width, height):
                raise RuntimeError(
                    "runtime intrinsics do not match aligned frame shape: "
                    f"intrinsics={(intrinsics.width, intrinsics.height)}, frame={(width, height)}"
                )

            color_timestamp_ns = int(color_frame.get_system_timestamp_us()) * 1000
            depth_timestamp_ns = int(depth_frame.get_system_timestamp_us()) * 1000
            sync_delta_ms = abs(color_timestamp_ns - depth_timestamp_ns) / 1_000_000.0
            if sync_delta_ms > self._max_sync_delta_ms:
                last_issue = (
                    f"RGB/depth timestamp delta {sync_delta_ms:.3f} ms exceeds "
                    f"{self._max_sync_delta_ms:.3f} ms"
                )
                continue

            return RGBDCapture(
                rgb=color_rgb,
                depth_mm=depth_mm,
                intrinsics=intrinsics,
                color_timestamp_ns=color_timestamp_ns,
                depth_timestamp_ns=depth_timestamp_ns,
                frame_index=int(color_frame.get_index()),
            )

        raise TimeoutError(
            f"Orbbec capture failed after {self._max_capture_attempts} attempts: {last_issue}"
        )

    @staticmethod
    def save_images(color_image: np.ndarray, depth_image: np.ndarray, save_dir: str) -> None:
        """Save color (RGB) and depth (mm) images to specified directory."""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        color_path = os.path.join(save_dir, f"color_{timestamp}.png")
        depth_path = os.path.join(save_dir, f"depth_{timestamp}.png")

        Image.fromarray(color_image).save(color_path)
        # Save depth as 16-bit png (clipped to uint16 millimeters).
        Image.fromarray(np.clip(depth_image, 0, 65535).astype(np.uint16)).save(depth_path)

        print(f"Saved color image to {color_path}")
        print(f"Saved depth image to {depth_path}")

    def destroy(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def __enter__(self) -> "OrbbecCameraCapture":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()


def get_capture() -> Tuple[np.ndarray, np.ndarray]:
    """One-shot capture (kept for parity with cam_capture.get_capture)."""
    with OrbbecCameraCapture() as cam:
        return cam.capture()


if __name__ == "__main__":
    # Minimal manual test: preview one frame pair and save to disk.
    cam = OrbbecCameraCapture()
    try:
        color, depth_mm = cam.capture()
        print(f"Captured color={color.shape} dtype={color.dtype}, depth={depth_mm.shape} dtype={depth_mm.dtype}")
        OrbbecCameraCapture.save_images(color, depth_mm, save_dir="output/orbbec_captures")
    finally:
        cam.destroy()
