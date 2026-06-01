"""
Orbbec camera capture utilities.

This module mirrors the spirit of `cam_capture.py` but uses `pyorbbecsdk`.
It provides a small, reusable capture class that keeps the pipeline open.
"""

from __future__ import annotations

import os
import time
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
    ):
        self._timeout_ms = int(timeout_ms)

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
        while True:
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
                continue

            try:
                raw_depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    (depth_frame.get_height(), depth_frame.get_width())
                )
            except ValueError:
                continue

            # Convert to millimeters (examples do the same scaling step).
            depth_mm = raw_depth.astype(np.float32) * float(depth_frame.get_depth_scale())

            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            return color_rgb, depth_mm

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
