"""Camera capture public interface.

The current AgenticLab camera backend is Orbbec-only. Kinect and RealSense
branches were intentionally removed so callers do not need optional SDKs unless
they actually use the Orbbec capture path.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from agenticlab_human.perception.camera.orbbec_capture import OrbbecCameraCapture

SUPPORTED_CAMERAS = ("Orbbec", "FemtoBolt", "Gemini305")


class CameraCapture:
    """Keep an Orbbec camera open for repeated RGB-D captures.

    Returns RGB color frames and aligned depth frames in millimeters.
    """

    def __init__(self, which_cam: str = "Orbbec"):
        if which_cam not in SUPPORTED_CAMERAS:
            supported = ", ".join(SUPPORTED_CAMERAS)
            raise ValueError(f"Unsupported camera type: {which_cam}. Use one of: {supported}")

        self.which_cam = which_cam
        self._camera = OrbbecCameraCapture()

    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture one RGB image and aligned depth image.

        Returns:
            color_image: RGB uint8 array, shape (H, W, 3)
            depth_image: float32 depth in millimeters, shape (H, W)
        """
        return self._camera.capture()

    def capture_pil(self) -> Image.Image:
        """Capture one RGB image as a PIL image for VLM/planner use."""
        color_image, _ = self.capture()
        return Image.fromarray(color_image).convert("RGB")

    def destroy(self) -> None:
        self._camera.destroy()

    def __enter__(self) -> "CameraCapture":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()


def get_capture(which_cam: str = "Orbbec") -> Tuple[np.ndarray, np.ndarray]:
    """One-shot RGB-D capture."""
    with CameraCapture(which_cam) as camera:
        return camera.capture()


def get_color_image(which_cam: str = "Orbbec") -> Image.Image:
    """One-shot RGB capture as a PIL image."""
    with CameraCapture(which_cam) as camera:
        return camera.capture_pil()


def save_images(color_image: np.ndarray, depth_image: np.ndarray, save_dir: str) -> None:
    """Save RGB color and depth images to a timestamped pair of PNG files."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    color_path = os.path.join(save_dir, f"color_{timestamp}.png")
    depth_path = os.path.join(save_dir, f"depth_{timestamp}.png")

    Image.fromarray(color_image).save(color_path)
    Image.fromarray(np.clip(depth_image, 0, 65535).astype(np.uint16)).save(depth_path)

    print(f"Saved color image to {color_path}")
    print(f"Saved depth image to {depth_path}")


def display_preview(color_image: np.ndarray, depth_image: np.ndarray) -> bool:
    """Display RGB + depth preview. Return True when the user presses q."""
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET,
    )

    display_color = cv2.resize(cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR), (640, 360))
    display_depth = cv2.resize(depth_colormap, (640, 360))
    combined = np.hstack([display_color, display_depth])

    cv2.imshow("Camera Preview: RGB (left) | Depth (right) - Press q to exit", combined)
    return (cv2.waitKey(30) & 0xFF) == ord("q")


def pixel_to_camera_3d(point_2d, depth_image, camera_intrinsics, factor_depth: float = 1000.0):
    """Convert a 2D RGB pixel to 3D camera coordinates using depth and intrinsics."""
    x, y = int(point_2d[0]), int(point_2d[1])
    depth = depth_image[y, x] / factor_depth

    fx = camera_intrinsics[0][0]
    fy = camera_intrinsics[1][1]
    cx = camera_intrinsics[0][2]
    cy = camera_intrinsics[1][2]

    x_cam = (x - cx) * depth / fx
    y_cam = (y - cy) * depth / fy
    z_cam = depth

    return np.array([x_cam, y_cam, z_cam])


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture RGB-D frames from an Orbbec camera.")
    parser.add_argument("--which-cam", default="Orbbec", choices=SUPPORTED_CAMERAS)
    parser.add_argument("--save-dir", default="output/orbbec_captures")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    with CameraCapture(args.which_cam) as camera:
        if args.preview:
            print(f"Starting {args.which_cam} preview. Press q to save and exit.")
            while True:
                color_img, depth_img = camera.capture()
                if display_preview(color_img, depth_img):
                    break
            cv2.destroyAllWindows()

        color_img, depth_img = camera.capture()
        save_images(color_img, depth_img, args.save_dir)
        print("Capture complete.")


if __name__ == "__main__":
    main()
