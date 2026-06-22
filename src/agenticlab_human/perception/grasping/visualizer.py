"""Short-lived Open3D visualization process for GraspNet HTTP results."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[4]
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
GRASPNESS_DIR = THIRD_PARTY_DIR / "graspness_unofficial"

for path in (THIRD_PARTY_DIR, GRASPNESS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


logger = logging.getLogger(__name__)


def visualize_manifest(manifest: dict[str, Any]) -> Path:
    import cv2
    import open3d as o3d
    from graspnetAPI import GraspGroup

    cloud = _load_point_cloud(manifest, o3d)
    grasp_rows = [_grasp_array(grasp) for grasp in manifest["grasps"]]
    grasps = GraspGroup(np.asarray(grasp_rows, dtype=np.float64))
    grasps = grasps.nms()
    grasps.sort_by_score()
    # select the best grasp
    grasps = grasps[:1]
    grippers = grasps.to_open3d_geometry_list()

    duration_s = max(0.0, float(manifest["visualize_seconds"]))
    print(f"Visualizing {len(grasps)} grasps for {duration_s:g} seconds")

    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name="Grasp Visualization", visible=True):
        raise RuntimeError("Open3D could not create the visualization window")

    output_path = _output_path(manifest)
    try:
        vis.add_geometry(cloud)
        for gripper in grippers:
            vis.add_geometry(gripper)

        view_ctrl = vis.get_view_control()
        view_ctrl.set_lookat(cloud.get_center())
        rotation = 2800.0 if manifest["camera_name"] == "RealSense" else 3141.59
        view_ctrl.rotate(0, rotation)

        for _ in range(5):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.03)

        vis.capture_screen_image(str(output_path))
        _crop_white_background(output_path, cv2)
        logger.info("Grasp visualization saved to %s", output_path)

        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            if vis.poll_events() is False:
                break
            vis.update_renderer()
            time.sleep(0.03)
    finally:
        vis.destroy_window()

    return output_path


def _load_point_cloud(manifest: dict[str, Any], o3d):
    rgb = np.asarray(Image.open(manifest["rgb_path"]).convert("RGB"), dtype=np.uint8)
    depth_path = Path(manifest["depth_path"])
    if depth_path.suffix.lower() == ".npy":
        depth_mm = np.load(depth_path)
    else:
        depth_mm = np.asarray(Image.open(depth_path))
    depth_mm = np.asarray(depth_mm, dtype=np.float32)

    camera_config = yaml.safe_load(Path(manifest["camera_config_path"]).read_text()) or {}
    camera = camera_config[manifest["camera_name"]]
    intrinsics = np.asarray(camera["intrinsic_matrix"], dtype=np.float32)
    factor_depth = float(camera.get("factor_depth", 1000.0))

    workspace = _workspace_mask(manifest, depth_mm.shape)
    valid = workspace & np.isfinite(depth_mm) & (depth_mm > 0)
    if not np.any(valid):
        raise ValueError("visualization workspace contains no valid depth points")

    rows, cols = np.indices(depth_mm.shape)
    z = depth_mm / factor_depth
    x = (cols - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.stack((x, y, z), axis=-1)[valid].astype(np.float64)
    colors = (rgb[valid].astype(np.float64) / 255.0)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def _workspace_mask(manifest: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    mask_path = manifest.get("workspace_mask_path")
    if mask_path:
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape != shape:
            raise ValueError(
                f"workspace mask shape mismatch: mask={mask.shape} depth={shape}"
            )
        return mask

    bbox = manifest.get("bbox_xyxy")
    if bbox is None:
        return np.ones(shape, dtype=bool)

    height, width = shape
    x1, y1, x2, y2 = bbox
    offset = int(manifest.get("mask_offset_px", 0))
    left = max(0, int(np.floor(x1)) - offset)
    top = max(0, int(np.floor(y1)) - offset)
    right = min(width, int(np.ceil(x2)) + offset)
    bottom = min(height, int(np.ceil(y2)) + offset)
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _grasp_array(grasp: dict[str, Any]) -> np.ndarray:
    pose = np.asarray(grasp["pose_4x4"], dtype=np.float64)
    return np.concatenate(
        (
            np.asarray(
                [
                    grasp["score"],
                    grasp["width"],
                    grasp["height"],
                    grasp["depth"],
                ],
                dtype=np.float64,
            ),
            pose[:3, :3].reshape(-1),
            pose[:3, 3],
            np.asarray([-1.0]),
        )
    )


def _output_path(manifest: dict[str, Any]) -> Path:
    output_dir = Path(manifest["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return output_dir / f"grasp_viz_{timestamp}_{manifest['request_id'][:8]}.png"


def _crop_white_background(path: Path, cv2) -> None:
    image = cv2.imread(str(path))
    if image is None:
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return
    x, y, width, height = cv2.boundingRect(np.concatenate(contours))
    cv2.imwrite(str(path), image[y : y + height, x : x + width])


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one GraspNet result manifest.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    manifest_path.unlink(missing_ok=True)
    visualize_manifest(manifest)


if __name__ == "__main__":
    main()
