"""Thin GraspNet inference wrapper used by the external HTTP service."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
GRASPNESS_DIR = THIRD_PARTY_DIR / "graspness_unofficial"

for path in (THIRD_PARTY_DIR, GRASPNESS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from graspnetAPI import GraspGroup  # noqa: E402
from graspness_unofficial.dataset.graspnet_dataset import (  # noqa: E402
    CameraInfo,
    create_point_cloud_from_depth_image,
    minkowski_collate_fn,
)
from graspness_unofficial.models.graspnet import (  # noqa: E402
    GraspNet,
    NoGraspablePointsError,
    pred_decode,
)
from graspness_unofficial.utils.collision_detector import ModelFreeCollisionDetector  # noqa: E402


logger = logging.getLogger(__name__)


class GraspNetWrapper:
    """Load one GraspNet model and return camera-frame grasp candidates."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        camera_config: dict[str, Any],
        camera_name: str = "Gemini335",
        checkpoint_path: str | None = None,
        device: str = "cuda:0",
        grasp_max_width: float | None = None,
    ) -> None:
        try:
            self.cfg = dict(config["Graspnet"])
        except KeyError as exc:
            raise ValueError("grasp config must contain a 'Graspnet' section") from exc

        try:
            camera_profile = camera_config[camera_name]
        except KeyError as exc:
            raise ValueError(f"camera config not found: {camera_name}") from exc
        self.camera_name = camera_name
        self.camera_profile = dict(camera_profile)
        self.camera_intrinsics = np.asarray(
            camera_profile["intrinsic_matrix"],
            dtype=np.float32,
        )
        if self.camera_intrinsics.shape != (3, 3):
            raise ValueError(f"{camera_name}.intrinsic_matrix must be 3x3")
        self.camera_width = int(camera_profile["width"])
        self.camera_height = int(camera_profile["height"])
        self.factor_depth = float(camera_profile.get("factor_depth", 1000.0))
        self.base_from_camera_rotation = np.asarray(
            camera_profile.get("handeye_calibration", {}).get(
                "rotation",
                np.eye(3),
            ),
            dtype=np.float64,
        )
        if self.base_from_camera_rotation.shape != (3, 3):
            raise ValueError(
                f"{camera_name}.handeye_calibration.rotation must be 3x3"
            )

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")

        self.grasp_max_width = float(
            grasp_max_width
            if grasp_max_width is not None
            else self.cfg.get("grasp_max_width", 0.1)
        )
        resolved_checkpoint = self._resolve_checkpoint(checkpoint_path)
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(f"GraspNet checkpoint not found: {resolved_checkpoint}")

        logger.info("Loading GraspNet checkpoint from %s onto %s", resolved_checkpoint, self.device)
        self.net = GraspNet(
            is_training=False,
            seed_feat_dim=int(self.cfg.get("seed_feat_dim", 512)),
            grasp_max_width=self.grasp_max_width,
        )
        checkpoint = torch.load(resolved_checkpoint, map_location="cpu", weights_only=False)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.to(self.device).eval()
        logger.info("GraspNet model loaded")

    def _resolve_checkpoint(self, checkpoint_path: str | None) -> Path:
        selected = checkpoint_path or self.cfg.get("checkpoint_path_realsense")
        if not selected:
            raise ValueError("checkpoint_path is required")
        path = Path(selected).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path

    def prepare_input(
        self,
        *,
        color_data: np.ndarray,
        depth_mm: np.ndarray,
        workspace_mask: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        color = np.asarray(color_data)
        depth = np.asarray(depth_mm)

        if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
            raise ValueError("color_data must be an HxWx3 uint8 RGB array")
        if depth.ndim != 2:
            raise ValueError("depth_mm must be an HxW array")
        if color.shape[:2] != depth.shape:
            raise ValueError(
                f"RGB/depth shape mismatch: rgb={color.shape[:2]} depth={depth.shape}"
            )
        height, width = depth.shape
        if (width, height) != (self.camera_width, self.camera_height):
            raise ValueError(
                f"RGB-D shape {(width, height)} does not match camera config "
                f"{self.camera_name} {(self.camera_width, self.camera_height)}"
            )
        camera = CameraInfo(
            self.camera_width,
            self.camera_height,
            float(self.camera_intrinsics[0, 0]),
            float(self.camera_intrinsics[1, 1]),
            float(self.camera_intrinsics[0, 2]),
            float(self.camera_intrinsics[1, 2]),
            self.factor_depth,
        )
        cloud = create_point_cloud_from_depth_image(
            depth.astype(np.float32, copy=False),
            camera,
            organized=True,
        )

        if workspace_mask is None:
            workspace = np.ones((height, width), dtype=bool)
        else:
            workspace = np.asarray(workspace_mask).astype(bool)
            if workspace.shape != depth.shape:
                raise ValueError(
                    f"workspace mask shape mismatch: mask={workspace.shape} depth={depth.shape}"
                )

        valid_mask = workspace & np.isfinite(depth) & (depth > 0)
        cloud_masked = cloud[valid_mask]
        if not len(cloud_masked):
            raise ValueError("workspace mask contains no valid depth points")

        num_points = int(self.cfg.get("num_point", 15000))
        if len(cloud_masked) >= num_points:
            indices = np.random.choice(len(cloud_masked), num_points, replace=False)
        else:
            original = np.arange(len(cloud_masked))
            repeated = np.random.choice(
                len(cloud_masked),
                num_points - len(cloud_masked),
                replace=True,
            )
            indices = np.concatenate([original, repeated])
        cloud_sampled = cloud_masked[indices].astype(np.float32)

        model_input = {
            "point_clouds": cloud_sampled,
            "coors": cloud_sampled / float(self.cfg.get("voxel_size", 0.005)),
            "feats": np.ones_like(cloud_sampled, dtype=np.float32),
        }
        return model_input, cloud_masked

    def infer(self, model_input: dict[str, np.ndarray]) -> GraspGroup:
        batch_data = minkowski_collate_fn([model_input])
        for key, value in batch_data.items():
            if "list" in key:
                for row in value:
                    for index, item in enumerate(row):
                        row[index] = item.to(self.device)
            else:
                batch_data[key] = value.to(self.device)

        with torch.inference_mode():
            try:
                end_points = self.net(batch_data)
            except NoGraspablePointsError as exc:
                logger.info("%s", exc)
                return GraspGroup()
            predictions = pred_decode(
                end_points,
                grasp_max_width=self.grasp_max_width,
            )
        return GraspGroup(predictions[0].detach().cpu().numpy())

    def collision_filter(
        self,
        grasps: GraspGroup,
        cloud_points: np.ndarray,
    ) -> GraspGroup:
        detector = ModelFreeCollisionDetector(
            cloud_points,
            voxel_size=float(self.cfg.get("voxel_size_cd", 0.01)),
        )
        collision_mask = detector.detect(
            grasps,
            approach_dist=float(self.cfg.get("approach_dist", 0.05)),
            collision_thresh=float(self.cfg.get("collision_thresh", 0.01)),
        )
        return grasps[~collision_mask]

    def filter_grasps(
        self,
        grasps: GraspGroup,
        angle_range: tuple[float, float] | None = None,
        cur_eef_pose: np.ndarray | None = None,
    ) -> GraspGroup:
        """Keep X5 grasps whose approach axis is near robot-base +X."""

        if not len(grasps):
            return GraspGroup()
        if angle_range is None:
            angle_threshold = float(self.cfg.get("angle_threshold", 10))
            angle_range = (0.0, angle_threshold)

        min_angle, max_angle = (float(value) for value in angle_range)
        if not 0.0 <= min_angle <= max_angle <= 180.0:
            raise ValueError("angle_range must satisfy 0 <= min <= max <= 180")

        base_from_camera = self.base_from_camera_rotation
        if cur_eef_pose is not None:
            eef_pose = np.asarray(cur_eef_pose, dtype=np.float64)
            if eef_pose.shape != (4, 4):
                raise ValueError("cur_eef_pose must be a 4x4 transform")
            base_from_camera = eef_pose[:3, :3] @ base_from_camera

        range_center = (min_angle + max_angle) / 2.0
        evaluated = []
        filtered = []
        for grasp in list(grasps[:50]):
            angle_deg = self._approach_angle_deg(
                grasp.rotation_matrix,
                base_from_camera,
            )
            evaluated.append((grasp, angle_deg, abs(angle_deg - range_center)))
            if min_angle <= angle_deg <= max_angle:
                filtered.append((grasp, angle_deg))

        if filtered:
            logger.info(
                "X5 angle filter kept %d/%d grasps within [%.1f, %.1f] degrees",
                len(filtered),
                len(evaluated),
                min_angle,
                max_angle,
            )
            return GraspGroup(
                np.asarray([grasp.grasp_array for grasp, _ in filtered])
            )

        best_grasp, best_angle, _ = min(evaluated, key=lambda item: item[2])
        logger.warning(
            "No grasp within X5 angle range [%.1f, %.1f] degrees; "
            "using closest grasp at %.2f degrees",
            min_angle,
            max_angle,
            best_angle,
        )
        return GraspGroup(np.asarray([best_grasp.grasp_array]))

    @staticmethod
    def _approach_angle_deg(
        camera_from_grasp_rotation: np.ndarray,
        base_from_camera_rotation: np.ndarray,
    ) -> float:
        base_from_grasp = (
            np.asarray(base_from_camera_rotation, dtype=np.float64)
            @ np.asarray(camera_from_grasp_rotation, dtype=np.float64)
        )
        approach_base = base_from_grasp[:, 0]
        norm = np.linalg.norm(approach_base)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("grasp approach axis must be finite and non-zero")
        cosine = np.clip(approach_base[0] / norm, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))


    def predict_candidates(
        self,
        *,
        color_data: np.ndarray,
        depth_mm: np.ndarray,
        workspace_mask: np.ndarray | None = None,
        max_grasps: int = 20,
        score_threshold: float = 0.0,
        collision_detection: bool = True,
        nms: bool = True,
    ) -> list[dict[str, Any]]:
        model_input, cloud_masked = self.prepare_input(
            color_data=color_data,
            depth_mm=depth_mm,
            workspace_mask=workspace_mask,
        )
        grasps = self.infer(model_input)
        logger.info("GraspNet produced %d raw candidates", len(grasps))

        if collision_detection and float(self.cfg.get("collision_thresh", 0.01)) > 0:
            grasps = self.collision_filter(grasps, cloud_masked)
            logger.info("%d candidates remain after collision filtering", len(grasps))
        if not len(grasps):
            return []

        if nms:
            grasps = grasps.nms()
        if score_threshold > 0:
            grasps = GraspGroup(
                grasps.grasp_group_array[grasps.scores >= float(score_threshold)]
            )
        if not len(grasps):
            return []

        grasps.sort_by_score()
        grasps = self.filter_grasps(grasps)
        selected = grasps[: min(int(max_grasps), len(grasps))]
        return [
            self._serialize_grasp(grasp)
            for grasp in selected
        ]

    def _serialize_grasp(self, grasp) -> dict[str, Any]:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.asarray(grasp.rotation_matrix, dtype=np.float32)
        pose[:3, 3] = np.asarray(grasp.translation, dtype=np.float32)

        x, y, z = pose[:3, 3]
        image_xy = None
        if z > 0:
            u = self.camera_intrinsics[0, 0] * x / z + self.camera_intrinsics[0, 2]
            v = self.camera_intrinsics[1, 1] * y / z + self.camera_intrinsics[1, 2]
            image_xy = (float(u), float(v))

        return {
            "pose_4x4": pose,
            "score": float(grasp.score),
            "width": float(grasp.width),
            "height": float(grasp.height),
            "depth": float(grasp.depth),
            "image_xy": image_xy,
            "metadata": {"source": "graspnet"},
        }

    def run_grasp_inference(
        self,
        color_data: np.ndarray,
        depth_data: np.ndarray,
        workspace_mask: np.ndarray | None = None,
    ):
        """Compatibility helper returning the best translation/rotation/width."""

        candidates = self.predict_candidates(
            color_data=color_data,
            depth_mm=depth_data,
            workspace_mask=workspace_mask,
            max_grasps=1,
        )
        if not candidates:
            return None, None, None
        pose = candidates[0]["pose_4x4"]
        return pose[:3, 3], pose[:3, :3], candidates[0]["width"]

    def close(self) -> None:
        net, self.net = getattr(self, "net", None), None
        if net is not None:
            net.cpu()
            del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
