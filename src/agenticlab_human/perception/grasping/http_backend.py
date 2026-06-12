"""Execution-facing adapter for the external GraspNet HTTP service."""

from __future__ import annotations

from pathlib import Path

from agenticlab_human.perception.backend.perception_backend import BBox
from agenticlab_human.perception.grasping.client import GraspNetHTTPClient
from agenticlab_human.perception.grasping.grasp_backend import GraspCandidate


class GraspNetHTTPBackend:
    """Request object-level grasp candidates from saved RGB-D files."""

    def __init__(
        self,
        server_url: str,
        *,
        timeout_s: float | None = 120.0,
        mask_offset_px: int = 10,
        max_grasps: int = 20,
        score_threshold: float = 0.0,
        collision_detection: bool = True,
        nms: bool = True,
        client: GraspNetHTTPClient | None = None,
    ) -> None:
        self.server_url = str(server_url).rstrip("/")
        self.mask_offset_px = int(mask_offset_px)
        self.max_grasps = int(max_grasps)
        self.score_threshold = float(score_threshold)
        self.collision_detection = bool(collision_detection)
        self.nms = bool(nms)
        self._client = client or GraspNetHTTPClient(
            self.server_url,
            timeout_s=timeout_s,
        )
        self._owns_client = client is None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        health = self._client.health()
        if not health.model_loaded:
            raise RuntimeError(f"GraspNet server is not ready: {health.detail}")
        self._initialized = True

    def shutdown(self) -> None:
        if self._owns_client:
            self._client.close()
        self._initialized = False

    def plan_for_object(
        self,
        *,
        rgb_path: str | Path,
        depth_path: str | Path,
        bbox: BBox,
        object_name: str,
    ) -> list[GraspCandidate]:
        if not self._initialized:
            raise RuntimeError("GraspNetHTTPBackend is not initialized")
        response = self._client.predict_from_files(
            rgb_path=rgb_path,
            depth_path=depth_path,
            bbox_xyxy=[float(value) for value in bbox.xyxy],
            object_label=object_name,
            mask_offset_px=self.mask_offset_px,
            max_grasps=self.max_grasps,
            score_threshold=self.score_threshold,
            collision_detection=self.collision_detection,
            nms=self.nms,
        )
        if not response.success:
            raise RuntimeError(response.error or "GraspNet prediction failed")
        return self._client.to_grasp_candidates(response)
