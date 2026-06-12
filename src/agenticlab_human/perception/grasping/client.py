"""Synchronous HTTP client for the external GraspNet service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

import requests

from agenticlab_human.perception.grasping.contracts import (
    GraspHealthResponse,
    GraspPredictRequest,
    GraspPredictResponse,
)
from agenticlab_human.perception.grasping.grasp_backend import GraspCandidate


class HTTPSession(Protocol):
    def get(self, url: str, **kwargs: Any): ...

    def post(self, url: str, **kwargs: Any): ...


class GraspNetHTTPClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float | None = 120.0,
        session: HTTPSession | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = None if timeout_s is None else float(timeout_s)
        self._owns_session = session is None
        self._session = session or requests.Session()

    def health(self) -> GraspHealthResponse:
        response = self._session.get(
            self._url("/v1/health"),
            **self._request_options(),
        )
        response.raise_for_status()
        return GraspHealthResponse.model_validate(response.json())

    def predict(self, request: GraspPredictRequest) -> GraspPredictResponse:
        response = self._session.post(
            self._url("/v1/grasp/predict"),
            json=request.model_dump(mode="json"),
            **self._request_options(),
        )
        response.raise_for_status()
        return GraspPredictResponse.model_validate(response.json())

    def predict_from_files(
        self,
        *,
        rgb_path: str | Path,
        depth_path: str | Path | None = None,
        workspace_mask_path: str | Path | None = None,
        bbox_xyxy: list[float] | None = None,
        object_label: str = "object",
        mask_offset_px: int = 10,
        max_grasps: int = 20,
        score_threshold: float = 0.0,
        collision_detection: bool = True,
        nms: bool = True,
    ) -> GraspPredictResponse:
        paths = infer_capture_paths(rgb_path)
        request = GraspPredictRequest(
            rgb_path=str(Path(rgb_path).expanduser().resolve()),
            depth_path=str(Path(depth_path or paths["depth"]).expanduser().resolve()),
            workspace_mask_path=(
                str(Path(workspace_mask_path).expanduser().resolve())
                if workspace_mask_path is not None
                else None
            ),
            bbox_xyxy=bbox_xyxy,
            object_label=object_label,
            mask_offset_px=mask_offset_px,
            max_grasps=max_grasps,
            score_threshold=score_threshold,
            collision_detection=collision_detection,
            nms=nms,
        )
        return self.predict(request)

    @staticmethod
    def to_grasp_candidates(response: GraspPredictResponse) -> list[GraspCandidate]:
        return [
            GraspCandidate(
                pose=grasp.pose_4x4,
                score=grasp.score,
                image_xy=tuple(grasp.image_xy) if grasp.image_xy is not None else None,
                object_name=grasp.object_label,
                metadata={
                    "frame": response.pose_frame,
                    "width": grasp.width,
                    "height": grasp.height,
                    "depth": grasp.depth,
                    **grasp.metadata,
                },
            )
            for grasp in response.grasps
        ]

    def close(self) -> None:
        if self._owns_session and hasattr(self._session, "close"):
            self._session.close()

    def __enter__(self) -> "GraspNetHTTPClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request_options(self) -> dict[str, float]:
        if self.timeout_s is None:
            return {}
        return {"timeout": self.timeout_s}


def infer_capture_paths(rgb_path: str | Path) -> dict[str, Path]:
    rgb = Path(rgb_path).expanduser().resolve()
    suffix = "_rgb.png"
    if not rgb.name.endswith(suffix):
        raise ValueError(f"RGB filename must end with {suffix!r}: {rgb}")
    stem = rgb.name[: -len(suffix)]
    return {
        "rgb": rgb,
        "depth": rgb.with_name(f"{stem}_depth_mm.npy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Request GraspNet inference from local RGB-D files.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8010")
    parser.add_argument("--rgb-path", required=True)
    parser.add_argument("--depth-path")
    parser.add_argument("--workspace-mask-path")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--object-label", default="object")
    parser.add_argument("--mask-offset-px", type=int, default=10)
    parser.add_argument("--max-grasps", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--no-collision-detection", action="store_true")
    parser.add_argument("--no-nms", action="store_true")
    args = parser.parse_args()

    with GraspNetHTTPClient(args.server_url) as client:
        health = client.health()
        if not health.model_loaded:
            raise RuntimeError(f"GraspNet server is not ready: {health.detail}")
        result = client.predict_from_files(
            rgb_path=args.rgb_path,
            depth_path=args.depth_path,
            workspace_mask_path=args.workspace_mask_path,
            bbox_xyxy=args.bbox,
            object_label=args.object_label,
            mask_offset_px=args.mask_offset_px,
            max_grasps=args.max_grasps,
            score_threshold=args.score_threshold,
            collision_detection=not args.no_collision_detection,
            nms=not args.no_nms,
        )
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
