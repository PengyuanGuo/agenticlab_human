"""FastAPI service for camera-frame GraspNet pose inference."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from PIL import Image

from agenticlab_human.perception.grasping.backend import (
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_CAMERA_NAME,
    DEFAULT_GRASPNET_CONFIG,
    GraspInferenceBackend,
    GraspNetInferenceBackend,
    MockGraspInferenceBackend,
)
from agenticlab_human.perception.grasping.contracts import (
    GraspHealthResponse,
    GraspPoseCandidate,
    GraspPredictRequest,
    GraspPredictResponse,
)


logger = logging.getLogger("uvicorn.error")
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VISUALIZATION_DIR = REPO_ROOT / "output" / "grasp_viz"


class GraspRuntime:
    """Keep model initialization and CUDA inference on one stable thread."""

    def __init__(self, backend: GraspInferenceBackend) -> None:
        self.backend = backend
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="graspnet")

    async def start(self) -> None:
        await self._call(self.backend.initialize)

    async def stop(self) -> None:
        try:
            await self._call(self.backend.shutdown)
        finally:
            self._executor.shutdown(wait=True)

    async def predict(self, **kwargs):
        return await self._call(self.backend.predict, **kwargs)

    async def health(self) -> tuple[bool, str]:
        return await self._call(self.backend.health)

    async def _call(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs),
        )


def create_app(
    backend: GraspInferenceBackend | None = None,
    *,
    visualize_seconds: float = 0.0,
    visualization_dir: str | Path = DEFAULT_VISUALIZATION_DIR,
) -> FastAPI:
    runtime = GraspRuntime(backend or MockGraspInferenceBackend())
    visualization_duration = max(0.0, float(visualize_seconds))
    visualization_output = Path(visualization_dir).expanduser().resolve()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await runtime.start()
            yield
        finally:
            await runtime.stop()

    app = FastAPI(
        title="AgenticLab GraspNet Server",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.grasp_runtime = runtime

    @app.get("/v1/health", response_model=GraspHealthResponse)
    async def health() -> GraspHealthResponse:
        ready, detail = await runtime.health()
        return GraspHealthResponse(
            status="ok" if ready else "degraded",
            backend=runtime.backend.name,
            model_loaded=ready,
            device=runtime.backend.device_name,
            detail=detail,
        )

    @app.post("/v1/grasp/predict", response_model=GraspPredictResponse)
    async def predict_grasp(
        request: GraspPredictRequest,
        background_tasks: BackgroundTasks,
    ) -> GraspPredictResponse:
        started_ns = time.perf_counter_ns()
        try:
            rgb, depth_mm, workspace_mask = _load_request_data(request)
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            candidates = await runtime.predict(
                rgb=rgb,
                depth_mm=depth_mm,
                workspace_mask=workspace_mask,
                max_grasps=request.max_grasps,
                score_threshold=request.score_threshold,
                collision_detection=request.collision_detection,
                nms=request.nms,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("GraspNet inference failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        grasps = [
            GraspPoseCandidate(
                pose_4x4=candidate.pose_4x4.tolist(),
                score=candidate.score,
                width=candidate.width,
                height=candidate.height,
                depth=candidate.depth,
                object_label=request.object_label,
                image_xy=list(candidate.image_xy) if candidate.image_xy is not None else None,
                metadata=candidate.metadata,
            )
            for candidate in candidates
        ]
        if grasps and visualization_duration > 0:
            camera_config_path = getattr(runtime.backend, "camera_config_path", None)
            camera_name = getattr(runtime.backend, "camera_name", None)
            if camera_config_path and camera_name:
                background_tasks.add_task(
                    _launch_visualization_subprocess,
                    {
                        "request_id": request.request_id,
                        "rgb_path": str(_required_file(request.rgb_path, "rgb_path")),
                        "depth_path": str(_required_file(request.depth_path, "depth_path")),
                        "workspace_mask_path": (
                            str(
                                _required_file(
                                    request.workspace_mask_path,
                                    "workspace_mask_path",
                                )
                            )
                            if request.workspace_mask_path
                            else None
                        ),
                        "bbox_xyxy": request.bbox_xyxy,
                        "mask_offset_px": request.mask_offset_px,
                        "camera_config_path": str(Path(camera_config_path).resolve()),
                        "camera_name": camera_name,
                        "grasps": [grasp.model_dump(mode="json") for grasp in grasps],
                        "visualize_seconds": visualization_duration,
                        "output_dir": str(visualization_output),
                    },
                )
            else:
                logger.warning(
                    "Grasp visualization skipped because the backend has no camera config"
                )
        duration_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        return GraspPredictResponse(
            request_id=request.request_id,
            success=bool(grasps),
            object_label=request.object_label,
            grasps=grasps,
            num_grasps=len(grasps),
            duration_ms=duration_ms,
            input_summary={
                "rgb_shape": list(rgb.shape),
                "depth_shape": list(depth_mm.shape),
                "depth_unit": "mm",
                "valid_depth_points": int(np.count_nonzero(depth_mm > 0)),
                "workspace_points": int(np.count_nonzero((depth_mm > 0) & workspace_mask)),
                "bbox_xyxy": request.bbox_xyxy,
                "camera_profile": getattr(runtime.backend, "camera_name", None),
            },
            error=None if grasps else "No grasp candidates found.",
        )

    return app


def _load_request_data(
    request: GraspPredictRequest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_path = _required_file(request.rgb_path, "rgb_path")
    depth_path = _required_file(request.depth_path, "depth_path")

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    if depth_path.suffix.lower() == ".npy":
        depth_mm = np.load(depth_path)
    else:
        depth_mm = np.asarray(Image.open(depth_path))
    depth_mm = np.asarray(depth_mm, dtype=np.float32)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image must have shape HxWx3, got {rgb.shape}")
    if depth_mm.ndim != 2:
        raise ValueError(f"depth image must have shape HxW, got {depth_mm.shape}")
    if rgb.shape[:2] != depth_mm.shape:
        raise ValueError(
            f"RGB/depth shape mismatch: rgb={rgb.shape[:2]} depth={depth_mm.shape}"
        )
    workspace_mask = _load_workspace_mask(request, shape=depth_mm.shape)
    return rgb, depth_mm, workspace_mask


def _load_workspace_mask(
    request: GraspPredictRequest,
    *,
    shape: tuple[int, int],
) -> np.ndarray:
    if request.workspace_mask_path:
        mask_path = _required_file(request.workspace_mask_path, "workspace_mask_path")
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape != shape:
            raise ValueError(
                f"workspace mask shape mismatch: mask={mask.shape} depth={shape}"
            )
        return mask

    height, width = shape
    if request.bbox_xyxy is None:
        return np.ones(shape, dtype=bool)

    x1, y1, x2, y2 = request.bbox_xyxy
    offset = request.mask_offset_px
    left = max(0, int(np.floor(x1)) - offset)
    top = max(0, int(np.floor(y1)) - offset)
    right = min(width, int(np.ceil(x2)) + offset)
    bottom = min(height, int(np.ceil(y2)) + offset)
    if right <= left or bottom <= top:
        raise ValueError("bbox does not overlap the image")

    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _required_file(value: str | None, field_name: str) -> Path:
    if not value:
        raise ValueError(f"{field_name} is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path


def _launch_visualization_subprocess(manifest: dict) -> None:
    try:
        output_dir = Path(manifest["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix=f"{manifest['request_id']}_",
            dir=output_dir,
            delete=False,
        ) as handle:
            json.dump(manifest, handle)
            manifest_path = Path(handle.name)

        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agenticlab_human.perception.grasping.visualizer",
                "--manifest",
                str(manifest_path),
            ],
            cwd=REPO_ROOT,
            close_fds=True,
        )
    except Exception:
        logger.exception("Failed to launch grasp visualization subprocess")


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AgenticLab GraspNet HTTP server.")
    parser.add_argument("--config", default=DEFAULT_GRASPNET_CONFIG)
    parser.add_argument("--camera-config", default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--visualize-seconds",
        type=float,
        default=0.0,
        help="Show each successful grasp result in a separate Open3D process.",
    )
    parser.add_argument(
        "--visualization-dir",
        default=str(DEFAULT_VISUALIZATION_DIR),
        help="Directory for grasp visualization screenshots.",
    )
    args = parser.parse_args()

    backend = GraspNetInferenceBackend(
        config_path=args.config,
        camera_config_path=args.camera_config,
        camera_name=args.camera_name,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    uvicorn.run(
        create_app(
            backend,
            visualize_seconds=args.visualize_seconds,
            visualization_dir=args.visualization_dir,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
