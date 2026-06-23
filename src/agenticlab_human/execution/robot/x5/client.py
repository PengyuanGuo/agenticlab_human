"""Synchronous HTTP client for the AgenticLab X5 control server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import requests
from PIL import Image

from agenticlab_human.execution.robot.x5.contracts import (
    GetStateCommand,
    HealthResponse,
    MoveJPointCommand,
    MoveJointsCommand,
    MoveLPointCommand,
    RGBDFrame,
    RGBD_MEDIA_TYPE,
    RobotCommand,
    RobotCommandRequest,
    RobotCommandResponse,
    SetGripperCommand,
    StopCommand,
    decode_rgbd_frame,
)
from agenticlab_human.execution.robot.x5.conversion import (
    tcp_pose_xyzw_to_xyz_rotvec,
)


class HTTPSession(Protocol):
    def get(self, url: str, **kwargs: Any): ...

    def post(self, url: str, **kwargs: Any): ...


class X5HTTPClient:
    """Request one RGB-D frame or execute one discrete robot command."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float | None = 30.0,
        session: HTTPSession | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = None if timeout_s is None else float(timeout_s)
        self._owns_session = session is None
        self._session = session or requests.Session()

    def health(self) -> HealthResponse:
        response = self._session.get(
            self._url("/v1/health"),
            **self._request_options(),
        )
        response.raise_for_status()
        return HealthResponse.model_validate(response.json())

    def capture_rgbd(self) -> RGBDFrame:
        response = self._session.post(
            self._url("/v1/camera/capture"),
            **self._request_options(),
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";", 1)[0]
        if content_type != RGBD_MEDIA_TYPE:
            raise ValueError(f"unexpected RGB-D content type: {content_type or '<missing>'}")
        return decode_rgbd_frame(response.content)

    def send_command(
        self,
        command: RobotCommand,
        *,
        request_id: str | None = None,
    ) -> RobotCommandResponse:
        request = RobotCommandRequest(
            command=command,
            **({"request_id": request_id} if request_id else {}),
        )
        response = self._session.post(
            self._url("/v1/robot/command"),
            json=request.model_dump(mode="json"),
            **self._request_options(),
        )
        if response.status_code == 400:
            return RobotCommandResponse.model_validate(response.json())
        response.raise_for_status()
        return RobotCommandResponse.model_validate(response.json())

    def get_state(self, arm: str = "all") -> RobotCommandResponse:
        return self.send_command(GetStateCommand(arm=arm))

    def move_joints(
        self,
        arm: str,
        joints_rad: list[float],
        *,
        torso_joints_deg: list[float] | None = None,
        speed_ratio: float = 0.1,
        wait: bool = True,
        request_id: str | None = None,
    ) -> RobotCommandResponse:
        return self.send_command(
            MoveJointsCommand(
                arm=arm,
                joints_rad=joints_rad,
                torso_joints_deg=torso_joints_deg,
                speed_ratio=speed_ratio,
                wait=wait,
            ),
            request_id=request_id,
        )

    def movej_point(
        self,
        arm: str,
        tcp_pose_xyz_rotvec: list[float],
        *,
        speed_ratio: float = 0.05,
        wait: bool = True,
        request_id: str | None = None,
    ) -> RobotCommandResponse:
        """Move the configured TCP to a world-frame target using joint motion."""

        return self.send_command(
            MoveJPointCommand(
                arm=arm,
                tcp_pose_xyz_rotvec=tcp_pose_xyz_rotvec,
                speed_ratio=speed_ratio,
                wait=wait,
            ),
            request_id=request_id,
        )

    def movel_point(
        self,
        arm: str,
        tcp_pose_xyz_rotvec: list[float],
        *,
        speed_ratio: float = 0.05,
        wait: bool = True,
        request_id: str | None = None,
    ) -> RobotCommandResponse:
        """Move the configured TCP to a world-frame target in a straight line."""

        return self.send_command(
            MoveLPointCommand(
                arm=arm,
                tcp_pose_xyz_rotvec=tcp_pose_xyz_rotvec,
                speed_ratio=speed_ratio,
                wait=wait,
            ),
            request_id=request_id,
        )

    def stop(self, arm: str = "all") -> RobotCommandResponse:
        return self.send_command(StopCommand(arm=arm))

    def set_gripper(
        self,
        position: float,
        *,
        arm: str = "left",
        wait: bool = True,
        request_id: str | None = None,
    ) -> RobotCommandResponse:
        """Set one gripper position: 0.0 closed, 1.0 fully open."""

        return self.send_command(
            SetGripperCommand(arm=arm, position=position, wait=wait),
            request_id=request_id,
        )

    def close_gripper(
        self,
        *,
        arm: str = "left",
        wait: bool = True,
        request_id: str | None = None,
    ) -> RobotCommandResponse:
        return self.set_gripper(0.0, arm=arm, wait=wait, request_id=request_id)

    def open_gripper(
        self,
        *,
        arm: str = "left",
        wait: bool = True,
        request_id: str | None = None,
    ) -> RobotCommandResponse:
        return self.set_gripper(1.0, arm=arm, wait=wait, request_id=request_id)

    def close(self) -> None:
        if self._owns_session and hasattr(self._session, "close"):
            self._session.close()

    def __enter__(self) -> "X5HTTPClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request_options(self) -> dict[str, float]:
        if self.timeout_s is None:
            return {}
        return {"timeout": self.timeout_s}


def save_rgbd_frame(frame: RGBDFrame, save_dir: str | Path) -> dict[str, Path]:
    """Save an HTTP-delivered RGB-D frame without requiring the Orbbec SDK."""

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = frame.frame_id.replace("/", "_").replace("\\", "_")

    rgb_path = output_dir / f"{stem}_rgb.png"
    depth_png_path = output_dir / f"{stem}_depth_mm.png"
    depth_npy_path = output_dir / f"{stem}_depth_mm.npy"
    depth_preview_path = output_dir / f"{stem}_depth_preview.png"
    metadata_path = output_dir / f"{stem}_metadata.json"

    Image.fromarray(frame.rgb).save(rgb_path)
    Image.fromarray(np.clip(frame.depth_mm, 0, 65535).astype(np.uint16)).save(depth_png_path)
    np.save(depth_npy_path, frame.depth_mm)
    Image.fromarray(_normalize_depth_for_preview(frame.depth_mm)).save(depth_preview_path)
    metadata_path.write_text(
        json.dumps(
            {
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "color_timestamp_ns": frame.color_timestamp_ns,
                "depth_timestamp_ns": frame.depth_timestamp_ns,
                "sync_delta_ms": frame.sync_delta_ms,
                "depth_unit": frame.depth_unit,
                "intrinsics": frame.intrinsics.model_dump(),
                "rgb_shape": list(frame.rgb.shape),
                "depth_shape": list(frame.depth_mm.shape),
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "rgb": rgb_path,
        "depth_png": depth_png_path,
        "depth_npy": depth_npy_path,
        "depth_preview": depth_preview_path,
        "metadata": metadata_path,
    }


def display_rgbd_frame(frame: RGBDFrame) -> None:
    """Display a received RGB-D frame until a key is pressed."""

    import cv2

    color_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    depth_u8 = _normalize_depth_for_preview(frame.depth_mm)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    combined = np.hstack([color_bgr, depth_color])

    max_width = 1600
    if combined.shape[1] > max_width:
        ratio = max_width / combined.shape[1]
        combined = cv2.resize(
            combined,
            (max_width, int(round(combined.shape[0] * ratio))),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imshow("X5 HTTP RGB-D: RGB | Depth - press any key to close", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _normalize_depth_for_preview(depth_mm: np.ndarray) -> np.ndarray:
    valid_depth = depth_mm[depth_mm > 0]
    if not valid_depth.size:
        return np.zeros(depth_mm.shape, dtype=np.uint8)
    near, far = np.percentile(valid_depth, [2, 98])
    scale = 255.0 / max(float(far - near), 1.0)
    preview = np.clip((depth_mm - near) * scale, 0, 255).astype(np.uint8)
    preview[depth_mm <= 0] = 0
    return preview

def _run_move_home(robot_config_path: str, camera_config_path: str, args: argparse.Namespace) -> int:
    """Temporary CLI helper: move the configured arm to home_joints_deg."""

    from agenticlab_human.execution.robot.x5.x5_remote_backend import (
        RemoteX5ActionBackend,
    )
    
    backend = RemoteX5ActionBackend(robot_config_path=robot_config_path, camera_config_path=camera_config_path, server_url=args.server_url)
    backend.initialize()
    try:
        result = backend.move_home()
    finally:
        backend.shutdown()
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.success else 1


def _run_capture(args: argparse.Namespace) -> int:
    with X5HTTPClient(args.server_url, timeout_s=args.timeout) as client:
        health = client.health()
        frame = client.capture_rgbd()

    saved = save_rgbd_frame(frame, args.save_dir)
    print(f"Server health: {health.status} camera={health.camera.backend} robot={health.robot.backend}")
    print(
        f"Received {frame.frame_id}: rgb={frame.rgb.shape} {frame.rgb.dtype}, "
        f"depth={frame.depth_mm.shape} {frame.depth_mm.dtype} {frame.depth_unit}"
    )
    print(f"Intrinsics: {frame.intrinsics.model_dump()}")
    print(f"RGB/depth sync delta: {frame.sync_delta_ms} ms")
    for name, path in saved.items():
        print(f"Saved {name}: {path}")

    if args.preview:
        display_rgbd_frame(frame)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="X5 HTTP client utilities (capture RGB-D or move home).",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--home", action="store_true", help="Move to home_joints_deg via RemoteX5ActionBackend.")
    parser.add_argument("--save-dir", default="output/x5_http_captures")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)

    if args.home:
        robot_config_path = "configs/robot/x5_config.yaml"
        camera_config_path = "configs/perception/camera_config.yaml"
        return _run_move_home(robot_config_path, camera_config_path, args)
    return _run_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
