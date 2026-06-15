"""FastAPI service for one-shot RGB-D capture and discrete X5 commands."""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

from agenticlab_human.execution.robot.x5.camera import (
    OrbbecRGBDCamera,
    RGBDCamera,
)
from agenticlab_human.execution.robot.x5.contracts import (
    ComponentHealth,
    HealthResponse,
    RGBD_MEDIA_TYPE,
    RobotCommandRequest,
    RobotCommandResponse,
    RobotState,
    SetGripperCommand,
    encode_rgbd_frame,
)
from agenticlab_human.execution.robot.x5.gripper_controller import (
    GripperService,
    MockGripperService,
    X5Gripper,
)
from agenticlab_human.execution.robot.x5.conversion import degrees_to_radians
from agenticlab_human.execution.robot.x5.mock_controller import MockX5Controller
from agenticlab_human.execution.robot.x5.x5_controller import (
    RealX5Controller,
    X5Controller,
)


DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parents[5] / "configs" / "robot" / "x5_config.yaml"
)
logger = logging.getLogger("uvicorn.error")


class HardwareRuntime:
    """Keep each hardware SDK on its own stable worker thread."""

    def __init__(
        self,
        camera: RGBDCamera,
        controller: X5Controller,
        gripper: X5Gripper,
    ) -> None:
        self.camera = camera
        self.controller = controller
        self.gripper = gripper
        self._camera_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="x5-camera",
        )
        self._robot_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="x5-robot",
        )
        self._gripper_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="x5-gripper",
        )

    async def start(self) -> None:
        await asyncio.gather(
            self._camera_call(self.camera.initialize),
            self._robot_call(self.controller.initialize),
            self._gripper_call(self.gripper.initialize),
        )

    async def stop(self) -> None:
        await asyncio.gather(
            self._camera_call(self.camera.shutdown),
            self._robot_call(self.controller.shutdown),
            self._gripper_call(self.gripper.shutdown),
            return_exceptions=True,
        )
        self._camera_executor.shutdown(wait=True)
        self._robot_executor.shutdown(wait=True)
        self._gripper_executor.shutdown(wait=True)

    async def capture(self):
        return await self._camera_call(self.camera.capture)

    async def camera_health(self):
        return await self._camera_call(self.camera.health)

    async def robot_health(self):
        arm_health, gripper_health = await asyncio.gather(
            self._robot_call(self.controller.health),
            self._gripper_call(self.gripper.health),
        )
        return ComponentHealth(
            ready=arm_health.ready and gripper_health.ready,
            backend=arm_health.backend,
            detail=(
                f"arm: {arm_health.detail}; "
                f"gripper({gripper_health.backend}): {gripper_health.detail}"
            ),
        )

    async def robot_state(self) -> RobotState:
        arm_state, gripper_state = await asyncio.gather(
            self._robot_call(self.controller.get_state),
            self._gripper_call(self.gripper.get_state),
        )
        return RobotState(
            arms=arm_state.arms,
            gripper=gripper_state,
            timestamp_ns=time.time_ns(),
        )

    async def execute(self, command):
        if isinstance(command, SetGripperCommand):
            return await self._gripper_call(self.gripper.execute, command)
        return await self._robot_call(self.controller.execute, command)

    async def _camera_call(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._camera_executor, func, *args)

    async def _robot_call(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._robot_executor, func, *args)

    async def _gripper_call(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._gripper_executor, func, *args)


def create_app(
    camera: RGBDCamera,
    controller: X5Controller | None = None,
    gripper: X5Gripper | None = None,
) -> FastAPI:
    """Create the HTTP app around an explicit camera implementation."""

    runtime = HardwareRuntime(
        camera=camera,
        controller=controller or MockX5Controller(),
        gripper=gripper or MockGripperService(),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await runtime.start()
            yield
        finally:
            await runtime.stop()

    app = FastAPI(
        title="AgenticLab X5 Control Server",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.hardware = runtime

    @app.get("/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        camera_health, robot_health = await asyncio.gather(
            runtime.camera_health(),
            runtime.robot_health(),
        )
        status = "ok" if camera_health.ready and robot_health.ready else "degraded"
        return HealthResponse(
            status=status,
            camera=camera_health,
            robot=robot_health,
        )

    @app.post(
        "/v1/camera/capture",
        response_class=Response,
        responses={200: {"content": {RGBD_MEDIA_TYPE: {}}}},
    )
    async def capture_rgbd() -> Response:
        frame = await runtime.capture()
        return Response(
            content=encode_rgbd_frame(frame),
            media_type=RGBD_MEDIA_TYPE,
            headers={
                "X-Frame-Id": frame.frame_id,
                "X-Timestamp-Ns": str(frame.timestamp_ns),
            },
        )

    @app.post("/v1/robot/command", response_model=RobotCommandResponse)
    async def robot_command(request: RobotCommandRequest):
        started_ns = time.perf_counter_ns()
        state_before = await runtime.robot_state()
        accepted_command = request.command.model_dump(mode="json")
        logger.info(
            "X5 command received: class=%s module=%s payload=%s",
            type(request.command).__name__,
            type(request.command).__module__,
            accepted_command,
        )
        try:
            await runtime.execute(request.command)
            state_after = await runtime.robot_state()
            return RobotCommandResponse(
                request_id=request.request_id,
                success=True,
                accepted_command=accepted_command,
                state_before=state_before,
                state_after=state_after,
                server_timestamp_ns=time.time_ns(),
                duration_ms=(time.perf_counter_ns() - started_ns) / 1_000_000.0,
            )
        except (RuntimeError, ValueError) as exc:
            state_after = await runtime.robot_state()
            response = RobotCommandResponse(
                request_id=request.request_id,
                success=False,
                accepted_command=accepted_command,
                state_before=state_before,
                state_after=state_after,
                server_timestamp_ns=time.time_ns(),
                duration_ms=(time.perf_counter_ns() - started_ns) / 1_000_000.0,
                error=str(exc),
            )
            return JSONResponse(status_code=400, content=response.model_dump(mode="json"))

    return app


def create_app_from_config(config_path: str = DEFAULT_CONFIG_PATH) -> tuple[FastAPI, dict[str, Any]]:
    config = yaml.safe_load(Path(config_path).read_text()) or {}
    camera_config = config.get("camera", {})
    robot_config = config.get("robot", {})

    camera_backend = camera_config.get("backend", "orbbec")
    if camera_backend != "orbbec":
        raise ValueError(f"unsupported camera backend: {camera_backend}")
    camera: RGBDCamera = OrbbecRGBDCamera(
        which_cam=str(camera_config.get("which_cam", "Orbbec")),
        color_size=(
            int(camera_config.get("width", 1280)),
            int(camera_config.get("height", 720)),
        ),
        color_fps=int(camera_config.get("fps", 30)),
        timeout_ms=int(camera_config.get("timeout_ms", 1000)),
        max_capture_attempts=int(camera_config.get("max_capture_attempts", 30)),
        max_sync_delta_ms=float(camera_config.get("max_sync_delta_ms", 20.0)),
    )

    robot_backend = robot_config.get("backend", "mock")
    if robot_backend == "mock":
        controller = MockX5Controller(
            arms=robot_config.get("arms", ["left", "right"]),
            initial_joints_rad=_mock_initial_joints_rad(robot_config),
        )
        gripper: X5Gripper = MockGripperService()
    elif robot_backend == "x5":
        controller = RealX5Controller(
            arm_configs=_build_x5_arm_configs(robot_config),
            mode=int(robot_config.get("mode", 100)),
            remote=bool(robot_config.get("remote", False)),
            enable_servo_on_start=bool(robot_config.get("enable_servo_on_start", True)),
            max_command_speed_ratio=float(robot_config.get("max_command_speed_ratio", 0.1)),
            move_timeout_ms=int(robot_config.get("move_timeout_ms", 60_000)),
            max_joint_delta_deg=float(robot_config.get("max_joint_delta_deg", 5.0)),
            joint_limits_deg=robot_config.get("joint_limits_deg"),
            stop_on_shutdown=bool(robot_config.get("stop_on_shutdown", False)),
        )
        gripper_config = robot_config.get("gripper", {})
        if not gripper_config.get("enabled", False):
            raise ValueError("robot.gripper.enabled must be true for X5 deployment")
        gripper = GripperService(gripper_config)
    else:
        raise ValueError(f"unsupported robot backend: {robot_backend}")
    return (
        create_app(camera=camera, controller=controller, gripper=gripper),
        config.get("server", {}),
    )


def _mock_initial_joints_rad(robot_config: dict[str, Any]) -> list[float]:
    if "initial_joints_rad" in robot_config:
        values = [float(value) for value in robot_config["initial_joints_rad"]]
    elif "initial_joints_deg" in robot_config:
        values = degrees_to_radians(robot_config["initial_joints_deg"])
    else:
        values = [0.0] * 7
    if len(values) < 7:
        raise ValueError("mock initial joints must contain at least 7 values")
    return values[:7]


def _build_x5_arm_configs(robot_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    arms = robot_config.get("arms", ["left"])
    arm_configs: dict[str, dict[str, Any]] = {}
    for arm in arms:
        arm_name = str(arm)
        arm_config = dict(robot_config.get(arm_name, {}))
        for key in (
            "robot_ip",
            "head_joints_deg",
            "tf_no",
            "tool_frame",
        ):
            if key not in arm_config and key in robot_config:
                arm_config[key] = robot_config[key]
        if "head_joints_deg" in arm_config:
            arm_config["head_joints_deg"] = _first_two_values(arm_config["head_joints_deg"])
        arm_configs[arm_name] = arm_config
    return arm_configs


def _first_two_values(values: Any) -> list[float]:
    converted = [float(value) for value in values]
    if len(converted) < 2:
        raise ValueError("expected at least 2 head joint values")
    return converted[:2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AgenticLab X5 HTTP server.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    configured_app, server_config = create_app_from_config(args.config)
    uvicorn.run(
        configured_app,
        host=args.host or server_config.get("host", "0.0.0.0"),
        port=args.port or int(server_config.get("port", 8000)),
    )


if __name__ == "__main__":
    main()
