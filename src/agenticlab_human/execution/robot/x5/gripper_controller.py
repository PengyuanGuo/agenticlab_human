"""Dahuan gripper device adapter and server-side gripper services."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from agenticlab_human.execution.robot.x5.contracts import (
    ComponentHealth,
    GripperState,
    SetGripperCommand,
)


class GripperController:
    """Low-level Modbus-RTU adapter for one Dahuan AG-series gripper."""

    def __init__(
        self,
        port: str = "COM11",
        baudrate: int = 115200,
        gripper_id: int = 1,
    ) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for the real X5 gripper"
            ) from exc
        self.ser = serial.Serial(port, baudrate, timeout=0.5)
        self.gripper_id = int(gripper_id)

    @staticmethod
    def _calculate_crc(data: bytes | bytearray) -> bytes:
        crc = 0xFFFF
        for value in data:
            crc ^= value
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        return crc.to_bytes(2, "little")

    def _send_command(self, func_code: int, register_addr: int, value: int) -> bytes:
        data = bytearray([self.gripper_id, func_code])
        data.extend(register_addr.to_bytes(2, "big"))
        data.extend(value.to_bytes(2, "big"))
        data.extend(self._calculate_crc(data))
        self.ser.write(data)
        return self.ser.read(8)

    def _read_registers(self, register_addr: int, count: int) -> bytes:
        data = bytearray([self.gripper_id, 0x03])
        data.extend(register_addr.to_bytes(2, "big"))
        data.extend(count.to_bytes(2, "big"))
        data.extend(self._calculate_crc(data))
        self.ser.write(data)
        return self.ser.read(5 + 2 * count)

    def init_gripper(
        self,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.5,
    ) -> None:
        """Initialize the gripper after power-on or a device fault."""

        self._send_command(0x06, 0x0100, 1)
        deadline = time.monotonic() + float(timeout_s)
        while self.get_init_status() != 1:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"gripper initialization timed out after {timeout_s:.1f}s"
                )
            time.sleep(float(poll_interval_s))

    def set_force(self, force: int) -> None:
        self._send_command(0x06, 0x0101, max(20, min(100, int(force))))

    def set_position(self, position: int) -> None:
        self._send_command(0x06, 0x0103, max(0, min(1000, int(position))))

    def get_init_status(self) -> int | None:
        return self._read_register(0x0200)

    def get_grip_status(self) -> int | None:
        return self._read_register(0x0201)

    def get_current_position(self) -> int | None:
        return self._read_register(0x0202)

    def close(self) -> None:
        self.ser.close()

    def _read_register(self, address: int) -> int | None:
        response = self._read_registers(address, 1)
        if len(response) < 5:
            return None
        return int.from_bytes(response[3:5], "big")


class GripperDevice(Protocol):
    def get_init_status(self) -> int | None: ...

    def init_gripper(
        self,
        *,
        timeout_s: float,
        poll_interval_s: float,
    ) -> None: ...

    def set_force(self, force: int) -> None: ...

    def set_position(self, position: int) -> None: ...

    def get_grip_status(self) -> int | None: ...

    def get_current_position(self) -> int | None: ...

    def close(self) -> None: ...


@runtime_checkable
class X5Gripper(Protocol):
    def initialize(self) -> None: ...

    def execute(self, command: SetGripperCommand) -> None: ...

    def get_state(self) -> GripperState: ...

    def health(self) -> ComponentHealth: ...

    def shutdown(self) -> None: ...


class GripperService:
    """Own the real gripper lifecycle, normalized command mapping, and state."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        device: GripperDevice | None = None,
    ) -> None:
        self._config = dict(config)
        self._device = device
        self._owns_device = device is None
        self._force = int(self._config.get("force", 100))
        self._closed_position = int(self._config.get("closed_position", 0))
        self._open_position = int(self._config.get("open_position", 1000))
        self._init_timeout_s = float(self._config.get("init_timeout_s", 10.0))
        self._move_timeout_s = float(self._config.get("move_timeout_s", 5.0))
        self._poll_interval_s = float(self._config.get("poll_interval_s", 0.1))
        self._initialized = False
        self._last_error = ""
        self._lock = threading.RLock()
        self._validate_config()

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            try:
                if self._device is None:
                    port = self._config.get("port")
                    if not port:
                        raise ValueError("robot.gripper.port is required")
                    self._device = GripperController(
                        port=str(port),
                        baudrate=int(self._config.get("baudrate", 115200)),
                        gripper_id=int(self._config.get("gripper_id", 1)),
                    )
                if self._device.get_init_status() != 1:
                    self._device.init_gripper(
                        timeout_s=self._init_timeout_s,
                        poll_interval_s=self._poll_interval_s,
                    )
                self._device.set_force(self._force)
                self._initialized = True
                self._last_error = ""
            except Exception as exc:
                self._last_error = str(exc)
                self._close_owned_device()
                raise

    def execute(self, command: SetGripperCommand) -> None:
        with self._lock:
            device = self._require_device()
            device.set_position(self._raw_position(command.position))
            if not command.wait:
                return

            deadline = time.monotonic() + self._move_timeout_s
            while True:
                status = device.get_grip_status()
                if status in (1, 2):
                    return
                if status == 3:
                    raise RuntimeError("gripper reported object-drop status")
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"gripper movement timed out after "
                        f"{self._move_timeout_s:.1f}s"
                    )
                time.sleep(self._poll_interval_s)

    def get_state(self) -> GripperState:
        with self._lock:
            if not self._initialized or self._device is None:
                return GripperState(connected=False, moving=False)
            status = self._device.get_grip_status()
            raw_position = self._device.get_current_position()
            return GripperState(
                connected=True,
                moving=status == 0,
                position=(
                    self._normalized_position(raw_position)
                    if raw_position is not None
                    else None
                ),
                raw_position=raw_position,
                grip_status=status,
            )

    def health(self) -> ComponentHealth:
        with self._lock:
            ready = self._initialized and self._device is not None
            detail = "ready" if ready else self._last_error or "not initialized"
        return ComponentHealth(
            ready=ready,
            backend="dahuan-modbus",
            detail=detail,
        )

    def shutdown(self) -> None:
        with self._lock:
            self._initialized = False
            self._close_owned_device()

    def _validate_config(self) -> None:
        if not 20 <= self._force <= 100:
            raise ValueError("gripper.force must be in [20, 100]")
        for name, value in (
            ("closed_position", self._closed_position),
            ("open_position", self._open_position),
        ):
            if not 0 <= value <= 1000:
                raise ValueError(f"gripper.{name} must be in [0, 1000]")
        if self._closed_position == self._open_position:
            raise ValueError(
                "gripper.closed_position and gripper.open_position must differ"
            )
        for name, value in (
            ("init_timeout_s", self._init_timeout_s),
            ("move_timeout_s", self._move_timeout_s),
            ("poll_interval_s", self._poll_interval_s),
        ):
            if value <= 0.0:
                raise ValueError(f"gripper.{name} must be positive")

    def _require_device(self) -> GripperDevice:
        if not self._initialized or self._device is None:
            raise RuntimeError("single gripper is not initialized")
        return self._device

    def _raw_position(self, normalized_position: float) -> int:
        return int(
            round(
                self._closed_position
                + float(normalized_position)
                * (self._open_position - self._closed_position)
            )
        )

    def _normalized_position(self, raw_position: int) -> float:
        span = self._open_position - self._closed_position
        normalized = (int(raw_position) - self._closed_position) / span
        return min(1.0, max(0.0, float(normalized)))

    def _close_owned_device(self) -> None:
        if self._device is not None and self._owns_device:
            try:
                self._device.close()
            finally:
                self._device = None


class MockGripperService:
    """In-memory single-gripper service for server tests."""

    def __init__(self) -> None:
        self._initialized = False
        self._state = GripperState(
            connected=False,
            moving=False,
            position=1.0,
            raw_position=1000,
            grip_status=1,
        )
        self._lock = threading.Lock()

    def initialize(self) -> None:
        with self._lock:
            self._initialized = True
            self._state.connected = True

    def execute(self, command: SetGripperCommand) -> None:
        with self._lock:
            if not self._initialized:
                raise RuntimeError("mock gripper is not initialized")
            self._state.position = float(command.position)
            self._state.raw_position = int(round(command.position * 1000.0))
            self._state.grip_status = 1

    def get_state(self) -> GripperState:
        with self._lock:
            return self._state.model_copy(deep=True)

    def health(self) -> ComponentHealth:
        with self._lock:
            ready = self._initialized
        return ComponentHealth(
            ready=ready,
            backend="mock",
            detail="ready" if ready else "not initialized",
        )

    def shutdown(self) -> None:
        with self._lock:
            self._initialized = False
            self._state.connected = False
            self._state.moving = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Directly control the X5 Dahuan gripper on the Server PC.",
    )
    parser.add_argument(
        "--port",
        help="Serial port override, e.g. COM6 or /dev/ttyUSB0.",
    )
    parser.add_argument("--baudrate", type=int, help="Serial baudrate override.")
    parser.add_argument("--gripper-id", type=int, help="Modbus gripper id override.")
    parser.add_argument("--force", type=int, help="Grip force override in [20, 100].")
    parser.add_argument("--init-timeout-s", type=float, help="Initialization timeout.")
    parser.add_argument("--state", action="store_true", help="Print gripper state.")
    parser.add_argument("--quit", action="store_true", help="Exit without touching the gripper.")

    action = parser.add_mutually_exclusive_group()
    action.add_argument("--open", action="store_true", help="Open the gripper.")
    action.add_argument("--close", action="store_true", help="Close the gripper.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.quit:
        return 0
    if not (args.open or args.close or args.state):
        parser.print_help()
        return 0

    config = yaml.safe_load(Path("configs/robot/x5_config.yaml").read_text()) or {}
    gripper_config = dict(config["robot"]["gripper"])
    for key in ("port", "baudrate", "force", "init_timeout_s"):
        value = getattr(args, key)
        if value is not None:
            gripper_config[key] = value
    if args.gripper_id is not None:
        gripper_config["gripper_id"] = args.gripper_id

    service = GripperService(gripper_config)
    try:
        service.initialize()
        if args.open:
            service.execute(SetGripperCommand(position=1.0, wait=True))
            print("Opened gripper.")
        elif args.close:
            service.execute(SetGripperCommand(position=0.0, wait=True))
            print("Closed gripper.")
        if args.state:
            print(json.dumps(service.get_state().model_dump(), indent=2))
        return 0
    finally:
        service.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
