#!/usr/bin/env python3
"""flexiv_gripper_controller.py — Flexiv gripper controller mirroring GripperController.

__init__(config, robot): bind to an already-created flexivrdk.Robot.
connect(): Enable device + Tool.Switch (updates gravity compensation).
init_gripper(): full open-close stroke for position reference (GN01: once per power cycle).
open/close_gripper(): position-controlled with force ceiling.
"""

import logging
import time
import flexivrdk

logger = logging.getLogger(__name__)


class FlexivGripperController:
    """Direct flexivrdk Gripper wrapper mirroring GripperController's interface."""

    def __init__(self, config: dict, robot: flexivrdk.Robot):
        if robot is None:
            raise ValueError("robot not initialized")

        cfg = config["Gripper"]
        self.gripper_name: str = cfg["name"]
        self.tool_name: str = cfg.get("tool_name", cfg["name"])
        self.open_width: float = cfg.get("open_width", 0.09)
        self.close_width: float = cfg.get("close_width", 0.005)
        self.default_velocity: float = cfg.get("default_velocity", 0.1)
        self.default_force_limit: float = cfg.get("default_force_limit", 20)
        self.init_wait_sec: float = cfg.get("init_wait_sec", 8)

        self._robot: flexivrdk.Robot = robot
        self._gripper: flexivrdk.Gripper | None = None
        self.is_connected: bool = False

    # ── connection ────────────────────────────────────────────────────────────

    def connect(self):
        """Enable gripper and switch tool on the bound robot."""
        try:
            # Tool.Switch requires robot in IDLE mode on Flexiv.
            self._robot.SwitchMode(flexivrdk.Mode.IDLE)
            time.sleep(0.1)

            self._gripper = flexivrdk.Gripper(self._robot)
            logger.info(f"Enabling gripper [{self.gripper_name}]")
            self._gripper.Enable(self.gripper_name)
            logger.info(f"Switching tool to [{self.tool_name}]")
            p = self._gripper.params()
            logger.info(
                f"Gripper ready — width=[{p.min_width:.4f},{p.max_width:.4f}] m  "
                f"force=[{p.min_force:.1f},{p.max_force:.1f}] N  "
                f"vel=[{p.min_vel:.4f},{p.max_vel:.4f}] m/s"
            )
            self.is_connected = True
        except Exception as e:
            logger.error(f"Failed to connect gripper: {e}")
            self.is_connected = False
            raise

    def disconnect(self):
        if self._gripper:
            try:
                self._gripper.Stop()
            except Exception:
                pass
        self.is_connected = False
        logger.info("Gripper disconnected")

    def init_gripper(self):
        """Trigger initialization stroke (needed once per power cycle for GN01)."""
        if not self._check_connected():
            return
        logger.info(f"Initializing gripper (timeout={self.init_wait_sec} s) …")
        self._gripper.Init()
        time.sleep(0.8)
        deadline = time.monotonic() + self.init_wait_sec
        while time.monotonic() < deadline:
            if not self._gripper.states().is_moving:
                break
            time.sleep(0.2)
        else:
            logger.warning("Gripper init timed out")
        logger.info(f"Gripper init done — width={self._gripper.states().width:.4f} m")

    # ── motion ────────────────────────────────────────────────────────────────

    def open_gripper(self, width=None, velocity=None, force_limit=None):
        """Open to width [m] (default: config open_width)."""
        if not self._check_connected():
            return
        w = width if width is not None else self.open_width
        v = velocity if velocity is not None else self.default_velocity
        f = force_limit if force_limit is not None else self.default_force_limit
        self._gripper.Move(w, v, f)
        logger.info(f"Gripper opening → {w:.4f} m  vel={v} m/s  force={f} N")
        time.sleep(1.2)

    def close_gripper(self, width=None, velocity=None, force_limit=None, slight_open: bool = False):
        """Close to width [m] (default: config close_width).

        slight_open=True: stop at 15% of stroke for gentle release.
        """
        if not self._check_connected():
            return
        w = (
            self.close_width + (self.open_width - self.close_width) * 0.15
            if slight_open
            else (width if width is not None else self.close_width)
        )
        v = velocity if velocity is not None else self.default_velocity
        f = force_limit if force_limit is not None else self.default_force_limit
        self._gripper.Move(w, v, f)
        logger.info(f"Gripper closing → {w:.4f} m  vel={v} m/s  force={f} N")
        time.sleep(1.2)

    def stop(self):
        if self._gripper:
            self._gripper.Stop()
            logger.info("Gripper stopped")

    def get_state(self) -> dict:
        if not self._check_connected():
            return {}
        s = self._gripper.states()
        return {"width": s.width, "force": s.force, "is_moving": s.is_moving}

    def get_params(self) -> dict:
        if not self._check_connected():
            return {}
        p = self._gripper.params()
        return {
            "name": p.name,
            "min_width": p.min_width,
            "max_width": p.max_width,
            "min_velocity": p.min_vel,
            "max_velocity": p.max_vel,
            "min_force": p.min_force,
            "max_force": p.max_force,
        }

    def _check_connected(self) -> bool:
        if not self.is_connected or self._gripper is None:
            logger.warning("Gripper not connected")
            return False
        return True


# ═════════════════════════════════════════════════════════════ interactive CLI
if __name__ == "__main__":
    from pathlib import Path

    import yaml

    from agenticlab_human.execution.robot.flexiv.flexiv_controller import FlexivController
    
    _REPO_ROOT = Path(__file__).resolve().parents[5]
    _CONFIG_PATH = _REPO_ROOT / "configs" / "robot" / "flexiv_config.yaml"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    with open(_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    rc = FlexivController(config)
    gr = None

    CMDS = "c=connect i=init o=open cl=close s=state p=params stop q=quit"
    print(f"\nCommands: {CMDS}")
    try:
        while (cmd := input("> ").strip().lower()) != "q":
            if cmd == "c":
                if rc.is_connected:
                    if gr is not None:
                        gr.disconnect()
                    rc.disconnect()
                if rc.connect():
                    gr = FlexivGripperController(config, rc.robot)
                    gr.connect()
                    rc.zero_ft_sensor()
                    print("Ready")
                else:
                    print("Connection failed")
            elif cmd == "i":
                gr.init_gripper()
            elif cmd == "o":
                gr.open_gripper()
                print(gr.get_state())
            elif cmd == "cl":
                gr.close_gripper()
                print(gr.get_state())
            elif cmd == "s":
                print(gr.get_state())
            elif cmd == "p":
                print(gr.get_params())
            elif cmd == "stop":
                gr.stop()
            else:
                print(f"? {CMDS}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if gr is not None:
            gr.disconnect()
        rc.disconnect()
