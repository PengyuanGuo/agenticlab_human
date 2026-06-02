#!/usr/bin/env python3
"""flexiv_controller.py — Flexiv Rizon controller mirroring RTDEController. Use [x, y, z, rx, ry, rz] to represent the TCP pose.


Public interface (matches RTDEController):
    connect / disconnect / move_to_home / move_cartesian / move_joints
    get_tcp_pose / get_joint_positions / get_joint_torques / get_ft_sensor
    get_full_state / set_free_drive / zero_ft_sensor / clear_fault / stop
    apply_rpy_rotation

Unit notes (flexivrdk native):
    states().q        — rad
    states().tcp_pose — [x,y,z, qw,qx,qy,qz]
    JPos(q_m)         — deg
    Coord orientation — Euler ZYX deg [rx,ry,rz]  (R = Rz·Ry·Rx)
    Model.reachable   — pose [x,y,z,qw,qx,qy,qz], seed q [rad], IK q [rad]
"""

import time
import logging
import numpy as np
import flexivrdk
from scipy.spatial.transform import Rotation, Slerp

logger = logging.getLogger(__name__)


class FlexivController:
    """Direct flexivrdk wrapper that mirrors RTDEController's public interface."""

    def __init__(self, config: dict):
        cfg = config["Flexiv"]
        self.robot_sn: str = cfg["robot_sn"]
        self.local_ip_whitelist: list = cfg.get("local_ip_whitelist", [])
        self.max_cartesian_speed: float = cfg.get("max_cartesian_speed", 0.25)
        self.max_cartesian_acceleration: float = cfg.get("max_cartesian_acceleration", 0.5)
        self.max_joint_speed: float = cfg.get("max_joint_speed", 1.0)
        self.max_joint_acceleration: float = cfg.get("max_joint_acceleration", 1.0)
        self.cartesian_limit: dict = cfg.get("cartesian_limits_m", {})
        self.home_joints_deg: list = cfg.get("home_joints_deg", [0, -45, 0, 90, 0, 40, 0])
        self.primitive_cfg: dict = cfg.get("primitive", {})
        self.joint_limits_rad: dict = cfg.get("joint_limits_rad", {})
        self.joint_limit_margin_rad: float = np.deg2rad(cfg.get("joint_limit_margin_deg", 1.0))
        self.singularity_score_threshold: float = cfg.get("singularity_score_threshold", 20.0)
        self.singularity_check_enabled: bool = cfg.get("singularity_check_enabled", True)
        self.ik_interpolation_enabled: bool = cfg.get("ik_interpolation_enabled", True)
        self.ik_max_linear_step_m: float = max(cfg.get("ik_max_linear_step_m", 0.03), 1e-4)
        self.ik_max_angular_step_rad: float = max(
            np.deg2rad(cfg.get("ik_max_angular_step_deg", 10.0)),
            1e-4,
        )
        self.moveptp_joint4_prefer_delta_deg = cfg.get("moveptp_joint4_prefer_delta_deg", None)
        self._collision_check_enabled: bool = cfg.get("collision_check_enabled", False)
        self.verbose: bool = cfg.get("verbose", False)
        self.robot: flexivrdk.Robot | None = None
        self.model: flexivrdk.Model | None = None
        self.is_connected: bool = False
        self._current_mode = None

    # ── connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect to robot, clear fault, enable, wait until operational."""
        if self.is_connected:
            self.disconnect()
        try:
            self.robot = flexivrdk.Robot(self.robot_sn, self.local_ip_whitelist)
            if self.robot.fault():
                logger.warning("Fault detected, trying to clear …")
                if not self.robot.ClearFault():
                    logger.error("Cannot clear fault — aborting connect")
                    return False
                logger.info("Fault cleared")
            logger.info("Enabling robot …")
            self.robot.Enable()
            while not self.robot.operational():
                time.sleep(1)
            self.model = flexivrdk.Model(self.robot)
            self.is_connected = True
            logger.info(f"Connected to Flexiv robot [{self.robot_sn}]")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to [{self.robot_sn}]: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Stop robot motion, try returning to IDLE, and mark as disconnected."""
        if self.robot:
            try:
                self.robot.Stop()
            except Exception:
                pass
            try:
                self.robot.SwitchMode(flexivrdk.Mode.IDLE)  # idle mode is needed for tool switch
            except Exception:
                pass
            self._current_mode = None
        self.model = None
        self.is_connected = False
        logger.info("Disconnected from Flexiv robot")

    # ── motion ────────────────────────────────────────────────────────────────

    def move_to_home(self) -> bool:
        """Move all joints to the home pose defined in config (MoveJ)."""
        if not self._check_connected():
            return False
        self._ensure_mode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("MoveJ", {"target": flexivrdk.JPos(self.home_joints_deg)})
        return self._wait_primitive("reachedTarget",
                                    timeout=self.primitive_cfg.get("home_timeout", 30))

    def move_cartesian(self, tcp_pose, speed=None, acceleration=None) -> bool:
        """Move TCP to [x,y,z,rx,ry,rz] (m, rad axis-angle) via IK.

        The public name is kept for compatibility with RTDEController and
        ActionWrapper. On Flexiv this samples a near-Cartesian TCP path, solves
        each waypoint with Model.reachable(), then executes small MoveJ segments.
        This avoids some MoveL wrist-flip/singularity failures from unusual
        GraspNet poses while keeping approach-to-grasp motion close to linear.
        """
        if not self._check_connected():
            return False
        tcp_pose = np.asarray(tcp_pose, dtype=float)
        if tcp_pose.shape != (6,):
            logger.error("tcp_pose must have 6 elements [x, y, z, rx, ry, rz]")
            return False
        if not self._check_cartesian_limits(tcp_pose[:3]):
            return False

        model = self._ensure_model()
        if model is None:
            logger.error("Flexiv Model is unavailable; cannot solve IK for move_cartesian")
            return False

        seed_joints = np.asarray(self.robot.states().q, dtype=float)
        waypoints = self._interpolate_tcp_pose6d(self.get_tcp_pose(), tcp_pose)
        joint_targets = []

        for waypoint in waypoints:
            target_pose = self._tcp_pose6d_to_pose7d(waypoint)
            is_reachable, ik_solution = self._solve_ik(target_pose, model, seed_joints)
            if not is_reachable or ik_solution is None:
                logger.error(f"Target TCP pose is not reachable by IK: {np.round(target_pose, 4)}")
                return False

            if not self._check_joint_limits(ik_solution):
                return False

            if self.singularity_check_enabled and self._is_near_singularity(ik_solution, model):
                return False

            # Collision checking can be added here now that IK gives us joint angles.
            if self._collision_check_enabled:
                pass

            joint_targets.append(ik_solution)
            seed_joints = ik_solution

        logger.info(f"IK planned {len(joint_targets)} joint waypoint(s)")
        self._ensure_mode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        joint_targets_deg = [np.rad2deg(q).tolist() for q in joint_targets]
        logger.info(f"IK final target joints: {np.round(joint_targets_deg[-1], 2)} deg")
        params = {"target": flexivrdk.JPos(joint_targets_deg[-1])}
        if len(joint_targets_deg) > 1:
            params["waypoints"] = [flexivrdk.JPos(q) for q in joint_targets_deg[:-1]]
        self.robot.ExecutePrimitive("MoveJ", params)
        return self._wait_primitive("reachedTarget",
                                    timeout=self.primitive_cfg.get("move_timeout", 30))

    def movel(self, tcp_pose, speed=None, acceleration=None) -> bool:
        """Move TCP to [x,y,z,rx,ry,rz] (m, rad axis-angle) via MoveL."""
        if not self._check_connected():
            return False
        tcp_pose = np.asarray(tcp_pose, dtype=float)
        if tcp_pose.shape != (6,):
            logger.error("tcp_pose must have 6 elements [x, y, z, rx, ry, rz]")
            return False
        if not self._check_cartesian_limits(tcp_pose[:3]):
            return False

        target = self._tcp_pose6d_to_coord(tcp_pose)
        vel = float(speed) if speed is not None else self.max_cartesian_speed
        acc = float(acceleration) if acceleration is not None else self.max_cartesian_acceleration

        self._ensure_mode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("MoveL", {
            "target": target,
            "maxVel": vel,
            "maxAcc": acc,
            "zoneRadius": "Z80",
        })
        return self._wait_primitive("reachedTarget",
                                    timeout=self.primitive_cfg.get("move_timeout", 30))

    def moveptp(self, tcp_pose, waypoints=None, jnt_vel_scale: int = 10,
                zone_radius: str = "Z50", target_toler_level: int = 1,
                prefer_current_joints: bool = False,
                prefer_joints_deg=None,
                joint4_prefer_delta_deg: float = None) -> bool:
        """Move TCP to [x,y,z,rx,ry,rz] via Flexiv MovePTP.

        MovePTP solves IK internally and moves in joint space through optional
        Cartesian waypoints. It is useful for fast obstacle-free Cartesian
        targets where the TCP path does not need to be straight. Set
        prefer_current_joints or prefer_joints_deg to bias the internal IK
        toward a specific 7-DOF posture. joint4_prefer_delta_deg clamps only
        A4 in that preferred posture around the current joint-4 angle.
        """
        if not self._check_connected():
            return False

        tcp_pose = np.asarray(tcp_pose, dtype=float)
        if tcp_pose.shape != (6,):
            logger.error("tcp_pose must have 6 elements [x, y, z, rx, ry, rz]")
            return False
        if not self._check_cartesian_limits(tcp_pose[:3]):
            return False

        try:
            if joint4_prefer_delta_deg is None:
                joint4_prefer_delta_deg = self.moveptp_joint4_prefer_delta_deg
            coord_prefer_joints_deg = self._get_preferred_joints_deg(
                prefer_joints_deg=prefer_joints_deg,
                joint4_delta_deg=joint4_prefer_delta_deg,
            )
        except ValueError as e:
            logger.error(str(e))
            return False

        params = {
            "target": self._tcp_pose6d_to_coord(
                tcp_pose,
                preferred_joints_deg=coord_prefer_joints_deg,
            ),
            "jntVelScale": int(np.clip(jnt_vel_scale, 1, 100)),
            "zoneRadius": zone_radius,
            "targetTolLevel": int(np.clip(target_toler_level, 0, 10)),
        }

        if prefer_current_joints or prefer_joints_deg is not None or joint4_prefer_delta_deg is not None:
            prefer_joints_deg = coord_prefer_joints_deg
            params["enablePreferJntPos"] = True
            params["preferJntPos"] = flexivrdk.JPos(prefer_joints_deg.tolist())

        if waypoints is not None:
            waypoint_coords = []
            for waypoint in waypoints:
                waypoint = np.asarray(waypoint, dtype=float)
                if waypoint.shape != (6,):
                    logger.error("Each MovePTP waypoint must have 6 elements [x, y, z, rx, ry, rz]")
                    return False
                if not self._check_cartesian_limits(waypoint[:3]):
                    return False
                waypoint_coords.append(
                    self._tcp_pose6d_to_coord(
                        waypoint,
                        preferred_joints_deg=coord_prefer_joints_deg,
                    )
                )
            if waypoint_coords:
                params["waypoints"] = waypoint_coords

        self._ensure_mode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("MovePTP", params)
        return self._wait_primitive("reachedTarget",
                                    timeout=self.primitive_cfg.get("move_timeout", 30))

    def move_joints(self, joints_num: int, angle: float) -> bool:
        """Move a single joint to *angle* (deg), keeping others at current positions."""
        if not self._check_connected():
            return False
        cur_q_deg = [np.rad2deg(v) for v in self.robot.states().q]
        cur_q_deg[joints_num] = float(angle)
        self._ensure_mode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.ExecutePrimitive("MoveJ", {"target": flexivrdk.JPos(cur_q_deg)}) # target unit in degrees
        return self._wait_primitive("reachedTarget",
                                    timeout=self.primitive_cfg.get("move_timeout", 30))

    def set_free_drive(self, enable: bool = True):
        """Enable/disable free-drive via floatingSoft primitive."""
        if not self._check_connected():
            return
        if enable:
            self._ensure_mode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
            self.robot.ExecutePrimitive("floatingSoft", {})
            logger.info("Free-drive enabled (floatingSoft)")
        else:
            self.robot.Stop()
            self._current_mode = None
            time.sleep(0.3)
            logger.info("Free-drive disabled (Stop → IDLE)")

    # ── state read ────────────────────────────────────────────────────────────

    def get_tcp_pose(self) -> np.ndarray:
        """Return [x,y,z,rx,ry,rz] (m, rad axis-angle). Converts from native quaternion."""
        tcp = self.robot.states().tcp_pose
        qw, qx, qy, qz = tcp[3], tcp[4], tcp[5], tcp[6]
        rotvec = Rotation.from_quat([qx, qy, qz, qw]).as_rotvec().tolist()
        return np.array(list(tcp[:3]) + rotvec)

    def get_joint_positions(self) -> np.ndarray:
        """Return 7 joint positions [rad]."""
        return np.array(self.robot.states().q)

    def get_joint_torques(self) -> np.ndarray:
        """Return estimated external joint torques [Nm] (7 values)."""
        return np.array(self.robot.states().tau_ext)

    def get_ft_sensor(self) -> np.ndarray:
        """Return external wrench at TCP in TCP frame [N, Nm] (6 values)."""
        return np.array(self.robot.states().ext_wrench_in_tcp)

    def get_full_state(self) -> dict:
        """Return a snapshot of all useful robot states."""
        s = self.robot.states()
        return {
            "tcp_pose": self.get_tcp_pose(),
            "tcp_pose_quat": list(s.tcp_pose),
            "joint_positions": np.array(s.q),
            "joint_velocities": np.array(s.dq),
            "joint_torques": np.array(s.tau),
            "external_torques": np.array(s.tau_ext),
            "ft_sensor_raw": np.array(s.ft_sensor_raw),
            "ext_wrench_in_tcp": np.array(s.ext_wrench_in_tcp),
            "ext_wrench_in_world": np.array(s.ext_wrench_in_world),
        }

    # ── utility ───────────────────────────────────────────────────────────────

    def zero_ft_sensor(self, timeout: int = None) -> bool:
        """Zero the force-torque sensor. Robot must be free of contact."""
        if not self._check_connected():
            return False
        t = timeout or self.primitive_cfg.get("ft_zero_timeout", 15)
        self._ensure_mode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        logger.warning("Zeroing FT sensor — ensure nothing contacts the robot")
        self.robot.ExecutePrimitive("ZeroFTSensor", {})
        result = self._wait_primitive("terminated", timeout=t)
        if result:
            logger.info("FT sensor zeroed")
        return result

    def clear_fault(self) -> bool:
        """Clear any active fault. Returns True if robot is fault-free."""
        if not self.robot:
            return False
        if self.robot.fault():
            logger.info("Clearing fault …")
            return self.robot.ClearFault()
        return True

    def has_fault(self) -> bool:
        """Return whether robot is currently in fault state."""
        return bool(self.robot and self.robot.fault())

    def stop(self):
        """Stop all robot motion and return to IDLE."""
        if self.robot:
            self.robot.Stop()
            self._current_mode = None
            logger.info("Robot stopped")


    @staticmethod
    def apply_rpy_rotation(T: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Apply extrinsic XYZ rotation (roll, pitch, yaw) onto SE(3) matrix T.

        Args:  T — 4×4 SE3,  roll/pitch/yaw — rad around X/Y/Z.
        Returns: new 4×4 SE3 (translation unchanged).
        """
        R_delta = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
        T_new = T.copy()
        T_new[:3, :3] = T[:3, :3] @ R_delta
        return T_new

    # ── internal ──────────────────────────────────────────────────────────────

    def _check_connected(self) -> bool:
        if not self.is_connected or self.robot is None:
            logger.warning("Robot not connected")
            return False
        return True

    def _ensure_mode(self, mode):
        if self._current_mode != mode:
            self.robot.SwitchMode(mode)
            self._current_mode = mode

    def _wait_primitive(self, state_key: str = "reachedTarget",
                        dt: float = None, timeout: float = None) -> bool:
        dt = dt or self.primitive_cfg.get("check_interval", 0.5)
        timeout = timeout or self.primitive_cfg.get("move_timeout", 30)
        deadline = time.time() + timeout
        while True:
            if self.robot.fault():
                raise RuntimeError("Robot entered fault during primitive execution")
            if self.robot.primitive_states().get(state_key):
                return True
            if time.time() > deadline:
                logger.error(f"Primitive timed out ({timeout}s) waiting for '{state_key}'")
                return False
            time.sleep(dt)

    def _tcp_pose6d_to_coord(self, tcp_pose: np.ndarray,
                             preferred_joints_deg=None) -> flexivrdk.Coord:
        """Build WORLD-frame Coord from [x,y,z,rx,ry,rz] (m, rad axis-angle)."""
        tcp_pose = np.asarray(tcp_pose, dtype=float).reshape(6)
        euler_deg = self._rotvec_to_euler_zyx_deg(tcp_pose[3:])
        if preferred_joints_deg is None:
            preferred_joints_deg = np.rad2deg(self.robot.states().q)
        preferred_joints_deg = np.asarray(preferred_joints_deg, dtype=float)
        if preferred_joints_deg.shape != (7,):
            raise ValueError("preferred_joints_deg must have 7 joint angles in degrees")
        return flexivrdk.Coord(
            tcp_pose[:3].tolist(),
            euler_deg,
            ["WORLD", "WORLD_ORIGIN"],
            preferred_joints_deg.tolist(),
        )

    def _get_preferred_joints_deg(self, prefer_joints_deg=None,
                                  joint4_delta_deg: float = None) -> np.ndarray:
        """Return preferred joints in degrees, optionally clamping only A4."""
        cur_q_deg = np.rad2deg(self.robot.states().q)
        if prefer_joints_deg is None:
            prefer_joints_deg = cur_q_deg.copy()
        else:
            prefer_joints_deg = np.asarray(prefer_joints_deg, dtype=float)
            if prefer_joints_deg.shape != (7,):
                raise ValueError("prefer_joints_deg must have 7 joint angles in degrees")

        # TODO: tested with commented joint4_delta_deg, didn't change the result    
        # if joint4_delta_deg is not None:
        #     delta = abs(float(joint4_delta_deg))
        #     prefer_joints_deg[3] = np.clip(
        #         prefer_joints_deg[3],
        #         cur_q_deg[3] - delta,
        #         cur_q_deg[3] + delta,
        #     )
        return prefer_joints_deg

    def _tcp_pose6d_to_pose7d(self, tcp_pose: np.ndarray) -> np.ndarray:
        """Convert [x,y,z,rx,ry,rz] axis-angle to Flexiv [x,y,z,qw,qx,qy,qz]."""
        tcp_pose = np.asarray(tcp_pose, dtype=float).reshape(6)
        return np.array(
            tcp_pose[:3].tolist() + self._rotvec_to_quat_wxyz(tcp_pose[3:]),
            dtype=float,
        )

    def _interpolate_tcp_pose6d(self, start_pose: np.ndarray,
                                target_pose: np.ndarray) -> list[np.ndarray]:
        """Interpolate TCP waypoints so IK follows a near-Cartesian path."""
        start_pose = np.asarray(start_pose, dtype=float).reshape(6)
        target_pose = np.asarray(target_pose, dtype=float).reshape(6)
        if not self.ik_interpolation_enabled:
            return [target_pose]

        linear_dist = np.linalg.norm(target_pose[:3] - start_pose[:3])
        start_rot = Rotation.from_rotvec(start_pose[3:])
        target_rot = Rotation.from_rotvec(target_pose[3:])
        angular_dist = (target_rot * start_rot.inv()).magnitude()
        linear_steps = int(np.ceil(linear_dist / self.ik_max_linear_step_m))
        angular_steps = int(np.ceil(angular_dist / self.ik_max_angular_step_rad))
        steps = max(1, linear_steps, angular_steps)

        slerp = Slerp([0.0, 1.0], Rotation.concatenate([start_rot, target_rot]))
        waypoints = []
        for i in range(1, steps + 1):
            ratio = i / steps
            pos = start_pose[:3] + (target_pose[:3] - start_pose[:3]) * ratio
            rotvec = slerp([ratio]).as_rotvec()[0]
            waypoints.append(np.concatenate([pos, rotvec]))
        return waypoints

    def _ensure_model(self) -> flexivrdk.Model | None:
        """Create the Flexiv model lazily if connect() did not already do it."""
        if self.model is not None:
            return self.model
        if self.robot is None:
            return None
        try:
            self.model = flexivrdk.Model(self.robot)
            return self.model
        except Exception as e:
            logger.error(f"Failed to initialize Flexiv Model: {e}")
            return None

    def _solve_ik(self, target_pose: np.ndarray, model: flexivrdk.Model,
                  seed_joints: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Solve IK with Model.reachable().

        Args:
            target_pose: target TCP pose [x, y, z, qw, qx, qy, qz].
            model: Flexiv model used for IK.
            seed_joints: seed joint positions [rad].

        Returns:
            (is_reachable, ik_solution): reachability and joint solution [rad].
        """
        if model is None:
            return False, None

        try:
            is_reachable, ik_solution = model.reachable(
                list(np.asarray(target_pose, dtype=float).reshape(7)),
                list(np.asarray(seed_joints, dtype=float).reshape(7)),
                False,
            )
            if is_reachable:
                return True, np.asarray(ik_solution, dtype=float)
            return False, None
        except Exception as e:
            if self.verbose:
                print(f"IK 求解失败: {e}")
            logger.error(f"IK solve failed: {e}")
            return False, None

    def _check_joint_limits(self, q_rad: np.ndarray) -> bool:
        """Check IK joint target against configured or robot-reported limits."""
        q_rad = np.asarray(q_rad, dtype=float).reshape(7)
        limits = self._get_joint_limits_rad()
        if limits is None:
            logger.error("Cannot verify joint limits because no Flexiv joint limits are available")
            return False

        q_min, q_max = limits
        below = q_rad < (q_min + self.joint_limit_margin_rad)
        above = q_rad > (q_max - self.joint_limit_margin_rad)
        if np.any(below | above):
            bad = np.where(below | above)[0]
            for idx in bad:
                logger.error(
                    "IK joint q%d=%.2f deg outside safe limit [%.2f, %.2f] deg "
                    "(margin %.2f deg)",
                    idx + 1,
                    np.rad2deg(q_rad[idx]),
                    np.rad2deg(q_min[idx]),
                    np.rad2deg(q_max[idx]),
                    np.rad2deg(self.joint_limit_margin_rad),
                )
            return False
        return True

    def _get_joint_limits_rad(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return joint limits [rad] from config first, then Flexiv robot info."""
        cfg_min = self.joint_limits_rad.get("min")
        cfg_max = self.joint_limits_rad.get("max")
        if cfg_min is not None and cfg_max is not None:
            q_min = np.asarray(cfg_min, dtype=float)
            q_max = np.asarray(cfg_max, dtype=float)
            if q_min.shape == (7,) and q_max.shape == (7,):
                return q_min, q_max
            logger.warning("Ignoring invalid Flexiv joint_limits_rad config; expected 7 min/max values")

        try:
            info = self.robot.info()
            q_min = np.asarray(info.q_min, dtype=float)
            q_max = np.asarray(info.q_max, dtype=float)
            if q_min.shape == (7,) and q_max.shape == (7,):
                return q_min, q_max
        except Exception as e:
            logger.error(f"Failed to read Flexiv joint limits from robot info: {e}")
        return None

    def _is_near_singularity(self, q_rad: np.ndarray,
                             model: flexivrdk.Model) -> bool:
        """Reject low manipulability configurations using Flexiv configuration_score()."""
        try:
            q_rad = np.asarray(q_rad, dtype=float).reshape(7)
            model.Update(q_rad.tolist(), [0.0] * 7)
            trans_score, orient_score = model.configuration_score()
        except Exception as e:
            logger.error(f"Failed to evaluate singularity score: {e}")
            return True

        threshold = self.singularity_score_threshold
        if trans_score < threshold or orient_score < threshold:
            logger.error(
                "IK solution is near singularity: translation_score=%.2f, "
                "orientation_score=%.2f, threshold=%.2f",
                trans_score,
                orient_score,
                threshold,
            )
            return True
        return False

    def _check_cartesian_limits(self, pos) -> bool:
        for axis, idx in [("x", 0), ("y", 1), ("z", 2)]:
            if axis in self.cartesian_limit:
                lo, hi = self.cartesian_limit[axis]
                if pos[idx] < lo or pos[idx] > hi:
                    logger.error(f"Target {axis}={pos[idx]:.4f} m outside [{lo}, {hi}]")
                    return False
        return True

    @staticmethod
    def _quat_wxyz_to_rotvec(quat_wxyz) -> list:
        """flexivrdk [qw,qx,qy,qz] → axis-angle [rx,ry,rz] rad."""
        qw, qx, qy, qz = quat_wxyz
        return Rotation.from_quat([qx, qy, qz, qw]).as_rotvec().tolist()

    @staticmethod
    def _rotvec_to_quat_wxyz(rotvec_rad) -> list:
        """axis-angle [rx,ry,rz] rad → flexivrdk [qw,qx,qy,qz]."""
        qx, qy, qz, qw = Rotation.from_rotvec(rotvec_rad).as_quat()
        return [qw, qx, qy, qz]

    @staticmethod
    def _rotvec_to_euler_zyx_deg(rotvec_rad) -> list:
        """axis-angle rad → Euler ZYX deg [rx,ry,rz] for Flexiv Coord.
        Flexiv convention: R = Rz·Ry·Rx  (intrinsic ZYX = scipy extrinsic 'xyz').
        """
        return Rotation.from_rotvec(rotvec_rad).as_euler("xyz", degrees=True).tolist()


# ═════════════════════════════════════════════════════════════ interactive CLI
if __name__ == "__main__":
    from pathlib import Path

    import yaml
    _REPO_ROOT = Path(__file__).resolve().parents[5]
    _CONFIG_PATH = _REPO_ROOT / "configs" / "robot" / "flexiv_config.yaml"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with open(_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    robot = FlexivController(config)

    CMDS = "c=connect h=home p=pose j=joints f=freedrive m=moveptp z=zero_ft q=quit"
    print(f"\nCommands: {CMDS}")
    try:
        while (cmd := input("> ").strip().lower()) != "q":
            if   cmd == "c": robot.connect()
            elif cmd == "h": print("OK" if robot.move_to_home() else "FAIL")
            elif cmd == "p": print(f"TCP: {robot.get_tcp_pose()}")
            elif cmd == "j": q = robot.get_joint_positions(); print(f"rad: {q}\ndeg: {np.rad2deg(q)}")
            elif cmd == "f": robot.set_free_drive(); input("Enter to exit …"); robot.set_free_drive(False); print(f"Pose: {robot.get_tcp_pose()}")
            elif cmd == "m": p = input("Enter pose in 6d format [x,y,z,rx,ry,rz]: "); p = np.fromstring(p, sep=" "); print("OK" if p.shape == (6,) and robot.moveptp(p, zone_radius="Z50") else "FAIL")
            elif cmd == "z": robot.zero_ft_sensor()
            else: print(f"? {CMDS}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        robot.disconnect()
