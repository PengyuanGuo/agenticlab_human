import cv2
import numpy as np
import yaml
import logging
import time
from scipy.spatial.transform import Rotation as R, Slerp
from vlm_robobench.modules.flexiv_controller import FlexivController
from vlm_robobench.modules.flexiv_gripper_controller import FlexivGripperController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionWrapper:
    """Flexiv-adapted action wrapper.
    Constructor signature differs from the UR5e version:
        UR5e:   ActionWrapper(cfg, rtde_cfg, gripper_cfg, camera_cfg)
        Flexiv: ActionWrapper(cfg, flexiv_cfg, camera_cfg)
    """

    def __init__(self, cfg, flexiv_cfg, camera_cfg):
        self.config = cfg['ActionWrapper']
        self.controller = FlexivController(flexiv_cfg)

        # self.wrist_cam_cfg = camera_cfg.get('RealSense', {})
        # self.shoulder_cam_cfg = camera_cfg.get('Kinect', {})
        self.shoulder_cam_cfg = camera_cfg.get('FemtoBolt', {})
        self.wrist_cam_cfg = camera_cfg.get('Gemini305', {}) # TODO: need to unify the interface in the future

        self.robot_speed = self.config.get('robot_speed', 0.3)
        self.robot_acceleration = self.config.get('robot_acceleration', 0.3)

        self.is_connected = self.controller.connect()
        if not self.is_connected:
            raise ConnectionError("Failed to connect to Flexiv robot.")
        self.gripper = FlexivGripperController(flexiv_cfg, self.controller.robot)
        self.gripper.connect()

        self.reset()
        self.use_wrist_camera = False
        self.DEFAULT_PLACE_ORIENTATION = np.array([-2.23985022, 2.15791994, 0.04289119]) # ideal value is [-2.22144, 2.22144, 0]

    def set_use_wrist_camera(self, use_wrist_camera: bool):
        self.use_wrist_camera = use_wrist_camera

    def convert_hand_eye(self, T_cam_grasp, in_se4=False, cur_eef_pose=None):
        """Convert grasp pose from camera frame to robot base frame. cur_eff_pose is 4x4 matrix"""
        if cur_eef_pose is None: # not using wrist cam
            handeye_rotation = np.array(self.shoulder_cam_cfg.get('handeye_calibration', {}).get('rotation', np.eye(3)))
            handeye_translation = np.array(self.shoulder_cam_cfg.get('handeye_calibration', {}).get('translation', np.zeros(3)))
            T_handeye = np.eye(4)
            T_handeye[:3, :3] = handeye_rotation
            T_handeye[:3, 3] = handeye_translation
        else:
            handeye_rotation = np.array(self.wrist_cam_cfg.get('handeye_calibration', {}).get('rotation', np.eye(3)))
            handeye_translation = np.array(self.wrist_cam_cfg.get('handeye_calibration', {}).get('translation', np.zeros(3)))
            T_handeye = np.eye(4)
            T_handeye[:3, :3] = handeye_rotation
            T_handeye[:3, 3] = handeye_translation
            T_base_ee = cur_eef_pose
            T_handeye = T_base_ee @ T_handeye

        T_base_grasp = T_handeye @ T_cam_grasp

        # Flexiv tool frame differs from UR5e; this R_eg accounts for graspnet's grasp frame → Flexiv flange frame alignment.
        if isinstance(self.controller, FlexivController):
            R_eg = np.array(
                [
                    [0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0],
                ],
                dtype=float,
            )
        else:
            # Fallback to identity if controller type is unexpected
            R_eg = np.eye(3, dtype=float)

        T_ge = np.eye(4)
        T_ge[:3, :3] = R_eg.T

        T_base_ee = T_base_grasp @ T_ge
        if in_se4:
            return T_base_ee
        return ActionWrapper.transfer_T_to_pose6d(T_base_ee)

    def _generate_approach_pose(self, T_cam_grasp, cur_eef_pose=None):
        """Create an approach pose by backing off along the grasp x-axis."""
        approach_distance = self.config.get('approach_distance', 0.1)
        T_cam_approach = T_cam_grasp.copy()
        R_cg = T_cam_grasp[:3, :3]
        t_cg = T_cam_grasp[:3, 3]
        delta = np.array([-approach_distance, 0.0, 0.0])
        T_cam_approach[:3, 3] = t_cg + R_cg @ delta # t_new = t + (R @ delta), transform delta in grasp frame to camera frame
        return self.convert_hand_eye(T_cam_approach, cur_eef_pose=cur_eef_pose)

    # ── pose conversion helpers ──────────────────────────────────────────────
    @staticmethod
    def transfer_T_to_pose6d(T):
        """4x4 SE3 → [x, y, z, rx, ry, rz] axis-angle (m, rad)."""
        translation = T[:3, 3]
        rotation_vec = R.from_matrix(T[:3, :3]).as_rotvec()
        return np.concatenate([translation, rotation_vec])

    @staticmethod
    def transfer_pose6d_to_T(pose6d):
        """[x, y, z, rx, ry, rz] axis-angle → 4x4 SE3."""
        translation = pose6d[:3]
        rotation_vec = pose6d[3:]
        rotation = R.from_rotvec(rotation_vec).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        return T

    @staticmethod
    def transfer_pos_rot_to_T(position, rotation):
        """(position, rotation_matrix) → 4x4 SE3."""
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position
        return T

    @staticmethod
    def interpolate_pose6d(start_pose, target_pose, ratio=0.5):
        """Interpolate [x,y,z,rx,ry,rz] with linear XYZ and SLERP orientation."""
        start_pose = np.asarray(start_pose, dtype=float).reshape(6)
        target_pose = np.asarray(target_pose, dtype=float).reshape(6)
        ratio = float(np.clip(ratio, 0.0, 1.0))

        pos = start_pose[:3] + (target_pose[:3] - start_pose[:3]) * ratio
        start_rot = R.from_rotvec(start_pose[3:])
        target_rot = R.from_rotvec(target_pose[3:])
        slerp = Slerp([0.0, 1.0], R.concatenate([start_rot, target_rot]))
        rotvec = slerp([ratio]).as_rotvec()[0]
        return np.concatenate([pos, rotvec])

    # ── motion primitives ────────────────────────────────────────────────────

    def pick(self, T_cam_grasp, cur_eef_pose=None, is_need_retreat=True, grasp_offset=[0, 0, 0]):
        """Execute pick sequence."""
        if not self.is_connected:
            raise ConnectionError("Robot is not connected.")
        self.use_wrist_camera = cur_eef_pose is not None

        grasp_pose = self.convert_hand_eye(T_cam_grasp, cur_eef_pose=cur_eef_pose)
        grasp_pose[:3] += np.array(grasp_offset)
        approach_pose = self._generate_approach_pose(T_cam_grasp, cur_eef_pose=cur_eef_pose)

        self.gripper.open_gripper()
        self.controller.moveptp(approach_pose, zone_radius="Z50")
        self.controller.movel(grasp_pose, speed=self.robot_speed, acceleration=self.robot_acceleration)
        self.gripper.close_gripper()
        time.sleep(1.0)
        if is_need_retreat:
            self.controller.movel(approach_pose, speed=self.robot_speed, acceleration=self.robot_acceleration)

    def move(self, T_target, frame='base', offset: list = [0, 0, 0],
             default_ort=None, pure_translation=True):
        """Move robot to target pose.

        Args:
            T_target: 4x4 SE3 or 6D pose
            frame: 'base' or 'camera'
            offset: XYZ offset applied in base frame
            default_ort: if set, override orientation part of the 6D pose
            pure_translation: keep current orientation, only change position
        """
        if not self.is_connected:
            raise ConnectionError("Robot is not connected.")

        if frame == 'camera':
            target_pose = self.convert_hand_eye(T_target)
        elif frame == 'base':
            if T_target.shape == (4, 4):
                target_pose = self.transfer_T_to_pose6d(T_target)
            else:
                target_pose = T_target.copy()
        else:
            raise ValueError(f"Invalid frame: {frame}")

        target_pose[0] += offset[0]
        target_pose[1] += offset[1]
        target_pose[2] += offset[2]

        if pure_translation:
            current_pose = self.controller.get_tcp_pose()
            target_pose[3:] = current_pose[3:]
            logger.info(f"Target Pose after applying pure translation: {target_pose}")
        if default_ort is not None:
            target_pose[3:] = default_ort

        self.controller.move_cartesian(target_pose, speed=self.robot_speed)

    def oriented_place(self, T_base_place, place_offset=0.05,
                       need_hand_eye_conversion=False,
                       yaw_rotation=0.0, pitch_rotation=0.0,
                       cur_eef_pose=None):
        """Place with specified yaw/roll adjustment for Flexiv, Reorientation task. Offset is in base frame x axis direction"""
        logger.info(f"T_cam_place: {T_base_place}")
        if need_hand_eye_conversion:
            place_pose = self.convert_hand_eye(T_base_place, in_se4=True)
            R_place_mat = R.from_rotvec(self.DEFAULT_PLACE_ORIENTATION).as_matrix()
            place_pose[:3, :3] = R_place_mat
            place_pose[2, 3] += place_offset
        else:
            place_pose = T_base_place
            place_pose[2, 3] += place_offset

        pitch_rad = np.deg2rad(pitch_rotation)
        yaw_rad = np.deg2rad(yaw_rotation)

        # FlexivController.apply_rpy_rotation(T, roll, pitch, yaw)
        place_pose_rotated = FlexivController.apply_rpy_rotation(
            place_pose, 0.0, pitch_rad, yaw_rad)
        place_pose_rotated[:3, 3] = place_pose[:3, 3]

        pose_rotate_yaw = FlexivController.apply_rpy_rotation(
            place_pose, 0.0, pitch_rad, 0.0)
        rotvec_new = R.from_matrix(pose_rotate_yaw[:3, :3]).as_rotvec()
        cur_translation = cur_eef_pose[:3, 3]
        self.controller.move_cartesian(
            np.concatenate([cur_translation, rotvec_new]), speed=self.robot_speed)

        # Flexiv is 7-DOF: wrist yaw is joint 6 (0-indexed)
        yaw_rotation = max(min(yaw_rotation, 0), -180)
        self.controller.move_joints(6, yaw_rotation)
        self.place(place_pose_rotated, place_offset=[0.0, 0.0, 0.0],
                   retract_distance=0.15)

    def place(self, T_base_place, place_offset=[0.0, 0.0, 0.065],
              need_hand_eye_conversion=False, cur_eef_pose=None,
              slight_open=False, retract_distance=0.05):
        """Execute place sequence."""
        if not self.is_connected:
            raise ConnectionError("Robot is not connected.")

        if need_hand_eye_conversion:
            place_pose = self.convert_hand_eye(T_base_place, in_se4=True,
                                               cur_eef_pose=cur_eef_pose)
            R_place_mat = R.from_rotvec(self.DEFAULT_PLACE_ORIENTATION).as_matrix()
            place_pose[:3, :3] = R_place_mat
            place_pose[:3, 3] += np.array(place_offset)
            place_pose = self.transfer_T_to_pose6d(place_pose)
        else:
            place_pose = self.transfer_T_to_pose6d(T_base_place)

        clearance_height = self.config.get('clearance_height', 0.18)
        logger.info(f"Place Pose (Base Frame): {place_pose}")
        # 1. Lift straight up from retreat pose (keep retreat orientation)
        up_pose = place_pose.copy()
        up_pose[2] += clearance_height
        self.controller.moveptp(up_pose, zone_radius="Z50")
        # 2: Move in XY plane at clearance height (transition to place orientation)
        xy_place_pose = place_pose.copy()
        xy_place_pose[2] = up_pose[2]
        self.controller.moveptp(xy_place_pose, zone_radius="Z50")
        # 3: Move straight down to place pose
        down_pose = place_pose.copy()
        self.controller.moveptp(down_pose, zone_radius="Z50")

        if slight_open:
            partial_width = (self.gripper.close_width
                             + (self.gripper.open_width - self.gripper.close_width) * 0.3)
            self.gripper.open_gripper(width=partial_width)
        else:
            self.gripper.open_gripper()

        retreat_pose = place_pose.copy()
        retreat_pose[2] += retract_distance
        self.controller.moveptp(retreat_pose, zone_radius="Z50")
        self.reset()

    # ── state / utility ──────────────────────────────────────────────────────

    def get_eef_pose(self):
        if self.is_connected:
            return self.controller.get_tcp_pose()
        raise ConnectionError("Robot is not connected.")

    def move_to_home(self):
        if self.is_connected:
            self.controller.move_to_home()
        else:
            raise ConnectionError("Robot is not connected.")

    def reset(self):
        if self.is_connected:
            self.controller.move_to_home()
            self.gripper.open_gripper()
        else:
            raise ConnectionError("Robot is not connected.")

    def shutdown(self, move_home: bool = False):
        """Safely teardown gripper + robot connections.

        Args:
            move_home: When True, try resetting to home before disconnect.
        """
        logger.info("ActionWrapper shutdown start (move_home=%s)", move_home)
        try:
            if move_home and self.is_connected:
                try:
                    self.reset()
                except Exception as exc:
                    logger.warning("Reset failed during shutdown: %s", exc)
        finally:
            try:
                self.gripper.disconnect()
            except Exception as exc:
                logger.warning("Gripper disconnect warning: %s", exc)
            try:
                self.controller.disconnect()
            except Exception as exc:
                logger.warning("Controller disconnect warning: %s", exc)
            self.is_connected = False
