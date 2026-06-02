import cv2
import numpy as np
from vlm_robobench.modules import rtde_controller
import yaml
import logging
import time
from scipy.spatial.transform import Rotation as R
from vlm_robobench.modules.gripper_controller import GripperController
from vlm_robobench.modules.rtde_controller import RTDEController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionWrapper:
    """Class for generating primitive actions for robotic manipulation."""

    def __init__(self, cfg, rtde_cfg, gripper_cfg, camera_cfg):
        self.config = cfg['ActionWrapper'] # TODO: Specify config structure
        self.gripper = GripperController(gripper_cfg)
        self.rtde = RTDEController(rtde_cfg)
        
        self.wrist_cam_cfg = camera_cfg['RealSense']
        self.shoulder_cam_cfg = camera_cfg['Kinect']

        self.robot_speed = self.config.get('robot_speed', 0.05)
        self.robot_acceleration = self.config.get('robot_acceleration', 0.1)
        # Connect to the robot
        self.is_connected = self.rtde.connect()
        if not self.is_connected:
            raise ConnectionError("Failed to connect to UR5e robot.")
        self.reset()
        self.use_wrist_camera = False
        self.DEFAULT_PLACE_ORIENTATION = np.array([2.18896741, -2.24723741, -0.00680875])  # Facing down

    def set_use_wrist_camera(self, use_wrist_camera: bool):
        self.use_wrist_camera = use_wrist_camera

    def convert_hand_eye(self, T_cam_grasp, in_se4 = False, cur_eef_pose=None):
        """Convert grasp pose from camera frame to robot base frame"""
        if cur_eef_pose is None: #not self.use_wrist_camera:
            handeye_rotation = np.array(self.shoulder_cam_cfg.get('handeye_calibration', {}).get('rotation', np.eye(3)))
            handeye_translation = np.array(self.shoulder_cam_cfg.get('handeye_calibration', {}).get('translation', np.zeros(3)))
            T_handeye = np.eye(4)
            T_handeye[:3, :3] = handeye_rotation
            T_handeye[:3, 3] = handeye_translation
        else:  # use_wrist_camera is True
            handeye_rotation = np.array(self.wrist_cam_cfg.get('handeye_calibration', {}).get('rotation', np.eye(3)))
            handeye_translation = np.array(self.wrist_cam_cfg.get('handeye_calibration', {}).get('translation', np.zeros(3)))
            T_handeye = np.eye(4)
            T_handeye[:3, :3] = handeye_rotation
            T_handeye[:3, 3] = handeye_translation
            T_base_ee = cur_eef_pose # 4x4 matrix
            T_handeye = T_base_ee @ T_handeye # T_be (target) = T_be (current) @ T_ec @ T_cg @ T_ge
        # 1. Transform grasp from camera frame to robot base frame
        T_base_grasp = T_handeye @ T_cam_grasp
        
        # 2. Get end-effector to grasp transformation
        # This represents how the end-effector should be oriented relative to the grasp
        R_eg = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])  # Grasp pose to end effector frame
        T_ge = np.eye(4)
        T_ge[:3, :3] = R_eg.T
        T_ge[:3, 3] = np.array([0, 0, 0])
        
        # 3. Compute end-effector pose in base frame
        T_base_ee = T_base_grasp @ T_ge
        if in_se4:
            return T_base_ee
        # For UR5e, we need to convert to 6D pose representation
        # 4. Convert to 6D pose representation
        pose_6d = self.transfer_T_to_pose6d(T_base_ee)

        return pose_6d

    def _generate_approach_pose(self, T_cam_grasp, cur_eef_pose=None):
        """Create a retreat pose by moving back from grasp pose"""
        approach_distance = self.config.get('approach_distance', 0.1)
        T_cam_approach = T_cam_grasp.copy()
        R_cg = T_cam_grasp[:3, :3]
        t_cg = T_cam_grasp[:3, 3]
        # local‐x offset
        delta = np.array([-approach_distance, 0.0, 0.0])
        T_cam_approach[:3, 3] = t_cg + R_cg @ delta
        return self.convert_hand_eye(T_cam_approach, cur_eef_pose=cur_eef_pose) # Convert to base frame

    def transfer_T_to_pose6d(self, T):
        """Convert 4x4 transformation matrix to 6D pose representation"""
        translation = T[:3, 3]
        rotation_vec = R.from_matrix(T[:3, :3]).as_rotvec()
        return np.concatenate([translation, rotation_vec])

    def transfer_pose6d_to_T(self, pose6d):
        """Convert 6D pose representation to 4x4 transformation matrix"""
        translation = pose6d[:3]
        rotation_vec = pose6d[3:]
        rotation = R.from_rotvec(rotation_vec).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        return T
    
    def transfer_pos_rot_to_T(self, position, rotation):
        """Convert position and rotation matrix to 4x4 transformation matrix"""
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position
        return T
    
    def pick(self, T_cam_grasp, cur_eef_pose=None, is_need_retreat=True, grasp_offset=[0, 0, 0]):
        """Execute pick sequence"""
        if not self.is_connected:
            logger.error("Robot is not connected.")
            raise ConnectionError("Robot is not connected.")
        if cur_eef_pose is not None:
            self.use_wrist_camera = True
        else:
            self.use_wrist_camera = False
        grasp_pose = self.convert_hand_eye(T_cam_grasp, cur_eef_pose=cur_eef_pose)
        grasp_pose[:3] += np.array(grasp_offset) # offset for the grasp pose in base frame
        approach_pose = self._generate_approach_pose(T_cam_grasp, cur_eef_pose=cur_eef_pose)
        # while True:
        #     time.sleep(1)
        # 1. Open gripper
        self.gripper.open_gripper()
        # 2. Move to approach pose
        self.rtde.move_cartesian(approach_pose, speed=self.robot_speed, acceleration=self.robot_acceleration)
        # 3. Move to grasp pose
        self.rtde.move_cartesian(grasp_pose, speed=self.robot_speed, acceleration=self.robot_acceleration)
        # 4. Close gripper
        self.gripper.close_gripper()
        # 5. Retreat to approach pose
        if is_need_retreat:
            self.rtde.move_cartesian(approach_pose, speed=self.robot_speed, acceleration=self.robot_acceleration)
        #TODO: Add error handling and verification steps
      
    def move(self, T_target, frame='base', offset: list = [0,0,0], default_ort = None, pure_translation=True):
        """Move robot to target pose.
        
        Args:
            T_target: Target pose as 4x4 transformation matrix
            frame: Reference frame of the input pose. Either 'base' (robot base frame, default) 
                   or 'camera' (camera frame)
            offset: List of 3 floats representing XYZ offset to apply to the target pose
        """
        if not self.is_connected:
            logger.error("Robot is not connected.")
            raise ConnectionError("Robot is not connected.")
        
        if frame == 'camera':
            # Convert from camera frame to base frame
            target_pose = self.convert_hand_eye(T_target)
            # logger.info(f"Target Pose (Camera Frame -> Base Frame): {target_pose}")
        elif frame == 'base':
            # Input is already in base frame, convert to 6D pose if needed
            if T_target.shape == (4, 4):
                target_pose = self.transfer_T_to_pose6d(T_target)
            else:
                target_pose = T_target  # Already in 6D format
            # logger.info(f"Target Pose (Base Frame): {target_pose}")
        else:
            raise ValueError(f"Invalid frame parameter: {frame}. Must be 'base' or 'camera'")
        # Apply offset
        target_pose[0] += offset[0]
        target_pose[1] += offset[1]
        target_pose[2] += offset[2]
        
        if pure_translation:
            # Keep current orientation
            current_pose = self.rtde.get_tcp_pose()
            target_pose[3:] = current_pose[3:]
            logger.info(f"Target Pose after applying pure translation: {target_pose}")
        if default_ort is not None:
            target_pose[3:] = default_ort

        self.rtde.move_cartesian(target_pose, speed=self.robot_speed, acceleration=self.robot_acceleration)    
    
    def oriented_place(self, T_base_place, place_offset=0.05, need_hand_eye_conversion=False, yaw_rotation=0.0, roll_rotation=0.0, cur_eef_pose=None):
        print(f"T_cam_place: {T_base_place}")
        """ T_base_place here is in camera frame """
        if need_hand_eye_conversion:
            place_pose = self.convert_hand_eye(T_base_place, in_se4=True) # robot base frame target pose
            print(f"T_base_place_converted: {place_pose}")
            R_place_mat = R.from_rotvec(self.DEFAULT_PLACE_ORIENTATION).as_matrix()
            place_pose[:3, :3] = R_place_mat
            place_pose[2, 3] += place_offset  # Add offset in Z direction
        else:
            place_pose = T_base_place
            place_pose[2, 3] += place_offset  # Add offset in Z direction
        # Modify rotation to make bottle upright with specified yaw and roll
        roll_rad = np.deg2rad(roll_rotation)
        yaw_rad = np.deg2rad(yaw_rotation)
        place_pose_rotated = self.rtde.apply_rpy_rotation(place_pose, yaw_rad, roll_rad, 0.0) 
        place_pose_rotated[:3, 3] = place_pose[:3, 3]
        print(f"place_pose_rotated: {place_pose_rotated}")
        

        # Rotation first, then place
        pose_rotate_yaw = self.rtde.apply_rpy_rotation(place_pose, 0.0, roll_rad, 0.0)
        rotvec_new = R.from_matrix(pose_rotate_yaw[:3, :3]).as_rotvec()
        cur_translation = cur_eef_pose[:3, 3]
        self.rtde.move_cartesian(np.concatenate([cur_translation, rotvec_new]), speed=self.robot_speed, acceleration=self.robot_acceleration)
        yaw_rotation = max(min(yaw_rotation, 0), -180)
        self.rtde.move_joints(5, yaw_rotation)
        self.place(place_pose_rotated, place_offset = [0.0, 0.0, 0.0], retract_distance=0.15)

        
    def place(self, T_base_place, place_offset=[0.0, 0.0, 0.065], need_hand_eye_conversion=False, cur_eef_pose=None, slight_open=False, retract_distance=0.05):
        """Execute place sequence"""
        if not self.is_connected:
            logger.error("Robot is not connected.")
            raise ConnectionError("Robot is not connected.")
        
        if need_hand_eye_conversion:
            place_pose = self.convert_hand_eye(T_base_place, in_se4=True, cur_eef_pose=cur_eef_pose)
            R_place_mat = R.from_rotvec(self.DEFAULT_PLACE_ORIENTATION).as_matrix()
            place_pose[:3, :3] = R_place_mat
            # Apply offset
            place_pose[:3, 3] += np.array(place_offset)
            place_pose = self.transfer_T_to_pose6d(place_pose)
        else:
            place_pose = self.transfer_T_to_pose6d(T_base_place)
        
        clearance_height = self.config.get('clearance_height', 0.18)
        logger.info(f"Place Pose (Base Frame): {place_pose}")
        # 1. Lift straight up from retreat pose (keep retreat orientation)
        up_pose = place_pose.copy()
        up_pose[2] += clearance_height
        self.rtde.move_cartesian(up_pose, speed=self.robot_speed, acceleration=self.robot_acceleration)
        # 2: Move in XY plane at clearance height (transition to place orientation)
        xy_place_pose = place_pose.copy()
        xy_place_pose[2] = up_pose[2]
        self.rtde.move_cartesian(xy_place_pose, speed=self.robot_speed, acceleration=self.robot_acceleration)
        # 3: Move straight down to place pose
        down_pose = place_pose.copy()
        self.rtde.move_cartesian(down_pose, speed=self.robot_speed, acceleration=self.robot_acceleration)
        # 4: Open gripper to release object
        self.gripper.open_gripper(slight_open=slight_open)
        # 5: Retreat straight up (5cm)
        retreat_pose = place_pose.copy()
        retreat_pose[2] += retract_distance
        self.rtde.move_cartesian(retreat_pose, speed=self.robot_speed, acceleration=self.robot_acceleration)
        self.reset()
    
    def get_eef_pose(self):
        if self.is_connected:
            return self.rtde.get_tcp_pose()
        else:
            raise ConnectionError("Robot is not connected.")
    def move_to_home(self):
        if self.is_connected:
            self.rtde.move_to_home()
        else:
            raise ConnectionError("Robot is not connected.")
        
    def reset(self):
        if self.is_connected:
            self.rtde.move_to_home()
            self.gripper.open_gripper()
        else:
            raise ConnectionError("Robot is not connected.")
        
