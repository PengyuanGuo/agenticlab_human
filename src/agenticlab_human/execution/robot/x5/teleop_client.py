#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X5 远程遥操作客户端

读取 T265/编码器，映射为机器人坐标系增量命令，发送给服务端。
支持实时延迟统计（可选）。
"""

import time
import threading
import numpy as np
from typing import Optional
import sys
import os
import socket
import argparse

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("错误: 需要安装 scipy")
    sys.exit(1)

# pynput 的导入在 run() 中处理
keyboard = None

# 导入本地模块（EncoderReader, T265Reader）
# 为避免重复代码，直接从 collect_data_single_person.py 导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from collect_data_single_person import EncoderReader, T265Reader, list_t265_devices, load_dual_arm_teleop_config

# 导入协议
from protocol import send_msg, recv_msg, create_control_command


class RemoteTeleopClient:
    """远程遥操作客户端"""
    
    def __init__(
        self,
        server_host: str,
        server_port: int,
        left_t265_serial: str = "",
        right_t265_serial: str = "",
        left_encoder_port: str = "",
        right_encoder_port: str = "",
        left_encoder_direction: int = 1,
        right_encoder_direction: int = 1,
        left_encoder_scale: float = 2.0,
        right_encoder_scale: float = 2.0,
        control_frequency: float = 30.0,
        position_scale: float = 1.5,
        # 旧的自适应滤波参数（已弃用，保留用于向后兼容）
        filter_alpha: float = 0.8,
        rot_deadband_deg: float = 1.0,
        rot_fullspeed_deg: float = 8.0,
        rot_alpha_min: float = 0.05,
        pos_deadband_mm: float = 5.0,
        pos_fullspeed_mm: float = 25.0,
        pos_alpha_min: float = 0.05,
        pos_alpha_max: float = 0.6,
        # 中值滤波 + 限幅保护参数（当前使用）
        pos_delta_limit_mm: float = 5.0,
        rot_delta_limit_deg: float = 0.5,
        t265_rot_center_offset_m: Optional[np.ndarray] = None,
        human_to_robot_direction: str = "opposite",
        enable_latency_display: bool = True,
        test_mode: bool = False,
        esc_long_press_threshold: float = 1.5,
        close_threshold: float = 0.2,
        open_threshold: float = 0.5,
        verbose: bool = True,
        single_hand_mode: str = None  # None, 'left', 'right'
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.position_scale = position_scale
        self.verbose = verbose
        self.enable_latency_display = enable_latency_display
        self.test_mode = test_mode
        self.esc_long_press_threshold = esc_long_press_threshold
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        
        # T265 旋转中心偏移
        if t265_rot_center_offset_m is None:
            self.t265_rot_center_offset_m = np.array([0.0, -0.04, 0.0], dtype=float)
        else:
            self.t265_rot_center_offset_m = np.array(t265_rot_center_offset_m, dtype=float).reshape(3)
        
        # 自适应滤波参数
        self.rot_deadband_rad = float(np.deg2rad(rot_deadband_deg))
        self.rot_fullspeed_rad = float(np.deg2rad(rot_fullspeed_deg))
        self.rot_alpha_min = float(rot_alpha_min)
        self.rot_alpha_max = float(filter_alpha)
        
        self.pos_deadband_m = float(pos_deadband_mm) / 1000.0
        self.pos_fullspeed_m = float(pos_fullspeed_mm) / 1000.0
        self.pos_alpha_min = float(pos_alpha_min)
        self.pos_alpha_max = float(pos_alpha_max)
        
        # 中值滤波 + 限幅保护参数（当前使用，内部存储为米和弧度）
        self.pos_delta_limit_m = float(pos_delta_limit_mm) / 1000.0  # mm -> m
        self.rot_delta_limit_rad = float(np.deg2rad(rot_delta_limit_deg))  # deg -> rad
        
        # 中值滤波缓冲区（存储最近3个时刻的增量）
        self.median_filter_window = 3
        self.left_pos_buffer: list = []  # 存储最近3个位置增量
        self.left_rotvec_buffer: list = []  # 存储最近3个旋转向量增量
        self.right_pos_buffer: list = []
        self.right_rotvec_buffer: list = []
        
        # 旧的滤波状态（已弃用，保留实现但不使用）
        self.left_last_filtered_pos: Optional[np.ndarray] = None
        self.left_last_filtered_quat: Optional[R] = None
        self.right_last_filtered_pos: Optional[np.ndarray] = None
        self.right_last_filtered_quat: Optional[R] = None
        
        # 旧的历史增量状态（已弃用，保留实现但不使用）
        self.left_last_delta_pos: Optional[np.ndarray] = None
        self.left_last_delta_rotvec: Optional[np.ndarray] = None
        self.left_prev_delta_pos: Optional[np.ndarray] = None  # 上上一次
        self.left_prev_delta_rotvec: Optional[np.ndarray] = None
        self.right_last_delta_pos: Optional[np.ndarray] = None
        self.right_last_delta_rotvec: Optional[np.ndarray] = None
        self.right_prev_delta_pos: Optional[np.ndarray] = None
        self.right_prev_delta_rotvec: Optional[np.ndarray] = None
        
        # T265 读取器
        self.left_t265 = T265Reader(serial_number=left_t265_serial)
        self.right_t265 = T265Reader(serial_number=right_t265_serial)
        
        # 编码器读取器
        self.left_encoder: Optional[EncoderReader] = None
        self.right_encoder: Optional[EncoderReader] = None
        self._left_encoder_port = left_encoder_port
        self._right_encoder_port = right_encoder_port
        self._left_encoder_direction = left_encoder_direction
        self._right_encoder_direction = right_encoder_direction
        self._left_encoder_scale = left_encoder_scale
        self._right_encoder_scale = right_encoder_scale
        
        # 夹爪控制
        self.gripper_max_angle = 90.0
        self.last_left_encoder_angle = 0.0
        self.last_right_encoder_angle = 0.0
        # 期望的夹爪宽度状态（0=闭合，1=张开）
        self.left_desired_gripper = 0.0
        self.right_desired_gripper = 0.0
        
        # T265 参考位姿
        self.left_t265_ref_pos: Optional[np.ndarray] = None
        self.left_t265_ref_rot: Optional[R] = None
        self.right_t265_ref_pos: Optional[np.ndarray] = None
        self.right_t265_ref_rot: Optional[R] = None

        # T265 上一帧位姿（用于逐周期增量）
        self.left_t265_last_pos: Optional[np.ndarray] = None
        self.left_t265_last_rot: Optional[R] = None
        self.right_t265_last_pos: Optional[np.ndarray] = None
        self.right_t265_last_rot: Optional[R] = None
        
        self.human_to_robot_direction = human_to_robot_direction
        
        # 旋转轴映射矩阵
        if self.human_to_robot_direction == "opposite":
            self.M_rotvec = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
        else:
            self.M_rotvec = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=float)
        
        # 离合器状态
        self.is_following = True
        
        # 状态机：WaitingInit / Running / IdleBetweenEpisodes
        self._state = "WaitingInit"
        self._running = False
        self._record_mode = True
        
        # Socket
        self._sock: Optional[socket.socket] = None
        
        # 序列号（用于延迟统计）
        self._seq = 0
        
        # 延迟统计
        self._latency_ema = 0.0  # 指数移动平均
        self._latency_ema_alpha = 0.1
        
        # 键盘监听
        self._kb_listener: Optional[keyboard.Listener] = None
        self._kb_lock = threading.Lock()
        self._key_pressed = {'esc': False}
        self._space_pressed = False
        self._esc_press_time = None  # ESC 按下时间（用于长按检测）
        self._pending_reset_reference = False  # 仅在 Space 松开时触发一次
        
        # 单手遥操作模式
        self._single_hand_mode = single_hand_mode  # None, 'left', 'right'
        self._swapped = False  # 是否已交换左右手映射
        self._enter_pressed = False  # ENTER 键状态（单手模式离合）
        
        # 初始化标志（延迟到 initialize() 调用）
        self._initialized = False
    
    def initialize(self) -> bool:
        """初始化传感器和连接（延迟到第一次 ESC 按下）"""
        if self._initialized:
            return True
        
        print("\n[Initializing...]")
        
        # 初始化 T265
        print("Connecting T265 devices...")
        if not self.left_t265.connect():
            print("Warning: Left T265 connection failed")
        if not self.right_t265.connect():
            print("Warning: Right T265 connection failed")
        
        # 初始化编码器
        print("Connecting encoders...")
        if self._left_encoder_port:
            self.left_encoder = EncoderReader(
                port=self._left_encoder_port,
                direction=self._left_encoder_direction,
                scale=self._left_encoder_scale
            )
            if self.left_encoder.connect():
                time.sleep(0.5)
                self.left_encoder.calibrate(print_info=True)
            else:
                print("Warning: Left encoder connection failed")
                self.left_encoder = None
        
        if self._right_encoder_port:
            self.right_encoder = EncoderReader(
                port=self._right_encoder_port,
                direction=self._right_encoder_direction,
                scale=self._right_encoder_scale
            )
            if self.right_encoder.connect():
                time.sleep(0.5)
                self.right_encoder.calibrate(print_info=True)
            else:
                print("Warning: Right encoder connection failed")
                self.right_encoder = None
        
        # 连接服务器（非测试模式）
        if not self.test_mode:
            if not self._connect_to_server():
                print("Failed to connect to server")
                return False
        else:
            print("[Test Mode] Skipping server connection")
        
        # 重置 T265 参考
        self._reset_t265_reference()
        
        self._initialized = True
        print("[Initialization Complete]")
        return True
    
    def _reset_t265_reference(self):
        """重置 T265 参考位姿"""
        if self.left_t265.is_connected:
            pos, quat = self.left_t265.get_pose()
            q = quat
            self.left_t265_ref_rot = R.from_quat([q[1], q[2], q[3], q[0]])
            self.left_t265_ref_pos = self._t265_center_pos(pos, self.left_t265_ref_rot)
            # 同步更新 last（确保下一步增量从 0 开始）
            self.left_t265_last_rot = self.left_t265_ref_rot
            self.left_t265_last_pos = self.left_t265_ref_pos.copy()
        
        if self.right_t265.is_connected:
            pos, quat = self.right_t265.get_pose()
            q = quat
            self.right_t265_ref_rot = R.from_quat([q[1], q[2], q[3], q[0]])
            self.right_t265_ref_pos = self._t265_center_pos(pos, self.right_t265_ref_rot)
            self.right_t265_last_rot = self.right_t265_ref_rot
            self.right_t265_last_pos = self.right_t265_ref_pos.copy()
        
        # 重置中值滤波缓冲区（避免使用旧的增量历史）
        self.left_pos_buffer.clear()
        self.left_rotvec_buffer.clear()
        self.right_pos_buffer.clear()
        self.right_rotvec_buffer.clear()
        
        # 重置旧的历史增量状态（已弃用，保留用于向后兼容）
        self.left_last_delta_pos = None
        self.left_last_delta_rotvec = None
        self.left_prev_delta_pos = None
        self.left_prev_delta_rotvec = None
        self.right_last_delta_pos = None
        self.right_last_delta_rotvec = None
        self.right_prev_delta_pos = None
        self.right_prev_delta_rotvec = None
        
        if self.verbose:
            print("T265 reference pose reset")
    
    def _t265_center_pos(self, pos: np.ndarray, rot: R) -> np.ndarray:
        """补偿旋转中心"""
        return np.array(pos, dtype=float).reshape(3) + rot.apply(self.t265_rot_center_offset_m)
    
    @staticmethod
    def _smoothstep01(x: float) -> float:
        """Smooth step function"""
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)
    
    def _adaptive_alpha(self, err: float, deadband: float, fullspeed: float,
                       alpha_min: float, alpha_max: float) -> float:
        """计算自适应滤波系数"""
        if err <= deadband:
            return 0.0
        if fullspeed <= deadband:
            return alpha_max
        t = (err - deadband) / (fullspeed - deadband)
        k = self._smoothstep01(t)
        return alpha_min + k * (alpha_max - alpha_min)
    
    def _apply_filter(self, target_pos: np.ndarray, target_rot: R,
                      last_filtered_pos: Optional[np.ndarray],
                      last_filtered_rot: Optional[R]) -> tuple:
        """应用自适应滤波"""
        if last_filtered_pos is None:
            last_filtered_pos = target_pos.copy()
        if last_filtered_rot is None:
            last_filtered_rot = target_rot
        
        # 位置滤波
        pos_err = float(np.linalg.norm(target_pos - last_filtered_pos))
        pos_alpha = self._adaptive_alpha(
            err=pos_err, deadband=self.pos_deadband_m,
            fullspeed=self.pos_fullspeed_m,
            alpha_min=self.pos_alpha_min, alpha_max=self.pos_alpha_max
        )
        
        if pos_alpha <= 0.0:
            filtered_pos = last_filtered_pos.copy()
        else:
            filtered_pos = last_filtered_pos + pos_alpha * (target_pos - last_filtered_pos)
        
        # 姿态滤波
        r_err = last_filtered_rot.inv() * target_rot
        rot_err = float(np.linalg.norm(r_err.as_rotvec()))
        
        rot_alpha = self._adaptive_alpha(
            err=rot_err, deadband=self.rot_deadband_rad,
            fullspeed=self.rot_fullspeed_rad,
            alpha_min=self.rot_alpha_min, alpha_max=self.rot_alpha_max
        )
        
        if rot_alpha <= 0.0:
            filtered_rot = last_filtered_rot
        else:
            # Slerp
            filtered_rot = last_filtered_rot * R.from_rotvec(rot_alpha * r_err.as_rotvec())
        
        return filtered_pos, filtered_rot
    
    def _apply_median_filter(self, buffer: list, new_value: np.ndarray) -> np.ndarray:
        """应用中值滤波
        
        Args:
            buffer: 缓冲区列表，存储最近 N 个时刻的增量
            new_value: 新的增量值
        
        Returns:
            中值滤波后的增量
        """
        # 添加新值到缓冲区
        buffer.append(new_value.copy())
        
        # 保持窗口大小
        if len(buffer) > self.median_filter_window:
            buffer.pop(0)
        
        # 如果缓冲区不足3个值，直接返回当前值
        if len(buffer) < 3:
            return new_value.copy()
        
        # 对每个分量分别计算中位数
        result = np.zeros(3)
        for i in range(3):
            values = [buf[i] for buf in buffer]
            values_sorted = sorted(values)
            result[i] = values_sorted[len(values_sorted) // 2]  # 中位数
        
        return result
    
    def _apply_clamp_limit(self, delta: np.ndarray, is_position: bool) -> np.ndarray:
        """应用限幅保护
        
        Args:
            delta: 增量向量（位置或旋转向量）
            is_position: True 表示位置增量，False 表示旋转向量增量
        
        Returns:
            限幅后的增量
        """
        if is_position:
            limit = self.pos_delta_limit_m
        else:
            limit = self.rot_delta_limit_rad
        
        # 计算增量向量的模长
        norm = np.linalg.norm(delta)
        
        # 如果超过阈值，则按比例缩放
        if norm > limit:
            return delta * (limit / norm)
        else:
            return delta.copy()
    
    @staticmethod
    def _detect_sign_flip(current: np.ndarray, last: Optional[np.ndarray], 
                          prev: Optional[np.ndarray]) -> np.ndarray:
        """检测符号反转并抑制噪声
        
        规则：
        1. 如果当前增量与上一次增量的符号相反，则把当前增量置为0
        2. 如果上一次增量为0，则判断当前增量是否与上上一次增量的符号相反，若相反则置为0
        3. 如果上两次增量均为0，则不做处理
        
        Args:
            current: 当前增量（3D向量）
            last: 上一次增量（3D向量，可能为None）
            prev: 上上一次增量（3D向量，可能为None）
        
        Returns:
            处理后的增量（可能被置为0）
        """
        result = current.copy()
        
        # 对每个分量分别判断
        for i in range(3):
            curr_val = current[i]
            
            # 如果上一次增量存在且不为0
            if last is not None and abs(last[i]) > 1e-9:
                # 判断符号是否相反
                if curr_val * last[i] < 0:  # 符号相反
                    result[i] = 0.0
            # 如果上一次增量为0或不存在，但上上一次增量存在且不为0
            elif prev is not None and abs(prev[i]) > 1e-9:
                # 判断当前增量是否与上上一次增量符号相反
                if curr_val * prev[i] < 0:  # 符号相反
                    result[i] = 0.0
            # 如果上两次增量均为0或不存在，则不做处理（保持当前值）
        
        return result
    
    def _compute_delta_command(self):
        """计算增量命令"""
        left_delta_pos = np.zeros(3)
        left_delta_rotvec = np.zeros(3)
        right_delta_pos = np.zeros(3)
        right_delta_rotvec = np.zeros(3)
        left_gripper = 0.0
        right_gripper = 0.0
        
        # 左臂（人类左手：来自 left_t265 / left_encoder）
        if self.left_t265.is_connected and self.left_t265_last_pos is not None and self.left_t265_last_rot is not None:
            try:
                t265_pos, t265_quat = self.left_t265.get_pose()
                curr_t265_rot = R.from_quat([t265_quat[1], t265_quat[2], t265_quat[3], t265_quat[0]])
                curr_t265_pos = self._t265_center_pos(t265_pos, curr_t265_rot)
                
                # T265 增量（上一帧 -> 当前帧）
                delta_p_t265 = curr_t265_pos - self.left_t265_last_pos
                delta_r_t265 = self.left_t265_last_rot.inv() * curr_t265_rot
                
                # 更新 last
                self.left_t265_last_pos = curr_t265_pos
                self.left_t265_last_rot = curr_t265_rot
                
                # 映射到临时变量 h_pos (人类左手在机器人空间中的位置增量)
                h_pos = np.zeros(3)
                if self.human_to_robot_direction == "opposite":
                    h_pos[0] = -delta_p_t265[1]
                    h_pos[1] = delta_p_t265[2]
                    h_pos[2] = -delta_p_t265[0]
                else:
                    h_pos[0] = -delta_p_t265[1]
                    h_pos[1] = -delta_p_t265[2]
                    h_pos[2] = delta_p_t265[0]
                
                h_pos *= self.position_scale
                
                # 姿态（先映射到机器人坐标系的旋转向量）
                rotvec_t265 = delta_r_t265.as_rotvec()
                h_rotvec = self.M_rotvec @ rotvec_t265

                # 应用中值滤波 + 限幅保护（使用左手缓冲区）
                h_pos = self._apply_median_filter(self.left_pos_buffer, h_pos)
                h_rotvec = self._apply_median_filter(self.left_rotvec_buffer, h_rotvec)
                
                # 限幅保护
                h_pos = self._apply_clamp_limit(h_pos, is_position=True)
                h_rotvec = self._apply_clamp_limit(h_rotvec, is_position=False)

                # 根据方向映射到相应的机器人手臂
                if self.human_to_robot_direction == "opposite":
                    # 人的左手 -> 机器人的右手
                    right_delta_pos = h_pos
                    right_delta_rotvec = h_rotvec
                else:
                    # 人的左手 -> 机器人的左手
                    left_delta_pos = h_pos
                    left_delta_rotvec = h_rotvec
            except Exception as e:
                if self.verbose:
                    print(f"Left T265 error: {e}")
        
        # 左夹爪（来自人类左侧编码器）
        if self.left_encoder is not None and self.left_encoder.is_connected:
            angle = self.left_encoder.get_angle()
            angle = max(0.0, min(self.gripper_max_angle, angle))
            gripper_width = angle / self.gripper_max_angle
            
            if self.left_desired_gripper == 0.0:
                if gripper_width > self.open_threshold:
                    self.left_desired_gripper = 1.0
            elif self.left_desired_gripper == 1.0:
                if gripper_width < self.close_threshold:
                    self.left_desired_gripper = 0.0
            
            # 根据方向映射到相应的机器人手臂
            if self.human_to_robot_direction == "opposite":
                # 人的左手夹爪 -> 机器人的右手夹爪
                right_gripper = self.left_desired_gripper
            else:
                left_gripper = self.left_desired_gripper
            self.last_left_encoder_angle = angle
        
        # 右臂（人类右手：来自 right_t265 / right_encoder）
        if self.right_t265.is_connected and self.right_t265_last_pos is not None and self.right_t265_last_rot is not None:
            try:
                t265_pos, t265_quat = self.right_t265.get_pose()
                curr_t265_rot = R.from_quat([t265_quat[1], t265_quat[2], t265_quat[3], t265_quat[0]])
                curr_t265_pos = self._t265_center_pos(t265_pos, curr_t265_rot)
                
                delta_p_t265 = curr_t265_pos - self.right_t265_last_pos
                delta_r_t265 = self.right_t265_last_rot.inv() * curr_t265_rot
                
                self.right_t265_last_pos = curr_t265_pos
                self.right_t265_last_rot = curr_t265_rot
                
                # 计算属于右手的本地映射增量
                h_pos = np.zeros(3)
                if self.human_to_robot_direction == "opposite":
                    h_pos[0] = -delta_p_t265[1]
                    h_pos[1] = -delta_p_t265[2]
                    h_pos[2] = delta_p_t265[0]
                else:
                    h_pos[0] = -delta_p_t265[2]
                    h_pos[1] = delta_p_t265[0]
                    h_pos[2] = -delta_p_t265[1]
                
                h_pos *= self.position_scale
                
                # 姿态映射
                rotvec_t265 = delta_r_t265.as_rotvec()
                h_rotvec = self.M_rotvec @ rotvec_t265

                # 应用中值滤波 + 限幅保护（使用右手缓冲区）
                h_pos = self._apply_median_filter(self.right_pos_buffer, h_pos)
                h_rotvec = self._apply_median_filter(self.right_rotvec_buffer, h_rotvec)
                
                # 限幅保护
                h_pos = self._apply_clamp_limit(h_pos, is_position=True)
                h_rotvec = self._apply_clamp_limit(h_rotvec, is_position=False)

                # 根据方向映射到相应的机器人手臂
                if self.human_to_robot_direction == "opposite":
                    # 人的右手 -> 机器人的左手
                    left_delta_pos = h_pos
                    left_delta_rotvec = h_rotvec
                else:
                    # 人的右手 -> 机器人的右手
                    right_delta_pos = h_pos
                    right_delta_rotvec = h_rotvec
            except Exception as e:
                if self.verbose:
                    print(f"Right T265 error: {e}")
        
        # 右夹爪（来自人类右侧编码器）
        if self.right_encoder is not None and self.right_encoder.is_connected:
            angle = self.right_encoder.get_angle()
            angle = max(0.0, min(self.gripper_max_angle, angle))
            gripper_width = angle / self.gripper_max_angle
            
            if self.right_desired_gripper == 0.0:
                if gripper_width > self.open_threshold:
                    self.right_desired_gripper = 1.0
            elif self.right_desired_gripper == 1.0:
                if gripper_width < self.close_threshold:
                    self.right_desired_gripper = 0.0
            
            # 根据方向映射到相应的机器人手臂
            if self.human_to_robot_direction == "opposite":
                # 人的右手夹爪 -> 机器人的左手夹爪
                left_gripper = self.right_desired_gripper
            else:
                right_gripper = self.right_desired_gripper
            self.last_right_encoder_angle = angle
        
        return left_delta_pos, left_delta_rotvec, right_delta_pos, right_delta_rotvec, left_gripper, right_gripper
    
    def _on_key_press(self, key):
        """键盘按下事件"""
        try:
            if key == keyboard.Key.esc:
                with self._kb_lock:
                    if self._esc_press_time is None:
                        self._esc_press_time = time.time()
            
            # 单手模式：ENTER 键负责离合（暂停）
            elif self._single_hand_mode and key == keyboard.Key.enter:
                with self._kb_lock:
                    if not self._enter_pressed:
                        self._enter_pressed = True
                        self.is_following = False
                        if self.verbose:
                            print("\n[Clutch] Disengaged (Enter Pressed)")
            
            # 单手模式：Space 键切换左右手映射
            elif self._single_hand_mode and key == keyboard.Key.space:
                with self._kb_lock:
                    if not self._space_pressed:
                        self._space_pressed = True
                        self._swapped = not self._swapped
                        swap_status = "Swapped (L<->R)" if self._swapped else "Normal"
                        if self.verbose:
                            print(f"\n[SingleHand] {swap_status}")
            
            # 双手模式：Space 键负责离合
            elif key == keyboard.Key.space:
                with self._kb_lock:
                    if not self._space_pressed:
                        self._space_pressed = True
                        self.is_following = False
                        if self.verbose:
                            print("\n[Clutch] Disengaged (Space Pressed)")
        except AttributeError:
            pass
    
    def _on_key_release(self, key):
        """键盘释放事件"""
        try:
            if key == keyboard.Key.esc:
                with self._kb_lock:
                    press_duration = 0.0
                    if self._esc_press_time is not None:
                        press_duration = time.time() - self._esc_press_time
                        self._esc_press_time = None
                    
                    # 判断是短按还是长按
                    if press_duration >= self.esc_long_press_threshold:
                        self._key_pressed['esc_long'] = True
                    else:
                        self._key_pressed['esc_short'] = True
            
            # 单手模式：ENTER 键恢复跟随
            elif self._single_hand_mode and key == keyboard.Key.enter:
                with self._kb_lock:
                    self._enter_pressed = False
                    self.is_following = True
                    self._pending_reset_reference = True
                    if self.verbose:
                        print("\n[Clutch] Engaged (Enter Released)")
                # 重置 T265 参考
                if self._initialized:
                    self._reset_t265_reference()
            
            # 单手模式：Space 键仅释放状态（不影响跟随）
            elif self._single_hand_mode and key == keyboard.Key.space:
                with self._kb_lock:
                    self._space_pressed = False
            
            # 双手模式：Space 键恢复跟随
            elif key == keyboard.Key.space:
                with self._kb_lock:
                    self._space_pressed = False
                    self.is_following = True
                    # 仅在 Space 松开时触发一次 reset_reference（服务端重置机器人参考；客户端重置 T265 参考）
                    self._pending_reset_reference = True
                    if self.verbose:
                        print("\n[Clutch] Engaged (Space Released)")
                # 重置 T265 参考（服务端也会重置机器人参考）
                if self._initialized:
                    self._reset_t265_reference()
        except AttributeError:
            pass
    
    def _handle_keyboard_input(self):
        """处理键盘输入（状态机）"""
        with self._kb_lock:
            # 长按 ESC：退出
            if self._key_pressed.get('esc_long', False):
                print("\n[ESC Long Press] Exiting...")
                self._running = False
                self._key_pressed['esc_long'] = False
                return
            
            # 短按 ESC：状态切换
            if self._key_pressed.get('esc_short', False):
                self._key_pressed['esc_short'] = False
                
                if self._state == "WaitingInit":
                    # 第一次短按：初始化并开始第一轮
                    print("\n[ESC] Initializing and starting episode 1...")
                    if self.initialize():
                        self._state = "Running"
                        self._record_mode = True
                        self._send_episode_control(start_episode=True, reset_reference=True)
                    else:
                        print("Initialization failed")
                
                elif self._state == "Running":
                    # 采集中短按：结束本轮
                    print("\n[ESC] Stopping current episode...")
                    self._state = "IdleBetweenEpisodes"
                    self._record_mode = False
                    self._send_episode_control(stop_episode=True, go_home=True)
                
                elif self._state == "IdleBetweenEpisodes":
                    # 空闲中短按：开始下一轮
                    print("\n[ESC] Starting next episode...")
                    self._state = "Running"
                    self._record_mode = True
                    self._reset_t265_reference()  # 重置 T265 参考位姿，防止跳变
                    self._send_episode_control(start_episode=True, reset_reference=True)
    
    def _send_episode_control(self, start_episode=False, stop_episode=False, go_home=False, reset_reference=False):
        """发送 episode 控制命令（立即发送，不等主循环）"""
        if self.test_mode or self._sock is None:
            if self.test_mode:
                print(f"[Test Mode] Would send: start={start_episode}, stop={stop_episode}, go_home={go_home}, reset={reset_reference}")
            return
        
        try:
            cmd = create_control_command(
                left_delta_pos_m=[0, 0, 0],
                left_delta_rotvec_rad=[0, 0, 0],
                right_delta_pos_m=[0, 0, 0],
                right_delta_rotvec_rad=[0, 0, 0],
                left_gripper=0.0,
                right_gripper=0.0,
                follow=False,
                record=False,
                start_episode=start_episode,
                stop_episode=stop_episode,
                go_home=go_home,
                reset_reference=reset_reference,
                seq=self._seq
            )
            send_msg(self._sock, cmd)
            self._seq += 1
        except Exception as e:
            if self.verbose:
                print(f"Send episode control failed: {e}")
    
    def _connect_to_server(self) -> bool:
        """连接到服务端"""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((self.server_host, self.server_port))
            print(f"Connected to server {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Connection to server failed: {e}")
            return False
    
    def run(self):
        """运行客户端主循环"""
        global keyboard
        if keyboard is None:
            try:
                from pynput import keyboard as kb_pkg
                keyboard = kb_pkg
            except ImportError:
                print("错误: 无法导入 pynput，请运行 'pip install pynput'")
                return
        
        print("\n" + "=" * 50)
        print("  X5 Remote Teleop Client")
        if self.test_mode:
            print("  Mode: TEST (no server connection)")
        else:
            print(f"  Server: {self.server_host}:{self.server_port}")
        
        if self._single_hand_mode:
            print(f"  Single Hand Mode: {self._single_hand_mode.upper()}")
            print("\n  [ESC] Short press: Init / Stop episode / Start next episode")
            print(f"  [ESC] Long press ({self.esc_long_press_threshold}s): Exit")
            print("  [Space] Press: Toggle L<->R mapping")
            print("  [Enter] Hold: Disengage (pause)")
        else:
            print("\n  [ESC] Short press: Init / Stop episode / Start next episode")
            print(f"  [ESC] Long press ({self.esc_long_press_threshold}s): Exit")
            print("  [Space] Hold: Disengage (pause recording)")
            print("  [Space] Release: Re-engage")
        if self.enable_latency_display:
            print("  Latency display: Enabled")
        print("=" * 50)
        print("\nWaiting for ESC to start...\n")
        
        # 启动键盘监听
        self._kb_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self._kb_listener.start()
        
        # 初始化键盘字典
        with self._kb_lock:
            self._key_pressed['esc_short'] = False
            self._key_pressed['esc_long'] = False
        
        self._running = True
        # reset_reference 改为事件触发（Space 松开时一次），不再周期触发
        
        try:
            while self._running:
                start_time = time.time()
                
                # 处理键盘输入（状态机切换）
                self._handle_keyboard_input()
                
                # 状态显示（每秒刷新一次）
                if self._seq % 30 == 0:
                    state_display = f"State: {self._state}"
                    if self._state == "WaitingInit":
                        print(f"\r{state_display} - Press ESC to start", end='', flush=True)
                    elif self._state == "IdleBetweenEpisodes":
                        print(f"\r{state_display} - Press ESC for next episode", end='', flush=True)
                
                # 只有在 Running 状态才周期发送控制命令
                if self._state == "Running" and self._initialized:
                    # Space 松开后仅触发一次 reset_reference
                    with self._kb_lock:
                        reset_reference_flag = bool(self._pending_reset_reference)
                        self._pending_reset_reference = False
                    
                    # 计算增量命令
                    left_dp, left_dr, right_dp, right_dr, left_g, right_g = self._compute_delta_command()
                    
                    # 单手遥操作模式处理：交换或置零
                    if self._single_hand_mode:
                        if self._single_hand_mode == 'left':
                            if self._swapped:
                                # 交换：左手数据发送给右臂，左臂置零
                                right_dp, right_dr, right_g = left_dp.copy(), left_dr.copy(), left_g
                                left_dp = np.zeros(3, dtype=float)
                                left_dr = np.zeros(3, dtype=float)
                                left_g = 0.0
                            else:
                                # 正常：左手数据发送给左臂，右臂置零
                                right_dp = np.zeros(3, dtype=float)
                                right_dr = np.zeros(3, dtype=float)
                                right_g = 0.0
                        else:  # right
                            if self._swapped:
                                # 交换：右手数据发送给左臂，右臂置零
                                left_dp, left_dr, left_g = right_dp.copy(), right_dr.copy(), right_g
                                right_dp = np.zeros(3, dtype=float)
                                right_dr = np.zeros(3, dtype=float)
                                right_g = 0.0
                            else:
                                # 正常：右手数据发送给右臂，左臂置零
                                left_dp = np.zeros(3, dtype=float)
                                left_dr = np.zeros(3, dtype=float)
                                left_g = 0.0
                    
                    # 创建控制命令
                    cmd = create_control_command(
                        left_delta_pos_m=left_dp.tolist(),
                        left_delta_rotvec_rad=left_dr.tolist(),
                        right_delta_pos_m=right_dp.tolist(),
                        right_delta_rotvec_rad=right_dr.tolist(),
                        left_gripper=float(left_g),
                        right_gripper=float(right_g),
                        follow=self.is_following,
                        record=self._record_mode and self.is_following,  # 离合时不记录
                        reset_reference=reset_reference_flag,
                        seq=self._seq,
                        client_send_ts=time.time()
                    )
                    
                    # 发送命令（非测试模式）
                    if not self.test_mode:
                        if not send_msg(self._sock, cmd):
                            print("\nSend command failed, connection lost")
                            break
                    
                    # 实时显示发送的数据
                    if self._seq % 2 == 0:
                        status_str = "FOLLOW" if self.is_following else "CLUTCH"
                        record_str = "REC" if (self._record_mode and self.is_following) else "OFF"
                        
                        # 格式化显示位置增量 (mm)
                        l_p = [x * 1000 for x in left_dp]
                        r_p = [x * 1000 for x in right_dp]
                        
                        display_text = (
                            f"[{self._state[:3].upper()}|{status_str}|{record_str}] "
                            f"L_dP(mm):{l_p[0]:+5.1f},{l_p[1]:+5.1f},{l_p[2]:+5.1f} G:{left_g:.1f} | "
                            f"R_dP(mm):{r_p[0]:+5.1f},{r_p[1]:+5.1f},{r_p[2]:+5.1f} G:{right_g:.1f}"
                        )
                        
                        if self.enable_latency_display and self._latency_ema > 0 and not self.test_mode:
                            display_text += f" | Latency: {self._latency_ema:.1f}ms"
                        
                        print(f"\r{display_text}", end='', flush=True)
                    
                    # 如果启用延迟显示，等待 ACK（非测试模式）
                    if self.enable_latency_display and not self.test_mode:
                        ack = recv_msg(self._sock, timeout=0.1)
                        if ack and ack.get('type') == 'ack' and ack.get('seq') == self._seq:
                            rtt = (time.time() - cmd['client_send_ts']) * 1000.0  # ms
                            self._latency_ema = (1.0 - self._latency_ema_alpha) * self._latency_ema + self._latency_ema_alpha * rtt
                    
                    self._seq += 1
                
                # 频率控制
                elapsed = time.time() - start_time
                sleep_time = max(0, self.dt - elapsed)
                time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\nInterrupted...")
        finally:
            # 发送退出命令（非测试模式）
            if not self.test_mode and self._sock:
                try:
                    # 如果正在录制，先发送 stop_episode 以保存数据
                    if self._state == "Running" and self._record_mode:
                        print("\n[Exit] Sending stop_episode to save data...")
                        stop_cmd = create_control_command(
                            left_delta_pos_m=[0,0,0], left_delta_rotvec_rad=[0,0,0],
                            right_delta_pos_m=[0,0,0], right_delta_rotvec_rad=[0,0,0],
                            left_gripper=0.0, right_gripper=0.0,
                            follow=False, stop_episode=True, go_home=True, seq=self._seq
                        )
                        send_msg(self._sock, stop_cmd)
                        self._seq += 1
                        time.sleep(0.1)  # 给服务端一点时间处理
                    
                    # 发送 exit 命令
                    print("[Exit] Sending exit command to server...")
                    exit_cmd = create_control_command(
                        left_delta_pos_m=[0,0,0], left_delta_rotvec_rad=[0,0,0],
                        right_delta_pos_m=[0,0,0], right_delta_rotvec_rad=[0,0,0],
                        left_gripper=0.0, right_gripper=0.0,
                        exit=True, seq=self._seq
                    )
                    send_msg(self._sock, exit_cmd)
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to send exit commands: {e}")
            
            self._cleanup()
    
    def _cleanup(self):
        """清理资源"""
        self._running = False
        
        if self._kb_listener:
            self._kb_listener.stop()
        
        if self._sock:
            try:
                self._sock.close()
            except:
                pass
        
        # 断开 T265
        if self.left_t265.is_connected:
            self.left_t265.disconnect()
        
        if self.right_t265.is_connected:
            self.right_t265.disconnect()
        
        # 断开编码器
        if self.left_encoder is not None and self.left_encoder.is_connected:
            self.left_encoder.disconnect()
        
        if self.right_encoder is not None and self.right_encoder.is_connected:
            self.right_encoder.disconnect()
        
        if self.verbose:
            print("Client closed")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="X5 Remote Teleop Client")
    parser.add_argument('--host', default='192.168.1.185', help='Server host')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--teleop-config', default='../dual_arm_teleop_config.yaml',
                        help='Teleop config file path')
    parser.add_argument('--env-config', default='../dual_arm_env_config.yaml',
                        help='Environment config file path')
    parser.add_argument('--disable-latency', action='store_true',
                        help='Disable latency display')
    parser.add_argument('--test', action='store_true',
                        help='Test mode (no server connection)')
    parser.add_argument('--esc-long-press', type=float, default=1.5,
                        help='ESC long press threshold in seconds (default: 1.5)')
    parser.add_argument('--single-hand', choices=['left', 'right'], default=None,
                        help='Single hand mode: use one hand to control both arms')
    args = parser.parse_args()
    
    # 相对路径转绝对路径
    config_path = os.path.join(os.path.dirname(__file__), args.teleop_config)
    teleop_config = load_dual_arm_teleop_config(config_path)
    
    # 读取环境配置文件以获取夹爪阈值
    env_config_path = os.path.join(os.path.dirname(__file__), args.env_config)
    try:
        import yaml
        with open(env_config_path, 'r', encoding='utf-8') as f:
            env_config = yaml.safe_load(f)
        gripper_config = env_config.get('gripper', {})
        close_threshold = gripper_config.get('close_threshold', 0.2)
        open_threshold = gripper_config.get('open_threshold', 0.5)
    except Exception as e:
        print(f"Warning: Failed to load env config from {env_config_path}: {e}")
        print("Using default gripper thresholds: close_threshold=0.2, open_threshold=0.5")
        close_threshold = 0.2
        open_threshold = 0.5
    
    # 列出可用的 T265 设备
    print("\nDetecting T265 devices...")
    available_t265 = list_t265_devices()
    print(f"Found {len(available_t265)} T265: {available_t265}")
    
    if len(available_t265) < 2:
        print("Warning: Less than 2 T265 detected!")
    
    # 从配置文件获取参数
    left_arm_config = teleop_config.get('left_arm', {})
    right_arm_config = teleop_config.get('right_arm', {})
    control_config = teleop_config.get('control', {})
    
    client = RemoteTeleopClient(
        server_host=args.host,
        server_port=args.port,
        left_t265_serial=left_arm_config.get('t265_serial', ''),
        right_t265_serial=right_arm_config.get('t265_serial', ''),
        left_encoder_port=left_arm_config.get('encoder_port', ''),
        right_encoder_port=right_arm_config.get('encoder_port', ''),
        left_encoder_direction=left_arm_config.get('encoder_direction', 1),
        right_encoder_direction=right_arm_config.get('encoder_direction', 1),
        left_encoder_scale=left_arm_config.get('encoder_scale', 2.0),
        right_encoder_scale=right_arm_config.get('encoder_scale', 2.0),
        control_frequency=control_config.get('frequency', 30.0),
        position_scale=control_config.get('position_scale', 1.5),
        # 旧的自适应滤波参数（已弃用，保留用于向后兼容）
        filter_alpha=control_config.get('alpha', 0.8),
        rot_deadband_deg=control_config.get('rot_deadband_deg', 1.0),
        rot_fullspeed_deg=control_config.get('rot_fullspeed_deg', 8.0),
        rot_alpha_min=control_config.get('rot_alpha_min', 0.05),
        pos_deadband_mm=control_config.get('pos_deadband_mm', 1.0),
        pos_fullspeed_mm=control_config.get('pos_fullspeed_mm', 10.0),
        pos_alpha_min=control_config.get('pos_alpha_min', 0.05),
        pos_alpha_max=control_config.get('pos_alpha_max', 0.6),
        # 中值滤波 + 限幅保护参数（当前使用）
        pos_delta_limit_mm=control_config.get('pos_delta_limit_mm', 50.0),
        rot_delta_limit_deg=control_config.get('rot_delta_limit_deg', 5.7),
        human_to_robot_direction=control_config.get('human_to_robot_direction', 'opposite'),
        enable_latency_display=not args.disable_latency,
        test_mode=args.test,
        esc_long_press_threshold=args.esc_long_press,
        close_threshold=close_threshold,
        open_threshold=open_threshold,
        verbose=True,
        single_hand_mode=args.single_hand
    )
    
    client.run()


if __name__ == "__main__":
    main()
