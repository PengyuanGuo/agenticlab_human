#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X5 远程遥操作服务端

接收客户端控制命令，控制机器人执行，并在本地记录数据和显示相机画面。
"""

import time
import numpy as np
from typing import Optional, Dict, Any
import sys
import os
import socket
import argparse
import cv2
import queue
import threading
from collections import deque
import faulthandler
import traceback

# 全局崩溃日志句柄（在 main() 中初始化），用于将线程内异常也落盘
_CRASH_FH = None

def _crash_log(text: str) -> None:
    global _CRASH_FH  # noqa: PLW0603
    try:
        if _CRASH_FH is None:
            return
        _CRASH_FH.write(text + "\n")
        _CRASH_FH.flush()
    except Exception:
        pass

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
# 添加父目录，以便导入 env_dual
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp
except ImportError:
    print("错误: 需要安装 scipy")
    sys.exit(1)

import xapi.api as x5
from env_dual import create_dual_arm_env_from_config

# 导入协议
sys.path.insert(0, os.path.dirname(__file__))
from protocol import send_msg, recv_msg, create_ack_response

# 导入辅助函数（从本地采集脚本）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from collect_data_single_person import save_episode_to_hdf5, display_observations


class RemoteTeleopServer:
    """远程遥操作服务端"""
    
    def __init__(
        self,
        env,
        host: str = "192.168.1.119",
        port: int = 5555,
        watchdog_timeout: float = 0.5,
        show_preview: bool = True,
        record_directory: str = "recorded_data",
        logs_directory: str = "logs",
        verbose: bool = True
    ):
        # x5 库互斥锁，防止多线程调用导致崩溃
        self._x5_lock = threading.Lock()
        
        self.env = env

        # ----------------------------
        # x5 线程亲和性（关键稳定性修复）
        # ----------------------------
        # 目标：发送线程启动后，主线程不再触碰 x5（避免“串行但跨线程”导致的 C 层静默崩溃）
        self._robot_state_cache_lock = threading.Lock()
        self._robot_state_cache = np.zeros(16, dtype=np.float32)
        self._x5_get_robot_state = None  # 原始的 env._get_robot_state（内部会调用 x5）
        self._last_robot_state_refresh_perf = 0.0
        self._robot_state_refresh_period_s = 0.1  # 10Hz 刷新即可满足记录/预览

        if hasattr(self.env, "_get_robot_state"):
            self._x5_get_robot_state = self.env._get_robot_state

            def cached_get_robot_state():
                with self._robot_state_cache_lock:
                    return self._robot_state_cache.copy()

            # 替换为“读缓存”版本：主线程 get_observation 不再调用 x5
            self.env._get_robot_state = cached_get_robot_state
            if verbose:
                print("Patched env._get_robot_state to cached mode (avoid x5 in main thread)")
        
        self.host = host
        self.port = port
        self.watchdog_timeout = watchdog_timeout
        self.show_preview = show_preview
        self.record_directory = record_directory
        self.logs_directory = logs_directory
        self.verbose = verbose
        
        # 参考位姿（用于增量控制）
        self.left_robot_ref_pos: Optional[np.ndarray] = None
        self.left_robot_ref_rot: Optional[R] = None
        self.right_robot_ref_pos: Optional[np.ndarray] = None
        self.right_robot_ref_rot: Optional[R] = None

        # 参考点位（保留 uf/tf/cfg/外轴，用于 servol / movj(Point) 时稳定 IK 分支）
        self.left_robot_ref_point: Optional[x5.Point] = None
        self.right_robot_ref_point: Optional[x5.Point] = None

        # 统一使用的 MovPointAdd 参数（与测试用例一致）
        self._movj_add = x5.MovPointAdd(vel=50, acc=50)
        
        # 记录数据
        self._episode_data = []
        self._record_mode = False
        self._episode_count = 0  # 已完成的 episode 数量
        
        # Socket
        self._server_sock: Optional[socket.socket] = None
        self._client_sock: Optional[socket.socket] = None
        
        # 运行状态
        self._running = False

        # 事件去抖/边沿触发：避免同一事件被连续触发导致刷屏/参考反复重置
        self._last_reset_reference_flag = False
        self._last_start_episode_flag = False
        self._last_stop_episode_flag = False
        
        # 初始化参考位姿
        self._reset_robot_reference()
        # 初始化一次机器人关节状态缓存（此时发送线程尚未启动，允许在主线程触碰 x5）
        self._refresh_robot_state_cache(force=True)

        # servol 参数（在线实时控制 TCP）
        self._use_servol = True
        # 注意：servol 的 cmdt 建议与服务端插值发送频率匹配（默认 50Hz => 0.02s）
        self._servol_cmdt = 1.0 / 50.0
        self._servol_gain = 30.0
        self._servol_vel = 100.0
        self._servol_acc = 100.0

        # 使能/报警恢复去抖（避免在异常情况下刷屏/频繁 reset）
        self._last_recover_time = 0.0
        self._recover_cooldown_s = 1.0

        # 服务端实时显示下发给机械臂的运动指令（已移除频率限制，实时打印）
        self._print_commands = True
        # 注意：_print_cmd_hz 和 _last_cmd_print_time 已不再使用，保留以兼容命令行参数
        self._print_cmd_hz = 30.0
        self._last_cmd_print_time = 0.0
        
        # 命令缓冲队列（用于30Hz频率发送）
        self._command_queue = queue.Queue()
        self._send_thread: Optional[threading.Thread] = None
        self._send_thread_running = False
        self._send_frequency = 50.0  # Hz (Interpolated control frequency)
        self._initial_queue_size = 3  # 启动时队列需要收集的数据条数
        self._send_started = False  # 标记是否已经开始发送（已收集够初始数据）
        self._is_idle = False  # 标记是否处于空闲状态（stop_episode 后为 True，start_episode 后为 False）
        
        # Calibration and Interpolation
        self._is_calibrating = True
        self._client_dt = 1.0 / 30.0  # Initial guess, will be estimated
        self._trajectory_buffer = []  # Stores target states for interpolation
        self._calibration_data = [] # Stores commands during calibration
        
        # 关节命令历史（用于保存时的时间戳匹配）
        # 每条记录: {'timestamp': float, 'left_joints': np.ndarray(8), 'right_joints': np.ndarray(8)}
        self._joint_command_history = []

        # 安全检查相关状态
        self._should_record_current_step = True  # 当前步骤是否应该录制（IK 验证失败时为 False）
        
        # 消息日志缓冲
        self._log_buffer = []
        self._log_lock = threading.Lock()
        
        # 数据采集线程（30Hz 固定频率）
        self._record_thread: Optional[threading.Thread] = None
        self._record_thread_running = False
        self._record_frequency = 30.0  # Hz
        self._record_started = False  # 标记是否已开始采集（发送线程启动后置为 True）

    def _log_msg(self, msg: str) -> None:
        """记录日志：在录制数据时写入缓冲，未录制时直接打印到终端（但严格过滤掉控制通信的高频刷屏消息）"""
        # 是否为高频通信指令
        is_control_msg = ("[RECV " in msg) or ("[SEND " in msg)
        
        if getattr(self, '_record_mode', False):
            # 只要在录制记录模式期间（包括最后保存计算数据时），所有日志统一落库不刷终端
            with self._log_lock:
                self._log_buffer.append(msg)
        else:
            # 未录制期间：高频控制消息直接丢弃；状态提示和告警正常输出到终端
            if not is_control_msg:
                print(msg, flush=True)

    def _save_log_buffer(self) -> None:
        """保存日志缓冲到文件"""
        with self._log_lock:
            if not self._log_buffer:
                return
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            safe_prompt = self.env._prompt.replace(" ", "_").replace("/", "_").replace("\\", "_") if getattr(self.env, '_prompt', None) else ""
            safe_prompt = safe_prompt.replace(",", "").replace(".", "")
            max_prompt_len = 80
            if len(safe_prompt) > max_prompt_len:
                safe_prompt = safe_prompt[:max_prompt_len]
                
            filename = f"episode_log_{timestamp_str}_{safe_prompt}.txt" if safe_prompt else f"episode_log_{timestamp_str}.txt"
            filepath = os.path.join(self.logs_directory, filename)
            
            try:
                os.makedirs(self.logs_directory, exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("\n".join(self._log_buffer))
                    f.write("\n")
                print(f"Log saved: {filepath}")
            except Exception as e:
                print(f"Failed to save log: {e}")
            self._log_buffer.clear()

    def _refresh_robot_state_cache(self, force: bool = False) -> None:
        """刷新 env.get_observation() 里使用的机器人关节状态缓存。

        说明：
        - env_dual.get_observation() 会调用 env._get_robot_state()。
        - 我们把 env._get_robot_state() patch 成“读缓存”，让主线程不再触碰 x5。
        - 缓存由发送线程定期刷新，从而把 x5 调用收敛到发送线程。
        """
        if self._x5_get_robot_state is None:
            return
        now = time.perf_counter()
        if (not force) and (now - float(self._last_robot_state_refresh_perf) < float(self._robot_state_refresh_period_s)):
            return
        self._last_robot_state_refresh_perf = now
        try:
            with self._x5_lock:
                state = self._x5_get_robot_state()
            if state is None:
                return
            state_np = np.array(state, dtype=np.float32)
            with self._robot_state_cache_lock:
                self._robot_state_cache = state_np.copy()
        except Exception as e:
            if self.verbose:
                print(f"[WARN] refresh robot state cache failed: {e}", flush=True)

    def _go_home_servoj(self, duration_s: float = 3.0, hz: float = 30.0) -> None:
        """用单线程 servoj 回零（避免 env.reset 内部再开线程导致 x5 跨线程）。"""
        cmdt = 1.0 / max(1e-6, float(hz))
        left_handle = getattr(getattr(self.env, "_left_robot", None), "handle", None)
        right_handle = getattr(getattr(self.env, "_right_robot", None), "handle", None)

        left_target = None
        right_target = None
        try:
            if isinstance(left_handle, int) and left_handle != -1 and getattr(self.env, "_left_robot_enabled", False):
                j = [float(v) for v in getattr(self.env, "_initial_left_joint_pos")][:7]
                try:
                    left_target = x5.Joint(j[0], j[1], j[2], j[3], j[4], j[5], j[6], 0.0, 0.0)
                except TypeError:
                    left_target = x5.Joint(j[0], j[1], j[2], j[3], j[4], j[5], j[6])
            
            if isinstance(right_handle, int) and right_handle != -1 and getattr(self.env, "_right_robot_enabled", False):
                j = [float(v) for v in getattr(self.env, "_initial_right_joint_pos")][:7]
                try:
                    right_target = x5.Joint(j[0], j[1], j[2], j[3], j[4], j[5], j[6], 0.0, 0.0)
                except TypeError:
                    right_target = x5.Joint(j[0], j[1], j[2], j[3], j[4], j[5], j[6])
        except Exception as e:
            if self.verbose:
                print(f"[WARN] build go_home targets failed: {e}", flush=True)
            return

        if left_target is None and right_target is None:
            print("[WARN] left_target and right_target are None")
            return

        if self.verbose:
            print("\n[Go Home] servoj to initial joints (single-thread)", flush=True)

        t0 = time.time()
        while self._send_thread_running and (time.time() - t0) < float(duration_s):
            with self._x5_lock:
                if left_target is not None and isinstance(left_handle, int) and left_handle != -1:
                    x5.servoj(left_handle, left_target, float(cmdt), 0, 5, 100, 100)
                if right_target is not None and isinstance(right_handle, int) and right_handle != -1:
                    x5.servoj(right_handle, right_target, float(cmdt), 0, 5, 100, 100)
            time.sleep(cmdt)

        # gripper home（仅在夹爪启用时操作）
        try:
            if getattr(self.env, "_left_gripper_enabled", False):
                getattr(self.env, "_left_gripper").set_position(int(0))
            if getattr(self.env, "_right_gripper_enabled", False):
                getattr(self.env, "_right_gripper").set_position(int(0))
        except Exception:
            pass

    @staticmethod
    def _extract_r123_from_point(p: x5.Point) -> tuple:
        """提取 Point 的 (r1,r2,r3) / (pose.e1,e2,e3)，用于 7 轴/外轴/冗余参数保持一致。"""
        r1 = getattr(p, "r1", None)
        r2 = getattr(p, "r2", None)
        r3 = getattr(p, "r3", None)
        if r1 is not None or r2 is not None or r3 is not None:
            return float(r1 or 0.0), float(r2 or 0.0), float(r3 or 0.0)
        pose = getattr(p, "pose", None)
        if pose is not None:
            e1 = getattr(pose, "e1", 0.0)
            e2 = getattr(pose, "e2", 0.0)
            e3 = getattr(pose, "e3", 0.0)
            return float(e1), float(e2), float(e3)
        return 0.0, 0.0, 0.0

    @staticmethod
    def _normalize_cfg(cfg_raw):
        """将 cfg 归一化为 tuple(int,...)（兼容 ctypes 数组等）。"""
        if cfg_raw is None:
            return (0, 0, 0, 1)
        if isinstance(cfg_raw, tuple):
            return cfg_raw
        try:
            return tuple(int(v) for v in cfg_raw)
        except Exception:
            return cfg_raw

    def _try_recover_enable(self, handle: int, tag: str = "") -> None:
        """当出现 XERR_STAT_ENABLE/系统未使能 时，尝试清队列+复位报警+重新上使能（去抖）。"""
        now = time.time()
        if now - float(self._last_recover_time) < float(self._recover_cooldown_s):
            return
        self._last_recover_time = now

        try:
            if hasattr(x5, "stop"):
                x5.stop(handle)
            if hasattr(x5, "abort"):
                x5.abort(handle)
        except Exception:
            pass

        try:
            alarms = x5.get_system_alarm_info(handle)
            if isinstance(alarms, list) and len(alarms) > 0:
                if self.verbose:
                    print(f"\n[WARN] {tag} 检测到报警，尝试 reset 清除: {alarms}")
                # x5.reset(handle)
                time.sleep(0.2)
        except Exception:
            pass

        try:
            x5.enable_servo(handle, False)
            x5.set_system_mode(handle, 100)
            x5.enable_servo(handle, True)
        except Exception:
            pass
    
    def _reset_robot_reference(self):
        """重置机器人参考位姿"""
        # 加锁保护 x5 库调用
        with self._x5_lock:
            self._reset_robot_reference_impl()

    def _reset_robot_reference_impl(self):
        """重置机器人参考位姿的实现（不加锁）"""
        # 左臂
        if self.env._left_robot is not None and self.env._left_robot_enabled:
            try:
                cp = self.env._left_robot.get_cpoint()
                # get_cpoint 在不同封装下可能返回 Point 或 Pose
                robot_pose = getattr(cp, "pose", None) or cp
                if robot_pose:
                    self.left_robot_ref_pos = np.array([
                        robot_pose.x / 1000.0,  # mm -> m
                        robot_pose.y / 1000.0,
                        robot_pose.z / 1000.0
                    ])
                    rot_euler_deg = np.array([robot_pose.a, robot_pose.b, robot_pose.c])
                    self.left_robot_ref_rot = R.from_euler('xyz', rot_euler_deg, degrees=True)
                    # 保存完整 Point（含 uf/tf/cfg/外轴），用于 servol 时避免 IK 分支跳变
                    if isinstance(cp, x5.Point):
                        self.left_robot_ref_point = cp
                    else:
                        # fallback：优先直接用 handle 向 xapi 取完整 Point（含 cfg/uf/tf/外轴）
                        left_handle = getattr(self.env._left_robot, "handle", None)
                        if isinstance(left_handle, int) and left_handle != -1:
                            try:
                                self.left_robot_ref_point = x5.get_cpoint(left_handle)
                            except Exception:
                                self.left_robot_ref_point = x5.Point(
                                    (robot_pose.x, robot_pose.y, robot_pose.z, robot_pose.a, robot_pose.b, robot_pose.c, 0.0, 0.0, 0.0),
                                    0, 0, (0, 0, 0, 1)
                                )
                        else:
                            self.left_robot_ref_point = x5.Point(
                                (robot_pose.x, robot_pose.y, robot_pose.z, robot_pose.a, robot_pose.b, robot_pose.c, 0.0, 0.0, 0.0),
                                0, 0, (0, 0, 0, 1)
                            )
                    if self.verbose:
                        print(f"Left robot ref pose: {self.left_robot_ref_pos}")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Cannot get left robot pose: {e}")
        
        # 右臂
        if self.env._right_robot is not None and self.env._right_robot_enabled:
            try:
                cp = self.env._right_robot.get_cpoint()
                robot_pose = getattr(cp, "pose", None) or cp
                if robot_pose:
                    self.right_robot_ref_pos = np.array([
                        robot_pose.x / 1000.0,
                        robot_pose.y / 1000.0,
                        robot_pose.z / 1000.0
                    ])
                    rot_euler_deg = np.array([robot_pose.a, robot_pose.b, robot_pose.c])
                    self.right_robot_ref_rot = R.from_euler('xyz', rot_euler_deg, degrees=True)
                    if isinstance(cp, x5.Point):
                        self.right_robot_ref_point = cp
                    else:
                        right_handle = getattr(self.env._right_robot, "handle", None)
                        if isinstance(right_handle, int) and right_handle != -1:
                            try:
                                self.right_robot_ref_point = x5.get_cpoint(right_handle)
                            except Exception:
                                self.right_robot_ref_point = x5.Point(
                                    (robot_pose.x, robot_pose.y, robot_pose.z, robot_pose.a, robot_pose.b, robot_pose.c, 0.0, 0.0, 0.0),
                                    0, 0, (0, 0, 0, 1)
                                )
                        else:
                            self.right_robot_ref_point = x5.Point(
                                (robot_pose.x, robot_pose.y, robot_pose.z, robot_pose.a, robot_pose.b, robot_pose.c, 0.0, 0.0, 0.0),
                                0, 0, (0, 0, 0, 1)
                            )
                    if self.verbose:
                        print(f"Right robot ref pose: {self.right_robot_ref_pos}")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Cannot get right robot pose: {e}")
        
        if self.verbose:
            print("Robot reference pose reset")
            # 调试：打印参考点的 cfg 信息
            if self.left_robot_ref_point is not None:
                cfg = getattr(self.left_robot_ref_point, "cfg", None)
                if cfg is not None:
                    try:
                        print(f"  Left ref cfg: {list(cfg)}")
                    except Exception:
                        print(f"  Left ref cfg: {cfg}")
            if self.right_robot_ref_point is not None:
                cfg = getattr(self.right_robot_ref_point, "cfg", None)
                if cfg is not None:
                    try:
                        print(f"  Right ref cfg: {list(cfg)}")
                    except Exception:
                        print(f"  Right ref cfg: {cfg}")
    
    def _send_thread_worker(self):
        """发送线程：先 0.5s 标定客户端频率，再插值到 50Hz 发送"""
        period = 1.0 / float(self._send_frequency)  # e.g. 50Hz => 0.02s
        next_time = time.perf_counter() + period

        # ----------------------------
        # Phase 1: Calibration (0.5s)
        # ----------------------------
        calib_cmds = []  # list[(recv_perf, cmd)]
        calib_start = None
        if self.verbose:
            print("\n[Send Thread] Calibration: collecting commands for 0.5s (no robot control)")

        while self._send_thread_running and self._is_calibrating:
            try:
                cmd = self._command_queue.get(timeout=0.1)
            except queue.Empty:
                # 检查是否应该退出
                if not self._send_thread_running:
                    return
                # 如果还没有开始收集数据，继续等待
                if calib_start is None:
                    # 标定阶段也刷新一次缓存，避免主线程 get_observation 读到全 0
                    self._refresh_robot_state_cache(force=False)
                    continue
                # 如果已经开始收集，检查是否已经超过0.5s
                elapsed = time.perf_counter() - calib_start
                if elapsed >= 0.5 and len(calib_cmds) >= 2:
                    break
                continue

            recv_perf = float(cmd.get("_server_recv_perf", time.perf_counter()))
            if calib_start is None:
                calib_start = recv_perf
            calib_cmds.append((recv_perf, cmd))

            # Stop when we have >=2 samples and elapsed >=0.5s
            if (recv_perf - calib_start) >= 0.5 and len(calib_cmds) >= 2:
                break

            # prevent unbounded growth
            if len(calib_cmds) > 500:
                calib_cmds = calib_cmds[-200:]
                calib_start = calib_cmds[0][0]

        if not self._send_thread_running:
            return

        # Estimate client dt from calibration samples (median inter-arrival)
        dts = []
        for i in range(1, len(calib_cmds)):
            dt = float(calib_cmds[i][0] - calib_cmds[i - 1][0])
            if dt > 1e-4:
                dts.append(dt)
        if len(dts) > 0:
            self._client_dt = float(np.median(np.array(dts, dtype=float)))
        else:
            self._client_dt = 1.0 / 20.0
        client_hz = 1.0 / max(1e-6, self._client_dt)
        if self.verbose:
            print(f"[Send Thread] Calibration done: estimated client rate ~ {client_hz:.2f} Hz (dt={self._client_dt:.4f}s), samples={len(calib_cmds)}")

        # Keep only last 2 commands from calibration window
        last_two = [c for (_, c) in calib_cmds[-2:]]
        calib_cmds.clear()

        # Reset reference to current robot pose before starting control, then apply last 2 deltas
        self._reset_robot_reference()
        self._refresh_robot_state_cache(force=True)

        # Trajectory buffer: deque[(t, state_dict)]
        self._trajectory_buffer = deque(maxlen=1000)
        traj_t_last = 0.0

        # Keyframe 0: current reference pose at t=0
        state0 = self._current_target_state(seq=-1)
        self._trajectory_buffer.append((traj_t_last, state0))

        # Apply last 2 commands to build initial motion (uses delta as incremental step)
        for cmd in last_two:
            state = self._process_accumulate_command(cmd)
            if state is None:
                continue
            traj_t_last += float(self._client_dt)
            self._trajectory_buffer.append((traj_t_last, state))

        play_wall_t0 = time.perf_counter()
        self._is_calibrating = False
        self._send_started = True

        # ----------------------------
        # Phase 2: 50Hz Control Loop
        # ----------------------------
        seg_key = None
        seg_slerp_left = None
        seg_slerp_right = None
        seg_t0 = 0.0
        seg_t1 = 0.0
        seg_s0 = None
        seg_s1 = None

        while self._send_thread_running:
            try:
                # consume all queued commands and append keyframes
                while True:
                    try:
                        cmd = self._command_queue.get_nowait()
                    except queue.Empty:
                        break

                    # 这些事件涉及 x5/参考位姿，必须在发送线程里处理（避免主线程触碰 x5）
                    if bool(cmd.get("_go_home_edge", False)):
                        self._go_home_servoj()
                        self._reset_robot_reference()
                        self._refresh_robot_state_cache(force=True)
                        traj_t_last = 0.0
                        play_wall_t0 = time.perf_counter()
                        self._trajectory_buffer.clear()
                        self._trajectory_buffer.append((0.0, self._current_target_state(seq=-1)))
                        seg_key = None
                        # 设置空闲状态，暂停发送命令和打印日志
                        self._is_idle = True
                        if self.verbose:
                            print("\n[INFO] Entered idle state (stop_episode)", flush=True)
                        continue
                    if bool(cmd.get("_reset_reference_edge", False)) or bool(cmd.get("_start_episode_edge", False)):
                        # start_episode 时退出空闲状态
                        if bool(cmd.get("_start_episode_edge", False)):
                            self._is_idle = False
                            if self.verbose:
                                print("\n[INFO] Exited idle state (start_episode)", flush=True)
                        self._reset_robot_reference()
                        self._refresh_robot_state_cache(force=True)
                        traj_t_last = 0.0
                        play_wall_t0 = time.perf_counter()
                        self._trajectory_buffer.clear()
                        self._trajectory_buffer.append((0.0, self._current_target_state(seq=-1)))
                        seg_key = None
                        continue

                    state = self._process_accumulate_command(cmd)
                    if state is None:
                        continue
                    traj_t_last += float(self._client_dt)
                    self._trajectory_buffer.append((traj_t_last, state))

                now = time.perf_counter()
                play_t = float(now - play_wall_t0)
                # 周期性刷新 robot_state 缓存（供主线程 observation 使用）
                self._refresh_robot_state_cache(force=False)

                # clamp to available trajectory end
                if len(self._trajectory_buffer) > 0:
                    last_t = float(self._trajectory_buffer[-1][0])
                    if play_t > last_t:
                        play_t = last_t

                # drop old frames so that buffer[0] <= play_t <= buffer[1]
                while len(self._trajectory_buffer) >= 2 and float(self._trajectory_buffer[1][0]) <= play_t:
                    self._trajectory_buffer.popleft()
                    seg_key = None  # segment changed

                # 空闲状态时跳过发送命令
                if self._is_idle:
                    pass
                elif len(self._trajectory_buffer) == 0:
                    # nothing to send yet
                    pass
                elif len(self._trajectory_buffer) == 1:
                    _, hold_state = self._trajectory_buffer[0]
                    with self._x5_lock:
                        if self._send_servo_command(hold_state):
                            # needs reset
                            if self.verbose:
                                print("\n[INFO] Bufffer reset triggered by hold state error", flush=True)
                            traj_t_last = 0.0
                            play_wall_t0 = time.perf_counter()
                            self._trajectory_buffer.clear()
                            self._trajectory_buffer.append((0.0, self._current_target_state(seq=-1)))
                            seg_key = None
                            
                else:
                    seg_t0, seg_s0 = self._trajectory_buffer[0]
                    seg_t1, seg_s1 = self._trajectory_buffer[1]
                    denom = float(seg_t1 - seg_t0)
                    if denom <= 1e-9:
                        alpha = 0.0
                    else:
                        alpha = float((play_t - float(seg_t0)) / denom)
                        if alpha < 0.0:
                            alpha = 0.0
                        elif alpha > 1.0:
                            alpha = 1.0

                    new_seg_key = (float(seg_t0), float(seg_t1))
                    if seg_key != new_seg_key:
                        seg_key = new_seg_key
                        # build slerp segment with normalized [0,1] time
                        ql0 = seg_s0["left_rot"].as_quat()
                        ql1 = seg_s1["left_rot"].as_quat()
                        qr0 = seg_s0["right_rot"].as_quat()
                        qr1 = seg_s1["right_rot"].as_quat()
                        try:
                            seg_slerp_left = Slerp([0.0, 1.0], R.from_quat([ql0, ql1]))
                        except Exception as e:
                            seg_slerp_left = None
                            if self.verbose:
                                print(f"\n[WARN] Left Slerp build failed: {e}", flush=True)
                        try:
                            seg_slerp_right = Slerp([0.0, 1.0], R.from_quat([qr0, qr1]))
                        except Exception as e:
                            seg_slerp_right = None
                            if self.verbose:
                                print(f"\n[WARN] Right Slerp build failed: {e}", flush=True)

                    left_pos = (1.0 - alpha) * seg_s0["left_pos_m"] + alpha * seg_s1["left_pos_m"]
                    right_pos = (1.0 - alpha) * seg_s0["right_pos_m"] + alpha * seg_s1["right_pos_m"]
                    left_rot = seg_slerp_left([alpha])[0] if seg_slerp_left is not None else seg_s0["left_rot"]
                    right_rot = seg_slerp_right([alpha])[0] if seg_slerp_right is not None else seg_s0["right_rot"]
                    interp_state = {
                        "seq": int(seg_s1.get("seq", -1)),
                        "left_pos_m": left_pos,
                        "left_rot": left_rot,
                        "right_pos_m": right_pos,
                        "right_rot": right_rot,
                        # keep original deltas for logging if needed
                        "left_delta_pos_m": seg_s1.get("left_delta_pos_m", np.zeros(3)),
                        "left_delta_rotvec_rad": seg_s1.get("left_delta_rotvec_rad", np.zeros(3)),
                        "right_delta_pos_m": seg_s1.get("right_delta_pos_m", np.zeros(3)),
                        "right_delta_rotvec_rad": seg_s1.get("right_delta_rotvec_rad", np.zeros(3)),
                        # pass through gripper state (step function, use target value)
                        "left_gripper": seg_s1.get("left_gripper", None),
                        "right_gripper": seg_s1.get("right_gripper", None),
                    }
                    with self._x5_lock:
                        if self._send_servo_command(interp_state):
                            # needs reset
                            if self.verbose:
                                print("\n[INFO] Bufffer reset triggered by servo command error", flush=True)
                            traj_t_last = 0.0
                            play_wall_t0 = time.perf_counter()
                            self._trajectory_buffer.clear()
                            self._trajectory_buffer.append((0.0, self._current_target_state(seq=-1)))
                            seg_key = None


                # timing: wait until next 50Hz tick
                current_time = time.perf_counter()
                wait_time = next_time - current_time
                if wait_time > 0:
                    time.sleep(wait_time)
                next_time = time.perf_counter() + period

            except KeyboardInterrupt:
                # 传播 KeyboardInterrupt 到主线程
                self._send_thread_running = False
                raise
            except Exception as e:
                if self.verbose:
                    print(f"Send thread error: {e}", flush=True)
                _crash_log("\n[SendThread Exception]\n" + traceback.format_exc())
                # 发生错误时短暂休眠，避免快速循环
                if self._send_thread_running:
                    time.sleep(0.01)
                next_time = time.perf_counter() + period
    
    def _current_target_state(self, seq: int = -1) -> Dict[str, Any]:
        """从当前参考位姿生成一个 target state（绝对位姿）"""
        # 这里假设 _reset_robot_reference 已经把 ref_pos/ref_rot 初始化好
        return {
            "seq": int(seq),
            "left_pos_m": np.array(self.left_robot_ref_pos, dtype=float) if self.left_robot_ref_pos is not None else np.zeros(3),
            "left_rot": self.left_robot_ref_rot if self.left_robot_ref_rot is not None else R.identity(),
            "right_pos_m": np.array(self.right_robot_ref_pos, dtype=float) if self.right_robot_ref_pos is not None else np.zeros(3),
            "right_rot": self.right_robot_ref_rot if self.right_robot_ref_rot is not None else R.identity(),
            "left_delta_pos_m": np.zeros(3),
            "left_delta_rotvec_rad": np.zeros(3),
            "right_delta_pos_m": np.zeros(3),
            "right_delta_rotvec_rad": np.zeros(3),
        }

    def _process_accumulate_command(self, cmd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理一条客户端命令：将 delta(相对上一时刻) 累加到参考位姿，并返回当前绝对 target state"""
        follow = bool(cmd.get("follow", False))
        if not follow:
            return None

        seq = int(cmd.get("seq", -1))

        # 累加更新参考位姿（delta 为相对上一时刻位姿的偏移量）
        if self.left_robot_ref_pos is not None and self.left_robot_ref_rot is not None:
            left_dp = np.array(cmd.get("left_delta_pos_m", [0, 0, 0]), dtype=float)
            left_dr = np.array(cmd.get("left_delta_rotvec_rad", [0, 0, 0]), dtype=float)
            self.left_robot_ref_pos = self.left_robot_ref_pos + left_dp
            self.left_robot_ref_rot = self.left_robot_ref_rot * R.from_rotvec(left_dr)
        else:
            left_dp = np.zeros(3, dtype=float)
            left_dr = np.zeros(3, dtype=float)

        if self.right_robot_ref_pos is not None and self.right_robot_ref_rot is not None:
            right_dp = np.array(cmd.get("right_delta_pos_m", [0, 0, 0]), dtype=float)
            right_dr = np.array(cmd.get("right_delta_rotvec_rad", [0, 0, 0]), dtype=float)
            self.right_robot_ref_pos = self.right_robot_ref_pos + right_dp
            self.right_robot_ref_rot = self.right_robot_ref_rot * R.from_rotvec(right_dr)
        else:
            right_dp = np.zeros(3, dtype=float)
            right_dr = np.zeros(3, dtype=float)

        # 夹爪：放入 state 中传递给发送线程与 robot_state 同步控制
        left_gripper_val = None
        if self.env._left_gripper_enabled:
             # 解析命令中的夹爪值，归一化为 0-1
             val = float(cmd.get("left_gripper", 0.0))
             left_gripper_val = 1.0 if val > 0.95 else (0.0 if val < 0.05 else val)

        right_gripper_val = None
        if self.env._right_gripper_enabled:
             val = float(cmd.get("right_gripper", 0.0))
             right_gripper_val = 1.0 if val > 0.95 else (0.0 if val < 0.05 else val)

        return {
            "seq": seq,
            "left_pos_m": np.array(self.left_robot_ref_pos, dtype=float) if self.left_robot_ref_pos is not None else np.zeros(3),
            "left_rot": self.left_robot_ref_rot if self.left_robot_ref_rot is not None else R.identity(),
            "right_pos_m": np.array(self.right_robot_ref_pos, dtype=float) if self.right_robot_ref_pos is not None else np.zeros(3),
            "right_rot": self.right_robot_ref_rot if self.right_robot_ref_rot is not None else R.identity(),
            "left_delta_pos_m": left_dp,
            "left_delta_rotvec_rad": left_dr,
            "right_delta_pos_m": right_dp,
            "right_delta_rotvec_rad": right_dr,
            "left_gripper": left_gripper_val,
            "right_gripper": right_gripper_val,
        }

    def _make_target_point(self, pos_m: np.ndarray, rot: R, ref_point: x5.Point) -> x5.Point:
        """将 (pos_m, rot) 转为 x5.Point，并继承参考点的 uf/tf/cfg/外轴"""
        pos_mm = np.array(pos_m, dtype=float) * 1000.0
        euler_deg = rot.as_euler("xyz", degrees=True)

        uf_raw = getattr(ref_point, "uf", 0)
        tf_raw = getattr(ref_point, "tf", 0)
        cfg_raw = getattr(ref_point, "cfg", (0, 0, 0, 1))
        try:
            uf = int(uf_raw)
        except Exception:
            uf = 0
        try:
            tf = int(tf_raw)
        except Exception:
            tf = 0
        cfg = self._normalize_cfg(cfg_raw)
        r1, r2, r3 = self._extract_r123_from_point(ref_point)
        return x5.Point(
            (
                float(pos_mm[0]),
                float(pos_mm[1]),
                float(pos_mm[2]),
                float(euler_deg[0]),
                float(euler_deg[1]),
                float(euler_deg[2]),
                float(r1),
                float(r2),
                float(r3),
            ),
            uf,
            tf,
            cfg,
        )

    def _send_servo_command(self, state: Dict[str, Any]) -> bool:
        """发送一条（可能是插值后的）绝对位姿给机器人（使用 env 的安全检查方法）
        
        Returns:
            bool: 如果发生了 IK 失败或严重错误导致需要重置轨迹缓冲区，返回 True；否则返回 False
        """
        seq = int(state.get("seq", -1))
        left_target_mm = None
        left_target_euler = None
        right_target_mm = None
        right_target_euler = None
        needs_reset = False

        # ---- 输入合法性兜底：避免把 NaN/inf 喂给 SDK 引发 C 层崩溃 ----
        try:
            lp = np.array(state.get("left_pos_m", [0, 0, 0]), dtype=float)
            rp = np.array(state.get("right_pos_m", [0, 0, 0]), dtype=float)
            lq = np.array(state.get("left_rot").as_quat(), dtype=float) if state.get("left_rot", None) is not None else np.zeros(4)
            rq = np.array(state.get("right_rot").as_quat(), dtype=float) if state.get("right_rot", None) is not None else np.zeros(4)
            if (not np.isfinite(lp).all()) or (not np.isfinite(rp).all()) or (not np.isfinite(lq).all()) or (not np.isfinite(rq).all()):
                if self.verbose:
                    print(f"\n[WARN] Non-finite target detected, skip send. seq={seq}", flush=True)
                return False
        except Exception:
            return False

        # 构造目标点位
        left_target_point = None
        right_target_point = None
        
        # 左臂
        if self.left_robot_ref_point is not None and self.env._left_robot is not None and self.env._left_robot_enabled:
            left_handle = getattr(self.env._left_robot, "handle", None)
            if isinstance(left_handle, int) and left_handle != -1:
                try:
                    left_target_point = self._make_target_point(state["left_pos_m"], state["left_rot"], self.left_robot_ref_point)
                    left_target_mm = np.array(state["left_pos_m"], dtype=float) * 1000.0
                    left_target_euler = state["left_rot"].as_euler("xyz", degrees=True)
                except Exception as e:
                    if self.verbose:
                        print(f"\n[WARN] Failed to build left target point: {e}", flush=True)

        # 右臂
        if self.right_robot_ref_point is not None and self.env._right_robot is not None and self.env._right_robot_enabled:
            right_handle = getattr(self.env._right_robot, "handle", None)
            if isinstance(right_handle, int) and right_handle != -1:
                try:
                    right_target_point = self._make_target_point(state["right_pos_m"], state["right_rot"], self.right_robot_ref_point)
                    right_target_mm = np.array(state["right_pos_m"], dtype=float) * 1000.0
                    right_target_euler = state["right_rot"].as_euler("xyz", degrees=True)
                except Exception as e:
                    if self.verbose:
                        print(f"\n[WARN] Failed to build right target point: {e}", flush=True)

        # 使用 env 的安全方法发送命令
        if left_target_point is not None or right_target_point is not None:
            servoj_params = {
                'cmdt': self._servol_cmdt,
                'gain': self._servol_gain,
                'vel': self._servol_vel,
                'acc': self._servol_acc,
            }
            
            try:
                left_success, right_success, should_record, left_joints, right_joints = self.env.apply_cartesian_command_with_safety(
                    left_target_point=left_target_point,
                    right_target_point=right_target_point,
                    left_gripper=state.get("left_gripper", None),  # 从 state 获取夹爪值
                    right_gripper=state.get("right_gripper", None),
                    servoj_params=servoj_params,
                )
                
                # 记录关节命令到历史（用于保存时的时间戳匹配）
                if self._record_mode and (left_joints is not None or right_joints is not None):
                    cmd_entry = {
                        'timestamp': time.time(),
                        'left_joints': left_joints,   # 8 维数组 (7关节+夹爪) 或 None
                        'right_joints': right_joints  # 8 维数组 (7关节+夹爪) 或 None
                    }
                    self._joint_command_history.append(cmd_entry)
                
                # 根据安全检查结果更新录制状态
                self._should_record_current_step = should_record
                
                # IK 验证失败时：
                # 1. 强制重置参考点到当前实际位置（确保下一次计算基于有效位置）
                # 2. 标记 needs_reset，通知发送线程清空轨迹缓冲区
                if (not left_success and left_target_point is not None) or \
                   (not right_success and right_target_point is not None):
                    if self.verbose:
                         print("\n[WARN] Safety check failed (IK/Limit), resetting reference...", flush=True)
                    # 必须重置所有 reference (pos, rot, point)，否则 _process_accumulate_command 会继续基于旧的无效位置累加
                    self._reset_robot_reference_impl()
                    needs_reset = True
                        
            except Exception as e:
                if self.verbose:
                    print(f"\n[ERROR] apply_cartesian_command_with_safety failed: {e}", flush=True)
                self._should_record_current_step = False
                # 发生异常通常也意味着状态不可信，重置比较安全
                needs_reset = False # 异常情况下可能连接都断了，暂时不强求 reset 逻辑，保持原样

        # logging (optional; can be expensive at 50Hz)
        if self._print_commands:
            now = time.time()
            period = 1.0 / max(1e-6, float(self._print_cmd_hz))
            if now - float(getattr(self, "_last_cmd_print_time", 0.0)) >= period:
                self._last_cmd_print_time = now
                send_ts_str = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"

                def _fmt_delta(dp_m, dr_rad):
                    dp_mm = np.array(dp_m, dtype=float) * 1000.0
                    dr_deg = np.rad2deg(np.array(dr_rad, dtype=float))
                    return dp_mm, dr_deg

                l_dp_mm, l_dr_deg = _fmt_delta(state.get("left_delta_pos_m", [0, 0, 0]), state.get("left_delta_rotvec_rad", [0, 0, 0]))
                r_dp_mm, r_dr_deg = _fmt_delta(state.get("right_delta_pos_m", [0, 0, 0]), state.get("right_delta_rotvec_rad", [0, 0, 0]))
                mode = "SERVOL" if self._use_servol else "MOVJ"

                def _fmt_target(tp_mm, te_deg):
                    if tp_mm is None or te_deg is None:
                        return "N/A"
                    return f"P(mm):{tp_mm[0]:.1f},{tp_mm[1]:.1f},{tp_mm[2]:.1f} A(deg):{te_deg[0]:.1f},{te_deg[1]:.1f},{te_deg[2]:.1f}"

                queue_size = self._command_queue.qsize()
                record_status = "REC" if getattr(self, "_should_record_current_step", True) else "PAUSE"
                msg = (
                    f"[SEND {send_ts_str} {mode} seq={seq} qsize={queue_size} {record_status}] "
                    f"L dP(mm):{l_dp_mm[0]:+6.1f},{l_dp_mm[1]:+6.1f},{l_dp_mm[2]:+6.1f} "
                    f"dR(deg):{l_dr_deg[0]:+5.1f},{l_dr_deg[1]:+5.1f},{l_dr_deg[2]:+5.1f} "
                    f"{_fmt_target(left_target_mm, left_target_euler)} | "
                    f"R dP(mm):{r_dp_mm[0]:+6.1f},{r_dp_mm[1]:+6.1f},{r_dp_mm[2]:+6.1f} "
                    f"dR(deg):{r_dr_deg[0]:+5.1f},{r_dr_deg[1]:+5.1f},{r_dr_deg[2]:+5.1f} "
                    f"{_fmt_target(right_target_mm, right_target_euler)}"
                )
                self._log_msg(msg)
        
        return needs_reset
    
    def _start_send_thread(self):
        """启动发送线程"""
        if self._send_thread is not None and self._send_thread.is_alive():
            return  # 线程已运行
        
        # 重置发送/标定状态
        self._send_started = False
        self._is_calibrating = True
        self._send_thread_running = True
        self._send_thread = threading.Thread(target=self._send_thread_worker, daemon=True)
        self._send_thread.start()
        if self.verbose:
            print(f"Send thread started (frequency: {self._send_frequency} Hz, initial queue size: {self._initial_queue_size})")
    
    def _stop_send_thread(self):
        """停止发送线程"""
        if self._send_thread is None:
            return
        
        self._send_thread_running = False
        if self._send_thread.is_alive():
            self._send_thread.join(timeout=1.0)
            if self._send_thread.is_alive():
                if self.verbose:
                    print("Warning: Send thread did not stop within timeout")
        self._send_thread = None
        if self.verbose:
            print("Send thread stopped")
    
    def _record_thread_worker(self):
        """数据采集线程：以固定 30Hz 频率采集数据"""
        period = 1.0 / float(self._record_frequency)  # 30Hz => 0.0333s
        next_time = time.perf_counter() + period
        
        if self.verbose:
            print(f"\n[Record Thread] Started (frequency: {self._record_frequency} Hz)")
        
        while self._record_thread_running:
            try:
                # 等待发送线程开始（_send_started）
                if not self._send_started:
                    time.sleep(0.01)
                    continue
                
                # 标记采集已开始
                if not self._record_started:
                    self._record_started = True
                    if self.verbose:
                        print("[Record Thread] Recording started (send thread active)")
                
                # 只有在录制模式且安全检查通过时才采集
                if self._record_mode and self._should_record_current_step:
                    try:
                        # 获取观测（使用缓存避免跨线程调用 x5）
                        obs = self.env.get_observation()
                        if obs is not None:
                            # 深拷贝观测数据，确保图像数组不被后续帧覆盖
                            import copy
                            obs_copy = {}
                            for key, value in obs.items():
                                if key == 'images':
                                    # 深拷贝图像字典中的每个 numpy 数组
                                    obs_copy['images'] = {
                                        cam_name: img.copy() for cam_name, img in value.items()
                                    }
                                elif isinstance(value, np.ndarray):
                                    obs_copy[key] = value.copy()
                                elif isinstance(value, dict):
                                    obs_copy[key] = copy.deepcopy(value)
                                else:
                                    obs_copy[key] = value
                            step_data = {'observation': obs_copy}
                            self._episode_data.append(step_data)
                    except Exception as e:
                        if self.verbose:
                            print(f"[Record Thread] Get observation error: {e}")
                
                # 精确计时等待
                now = time.perf_counter()
                sleep_time = float(next_time - now)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                next_time = max(time.perf_counter() + period * 0.5, next_time + period)
                
            except Exception as e:
                if self.verbose:
                    print(f"[Record Thread] Error: {e}")
                time.sleep(0.01)
        
        if self.verbose:
            print("[Record Thread] Stopped")
    
    def _start_record_thread(self):
        """启动数据采集线程"""
        if self._record_thread is not None:
            return
        
        self._record_thread_running = True
        self._record_started = False
        self._record_thread = threading.Thread(target=self._record_thread_worker, daemon=True)
        self._record_thread.start()
        if self.verbose:
            print(f"Record thread started (frequency: {self._record_frequency} Hz)")
    
    def _stop_record_thread(self):
        """停止数据采集线程"""
        if self._record_thread is None:
            return
        
        self._record_thread_running = False
        if self._record_thread.is_alive():
            self._record_thread.join(timeout=1.0)
            if self._record_thread.is_alive():
                if self.verbose:
                    print("Warning: Record thread did not stop within timeout")
        self._record_thread = None
        self._record_started = False
        if self.verbose:
            print("Record thread stopped")
    
    def _start_server(self) -> bool:
        """启动服务器"""
        try:
            self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_sock.bind((self.host, self.port))
            self._server_sock.listen(1)
            # 设置超时，让它在等待 accept 时也能响应信号
            self._server_sock.settimeout(1.0)
            print(f"Server listening on {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Start server failed: {e}")
            return False
    
    def _wait_for_client(self) -> bool:
        """等待客户端连接"""
        print("Waiting for client connection... (Press CTRL+C to stop)")
        while self._running:
            try:
                self._client_sock, client_addr = self._server_sock.accept()
                print(f"Client connected from {client_addr}")
                # 连接成功后设置较短的超时，方便在循环中检查 _running 状态
                self._client_sock.settimeout(0.1)
                return True
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"Accept client failed: {e}")
                return False
        return False

    def run(self):
        """运行服务端主循环"""
        print("\n" + "=" * 50)
        print("  X5 Remote Teleop Server")
        print(f"  Listening: {self.host}:{self.port}")
        print(f"  Preview: {'Enabled' if self.show_preview else 'Disabled'}")
        print(f"  Record dir: {self.record_directory}")
        print("=" * 50 + "\n")
        
        self._running = True
        
        if not self._start_server():
            return
        
        try:
            if not self._wait_for_client():
                return
            
            # 启动发送线程（以30Hz频率发送命令）
            self._start_send_thread()
            
            # 启动数据采集线程（30Hz 固定频率）
            self._start_record_thread()
            
            # 创建显示窗口
            window_name = "X5 Server - Observations"
            if self.show_preview:
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            
            last_cmd_time = time.time()
            
            while self._running:
                # 接收命令
                try:
                    cmd = recv_msg(self._client_sock, timeout=0.1)
                except Exception as e:
                    if self.verbose:
                        print(f"Receive message exception: {e}")
                    # 接收消息异常，可能是连接关闭，退出循环
                    break
                
                if cmd is not None:
                    last_cmd_time = time.time()
                    seq = cmd.get('seq', 0)
                    follow = cmd.get('follow', False)
                    # 为发送线程记录“到达服务端的单调时间戳”，用于估计客户端发送频率
                    cmd["_server_recv_perf"] = time.perf_counter()
                    
                    # 发送 ACK
                    try:
                        ack = create_ack_response(
                            seq=seq,
                            server_recv_ts=time.time()
                        )
                        if not send_msg(self._client_sock, ack):
                            # 发送ACK失败，可能是连接关闭
                            if self.verbose:
                                print("Failed to send ACK, connection may be closed")
                            break
                    except Exception as e:
                        if self.verbose:
                            print(f"Send ACK exception: {e}")
                        # 发送ACK异常，可能是连接关闭，退出循环
                        break
                    
                    # 处理 episode 控制（边沿触发）
                    start_episode_flag = bool(cmd.get('start_episode', False))
                    stop_episode_flag = bool(cmd.get('stop_episode', False))

                    start_episode_edge = bool(start_episode_flag and not self._last_start_episode_flag)
                    stop_episode_edge = bool(stop_episode_flag and not self._last_stop_episode_flag)

                    if start_episode_edge:
                        print("\n[Start Episode] Clearing buffer and resetting reference...")
                        self._episode_data = []
                        with self._log_lock:
                            self._log_buffer.clear()
                        self._record_mode = True
                    
                    if stop_episode_edge:
                        print("\n[Stop Episode] Saving data and returning home...")
                        
                        # 保存数据
                        if self._episode_data:
                            self._episode_count += 1
                            # episode_prompt = f"{self.env._prompt}_ep{self._episode_count}" if self.env._prompt else f"episode_{self._episode_count}"
                            episode_prompt = f"{self.env._prompt}" if self.env._prompt else f"episode_{self._episode_count}"
                            # 传入机器人句柄和关节命令历史
                            left_handle = getattr(self.env._left_robot, 'handle', None)
                            right_handle = getattr(self.env._right_robot, 'handle', None)
                            save_episode_to_hdf5(
                                list(self._episode_data), self.record_directory, episode_prompt,
                                left_handle, right_handle, self._joint_command_history,
                                fps=self._record_frequency
                            )
                            print(f"Episode {self._episode_count} saved ({len(self._episode_data)} steps)")
                            self._episode_data = []
                            self._joint_command_history = []  # 清空关节命令历史
                        
                        # 保存日志
                        self._save_log_buffer()
                        
                        # 回到初始位置
                        # 注意：go_home 会触发 x5 控制，必须在发送线程中执行，避免主线程触碰 x5
                        # 通过 cmd["_go_home_edge"] 交给发送线程处理
                        
                        # 在保存日志最后，才把记录模式关掉，以便拦截计算期间的日志
                        self._record_mode = False

                    self._last_start_episode_flag = start_episode_flag
                    self._last_stop_episode_flag = stop_episode_flag
                    
                    # 处理退出
                    if cmd.get('exit', False):
                        print("\nReceived exit command")
                        self._running = False
                        break
                    
                    # 重置参考位姿（Space 离合用）——边沿触发，避免连续触发
                    reset_reference_flag = bool(cmd.get('reset_reference', False))
                    reset_reference_edge = bool(reset_reference_flag and not self._last_reset_reference_flag)
                    self._last_reset_reference_flag = reset_reference_flag

                    # 将这些“涉及 x5 的事件”作为边沿信号传给发送线程处理
                    cmd["_start_episode_edge"] = start_episode_edge
                    cmd["_stop_episode_edge"] = stop_episode_edge
                    cmd["_reset_reference_edge"] = reset_reference_edge
                    # stop_episode 时始终触发回零
                    cmd["_go_home_edge"] = stop_episode_edge
                    
                    # 更新记录模式
                    self._record_mode = cmd.get('record', self._record_mode)
                    
                    # 命令入队：用于(1)估计客户端发送频率 (2)生成插值轨迹
                    # 注意：follow=False 的命令也会入队，但不会驱动机器人（_process_accumulate_command 会忽略）
                    self._command_queue.put(cmd)
                    
                    # 实时打印接收到的客户端消息（带时间戳，在入队后打印以显示准确的队列大小）
                    if self._print_commands:
                        recv_ts = time.time()
                        recv_ts_str = time.strftime("%H:%M:%S", time.localtime(recv_ts)) + f".{int((recv_ts % 1) * 1000):03d}"
                        follow_str = "FOLLOW" if follow else "CLUTCH"
                        record_str = "REC" if cmd.get('record', False) else "OFF"
                        l_dp = cmd.get('left_delta_pos_m', [0, 0, 0])
                        r_dp = cmd.get('right_delta_pos_m', [0, 0, 0])
                        l_dp_mm = [x * 1000 for x in l_dp]
                        r_dp_mm = [x * 1000 for x in r_dp]
                        queue_size = self._command_queue.qsize()  # 入队后的队列大小
                        recv_msg_str = (
                            f"[RECV {recv_ts_str} seq={seq} {follow_str} {record_str} qsize={queue_size}] "
                            f"L dP(mm):{l_dp_mm[0]:+6.1f},{l_dp_mm[1]:+6.1f},{l_dp_mm[2]:+6.1f} | "
                            f"R dP(mm):{r_dp_mm[0]:+6.1f},{r_dp_mm[1]:+6.1f},{r_dp_mm[2]:+6.1f}"
                        )
                        # 换行显示接收消息，便于与发送消息区分
                        self._log_msg(recv_msg_str)
                
                else:
                    # Watchdog: 超时未收到命令
                    if time.time() - last_cmd_time > self.watchdog_timeout:
                        if self.verbose:
                            print("Watchdog timeout, holding position", end='\r')
                
                # 获取观测（用于预览显示，数据采集由独立线程负责）
                try:
                    # 注意：x5 库调用保护已通过 Monkey Patch 在 _get_robot_state 中实现
                    obs = self.env.get_observation()
                    
                    # 显示相机预览
                    if self.show_preview:
                        recording_status = self._record_mode and self._should_record_current_step
                        display_observations(obs, window_name=window_name, recording=recording_status)
                    
                    # 打印录制进度（每 50 步）
                    if self._record_mode and len(self._episode_data) > 0 and len(self._episode_data) % 50 == 0:
                        if self.verbose:
                            print(f"Recording... Steps: {len(self._episode_data)}", end='\r')
                
                except Exception as e:
                    if self.verbose:
                        print(f"Get observation error: {e}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user (CTRL+C)...")
            self._running = False
        finally:
            # 保存未保存的数据（仅当有数据且正在录制时才保存）
            if self._episode_data and self._record_mode:
                print("\n保存未完成的录制数据...")
                self._episode_count += 1
                # episode_prompt = f"{self.env._prompt}_ep{self._episode_count}" if self.env._prompt else f"episode_{self._episode_count}"
                episode_prompt = f"{self.env._prompt}" if self.env._prompt else f"episode_{self._episode_count}"
                # 传入机器人句柄和关节命令历史
                left_handle = getattr(self.env._left_robot, 'handle', None)
                right_handle = getattr(self.env._right_robot, 'handle', None)
                save_episode_to_hdf5(
                    list(self._episode_data), self.record_directory, episode_prompt,
                    left_handle, right_handle, self._joint_command_history,
                    fps=self._record_frequency
                )
                print(f"数据已保存 ({len(self._episode_data)} 步)")
                self._episode_data = []
                self._joint_command_history = []
            
            # 同样退出时保存日志
            self._save_log_buffer()
            
            self._cleanup()
    
    def _cleanup(self):
        """清理资源"""
        # 先停止采集线程和发送线程（设置标志让线程退出）
        self._record_thread_running = False
        self._send_thread_running = False
        self._running = False
        
        # 停止采集线程（等待线程结束）
        self._stop_record_thread()
        
        # 停止发送线程（等待线程结束）
        self._stop_send_thread()
        
        if self._client_sock:
            try:
                self._client_sock.close()
            except:
                pass
        
        if self._server_sock:
            try:
                self._server_sock.close()
            except:
                pass
        
        if self.show_preview:
            cv2.destroyAllWindows()
        
        if self.verbose:
            print("Server closed")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="X5 Remote Teleop Server")
    parser.add_argument('--host', default='192.168.1.185', help='Listen host')
    parser.add_argument('--port', type=int, default=5555, help='Listen port')
    parser.add_argument('--env-config', default='../dual_arm_env_config.yaml',
                        help='Environment config file path')
    parser.add_argument('--record-dir', default='recorded_data', help='Data save directory')
    parser.add_argument('--no-preview', action='store_true', help='Disable camera preview')
    parser.add_argument('--prompt', default='', help='Task prompt')
    parser.add_argument('--control-mode', choices=['servol', 'movj'], default='servol',
                        help='Robot control mode: servol (realtime TCP) or movj (queued)')
    parser.add_argument('--servol-cmdt', type=float, default=0.02, help='servol cmdt seconds (e.g. 0.02 for 50Hz)')
    parser.add_argument('--servol-gain', type=float, default=15.0, help='servol gain (0~50)')
    parser.add_argument('--servol-vel', type=float, default=100.0, help='servol vel percent (0~100)')
    parser.add_argument('--servol-acc', type=float, default=100.0, help='servol acc percent (0~100)')
    parser.add_argument('--no-print-cmd', action='store_true', help='Disable real-time command print')
    parser.add_argument('--print-cmd-hz', type=float, default=30.0, help='Command print refresh rate (Hz)')
    parser.add_argument('--send-frequency', type=float, default=50.0, help='Command send frequency (Hz), default 50Hz')
    args = parser.parse_args()

    # ---- 崩溃诊断：静默崩溃时尽量落盘堆栈 ----
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    try:
        crash_log_path = os.path.join(logs_dir, "teleop_server_crash.log")
        global _CRASH_FH  # noqa: PLW0603
        _CRASH_FH = open(crash_log_path, "a", buffering=1, encoding="utf-8")
        _CRASH_FH.write(f"\n===== START {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        _CRASH_FH.flush()
        faulthandler.enable(file=_CRASH_FH, all_threads=True)
        
        def _thread_excepthook(args_):
            try:
                _CRASH_FH.write("\n[threading.excepthook]\n")
                _CRASH_FH.write(f"thread={getattr(args_, 'thread', None)}\n")
                _CRASH_FH.write("".join(traceback.format_exception(args_.exc_type, args_.exc_value, args_.exc_traceback)))
                _CRASH_FH.flush()
            except Exception:
                pass

        threading.excepthook = _thread_excepthook
    except Exception:
        _CRASH_FH = None
    
    # 相对路径转绝对路径
    config_path = os.path.join(os.path.dirname(__file__), args.env_config)
    
    # 创建环境
    print(f"\nLoading config: {config_path}")
    # 仅当命令行显式指定了 prompt 时才覆盖配置文件中的值
    env_kwargs = {}
    if args.prompt:
        env_kwargs['prompt'] = args.prompt
    env = create_dual_arm_env_from_config(config_path, **env_kwargs)
    
    try:
        print("Resetting environment...")
        env.reset()
        
        # 创建服务端
        server = RemoteTeleopServer(
            env=env,
            host=args.host,
            port=args.port,
            show_preview=not args.no_preview,
            record_directory=args.record_dir,
            logs_directory=logs_dir,
            verbose=True
        )
        # Apply control params
        server._use_servol = (args.control_mode == 'servol')  # noqa: SLF001
        server._servol_cmdt = float(args.servol_cmdt)  # noqa: SLF001
        server._servol_gain = float(args.servol_gain)  # noqa: SLF001
        server._servol_vel = float(args.servol_vel)  # noqa: SLF001
        server._servol_acc = float(args.servol_acc)  # noqa: SLF001
        server._print_commands = (not args.no_print_cmd)  # noqa: SLF001
        server._print_cmd_hz = float(args.print_cmd_hz)  # noqa: SLF001
        server._send_frequency = float(args.send_frequency)  # noqa: SLF001
        
        # 挂载日志记录回调到环境
        env.set_log_callback(server._log_msg)
        
        # 运行服务端
        server.run()
    
    except KeyboardInterrupt:
        print("\nServer stopped by user (KeyboardInterrupt)")
    except Exception as e:
        print(f"\nServer error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("Program exited")


if __name__ == "__main__":
    main()
