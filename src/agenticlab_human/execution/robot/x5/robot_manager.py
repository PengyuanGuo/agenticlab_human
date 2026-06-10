"""机器人管理器模块。

负责机器人的连接、控制、状态监测等核心功能。
基于 xapi SDK 实现，专注于机器人运动控制与基础 IO 操作。
"""

import time
import threading
import logging
from typing import Optional, Union, List, Dict, Tuple, Any
import xapi.api as x5
from waiting import wait, TimeoutExpired

# 设置日志
logger = logging.getLogger("RobotManager")

class RobotManager:
    """机器人管理器类。

    提供机器人的连接、控制、状态监测等功能。
    每个实例独立管理一个机器人连接，支持多实例（例如双臂）。
    """

    def __init__(self):
        """初始化机器人管理器实例。"""
        self.logger = logger
        self._handle = -1
        self._connected = False
        self.logger.debug("机器人管理器初始化完成")

    @property
    def handle(self) -> int:
        return self._handle

    @property
    def is_connected(self) -> bool:
        return self._connected and self._handle != -1

    def _check_handle(self) -> bool:
        result = self.is_connected
        if not result:
            self.logger.warning(f"机器人未连接或句柄无效: handle={self._handle}")
        return result

    def connect(self, ip_address: str) -> bool:
        """连接到机器人"""
        try:
            self._handle = x5.connect(ip_address)
            if self._handle != -1:
                self._connected = True
                self.logger.info(f"机器人连接成功: IP={ip_address}, handle={self._handle}")
                return True
            else:
                self.logger.error(f"连接机器人失败: {ip_address}")
                return False
        except Exception as e:
            self.logger.error(f"连接异常: {e}", exc_info=True)
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        try:
            if self._connected and self._handle != -1:
                x5.disconnect(self._handle)
                self._connected = False
                self._handle = -1
                self.logger.info("机器人连接已断开")
            return True
        except Exception as e:
            self.logger.error(f"断开连接异常: {e}")
            return False

    # ==================== 状态与报警 ====================

    def get_system_state(self) -> Union[x5.SystemState, bool]:
        """获取系统状态"""
        if not self._check_handle(): return False
        try:
            return x5.get_system_state(self._handle)
        except Exception as e:
            self.logger.error(f"获取系统状态失败: {e}")
            return False

    def get_controller_alarm(self) -> Union[List[Dict], bool]:
        """获取控制器报警信息"""
        if not self._check_handle(): return False
        try:
            return x5.get_system_alarm_info(self._handle)
        except Exception as e:
            self.logger.error(f"获取报警信息失败: {e}")
            return False

    def check_alarms(self) -> Tuple[bool, str]:
        """检查是否有报警并返回格式化信息"""
        alarms = self.get_controller_alarm()
        active = isinstance(alarms, list) and len(alarms) > 0
        if active:
            blocks = []
            for i, a in enumerate(alarms, 1):
                code = a.get("code", "unknown")
                text = a.get("content", a.get("message", "未知报警"))
                blocks.append(f"[{i}] 代码 {code}: {text}")
            return True, "\n".join(blocks)
        return False, ""

    def check_emergency_stop(self) -> bool:
        """检查是否急停"""
        has_alarm, info = self.check_alarms()
        return "急停" in info if has_alarm else False

    # ==================== 数字 IO 控制 ====================

    def set_do(self, port: int, value: int) -> bool:
        if not self._check_handle(): return False
        try:
            x5.set_do(self._handle, port, value)
            return True
        except Exception as e:
            self.logger.error(f"设置 DO 端口 {port} 失败: {e}")
            return False

    def get_do(self, port: int) -> Union[int, bool]:
        if not self._check_handle(): return False
        try:
            return x5.get_do(self._handle, port)
        except Exception as e:
            self.logger.error(f"获取 DO 端口 {port} 失败: {e}")
            return False

    def get_di(self, port: int) -> Union[int, bool]:
        if not self._check_handle(): return False
        try:
            return x5.get_di(self._handle, port)
        except Exception as e:
            self.logger.error(f"获取 DI 端口 {port} 失败: {e}")
            return False

    # ==================== 运动控制 ====================

    def set_speed(self, speed: int) -> bool:
        """设置全局速度百分比 (1-100)"""
        if not self._check_handle(): return False
        try:
            x5.set_speed(self._handle, speed)
            return True
        except Exception as e:
            self.logger.error(f"设置速度失败: {e}")
            return False

    def mov_j(
        self,
        target: Union[x5.Joint, x5.Point],
        block: bool = True,
        movpointadd: Optional[x5.MovPointAdd] = None,
        cnt: int = 0,
    ) -> bool:
        """关节运动（统一使用 x5.movj(handle, target, add_data) 接口）

        - target 可以是 x5.Joint 或 x5.Point
        - movpointadd 用于速度/加速度/cnt/offset 等附加参数
        - block=True 时会调用 x5.wait_move_done(handle)
        """
        if not self._check_handle(): return False
        try:
            if cnt != 0 and movpointadd is None:
                movpointadd = x5.MovPointAdd()
                movpointadd.cnt = cnt

            if movpointadd is not None:
                x5.movj(self._handle, target, movpointadd)
            else:
                x5.movj(self._handle, target)

            if block:
                x5.wait_move_done(self._handle)
            return True
        except Exception as e:
            self.logger.error(f"MovJ 运动异常: {e}")
            return False

    def mov_l(self, target: x5.Point, block: bool = True) -> bool:
        """直线运动"""
        if not self._check_handle(): return False
        try:
            x5.movl(self._handle, target)
            if block: x5.wait_move_done(self._handle)
            return True
        except Exception as e:
            self.logger.error(f"MovL 运动异常: {e}")
            return False

    def servoj(
        self,
        target: x5.Joint,
        cmdt: float = 0.033,
        lookahead: int = 0,
        gain: int = 5,
        vel: int = 100,
        acc: int = 100,
    ) -> bool:
        """在线实时控制关节位置（servoj）。

        说明：
        - 这是“实时跟随”接口，通常需要按固定频率连续调用（例如 30Hz）。
        - 不要在每步里调用 wait_move_done。
        """
        if not self._check_handle():
            return False
        try:
            x5.servoj(self._handle, target, float(cmdt), float(lookahead), float(gain), float(vel), float(acc))
            return True
        except Exception as e:
            self.logger.error(f"servoj 失败: {e}")
            return False

    def jump(self, target: x5.Point, jump_data: List[float], block: bool = True) -> bool:
        """门型运动"""
        if not self._check_handle(): return False
        try:
            x5.jump(self._handle, target, jump_data)
            if block: x5.wait_move_done(self._handle)
            return True
        except Exception as e:
            self.logger.error(f"Jump 运动异常: {e}")
            return False

    def stop(self) -> bool:
        """停止所有运动"""
        if not self._check_handle(): return False
        try:
            x5.stop(self._handle)
            x5.abort(self._handle)
            return True
        except Exception as e:
            self.logger.error(f"停止指令执行失败: {e}")
            return False

    def wait_move_done(self, timeout_ms: int = 60000) -> bool:
        """等待运动完成"""
        if not self._check_handle(): return False
        try:
            x5.wait_move_done(self._handle, timeout_ms)
            return True
        except Exception as e:
            self.logger.error(f"等待运动完成超时或失败: {e}")
            return False

    # ==================== 坐标获取与设置 ====================

    def get_cpoint(self) -> Union[x5.Point, bool]:
        """获取当前点位 (用户坐标系)"""
        if not self._check_handle(): return False
        try:
            return x5.get_cpoint(self._handle)
        except Exception as e:
            self.logger.error(f"获取当前坐标点失败: {e}")
            return False

    def get_cjoint(self) -> Union[x5.Joint, bool]:
        """获取当前关节角度"""
        if not self._check_handle(): return False
        try:
            return x5.get_cjoint(self._handle)
        except Exception as e:
            self.logger.error(f"获取当前关节角度失败: {e}")
            return False

    def set_uf_no(self, uf_no: int) -> bool:
        """设置用户坐标系编号"""
        if not self._check_handle(): return False
        try:
            x5.set_ufno(self._handle, uf_no)
            return True
        except Exception as e:
            self.logger.error(f"设置用户坐标系 {uf_no} 失败: {e}")
            return False

    def set_tf_no(self, tf_no: int) -> bool:
        """设置工具坐标系编号"""
        if not self._check_handle(): return False
        try:
            x5.set_tfno(self._handle, tf_no)
            return True
        except Exception as e:
            self.logger.error(f"设置工具坐标系 {tf_no} 失败: {e}")
            return False

    # ==================== 模式与使能 ====================

    def set_mode(self, mode: int = 100, enable: bool = True) -> bool:
        """设置系统模式并上使能"""
        if not self._check_handle(): return False
        try:
            # 切换模式前先关闭使能
            x5.enable_servo(self._handle, False)
            x5.set_system_mode(self._handle, mode)
            wait(lambda: x5.get_system_state(self._handle).mode == mode, timeout_seconds=5)
            
            if enable:
                x5.enable_servo(self._handle, True)
                wait(lambda: x5.get_system_state(self._handle).enable == 1, timeout_seconds=5)
            return True
        except Exception as e:
            self.logger.error(f"设置工作模式/使能失败: {e}")
            return False

    def reset(self) -> bool:
        """重置控制器错误"""
        if not self._check_handle(): return False
        try:
            x5.reset(self._handle)
            return True
        except Exception as e:
            self.logger.error(f"重置失败: {e}")
            return False

    # ==================== Lua/消息处理 ====================

    def execute_lua(self, lua_cmd: str) -> bool:
        """执行自定义 Lua 脚本字符串"""
        if not self._check_handle(): return False
        try:
            x5.execute_lua(self._handle, lua_cmd)
            return True
        except Exception as e:
            self.logger.error(f"执行 Lua 失败: {lua_cmd}, error: {e}")
            return False

    def get_lua_message(self) -> Tuple[Optional[str], int]:
        """获取 Lua 反馈消息"""
        if not self._check_handle(): return None, 0
        try:
            return x5.get_lua_message(self._handle, 4096)
        except:
            return None, 0

if __name__ == "__main__":
    left_arm = RobotManager()
    right_arm = RobotManager()
    left_arm.connect("192.168.1.7")
    left_arm.set_mode(100, True)
    left_arm.mov_j(x5.Joint(-24, 50, 6, 35, -32, 83, 80), True)
    right_arm.connect("192.168.1.8")
    right_arm.set_mode(100, True)
    right_arm.mov_j(x5.Joint(24, 50, -6, 35, 32, 83, -80), True)
    left_arm.disconnect()
    right_arm.disconnect()