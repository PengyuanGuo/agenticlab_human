#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
import serial

class GripperController:
    """
    大寰机器人 AG 系列手爪 Modbus-RTU 控制类
    适用型号：AG-160-95, AG-105-145 等
    """
    def __init__(self, port: str = "COM11", baudrate: int = 115200, gripper_id: int = 1):
        """
        初始化串口连接
        :param port: 串口号 (如 "COM11" 或 "/dev/ttyUSB0")
        :param baudrate: 波特率 (默认 115200)
        :param gripper_id: 设备 ID (默认 1)
        """
        self.ser = serial.Serial(port, baudrate, timeout=0.5)
        self.gripper_id = gripper_id

    def _calculate_crc(self, data):
        """计算 Modbus CRC16 校验码"""
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for i in range(8):
                if (crc & 1) != 0:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return crc.to_bytes(2, 'little')

    def _send_command(self, func_code, register_addr, value):
        """
        发送写指令 (功能码 0x06)
        """
        data = bytearray([self.gripper_id, func_code])
        data.extend(register_addr.to_bytes(2, 'big'))
        data.extend(value.to_bytes(2, 'big'))
        data.extend(self._calculate_crc(data))
        
        self.ser.write(data)
        response = self.ser.read(8) # 0x06 返回 8 字节
        return response

    def _read_registers(self, register_addr, count):
        """
        发送读指令 (功能码 0x03)
        """
        data = bytearray([self.gripper_id, 0x03])
        data.extend(register_addr.to_bytes(2, 'big'))
        data.extend(count.to_bytes(2, 'big'))
        data.extend(self._calculate_crc(data))
        
        self.ser.write(data)
        response = self.ser.read(5 + 2 * count) # 0x03 返回 5 + 2*N 字节
        return response

    def init_gripper(self, timeout_s: float = 10.0, poll_interval_s: float = 0.5):
        """
        初始化夹爪。在断电或异常后需要先调用此接口。
        寄存器：0x0100
        """
        print(f"正在初始化夹爪 ID: {self.gripper_id}...")
        self._send_command(0x06, 0x0100, 1)
        deadline = time.monotonic() + float(timeout_s)
        while True:
            if self.get_init_status() == 1:
                print("夹爪初始化成功")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"夹爪初始化超时: {timeout_s:.1f}s")
            time.sleep(float(poll_interval_s))

    def set_force(self, force):
        """
        设置夹持力值
        :param force: 力值百分比 (20 - 100)
        寄存器：0x0101
        """
        if not (20 <= force <= 100):
            print("警告: 力值范围应在 20-100 之间")
            force = max(20, min(100, force))
        self._send_command(0x06, 0x0101, force)

    def set_position(self, position):
        """
        设置目标位置
        :param position: 目标位置 (0 - 1000)。0 为完全闭合，1000 为完全打开。
        寄存器：0x0103
        """
        if not (0 <= position <= 1000):
            print("警告: 位置范围应在 0-1000 之间")
            position = max(0, min(1000, position))
        self._send_command(0x06, 0x0103, position)

    def get_init_status(self):
        """
        获取初始化状态
        :return: 0: 未初始化, 1: 已初始化
        寄存器：0x0200
        """
        res = self._read_registers(0x0200, 1)
        if len(res) >= 5:
            return int.from_bytes(res[3:5], 'big')
        return None

    def get_grip_status(self):
        """
        获取夹持状态反馈
        :return: 
            0: 夹爪正在运动中
            1: 夹爪已运动至目标位置，未夹持到物体
            2: 夹爪已夹持到物体
            3: 夹爪在夹持物体后发生掉落
        寄存器：0x0201
        """
        res = self._read_registers(0x0201, 1)
        if len(res) >= 5:
            return int.from_bytes(res[3:5], 'big')
        return None

    def get_current_position(self):
        """
        获取当前位置反馈
        :return: 0 - 1000 (0 为闭合，1000 为打开)
        寄存器：0x0202
        """
        res = self._read_registers(0x0202, 1)
        if len(res) >= 5:
            return int.from_bytes(res[3:5], 'big')
        return None

    def close(self):
        """关闭串口连接"""
        self.ser.close()

if __name__ == "__main__":
    # 示例代码
    left_hand = GripperController(port="COM6", baudrate=115200, gripper_id=1)
    # right_hand = GripperController(port="COM8", baudrate=115200, gripper_id=1)
    
    # 1. 初始化
    if left_hand.get_init_status() != 1:
        left_hand.init_gripper()
    
    # if right_hand.get_init_status() != 1:
    #     right_hand.init_gripper()
    # 2. 设置力值 (50%)
    left_hand.set_force(100)
    # right_hand.set_force(70)

    # 3. 闭合夹爪 (位置 0)
    print("正在闭合夹爪...")
    # right_hand.set_position(1000)
    left_hand.set_position(1000)

    # 4. 轮询状态直到停止运动
    while True:
        status = left_hand.get_grip_status()
        pos = left_hand.get_current_position()
        if status == 0:
            print(f"运动中... 当前位置: {pos}")
        elif status == 1:
            print(f"已到达目标位置 (未夹到物体), 位置: {pos}")
            break
        elif status == 2:
            print(f"已夹到物体, 位置: {pos}")
            break
        elif status == 3:
            print("报警: 物体掉落！")
            break
        time.sleep(0.2)

    # 3. 闭合夹爪 (位置 0)
    print("正在闭合夹爪...")
    left_hand.set_position(0)
    # right_hand.set_position(0)

    # 4. 轮询状态直到停止运动
    while True:
        status = left_hand.get_grip_status()
        pos = left_hand.get_current_position()
        if status == 0:
            print(f"运动中... 当前位置: {pos}")
        elif status == 1:
            print(f"已到达目标位置 (未夹到物体), 位置: {pos}")
            break
        elif status == 2:
            print(f"已夹到物体, 位置: {pos}")
            break
        elif status == 3:
            print("报警: 物体掉落！")
            break
        time.sleep(0.2)

    left_hand.close()
    # right_hand.close()
