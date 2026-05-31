"""
云台串口通信模块 — 香橙派5 Max ↔ STM32 (USART6)
UART2: GPIO1_B5=TX, GPIO1_B6=RX, 设备 /dev/ttyS2

协议:
  发送(12B): float yaw_speed, float pitch_speed, uint8*3 cmd, uint8 checksum
  接收(32B): float imu[3], float yaw/pitch_imu_angle, float yaw/pitch_motor_angle, uint8*3 state, uint8 checksum
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Optional

# ============================================================
#  协议结构体
# ============================================================

RECEIVE_PKG_SIZE = 12   # 香橙派 → STM32
TRANSMIT_PKG_SIZE = 32  # STM32 → 香橙派


@dataclass(slots=True)
class CommandPacket:
    """发送给 STM32 的控制指令"""
    yaw_speed: float = 0.0        # rpm, ±50
    pitch_speed: float = 0.0      # rpm, ±50
    laser_enabled: int = 2        # 0关 1开 other不变
    enabled: int = 2              # 0关 1开 other不变
    stability_enabled: int = 2    # 0关 1开 other不变

    def pack(self) -> bytes:
        """打包为 12 字节二进制帧"""
        data = struct.pack(
            "<ffBBB",
            float(self.yaw_speed),
            float(self.pitch_speed),
            int(self.laser_enabled) & 0xFF,
            int(self.enabled) & 0xFF,
            int(self.stability_enabled) & 0xFF,
        )
        chk = sum(data) & 0xFF
        return data + bytes([chk])


@dataclass(slots=True)
class TelemetryPacket:
    """从 STM32 接收的遥测数据"""
    imu_yaw: float = 0.0           # rad
    imu_pitch: float = 0.0         # rad
    imu_roll: float = 0.0          # rad
    yaw_imu_angle: float = 0.0     # rad
    pitch_imu_angle: float = 0.0   # rad
    yaw_motor_angle: float = 0.0   # rad
    pitch_motor_angle: float = 0.0 # rad
    laser_enabled: int = 0
    enabled: int = 0
    stability_enabled: int = 0

    @classmethod
    def unpack(cls, data: bytes) -> Optional["TelemetryPacket"]:
        """从 32 字节二进制帧解包，校验失败返回 None"""
        if len(data) != TRANSMIT_PKG_SIZE:
            return None
        if (sum(data[:31]) & 0xFF) != data[31]:
            return None
        vals = struct.unpack("<7fBBBB", data)
        return cls(*vals)


# ============================================================
#  串口管理
# ============================================================

class GimbalSerial:
    """全双工串口通信 — 发送 CommandPacket，接收 TelemetryPacket"""

    def __init__(
        self,
        port: str = "/dev/ttyS2",      # Orange Pi 5 Max UART2
        baudrate: int = 1_152_000,     # STM32 USART6 实际波特率
        timeout: float = 0.05,
    ):
        import serial
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )

    def send(self, cmd: CommandPacket) -> None:
        """发送控制指令"""
        self.ser.write(cmd.pack())

    def recv(self) -> Optional[TelemetryPacket]:
        """非阻塞读取一帧遥测数据，无数据返回 None"""
        if self.ser.in_waiting >= TRANSMIT_PKG_SIZE:
            return TelemetryPacket.unpack(self.ser.read(TRANSMIT_PKG_SIZE))
        return None

    def drain(self) -> None:
        """清空接收缓冲区"""
        self.ser.reset_input_buffer()

    def close(self) -> None:
        self.ser.close()
