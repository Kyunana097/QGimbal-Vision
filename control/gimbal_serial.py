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
    laser_enabled: int = 2        # 0=关 1=开 其他=不变
    enabled: int = 2              # 0=禁用 1=使能 其他=不变
    stability_enabled: int = 2    # 0=关 1=开 其他=不变

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
        vals = struct.unpack("<7fBBBB", data)  # 11 个值: 7f + 4B
        return cls(*vals[:10])                  # 前 10 个是字段, 第 11 个是校验和

    @classmethod
    def _raw_unpack(cls, data: bytes) -> Optional["TelemetryPacket"]:
        """不校验直接解包 (帧同步退化模式)"""
        if len(data) != TRANSMIT_PKG_SIZE:
            return None
        vals = struct.unpack("<7fBBBB", data)
        return cls(*vals[:10])


# ============================================================
#  串口管理
# ============================================================

class GimbalSerial:
    """全双工串口通信 — 发送 CommandPacket，接收 TelemetryPacket"""

    def __init__(
        self,
        port: str = "/dev/ttyS2",
        baudrate: int = 115200,
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
            write_timeout=0.5,
        )
        self._cmd_pkg_size = RECEIVE_PKG_SIZE
        self._tlm_pkg_size = TRANSMIT_PKG_SIZE
        self._sync_offset = None  # 帧同步偏移

    def send(self, cmd: CommandPacket) -> None:
        """发送控制指令 — write + flush 确保完整帧发送"""
        pkt = cmd.pack()
        assert len(pkt) == self._cmd_pkg_size, \
            f"命令包长度错误: {len(pkt)} != {self._cmd_pkg_size}"
        self.ser.write(pkt)
        self.ser.flush()  # 等待硬件 FIFO 发送完成, 保证帧边界

    def send_raw(self, data: bytes) -> None:
        """发送原始二进制数据 (用于测试)"""
        self.ser.write(data)
        self.ser.flush()

    def recv(self) -> Optional[TelemetryPacket]:
        """非阻塞读取一帧遥测数据

        三级帧同步 (按优先级):
          1. 校验和匹配 (搜索所有偏移)
          2. 签名匹配: 状态字节 laser/ena/stab ∈ [0,2] (退化模式)
        """
        n = self.ser.in_waiting
        if n < TRANSMIT_PKG_SIZE:
            return None

        buf = self.ser.read(n)

        # 策略1: 全搜索 — 校验和验证
        for off in range(len(buf) - TRANSMIT_PKG_SIZE + 1):
            pkg = TelemetryPacket.unpack(buf[off:off + TRANSMIT_PKG_SIZE])
            if pkg is not None:
                return pkg

        # 策略2: 签名匹配 — 状态字节应取值 0/1/2
        for off in range(len(buf) - TRANSMIT_PKG_SIZE + 1):
            b28, b29, b30 = buf[off + 28], buf[off + 29], buf[off + 30]
            if 0 <= b28 <= 2 and 0 <= b29 <= 2 and 0 <= b30 <= 2:
                pkg = TelemetryPacket._raw_unpack(buf[off:off + TRANSMIT_PKG_SIZE])
                if pkg is not None:
                    return pkg

        return None

    def drain(self) -> None:
        """清空接收缓冲区"""
        self.ser.reset_input_buffer()

    def close(self) -> None:
        self.ser.close()
