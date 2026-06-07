#!/usr/bin/env python3
"""
串口监控 — 接收并解析 STM32 遥测数据
用法:
  python3 tools/serial_monitor.py
  python3 tools/serial_monitor.py --port /dev/ttyS2 --baud 115200
  python3 tools/serial_monitor.py --raw   # 仅显示原始hex
"""

import argparse
import math
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.gimbal_serial import GimbalSerial, TelemetryPacket, TRANSMIT_PKG_SIZE


def print_telemetry(pkg: TelemetryPacket, count: int) -> None:
    """格式化打印遥测数据"""
    def rad2deg(r):
        return math.degrees(r)

    print(f"\r[{count:5d}] ", end="")
    print(f"IMU: Y={rad2deg(pkg.imu_yaw):+7.2f}° "
          f"P={rad2deg(pkg.imu_pitch):+7.2f}° "
          f"R={rad2deg(pkg.imu_roll):+7.2f}°  |  ", end="")
    print(f"Motor: Y={rad2deg(pkg.yaw_motor_angle):+7.2f}° "
          f"P={rad2deg(pkg.pitch_motor_angle):+7.2f}°  |  ", end="")
    print(f"Ena={pkg.enabled} Stab={pkg.stability_enabled} "
          f"Laser={pkg.laser_enabled}", end="")
    sys.stdout.flush()


def monitor_raw(port: str, baud: int) -> None:
    """原始 hex 模式 — 显示所有接收到的字节"""
    import serial
    ser = serial.Serial(port, baud, timeout=0.1)
    print(f"监听 {port} @ {baud} (原始hex模式, Ctrl+C 退出)")
    print("-" * 60)
    try:
        while True:
            if ser.in_waiting:
                data = ser.read(ser.in_waiting)
                hex_str = data.hex(" ")
                print(f"[+{len(data):3d}B] {hex_str}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


def monitor_parsed(port: str, baud: int) -> None:
    """解析模式 — 尝试解析 32 字节遥测帧"""
    try:
        gs = GimbalSerial(port=port, baudrate=baud)
    except Exception as e:
        print(f"✗ 串口打开失败: {e}")
        print(f"  检查: ls {port}")
        return

    print(f"监听 {port} @ {baud} — 等待 STM32 遥测帧 (Ctrl+C 退出)")
    print("-" * 60)
    print(f"协议: 32B/帧, ~1kHz, 校验=前31B累加和")
    print(f"帧结构: 7×float32 + 3×uint8 + uint8校验")
    print()

    pkg_count = 0
    raw_count = 0
    t0 = time.time()
    try:
        while True:
            if gs.ser.in_waiting:
                buf = gs.ser.read(gs.ser.in_waiting)
                raw_count += len(buf)

                # 滑动窗口搜索 32 字节有效帧
                max_offset = min(len(buf) - TRANSMIT_PKG_SIZE + 1, 64)
                for offset in range(max(0, max_offset)):
                    candidate = buf[offset:offset + TRANSMIT_PKG_SIZE]
                    pkg = TelemetryPacket.unpack(candidate)
                    if pkg is not None:
                        pkg_count += 1
                        if pkg_count % 10 == 0:  # 每10帧打印一次(减少刷屏)
                            print_telemetry(pkg, pkg_count)
                        break
                else:
                    # 无有效帧, 显示原始 hex
                    if raw_count < 500:  # 只在前500字节时显示
                        print(f"[raw {len(buf)}B] {buf.hex(' ')}")
            else:
                time.sleep(0.001)

            # 每秒统计
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps = pkg_count / elapsed
                bps = raw_count / elapsed
                print(f"\n  ── {elapsed:.0f}s: {pkg_count}帧 ({fps:.0f}fps) "
                      f"{raw_count}字节 ({bps:.0f}B/s) ──\n")
                t0 = time.time()
                pkg_count = 0
                raw_count = 0

    except KeyboardInterrupt:
        print("\n\n监控结束")
    finally:
        gs.close()


def main():
    parser = argparse.ArgumentParser(
        description="串口监控 — 接收 STM32 遥测数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 tools/serial_monitor.py                          # 解析模式
  python3 tools/serial_monitor.py --raw                    # 原始hex模式
  python3 tools/serial_monitor.py --port /dev/ttyS2 --baud 115200
        """,
    )
    parser.add_argument("--port", default="/dev/ttyS2", help="串口设备")
    parser.add_argument("--baud", type=int, default=115200, help="波特率")
    parser.add_argument("--raw", action="store_true", help="原始hex模式(不解析)")
    args = parser.parse_args()

    if args.raw:
        monitor_raw(args.port, args.baud)
    else:
        monitor_parsed(args.port, args.baud)


if __name__ == "__main__":
    main()
