#!/usr/bin/env python3
"""
云台定位性能测试工具
用法 (在香橙派5 Max上):
  python3 tools/gimbal_test.py monitor
  python3 tools/gimbal_test.py step
  python3 tools/gimbal_test.py sweep
  python3 tools/gimbal_test.py log --duration 30
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

# 允许从项目根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.gimbal_serial import GimbalSerial, CommandPacket, TelemetryPacket


# ============================================================
#  测试用例
# ============================================================

def cmd_enable() -> CommandPacket:
    return CommandPacket(enabled=1, stability_enabled=1)


def cmd_disable() -> CommandPacket:
    return CommandPacket(enabled=0, stability_enabled=0)


def cmd_speed(yaw: float = 0.0, pitch: float = 0.0) -> CommandPacket:
    return CommandPacket(yaw_speed=yaw, pitch_speed=pitch,
                         enabled=1, stability_enabled=1)


def test_enable(gs: GimbalSerial) -> None:
    """使能 → 等待 → 监控状态"""
    print("[TEST] 使能云台 + 增稳...")
    gs.send(cmd_enable())
    time.sleep(0.3)

    for attempt in range(20):
        pkg = gs.recv()
        if pkg:
            print(f"  enabled={pkg.enabled}  stability={pkg.stability_enabled}"
                  f"  imu_pitch={math.degrees(pkg.imu_pitch):.1f}°"
                  f"  motor_pitch={math.degrees(pkg.pitch_motor_angle):.1f}°")
            if pkg.enabled and pkg.stability_enabled:
                print("  ✓ 增稳已启动")
                return
        time.sleep(0.1)
    print("  ✗ 未收到增稳确认")


def test_step_response(gs: GimbalSerial) -> None:
    """Yaw 阶跃响应测试"""
    print("[TEST] Yaw 阶跃响应 (±15 rpm)...")
    gs.drain()
    gs.send(cmd_enable())
    time.sleep(0.3)

    for speed in [15, -15, 15, -15, 0]:
        print(f"\n-- yaw_speed = {speed} rpm --")
        gs.send(cmd_speed(yaw=speed))
        t0 = time.time()
        while time.time() - t0 < 1.5:
            pkg = gs.recv()
            if pkg:
                t = time.time() - t0
                print(f"  t={t:.2f}s  imu_yaw={math.degrees(pkg.imu_yaw):.2f}°  "
                      f"motor_yaw={math.degrees(pkg.yaw_motor_angle):.2f}°")
            time.sleep(0.02)

    gs.send(cmd_speed(yaw=0))


def test_sine_sweep(gs: GimbalSerial, duration: float = 15.0, freq: float = 0.5) -> None:
    """Yaw 正弦扫频测试"""
    print(f"[TEST] Yaw 正弦扫频 {freq}Hz ±20rpm, {duration}s...")
    gs.drain()
    gs.send(cmd_enable())
    time.sleep(0.3)

    t0 = time.time()
    while time.time() - t0 < duration:
        t = time.time() - t0
        yaw = 20.0 * math.sin(2 * math.pi * freq * t)
        gs.send(cmd_speed(yaw=yaw))
        time.sleep(0.01)

    gs.send(cmd_speed(yaw=0))
    print("  扫频完成")


def test_logging(gs: GimbalSerial, duration: float = 30.0,
                 filename: str = "gimbal_log.csv") -> None:
    """持续记录遥测数据到 CSV"""
    print(f"[TEST] 记录 {duration}s → {filename} ...")
    gs.drain()
    gs.send(cmd_enable())
    time.sleep(0.3)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time", "imu_yaw_rad", "imu_pitch_rad", "imu_roll_rad",
            "yaw_imu_angle_rad", "pitch_imu_angle_rad",
            "yaw_motor_angle_rad", "pitch_motor_angle_rad",
            "enabled", "stability_enabled",
        ])
        t0 = time.time()
        count = 0
        while time.time() - t0 < duration:
            pkg = gs.recv()
            if pkg:
                writer.writerow([
                    time.time() - t0,
                    pkg.imu_yaw, pkg.imu_pitch, pkg.imu_roll,
                    pkg.yaw_imu_angle, pkg.pitch_imu_angle,
                    pkg.yaw_motor_angle, pkg.pitch_motor_angle,
                    pkg.enabled, pkg.stability_enabled,
                ])
                count += 1
            time.sleep(0.001)
    print(f"  完成: {count} 帧, {filename}")


def test_monitor(gs: GimbalSerial) -> None:
    """实时监控云台状态"""
    print("[MONITOR] 实时状态 (Ctrl+C 退出)...")
    gs.drain()
    try:
        while True:
            pkg = gs.recv()
            if pkg:
                yaw_d = math.degrees(pkg.imu_yaw)
                pitch_d = math.degrees(pkg.imu_pitch)
                myaw_d = math.degrees(pkg.yaw_motor_angle)
                mpitch_d = math.degrees(pkg.pitch_motor_angle)
                print(
                    f"\r  IMU: Y={yaw_d:+8.2f}° P={pitch_d:+8.2f}°  "
                    f"Motor: Y={myaw_d:+8.2f}° P={mpitch_d:+8.2f}°  "
                    f"Ena={pkg.enabled} Stab={pkg.stability_enabled}  ",
                    end="", flush=True,
                )
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n  监控结束")


# ============================================================
#  CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="云台定位性能测试工具 — Orange Pi 5 Max",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
接线:  香橙派5Max                    STM32
       Pin8  GPIO1_B5 (UART2_TX) → PG9  (USART6_RX)
       Pin10 GPIO1_B6 (UART2_RX) ← PG14 (USART6_TX)
       GND  ───────────────────── → GND

注意:
  1. 波特率必须是 1152000 (与 STM32 usart.c 一致)
  2. 如果连接失败，检查是否已启用 UART2:
     sudo orangepi-config → System → Hardware → 启用 UART2
""",
    )
    parser.add_argument(
        "--port", default="/dev/ttyS2",
        help="串口设备 (Orange Pi 5 Max UART2 默认: /dev/ttyS2)",
    )
    parser.add_argument(
        "--baud", type=int, default=1_152_000,
        help="波特率 (STM32 默认: 1152000)",
    )
    parser.add_argument(
        "action", nargs="?", default="monitor",
        choices=["enable", "step", "sweep", "log", "monitor"],
        help="测试动作 (默认: monitor)",
    )
    parser.add_argument("--duration", type=float, default=30.0,
                        help="log/sweep 时长(秒)")
    parser.add_argument("--output", default="gimbal_log.csv",
                        help="CSV 输出路径")

    args = parser.parse_args()

    print(f"连接云台: {args.port} @ {args.baud:,} baud")
    try:
        gs = GimbalSerial(port=args.port, baudrate=args.baud)
    except Exception as e:
        print(f"✗ 串口打开失败: {e}")
        print("  检查: 1) 设备存在?  2) 权限? (sudo usermod -aG dialout $USER)")
        print(f"        3) UART2 已启用?  4) 波特率 {args.baud} 是否被硬件支持?")
        return

    try:
        if args.action == "enable":
            test_enable(gs)
        elif args.action == "step":
            test_step_response(gs)
        elif args.action == "sweep":
            test_sine_sweep(gs, duration=args.duration)
        elif args.action == "log":
            test_logging(gs, duration=args.duration, filename=args.output)
        elif args.action == "monitor":
            test_monitor(gs)
    finally:
        print("\n关闭串口...")
        gs.send(cmd_disable())
        gs.close()


if __name__ == "__main__":
    main()
