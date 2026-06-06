#!/usr/bin/env python3
"""
云台画圆测试 — 让云台以圆形轨迹运动
用法 (在香橙派5 Max上):
  python3 tools/circle_test.py
  python3 tools/circle_test.py --radius 5 --freq 0.3 --duration 20
  python3 tools/circle_test.py --port /dev/ttyS2 --baud 1152000

原理:
  向 yaw/pitch 轴发送正交的正弦波速度指令（相位差 90°），
  云台指向在空间中画出圆形轨迹。

  yaw_speed(t)  = A * ω * cos(ωt)      (导数 d/dt[sin])
  pitch_speed(t) = A * ω * sin(ωt)     (导数 d/dt[-cos])

  位置: yaw(t) ≈ A*sin(ωt), pitch(t) ≈ -A*cos(ωt)  → 半径为 A 的圆

接线:  香橙派5Max                    STM32
       Pin8  GPIO1_B5 (UART2_TX) → PG9  (USART6_RX)
       Pin10 GPIO1_B6 (UART2_RX) ← PG14 (USART6_TX)
       GND  ───────────────────── → GND
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.gimbal_serial import GimbalSerial, CommandPacket


def draw_circle(
    gs: GimbalSerial,
    radius_rpm: float = 3.0,
    freq_hz: float = 0.5,
    duration_s: float = 20.0,
    step_s: float = 0.02,
    verbose: bool = True,
) -> None:
    """
    让云台画出圆形轨迹。

    参数:
        gs:          GimbalSerial 实例
        radius_rpm:  圆半径（速度幅值，RPM），越大圆越大。建议 1~10
        freq_hz:     旋转频率（Hz），越大转得越快。建议 0.2~1.0
        duration_s:  总运行时间（秒）
        step_s:      控制周期（秒），默认 20ms = 50Hz
        verbose:     是否打印实时信息
    """
    omega = 2.0 * math.pi * freq_hz  # 角频率 (rad/s)
    amplitude = float(radius_rpm)     # 速度幅值

    print(f"\n{'='*60}")
    print(f"  云台画圆测试")
    print(f"  半径 = {radius_rpm} rpm | 频率 = {freq_hz} Hz")
    print(f"  周期 = {1.0/freq_hz:.1f}s/圈 | 总时长 = {duration_s}s")
    print(f"  预计画 {duration_s * freq_hz:.0f} 个圆")
    print(f"{'='*60}\n")

    # ── 1. 先使能云台 + 增稳 ──
    print("[1/3] 使能云台 + 增稳...")
    gs.drain()
    enable_cmd = CommandPacket(enabled=1, stability_enabled=1)
    gs.send(enable_cmd)
    time.sleep(0.5)

    # 等待确认使能成功
    for attempt in range(15):
        pkg = gs.recv()
        if pkg and pkg.enabled:
            print(f"  ✓ 云台已使能 (stability={pkg.stability_enabled})")
            break
        time.sleep(0.1)
    else:
        print("  ⚠ 未收到使能确认，继续运行...")

    # ── 2. 画圆 ──
    print(f"\n[2/3] 开始画圆... (Ctrl+C 提前停止)")
    t0 = time.time()
    loop_count = 0

    try:
        while True:
            t = time.time() - t0
            if t > duration_s:
                break

            # 正交正弦波速度指令
            # yaw:  A*cos(ωt) → 积分后为 (A/ω)*sin(ωt)
            # pitch: A*sin(ωt) → 积分后为 -(A/ω)*cos(ωt)
            yaw_rpm = amplitude * math.cos(omega * t)
            pitch_rpm = amplitude * math.sin(omega * t)

            cmd = CommandPacket(
                yaw_speed=yaw_rpm,
                pitch_speed=pitch_rpm,
                enabled=1,
                stability_enabled=1,
            )
            gs.send(cmd)

            loop_count += 1

            if verbose and loop_count % 25 == 0:  # 每 0.5s 打印一次
                pkg = gs.recv()
                yaw_deg = math.degrees(pkg.yaw_motor_angle) if pkg else 0
                pitch_deg = math.degrees(pkg.pitch_motor_angle) if pkg else 0
                print(
                    f"  t={t:5.1f}s  "
                    f"yaw_spd={yaw_rpm:+6.2f}  pitch_spd={pitch_rpm:+6.2f}  "
                    f"motor_yaw={yaw_deg:+7.2f}°  motor_pitch={pitch_deg:+7.2f}°"
                )

            # 非阻塞等待
            elapsed = time.time() - t0
            next_tick = (loop_count + 1) * step_s
            sleep_time = next_tick - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n  ⚠ 用户中断")

    # ── 3. 停止并关闭 ──
    print(f"\n[3/3] 停止云台...")
    stop_cmd = CommandPacket(yaw_speed=0, pitch_speed=0, enabled=1, stability_enabled=1)
    for _ in range(5):  # 多发送几次确保收到
        gs.send(stop_cmd)
        time.sleep(0.02)

    time.sleep(0.3)
    print(f"  完成! 共发送 {loop_count} 帧, 运行 {time.time() - t0:.1f}s")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="云台画圆测试 — 正交正弦波速度控制",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
接线:  香橙派5Max                    STM32
       Pin8  GPIO1_B5 (UART2_TX) → PG9  (USART6_RX)
       Pin10 GPIO1_B6 (UART2_RX) ← PG14 (USART6_TX)
       GND  ───────────────────── → GND

示例:
  python3 tools/circle_test.py                           # 默认: 半径3rpm, 0.5Hz, 20s
  python3 tools/circle_test.py --radius 5 --freq 0.3     # 大圆慢速
  python3 tools/circle_test.py --radius 2 --freq 1.0     # 小圆快速
  python3 tools/circle_test.py --duration 60             # 运行60秒

前提条件:
  1. UART2 已启用 (sudo orangepi-config → System → Hardware → UART2)
  2. 如果没有 /dev/ttyS2，需要启用 overlay:
     sudo nano /boot/orangepiEnv.txt
     添加: overlays=uart2-m0
     然后: sudo reboot
  3. 用户须在 dialout 组: sudo usermod -aG dialout $USER
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
        "--radius", type=float, default=3.0,
        help="圆半径/速度幅值 RPM (默认 3, 建议 1~10)",
    )
    parser.add_argument(
        "--freq", type=float, default=0.5,
        help="旋转频率 Hz (默认 0.5, 建议 0.2~1.0)",
    )
    parser.add_argument(
        "--duration", type=float, default=20.0,
        help="运行时长 秒 (默认 20)",
    )
    parser.add_argument(
        "--step", type=float, default=0.02,
        help="控制周期 秒 (默认 0.02 = 50Hz)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="减少输出",
    )

    args = parser.parse_args()

    print(f"连接云台: {args.port} @ {args.baud:,} baud")
    try:
        gs = GimbalSerial(port=args.port, baudrate=args.baud)
    except Exception as e:
        print(f"✗ 串口打开失败: {e}")
        print("\n请检查:")
        print("  1. UART2 是否已启用? 运行: ls /dev/ttyS2")
        print("  2. 如不存在, 启用 overlay: sudo nano /boot/orangepiEnv.txt")
        print("     添加: overlays=uart2-m0")
        print("     然后: sudo reboot")
        print("  3. 权限: sudo usermod -aG dialout $USER")
        print("  4. 波特率是否与 STM32 一致?")
        return

    try:
        draw_circle(
            gs,
            radius_rpm=args.radius,
            freq_hz=args.freq,
            duration_s=args.duration,
            step_s=args.step,
            verbose=not args.quiet,
        )
    finally:
        # 确保云台停止
        stop_cmd = CommandPacket(yaw_speed=0, pitch_speed=0, enabled=1, stability_enabled=1)
        gs.send(stop_cmd)
        gs.send(stop_cmd)
        time.sleep(0.1)
        gs.close()
        print("串口已关闭")


if __name__ == "__main__":
    main()
