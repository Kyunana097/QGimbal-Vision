#!/usr/bin/env python3
"""
UART2 回环测试 — 发送数据并验证是否能收到
用法:
  python3 tools/uart_loopback_test.py

前提:
  将 Orange Pi 5 Max 的 Pin8 (TX) 和 Pin10 (RX) 用杜邦线短接
"""

import time
import sys


def main():
    import serial

    PORT = "/dev/ttyS2"
    BAUD = 115200

    print(f"打开 {PORT} @ {BAUD}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1.0)
    except Exception as e:
        print(f"✗ 无法打开串口: {e}")
        return

    # 清空缓冲区
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # 发送测试数据
    test_msg = b"Hello World [loopback test]\n"
    print(f"发送: {test_msg!r}")
    written = ser.write(test_msg)
    print(f"已写入 {written} 字节")

    # 立即尝试读取
    ser.timeout = 1.0
    time.sleep(0.1)
    received = ser.read(ser.in_waiting or 100)
    ser.close()

    if received:
        print(f"✓ 回环成功! 收到 {len(received)} 字节: {received!r}")
    else:
        print("✗ 回环失败 — 未收到任何数据")
        print()
        print("可能原因:")
        print("  1. Pin8 (TX) 和 Pin10 (RX) 没有短接 — 请用杜邦线连接")
        print("  2. /dev/ttyS2 不是正确的 UART2 设备")
        print("  3. UART2 未正确启用 — 运行: sudo orangepi-config")
        print()
        print("尝试扫描所有串口设备:")
        import os
        for f in sorted(os.listdir("/dev")):
            if f.startswith("ttyS") or f.startswith("ttyAMA"):
                print(f"  /dev/{f}")


if __name__ == "__main__":
    main()
