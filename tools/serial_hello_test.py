#!/usr/bin/env python3
"""
串口测试程序 — 每0.5s向 /dev/ttyS2 发送 "Hello World"
用法:
  python3 tools/serial_hello_test.py
  python3 tools/serial_hello_test.py --port /dev/ttyS2 --baud 115200 --interval 0.5
"""

import argparse
import time
import sys


def main():
    parser = argparse.ArgumentParser(description="串口 Hello World 测试")
    parser.add_argument("--port", default="/dev/ttyS2", help="串口设备 (默认 /dev/ttyS2)")
    parser.add_argument("--baud", type=int, default=115200, help="波特率 (默认 115200)")
    parser.add_argument("--interval", type=float, default=0.5, help="发送间隔 秒 (默认 0.5)")
    args = parser.parse_args()

    # 打开串口
    try:
        import serial
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
        )
    except Exception as e:
        print(f"✗ 串口打开失败: {e}")
        print(f"  设备: {args.port}")
        print(f"  检查: ls {args.port}")
        sys.exit(1)

    print(f"串口已打开: {args.port} @ {args.baud} baud")
    print(f"每 {args.interval}s 发送 'Hello World' (Ctrl+C 停止)")
    print("-" * 50)

    count = 0
    try:
        while True:
            msg = f"Hello World [{count}]\n"
            data = msg.encode("utf-8")
            ser.write(data)
            print(f"[{count:4d}] TX: {msg.rstrip()}")
            count += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        ser.close()
        print(f"串口已关闭，共发送 {count} 条")


if __name__ == "__main__":
    main()
