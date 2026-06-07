"""
QGimbal-Vision — 二维云台视觉追踪 + Flask 实时调参

功能：
  1. 矩形检测（开源算法 + A4纸验证）
  2. 双轴 PID 追踪控制（yaw/pitch）
  3. STM32 串口通信
  4. Flask Web 界面：MJPEG 视频流 + 实时参数滑块调优

运行：
  python main.py --camera 0
  python main.py --camera 0 --serial-port COM3
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

# ── 开源项目模块 ──────────────────────────────────────────────────
from control.config import ControlConfig, PIDConfig
from control.gimbal_serial import GimbalSerial, CommandPacket, TelemetryPacket
from control.tracker_control import GimbalTracker

# ══════════════════════════════════════════════════════════════════
# 参数定义
# ══════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = {
    # ── 预处理 ──
    'blur_ksize': 3,           # 高斯模糊核大小
    'canny_low': 25,           # Canny 低阈值
    'canny_high': 75,          # Canny 高阈值
    'close_ksize': 3,          # 形态学闭运算核大小
    'close_iter': 1,           # 闭运算迭代次数
    'use_clahe': 0,            # 是否使用 CLAHE (0/1)

    # ── 矩形检测（开源算法参数） ──
    'min_area_ratio': 0.005,   # 最小面积比例（相对图像）
    'max_area_ratio': 0.5,     # 最大面积比例（相对图像）
    'approx_eps': 0.02,        # 多边形逼近 epsilon
    'angle_tol': 25.0,         # 直角容差（度）

    # ── A4 纸额外验证（Flask 调试参数） ──
    'min_edge': 40,            # 最小边长 (px)
    'persp_min': 0.25,         # 对边最小比例
    'ratio_min': 1.08,         # 长宽比下限
    'ratio_max': 2.00,         # 长宽比上限

    # ── 追踪控制 ──
    'control_enabled': 0,      # 是否启用 PID 控制 (默认关, 网页开启才发)
    'deadband_px': 0.0,        # 像素死区
    'lost_timeout_s': 0.4,     # 丢目标超时 (s)
    'max_rpm_yaw': 20.0,       # Yaw 最大转速
    'max_rpm_pitch': 20.0,     # Pitch 最大转速
    'invert_yaw': 1,           # 反转 Yaw 方向
    'invert_pitch': 0,         # 反转 Pitch 方向

    # ── PID — Yaw 轴 ──
    'yaw_kp': 4.0,
    'yaw_ki': 0.80,
    'yaw_kd': 0.08,
    'yaw_integral_limit': 0.2,
    'yaw_output_limit': 1.0,

    # ── PID — Pitch 轴 ──
    'pitch_kp': 3.0,
    'pitch_ki': 0.6,
    'pitch_kd': 0.06,
    'pitch_integral_limit': 0.2,
    'pitch_output_limit': 1.0,
}

# 哪些参数是整数类型
INT_PARAMS = {
    'blur_ksize', 'canny_low', 'canny_high', 'close_ksize', 'close_iter',
    'min_edge', 'control_enabled', 'use_clahe', 'invert_yaw', 'invert_pitch',
}

# ══════════════════════════════════════════════════════════════════
# 全局状态
# ══════════════════════════════════════════════════════════════════

params = dict(DEFAULT_PARAMS)
params_lock = threading.Lock()

main_jpeg = None
debug_jpeg = None
frame_lock = threading.Lock()
frame_event = threading.Event()
stop_event = threading.Event()

latest_data = {}
data_lock = threading.Lock()

A4_RATIO = np.sqrt(2)

# PID/串口 组件（在 main 中初始化）
tracker: GimbalTracker | None = None
gimbal_serial: GimbalSerial | None = None
tracker_lock = threading.Lock()

# 控制模式
control_mode: str = 'idle'         # 'track' | 'manual' | 'test' | 'idle'  (默认待机不发信号)
control_mode_lock = threading.Lock()

# 手动控制当前速度
manual_yaw_rpm: float = 0.0
manual_pitch_rpm: float = 0.0
MANUAL_SPEED = 5.0                 # 慢速移动 RPM

# 测试信号线程
test_thread: threading.Thread | None = None
test_stop = threading.Event()

# 串口遥测
latest_telemetry: dict = {}
telemetry_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════

def line_intersection(p1, p2, p3, p4):
    """计算两条对角线的交点"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(det) < 1e-8:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det
    return int(px), int(py)


def order_corners(pts):
    """将 4 个角点排序为 [TL, TR, BR, BL]"""
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered.astype(int)


def validate_a4(corners, area, p):
    """A4 纸额外验证：边长、对边比例、长宽比"""
    edges = [
        corners[1] - corners[0],
        corners[2] - corners[1],
        corners[2] - corners[3],
        corners[3] - corners[0],
    ]
    lengths = [float(np.linalg.norm(e.astype(float))) for e in edges]

    if min(lengths) < p['min_edge']:
        return False, 0

    h_ratio = min(lengths[0], lengths[2]) / max(lengths[0], lengths[2])
    v_ratio = min(lengths[1], lengths[3]) / max(lengths[1], lengths[3])
    if h_ratio < p['persp_min'] or v_ratio < p['persp_min']:
        return False, 0

    all_pts = corners.reshape(4, 1, 2).astype(np.float32)
    rect = cv2.minAreaRect(all_pts)
    rw, rh = rect[1]
    if rw < 5 or rh < 5:
        return False, 0
    rect_ratio = max(rw, rh) / min(rw, rh)

    if rect_ratio < p['ratio_min'] or rect_ratio > p['ratio_max']:
        return False, 0

    ratio_error = abs(rect_ratio - A4_RATIO)
    area_score = min(area / 60000.0, 1.0)
    ratio_score = max(0.0, 1.0 - ratio_error / 0.6)
    score = area_score * 0.25 + ratio_score * 0.45 + (h_ratio + v_ratio) * 0.15
    return True, score


# ══════════════════════════════════════════════════════════════════
# 帧处理（混合算法 + 调试条）
# ══════════════════════════════════════════════════════════════════

def process_frame(frame: np.ndarray, p: dict):
    """
    对一帧执行完整的视觉检测流程：
      1. 预处理（CLAHE / 高斯模糊 / Canny / 闭运算）
      2. 轮廓检测 + 多边形逼近
      3. 直角验证（开源算法）+ A4 验证
      4. 绘制结果 + 生成调试条
    返回 (result_frame, debug_strip, target_center)
    """
    global latest_data

    h, w = frame.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── CLAHE（可选） ──
    if p['use_clahe']:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # ── 高斯模糊 ──
    bk = p['blur_ksize']
    if bk % 2 == 0:
        bk += 1
    blur = cv2.GaussianBlur(gray, (bk, bk), 1)

    # ── Canny 边缘检测 ──
    edges = cv2.Canny(blur, p['canny_low'], p['canny_high'])

    # ── 形态学闭运算 ──
    ck = p['close_ksize']
    ci = p['close_iter']
    kernel = np.ones((ck, ck), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=ci)

    # ── 轮廓检测 ──
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area_abs = img_area * p['min_area_ratio']
    max_area_abs = img_area * p['max_area_ratio']

    candidates = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area_abs or area > max_area_abs:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, p['approx_eps'] * peri, True)

        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        pts = approx.reshape(4, 2).astype(np.float32)

        # ── 直角验证（开源算法核心） ──
        from vision.rect_detect import angle_between
        angles = []
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i + 1) % 4]
            p2 = pts[(i + 2) % 4]
            angles.append(angle_between(p0 - p1, p2 - p1))

        if not all(abs(a - 90) < p['angle_tol'] for a in angles):
            continue

        # ── 边长比检查（开源算法：不超过 3:1） ──
        dists = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)]
        if min(dists) <= 0 or max(dists) / min(dists) > 3:
            continue

        # ── 排序角点 ──
        corners = order_corners(approx)

        # ── A4 验证 ──
        valid, score = validate_a4(corners, area, p)
        if not valid:
            continue

        # ── 中心点 ──
        center_arr = np.mean(pts, axis=0)
        cx, cy = float(center_arr[0]), float(center_arr[1])

        candidates.append({
            'score': score,
            'area': area,
            'approx': approx,
            'corners': corners,
            'center': (cx, cy),
            'dists': dists,
        })

    # ── 构建调试条 ──
    debug_h = 180
    dw = w // 3

    def to_debug_panel(arr, title1, title2=None):
        if len(arr.shape) == 2:
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            bgr = arr.copy()
        bgr = cv2.resize(bgr, (dw, debug_h))
        cv2.putText(bgr, title1, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if title2:
            cv2.putText(bgr, title2, (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        return bgr

    db_blur = to_debug_panel(blur, f'Gaussian Blur', f'ksize={bk}')
    db_canny = to_debug_panel(edges, f'Canny', f'lo={p["canny_low"]} hi={p["canny_high"]}')
    db_closed = to_debug_panel(closed, f'Morph Close', f'k={ck} iter={ci}')

    debug_strip = np.hstack([db_blur, db_canny, db_closed])
    if debug_strip.shape[1] < w:
        pad = np.zeros((debug_h, w - debug_strip.shape[1], 3), dtype=np.uint8)
        debug_strip = np.hstack([debug_strip, pad])

    # ── 绘制结果 ──
    result = frame.copy()

    # 画面中心十字
    cv2.drawMarker(result, (w // 2, h // 2), (255, 0, 0),
                   markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

    if candidates:
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best = candidates[0]

        cv2.drawContours(result, [best['approx']], -1, (0, 255, 0), 3)

        labels = ['TL', 'TR', 'BR', 'BL']
        for i, pt in enumerate(best['corners']):
            cv2.circle(result, tuple(int(v) for v in pt), 6, (255, 0, 0), -1)
            cv2.putText(result, f'{labels[i]}({int(pt[0])},{int(pt[1])})',
                        (int(pt[0]) + 10, int(pt[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        center = line_intersection(best['corners'][0], best['corners'][2],
                                   best['corners'][1], best['corners'][3])
        if center is not None:
            cx_d, cy_d = center
            cv2.circle(result, (cx_d, cy_d), 8, (0, 0, 255), -1)
            cv2.line(result, tuple(int(v) for v in best['corners'][0]),
                     tuple(int(v) for v in best['corners'][2]), (255, 0, 255), 1)
            cv2.line(result, tuple(int(v) for v in best['corners'][1]),
                     tuple(int(v) for v in best['corners'][3]), (255, 0, 255), 1)

        edge_lengths = best['dists']
        ratio_h = edge_lengths[0] / max(edge_lengths[2], 1)
        ratio_v = edge_lengths[3] / max(edge_lengths[1], 1)

        cv2.putText(result, f'Center:({best["center"][0]:.0f},{best["center"][1]:.0f}) Score:{best["score"]:.2f}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(result, f'Area:{best["area"]:.0f} H-ratio:{ratio_h:.2f} V-ratio:{ratio_v:.2f}',
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(result, f'T:{edge_lengths[0]:.0f} B:{edge_lengths[2]:.0f} L:{edge_lengths[3]:.0f} R:{edge_lengths[1]:.0f}',
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        with data_lock:
            latest_data = {
                'found': True, 'timestamp': time.time(),
                'center': {'x': round(best['center'][0], 1), 'y': round(best['center'][1], 1)},
                'corners': [{'x': int(c[0]), 'y': int(c[1])} for c in best['corners']],
                'area_px': int(best['area']), 'score': round(best['score'], 3),
                'edges_px': [round(e, 1) for e in edge_lengths],
                'h_ratio': round(ratio_h, 3), 'v_ratio': round(ratio_v, 3),
                'num_candidates': len(candidates),
            }
        return result, debug_strip, best['center']

    else:
        cv2.putText(result, 'No target detected', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        with data_lock:
            latest_data = {
                'found': False, 'timestamp': time.time(),
                'num_candidates': 0,
            }
        return result, debug_strip, None


# ══════════════════════════════════════════════════════════════════
# 后台采集线程
# ══════════════════════════════════════════════════════════════════

def capture_loop(camera_idx: int, flip_frame: bool = True):
    """后台线程：持续采集摄像头、处理帧、运行 PID 控制"""
    global main_jpeg, debug_jpeg, latest_data

    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {camera_idx}")
        cap.release()
        return

    prev_time = time.time()
    fps = 0.0

    try:
        while not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                time.sleep(0.01)
                continue

            if flip_frame:
                frame = cv2.flip(frame, -1)

            with params_lock:
                p = dict(params)

            # ── 视觉处理 ──
            processed, debug_strip, target_center = process_frame(frame, p)

            # ── FPS 计算 ──
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                alpha = 0.98
                inst_fps = 1.0 / dt
                fps = alpha * fps + (1 - alpha) * inst_fps if fps > 0 else inst_fps

            # ── FPS 叠加 ──
            cv2.putText(processed, f"FPS: {fps:.1f}", (10, processed.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ── PID 追踪控制 + 模式化串口发送 ──
            ctrl_out = None
            with control_mode_lock:
                mode = control_mode

            if tracker is not None and mode == 'track':
                ret, ctrl_out = tracker.update(
                    frame_w=frame.shape[1],
                    frame_h=frame.shape[0],
                    target_center=target_center,
                    dt=max(dt, 1e-6),
                    now=now,
                )
                if ret and gimbal_serial:
                    with tracker_lock:
                        gimbal_serial.send(CommandPacket(
                            yaw_speed=float(ctrl_out.yaw_rpm),
                            pitch_speed=float(ctrl_out.pitch_rpm),
                            # enabled=2, stab=2 避免每帧重置PID积分
                        ))

            elif mode == 'manual':
                with tracker_lock:
                    if gimbal_serial:
                        gimbal_serial.send(CommandPacket(
                            yaw_speed=float(manual_yaw_rpm),
                            pitch_speed=float(manual_pitch_rpm),
                        ))
            # 'test': 测试线程独立发送
            # 'idle': 不发送任何指令

            # ── 控制信息叠加到主画面 ──
            if ctrl_out is not None:
                cv2.putText(
                    processed,
                    f"err(px)=({ctrl_out.err_x_px:.0f},{ctrl_out.err_y_px:.0f}) "
                    f"rpm=({ctrl_out.yaw_rpm:.1f},{ctrl_out.pitch_rpm:.1f})",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                )

            # ── 编码 JPEG ──
            _, main_buf = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            _, debug_buf = cv2.imencode('.jpg', debug_strip, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            with frame_lock:
                main_jpeg = main_buf.tobytes()
                debug_jpeg = debug_buf.tobytes()

            frame_event.set()

            # ── 更新数据面板 ──
            with data_lock:
                d = dict(latest_data)
                if ctrl_out is not None:
                    d['yaw_rpm'] = round(ctrl_out.yaw_rpm, 2)
                    d['pitch_rpm'] = round(ctrl_out.pitch_rpm, 2)
                    d['err_x_px'] = round(ctrl_out.err_x_px, 1)
                    d['err_y_px'] = round(ctrl_out.err_y_px, 1)
                else:
                    d['yaw_rpm'] = 0
                    d['pitch_rpm'] = 0
                    d['err_x_px'] = 0
                    d['err_y_px'] = 0
                d['fps'] = round(fps, 1)
                latest_data = d

            time.sleep(0.005)

    finally:
        cap.release()
        print("Camera released")


# ══════════════════════════════════════════════════════════════════
# MJPEG 生成器
# ══════════════════════════════════════════════════════════════════

def generate_main_frames():
    while not stop_event.is_set():
        frame_event.wait(1.0)
        frame_event.clear()
        with frame_lock:
            if main_jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + main_jpeg + b'\r\n')


def generate_debug_frames():
    while not stop_event.is_set():
        frame_event.wait(1.0)
        frame_event.clear()
        with frame_lock:
            if debug_jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + debug_jpeg + b'\r\n')


# ══════════════════════════════════════════════════════════════════
# 后台串口线程
# ══════════════════════════════════════════════════════════════════

def _telemetry_reader():
    """后台线程：持续读取 STM32 遥测数据"""
    while not stop_event.is_set():
        try:
            with tracker_lock:
                if gimbal_serial:
                    pkg = gimbal_serial.recv()
                    if pkg:
                        with telemetry_lock:
                            latest_telemetry.update({
                                'connected': True,
                                'imu_yaw': round(pkg.imu_yaw, 4),
                                'imu_pitch': round(pkg.imu_pitch, 4),
                                'imu_roll': round(pkg.imu_roll, 4),
                                'yaw_imu_angle': round(pkg.yaw_imu_angle, 4),
                                'pitch_imu_angle': round(pkg.pitch_imu_angle, 4),
                                'yaw_motor_angle': round(pkg.yaw_motor_angle, 4),
                                'pitch_motor_angle': round(pkg.pitch_motor_angle, 4),
                                'enabled': pkg.enabled,
                                'stability': pkg.stability_enabled,
                                'laser': pkg.laser_enabled,
                            })
            time.sleep(0.01)
        except Exception:
            time.sleep(0.1)  # 出错时等久一点


def _test_signal_runner(signal_type: str):
    """后台线程：生成测试信号波形（画圆 / 点头）"""
    global control_mode
    with control_mode_lock:
        control_mode = 'test'

    omega = 2.0 * math.pi * 0.5     # 0.5 Hz
    amplitude = 3.0                  # 速度幅值 RPM
    t0 = time.time()

    try:
        while not test_stop.is_set():
            t = time.time() - t0
            if signal_type == 'circle':
                yaw_rpm = amplitude * math.cos(omega * t)
                pitch_rpm = amplitude * math.sin(omega * t)
            elif signal_type == 'nod':
                yaw_rpm = 0.0
                pitch_rpm = amplitude * math.sin(omega * t)
            else:
                break

            with tracker_lock:
                if gimbal_serial:
                    gimbal_serial.send(CommandPacket(
                        yaw_speed=float(yaw_rpm),
                        pitch_speed=float(pitch_rpm),
                        # 默认 enabled=2, stab=2 不重置PID
                    ))
            time.sleep(0.02)
    finally:
        # 停止后发送零速指令
        with tracker_lock:
            if gimbal_serial:
                gimbal_serial.send(CommandPacket(
                    yaw_speed=0.0, pitch_speed=0.0,
                    
                ))
        with control_mode_lock:
            control_mode = 'idle'


# ══════════════════════════════════════════════════════════════════
# Flask 路由
# ══════════════════════════════════════════════════════════════════

app = Flask(__name__)


@app.route('/')
def index():
    return HTML_PAGE


@app.route('/video_feed')
def video_feed():
    return Response(generate_main_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/debug_feed')
def debug_feed():
    return Response(generate_debug_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/data')
def data():
    with data_lock:
        return jsonify(dict(latest_data))


@app.route('/set_param', methods=['POST'])
def set_param():
    global control_mode
    data = request.get_json()
    name = data['name']
    value = data['value']

    with params_lock:
        if name in params:
            if name in INT_PARAMS:
                value = int(value)
            else:
                value = float(value)
            params[name] = value

            # 同步 control_enabled 滑块 ↔ control_mode
            if name == 'control_enabled':
                with control_mode_lock:
                    if value == 1 and control_mode != 'test':
                        control_mode = 'track'
                    elif value == 0:
                        control_mode = 'idle'

    return jsonify({'status': 'ok'})


@app.route('/get_params')
def get_params():
    with params_lock:
        return jsonify(dict(params))


@app.route('/reset_params', methods=['POST'])
def reset_params():
    global tracker
    with params_lock:
        for k, v in DEFAULT_PARAMS.items():
            params[k] = v

    # 重建 PID 控制器
    with tracker_lock:
        cfg = _build_control_config(params)
        tracker = GimbalTracker(cfg)

    return jsonify({'status': 'ok', 'params': dict(params)})


@app.route('/reconnect_serial', methods=['POST'])
def reconnect_serial():
    """重新连接串口"""
    global gimbal_serial
    data = request.get_json() or {}
    port = data.get('port', None)
    baud = int(data.get('baud', 115200))

    with tracker_lock:
        if gimbal_serial is not None:
            gimbal_serial.close()
        if port:
            try:
                gimbal_serial = GimbalSerial(port=port, baudrate=baud)
            except Exception as e:
                gimbal_serial = None
                return jsonify({'status': 'error', 'message': str(e)})
        else:
            gimbal_serial = None
    return jsonify({'status': 'ok', 'port': port, 'baud': baud})


@app.route('/manual_control', methods=['POST'])
def manual_control():
    """手动方向控制"""
    global control_mode, manual_yaw_rpm, manual_pitch_rpm
    data = request.get_json()
    direction = data.get('direction', 'stop')

    # 保持锁顺序: params_lock → control_mode_lock（与 set_param 一致，避免死锁）
    if direction != 'stop':
        with params_lock:
            params['control_enabled'] = 0  # 手动模式下关闭 PID

    with control_mode_lock:
        if direction == 'stop':
            manual_yaw_rpm = 0.0
            manual_pitch_rpm = 0.0
        elif control_mode != 'test':
            if control_mode != 'manual':
                # 进入手动模式: 关增稳→速度模式 (IMU不在pitch轴)
                with tracker_lock:
                    if gimbal_serial:
                        gimbal_serial.send(CommandPacket(
                            yaw_speed=0.0, pitch_speed=0.0,
                            stability_enabled=0,
                        ))
            control_mode = 'manual'
            if direction == 'up':
                manual_yaw_rpm = 0.0
                manual_pitch_rpm = -MANUAL_SPEED
            elif direction == 'down':
                manual_yaw_rpm = 0.0
                manual_pitch_rpm = MANUAL_SPEED
            elif direction == 'left':
                manual_yaw_rpm = -MANUAL_SPEED
                manual_pitch_rpm = 0.0
            elif direction == 'right':
                manual_yaw_rpm = MANUAL_SPEED
                manual_pitch_rpm = 0.0
            else:
                manual_yaw_rpm = 0.0
                manual_pitch_rpm = 0.0

    return jsonify({
        'status': 'ok',
        'mode': control_mode,
        'yaw_rpm': manual_yaw_rpm,
        'pitch_rpm': manual_pitch_rpm,
    })


@app.route('/set_mode', methods=['POST'])
def set_mode():
    """切换控制模式"""
    global control_mode
    data = request.get_json()
    mode = data.get('mode', 'track')

    with control_mode_lock:
        if mode in ('track', 'idle'):
            # 先停止测试线程
            if test_thread and test_thread.is_alive():
                test_stop.set()
            control_mode = mode

    # 切换到 track: 使能+开增稳 (PID闭环, IMU反馈)
    if mode == 'track':
        with tracker_lock:
            if gimbal_serial:
                gimbal_serial.send(CommandPacket(
                    yaw_speed=0.0, pitch_speed=0.0,
                    enabled=1, stability_enabled=1,
                ))
    # 切换到 idle: 关闭增稳 (开环速度模式, 安全)
    elif mode == 'idle':
        with tracker_lock:
            if gimbal_serial:
                gimbal_serial.send(CommandPacket(
                    yaw_speed=0.0, pitch_speed=0.0,
                    stability_enabled=0,
                ))

    # 同步 params 中的 control_enabled
    with params_lock:
        params['control_enabled'] = 1 if mode == 'track' else 0

    return jsonify({'status': 'ok', 'mode': control_mode})


@app.route('/test_signal', methods=['POST'])
def test_signal():
    """启动 / 停止测试信号"""
    global test_thread, test_stop, control_mode
    data = request.get_json()
    signal = data.get('signal', 'stop')

    if signal == 'stop':
        test_stop.set()
        if test_thread and test_thread.is_alive():
            test_thread.join(timeout=2.0)
            test_thread = None
        with control_mode_lock:
            control_mode = 'idle'
    else:
        # 关增稳→速度模式 (IMU不在pitch轴, PID闭环会振荡)
        with tracker_lock:
            if gimbal_serial:
                gimbal_serial.send(CommandPacket(
                    yaw_speed=0.0, pitch_speed=0.0,
                    stability_enabled=0,
                ))
        # 先停止之前的线程
        test_stop.set()
        if test_thread and test_thread.is_alive():
            test_thread.join(timeout=2.0)
        test_stop.clear()
        test_thread = threading.Thread(
            target=_test_signal_runner,
            args=(signal,),
            daemon=True,
        )
        test_thread.start()

    # 同步 params: 测试期间关闭 PID
    with params_lock:
        params['control_enabled'] = 0 if signal != 'stop' else params['control_enabled']

    return jsonify({'status': 'ok', 'signal': signal, 'mode': control_mode})


@app.route('/gimbal_reset', methods=['POST'])
def gimbal_reset():
    """复位云台 — 发送零速 + 切换到 idle 模式"""
    global control_mode, manual_yaw_rpm, manual_pitch_rpm

    with control_mode_lock:
        control_mode = 'idle'
    manual_yaw_rpm = 0.0
    manual_pitch_rpm = 0.0

    # 停止测试
    test_stop.set()

    # 同步 params
    with params_lock:
        params['control_enabled'] = 0

    # 发送零速
    with tracker_lock:
        if gimbal_serial:
            for _ in range(3):
                gimbal_serial.send(CommandPacket(
                    yaw_speed=0.0, pitch_speed=0.0,
                    
                ))
                time.sleep(0.02)

    return jsonify({'status': 'ok'})


@app.route('/telemetry')
def telemetry():
    """获取串口遥测数据"""
    with telemetry_lock:
        return jsonify(dict(latest_telemetry))


@app.route('/update_tracker', methods=['POST'])
def update_tracker():
    """根据当前参数重建 PID 控制器"""
    global tracker
    with params_lock:
        p = dict(params)

    with tracker_lock:
        cfg = _build_control_config(p)
        tracker = GimbalTracker(cfg)
    return jsonify({'status': 'ok'})


def _build_control_config(p: dict) -> ControlConfig:
    """从参数字典构建 ControlConfig"""
    return ControlConfig(
        enabled=bool(p['control_enabled']),
        deadband_px=float(p['deadband_px']),
        lost_timeout_s=float(p['lost_timeout_s']),
        max_rpm_yaw=float(p['max_rpm_yaw']),
        max_rpm_pitch=float(p['max_rpm_pitch']),
        invert_yaw=bool(p['invert_yaw']),
        invert_pitch=bool(p['invert_pitch']),
        yaw_pid=PIDConfig(
            kp=float(p['yaw_kp']), ki=float(p['yaw_ki']), kd=float(p['yaw_kd']),
            integral_limit=float(p['yaw_integral_limit']),
            output_limit=float(p['yaw_output_limit']),
        ),
        pitch_pid=PIDConfig(
            kp=float(p['pitch_kp']), ki=float(p['pitch_ki']), kd=float(p['pitch_kd']),
            integral_limit=float(p['pitch_integral_limit']),
            output_limit=float(p['pitch_output_limit']),
        ),
    )


# ══════════════════════════════════════════════════════════════════
# HTML 页面（含所有参数滑块）
# ══════════════════════════════════════════════════════════════════

HTML_PAGE = r'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QGimbal-Vision — 云台视觉追踪调参</title>
<style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
        background:#0a0a14; color:#d0d0d0;
        font-family:'Segoe UI','PingFang SC','Microsoft YaHei',monospace;
        padding:12px;
    }
    .header { text-align:center; margin-bottom:12px; }
    .header h1 { font-size:20px; color:#4ade80; letter-spacing:1px; }
    .header .dot {
        display:inline-block; width:8px; height:8px;
        background:#ef4444; border-radius:50%; margin-right:6px;
        animation:blink 1.2s infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
    .header .live-text { font-size:11px; color:#94a3b8; margin-top:2px; }

    .layout { display:flex; gap:12px; flex-wrap:wrap; justify-content:center; }

    .video-col { flex:0 0 auto; }
    .video-box {
        border:2px solid #1e293b; border-radius:8px; overflow:hidden;
        background:#020617; line-height:0;
        box-shadow:0 0 30px rgba(74,222,128,0.06),0 4px 20px rgba(0,0,0,0.4);
    }
    .video-box img { display:block; width:640px; height:480px; }
    .debug-box {
        margin-top:8px; border:2px solid #1e293b; border-radius:8px;
        overflow:hidden; background:#020617; line-height:0;
        box-shadow:0 0 20px rgba(96,165,250,0.06);
    }
    .debug-box img { display:block; width:640px; height:180px; }
    .section-label {
        font-size:10px; color:#64748b; letter-spacing:1px;
        margin:4px 0 2px 2px;
    }

    .data-panel {
        background:#0f1019; border:1px solid #1e293b; border-radius:8px;
        padding:14px; width:290px; box-shadow:0 4px 20px rgba(0,0,0,0.3);
        height:fit-content; font-size:11px;
    }
    .data-panel h3 { font-size:13px; color:#4ade80; margin-bottom:10px; }
    .data-row { display:flex; justify-content:space-between; padding:5px 0;
        border-bottom:1px solid #141520; font-size:11px; }
    .data-row .label { color:#94a3b8; }
    .data-row .value { color:#e2e8f0; font-weight:500; }
    .data-row .value.hl { color:#4ade80; font-size:13px; }
    .status-badge { display:inline-block; padding:2px 8px; border-radius:8px;
        font-size:10px; font-weight:600; }
    .status-badge.found { background:#065f46; color:#4ade80; }
    .status-badge.lost  { background:#7f1d1d; color:#fca5a5; }
    .coord-grid { display:grid; grid-template-columns:1fr 1fr; gap:4px; margin-top:4px; }
    .coord-cell { background:#141520; border-radius:4px; padding:5px; text-align:center; font-size:10px; }
    .coord-cell .cl { color:#60a5fa; font-weight:600; }
    .coord-cell .cv { color:#cbd5e1; margin-top:1px; }
    .data-section { font-size:11px; color:#f59e0b; margin:10px 0 4px;
        padding-bottom:3px; border-bottom:1px solid #1e293b; }

    .sliders-panel {
        background:#0f1019; border:1px solid #1e293b; border-radius:8px;
        padding:14px; width:100%; max-width:950px; margin-top:10px;
        box-shadow:0 4px 20px rgba(0,0,0,0.3);
    }
    .sliders-panel h3 { font-size:13px; color:#f59e0b; margin-bottom:4px; }
    .slider-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px 16px; }
    @media (max-width:700px) { .slider-grid { grid-template-columns:1fr; } }
    .slider-group { }
    .slider-group label { display:flex; justify-content:space-between;
        font-size:10px; color:#94a3b8; margin-bottom:2px; }
    .slider-group label .val { color:#4ade80; font-weight:600; }
    .slider-group input[type=range] { width:100%; accent-color:#4ade80; }
    .section-title {
        font-size:11px; color:#60a5fa; font-weight:600;
        margin:10px 0 4px; padding-bottom:3px; border-bottom:1px solid #1e293b;
    }
    .section-title.first { margin-top:4px; }
    .btn-row { margin-top:10px; display:flex; gap:8px; flex-wrap:wrap; }
    .btn {
        padding:5px 14px; border:none; border-radius:5px; font-size:11px;
        cursor:pointer; font-weight:600;
    }
    .btn-reset { background:#7f1d1d; color:#fca5a5; }
    .btn-reset:hover { background:#991b1b; }
    .btn-apply { background:#1e3a5f; color:#60a5fa; }
    .btn-apply:hover { background:#1e4a6f; }
    .btn-serial { background:#1e3a5f; color:#f59e0b; }
    .btn-serial:hover { background:#1e4a6f; }
    .info-bar { margin-top:10px; display:flex; gap:10px; font-size:10px; color:#64748b;
        justify-content:center; flex-wrap:wrap; }
    .info-bar span { background:#141520; padding:3px 8px; border-radius:10px; }
    .serial-row { display:flex; gap:8px; align-items:center; margin-top:8px; flex-wrap:wrap; }
    .serial-row input {
        background:#141520; border:1px solid #1e293b; color:#e2e8f0;
        padding:4px 8px; border-radius:4px; font-size:11px; width:120px;
    }
    .serial-row label { font-size:11px; color:#94a3b8; }

    /* ── 控制面板 ── */
    .control-panel {
        background:#0f1019; border:1px solid #1e293b; border-radius:8px;
        padding:14px; width:100%; max-width:950px; margin-top:10px;
        box-shadow:0 4px 20px rgba(0,0,0,0.3);
    }
    .control-panel h3 { font-size:13px; color:#60a5fa; margin-bottom:10px; }
    .control-row { display:flex; gap:16px; flex-wrap:wrap; }
    .control-col { flex:1; min-width:130px; }
    .control-col-title { font-size:10px; color:#64748b; margin-bottom:6px; letter-spacing:1px; }

    /* 方向键十字布局 */
    .dpad { display:grid; grid-template-columns:48px 48px 48px; grid-template-rows:48px 48px 48px; gap:3px; }
    .dpad .corner { visibility:hidden; }
    .btn-dir {
        background:#1e3a5f; border:1px solid #2a4a7f; border-radius:6px;
        color:#e2e8f0; font-size:18px; cursor:pointer;
        display:flex; align-items:center; justify-content:center;
        user-select:none; -webkit-user-select:none;
        transition:background 0.1s;
    }
    .btn-dir:hover { background:#2a5a8f; }
    .btn-dir:active, .btn-dir.active { background:#4ade80; color:#020617; }
    .btn-dir.stop { background:#7f1d1d; color:#fca5a5; font-size:12px; font-weight:700; }
    .btn-dir.stop:hover { background:#991b1b; }

    /* 按钮样式 */
    .btn-ctrl {
        display:block; width:100%; padding:8px 10px; margin-bottom:5px;
        border:none; border-radius:5px; font-size:11px; cursor:pointer; font-weight:600;
        text-align:center; transition:background 0.15s;
    }
    .btn-track { background:#065f46; color:#4ade80; }
    .btn-track:hover { background:#0a7a56; }
    .btn-track.active { background:#4ade80; color:#020617; }
    .btn-idle { background:#7f651d; color:#fbbf24; }
    .btn-idle:hover { background:#9a7b2a; }
    .btn-idle.active { background:#fbbf24; color:#020617; }
    .btn-reset-ctrl { background:#7f1d1d; color:#fca5a5; }
    .btn-reset-ctrl:hover { background:#991b1b; }
    .btn-test { background:#1e3a5f; color:#60a5fa; }
    .btn-test:hover { background:#2a5a8f; }
    .btn-test.running { background:#4ade80; color:#020617; }
    .btn-stop-test { background:#7f1d1d; color:#fca5a5; margin-top:10px; }
    .btn-stop-test:hover { background:#991b1b; }

    /* 遥测面板 */
    .telemetry-box {
        background:#020617; border:1px solid #1e293b; border-radius:6px;
        padding:10px; margin-top:10px; font-family:'Courier New',monospace; font-size:11px;
    }
    .telemetry-box .t-title { color:#64748b; font-size:10px; margin-bottom:6px; }
    .telemetry-box .t-row { display:flex; gap:16px; flex-wrap:wrap; color:#94a3b8; }
    .telemetry-box .t-row .t-val { color:#4ade80; }
    .telemetry-box .t-row .t-warn { color:#fbbf24; }
    .telemetry-box .t-row .t-off { color:#ef4444; }
</style>
</head>
<body>

<div class="header">
    <h1><span class="dot" id="status_dot"></span>QGimbal-Vision</h1>
    <p class="live-text">LIVE &middot; 640x480 &middot; <span id="fps_disp">--</span> FPS</p>
</div>

<div class="layout">
    <div class="video-col">
        <div class="section-label">主画面 — 检测结果 + PID 控制信息</div>
        <div class="video-box">
            <img src="/video_feed" width="640" height="480" alt="Main Stream">
        </div>
        <div class="section-label">调试视图 — 高斯模糊 | Canny 边缘 | 闭运算</div>
        <div class="debug-box">
            <img src="/debug_feed" width="640" height="180" alt="Debug Stream">
        </div>
    </div>

    <div class="data-panel">
        <h3>检测数据</h3>
        <div class="data-row"><span class="label">状态</span><span class="value" id="status">--</span></div>
        <div class="data-row"><span class="label">评分</span><span class="value" id="score">--</span></div>
        <div class="data-row"><span class="label">中心 X</span><span class="value hl" id="cx">--</span></div>
        <div class="data-row"><span class="label">中心 Y</span><span class="value hl" id="cy">--</span></div>
        <div class="data-row"><span class="label">面积 (px)</span><span class="value" id="area">--</span></div>
        <div class="data-row"><span class="label">候选数</span><span class="value" id="num_candidates">--</span></div>
        <div class="data-row"><span class="label">H-ratio</span><span class="value" id="hratio">--</span></div>
        <div class="data-row"><span class="label">V-ratio</span><span class="value" id="vratio">--</span></div>
        <div class="data-row"><span class="label">上 / 下 (px)</span><span class="value" id="edges_h">--</span></div>
        <div class="data-row"><span class="label">左 / 右 (px)</span><span class="value" id="edges_v">--</span></div>

        <div class="data-section">PID 控制输出</div>
        <div class="data-row"><span class="label">Yaw RPM</span><span class="value hl" id="yaw_rpm">--</span></div>
        <div class="data-row"><span class="label">Pitch RPM</span><span class="value hl" id="pitch_rpm">--</span></div>
        <div class="data-row"><span class="label">Err X (px)</span><span class="value" id="err_x_px">--</span></div>
        <div class="data-row"><span class="label">Err Y (px)</span><span class="value" id="err_y_px">--</span></div>

        <div style="margin-top:6px; font-size:10px; color:#94a3b8;">四角坐标</div>
        <div class="coord-grid" id="corners_grid">
            <div class="coord-cell"><div class="cl">TL</div><div class="cv">--</div></div>
            <div class="coord-cell"><div class="cl">TR</div><div class="cv">--</div></div>
            <div class="coord-cell"><div class="cl">BL</div><div class="cv">--</div></div>
            <div class="coord-cell"><div class="cl">BR</div><div class="cv">--</div></div>
        </div>
    </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════ -->
<!--  云台控制面板                                                      -->
<!-- ═══════════════════════════════════════════════════════════════ -->
<div class="control-panel">
    <h3>🎮 云台控制</h3>
    <div class="control-row">

        <!-- 方向控制 -->
        <div class="control-col">
            <div class="control-col-title">方向控制 (按住移动，松开停止)</div>
            <div class="dpad" id="dpad">
                <div class="corner"></div>
                <button class="btn-dir" id="btn-up"    data-dir="up">▲</button>
                <div class="corner"></div>
                <button class="btn-dir" id="btn-left"  data-dir="left">◀</button>
                <button class="btn-dir stop" id="btn-stop-dpad" data-dir="stop">STOP</button>
                <button class="btn-dir" id="btn-right" data-dir="right">▶</button>
                <div class="corner"></div>
                <button class="btn-dir" id="btn-down"  data-dir="down">▼</button>
                <div class="corner"></div>
            </div>
        </div>

        <!-- 模式切换 -->
        <div class="control-col">
            <div class="control-col-title">模式切换</div>
            <button class="btn-ctrl btn-track active" id="btn-track" onclick="setMode('track')">🎯 开始跟踪</button>
            <button class="btn-ctrl btn-idle" id="btn-idle" onclick="setMode('idle')">⏸ 待机</button>
            <button class="btn-ctrl btn-reset-ctrl" id="btn-reset" onclick="resetGimbal()">🔄 复位云台</button>
        </div>

        <!-- 测试信号 -->
        <div class="control-col">
            <div class="control-col-title">测试信号</div>
            <button class="btn-ctrl btn-test" id="btn-circle" onclick="startTest('circle')">⭕ 画圆</button>
            <button class="btn-ctrl btn-test" id="btn-nod" onclick="startTest('nod')">↕ 点头</button>
            <button class="btn-ctrl btn-stop-test" id="btn-stop-test" onclick="stopTest()">⏹ 停止测试</button>
            <div class="control-col-title" style="margin-top:8px;">当前模式: <span id="mode_disp" style="color:#4ade80;">track</span></div>
        </div>
    </div>

    <!-- 串口遥测 -->
    <div class="telemetry-box">
        <div class="t-title">📡 串口遥测 (STM32)</div>
        <div class="t-row">
            <span>IMU: Y=<span class="t-val" id="t_imu_yaw">--</span>° P=<span class="t-val" id="t_imu_pitch">--</span>° R=<span class="t-val" id="t_imu_roll">--</span>°</span>
            <span>Motor: Y=<span class="t-val" id="t_motor_yaw">--</span>° P=<span class="t-val" id="t_motor_pitch">--</span>°</span>
            <span>使能:<span class="t-val" id="t_enabled">--</span></span>
            <span>增稳:<span class="t-val" id="t_stability">--</span></span>
        </div>
    </div>
</div>

<div class="sliders-panel">
    <h3>参数调优</h3>

    <!-- 预处理 -->
    <div class="section-title first">预处理</div>
    <div class="slider-grid">
        <div class="slider-group">
            <label>高斯模糊核 <span class="val" id="v_blur_ksize">3</span></label>
            <input type="range" id="blur_ksize" min="1" max="15" step="2" value="3"
                   oninput="setP('blur_ksize',this.value)">
        </div>
        <div class="slider-group">
            <label>CLAHE (0=关 1=开) <span class="val" id="v_use_clahe">0</span></label>
            <input type="range" id="use_clahe" min="0" max="1" step="1" value="0"
                   oninput="setP('use_clahe',this.value)">
        </div>
        <div class="slider-group">
            <label>Canny 低阈值 <span class="val" id="v_canny_low">25</span></label>
            <input type="range" id="canny_low" min="0" max="255" value="25"
                   oninput="setP('canny_low',this.value)">
        </div>
        <div class="slider-group">
            <label>Canny 高阈值 <span class="val" id="v_canny_high">75</span></label>
            <input type="range" id="canny_high" min="0" max="255" value="75"
                   oninput="setP('canny_high',this.value)">
        </div>
        <div class="slider-group">
            <label>闭运算核大小 <span class="val" id="v_close_ksize">3</span></label>
            <input type="range" id="close_ksize" min="1" max="7" value="3"
                   oninput="setP('close_ksize',this.value)">
        </div>
        <div class="slider-group">
            <label>闭运算迭代 <span class="val" id="v_close_iter">1</span></label>
            <input type="range" id="close_iter" min="0" max="3" step="1" value="1"
                   oninput="setP('close_iter',this.value)">
        </div>
    </div>

    <!-- 矩形检测（开源算法参数） -->
    <div class="section-title">矩形检测（开源算法）</div>
    <div class="slider-grid">
        <div class="slider-group">
            <label>多边形逼近 eps <span class="val" id="v_approx_eps">0.020</span></label>
            <input type="range" id="approx_eps" min="0.005" max="0.100" step="0.005" value="0.02"
                   oninput="setP('approx_eps',this.value)">
        </div>
        <div class="slider-group">
            <label>直角容差 (deg) <span class="val" id="v_angle_tol">25.0</span></label>
            <input type="range" id="angle_tol" min="5" max="45" step="0.5" value="25"
                   oninput="setP('angle_tol',this.value)">
        </div>
        <div class="slider-group">
            <label>最小面积比 <span class="val" id="v_min_area_ratio">0.005</span></label>
            <input type="range" id="min_area_ratio" min="0.001" max="0.05" step="0.001" value="0.005"
                   oninput="setP('min_area_ratio',this.value)">
        </div>
        <div class="slider-group">
            <label>最大面积比 <span class="val" id="v_max_area_ratio">0.500</span></label>
            <input type="range" id="max_area_ratio" min="0.05" max="0.95" step="0.01" value="0.5"
                   oninput="setP('max_area_ratio',this.value)">
        </div>
    </div>

    <!-- A4 验证 -->
    <div class="section-title">A4 纸验证</div>
    <div class="slider-grid">
        <div class="slider-group">
            <label>最小边长 (px) <span class="val" id="v_min_edge">40</span></label>
            <input type="range" id="min_edge" min="10" max="200" value="40"
                   oninput="setP('min_edge',this.value)">
        </div>
        <div class="slider-group">
            <label>对边最小比例 <span class="val" id="v_persp_min">0.25</span></label>
            <input type="range" id="persp_min" min="0.05" max="0.50" step="0.01" value="0.25"
                   oninput="setP('persp_min',this.value)">
        </div>
        <div class="slider-group">
            <label>长宽比下限 <span class="val" id="v_ratio_min">1.08</span></label>
            <input type="range" id="ratio_min" min="0.50" max="1.50" step="0.01" value="1.08"
                   oninput="setP('ratio_min',this.value)">
        </div>
        <div class="slider-group">
            <label>长宽比上限 <span class="val" id="v_ratio_max">2.00</span></label>
            <input type="range" id="ratio_max" min="1.50" max="3.00" step="0.01" value="2.00"
                   oninput="setP('ratio_max',this.value)">
        </div>
    </div>

    <!-- PID Yaw -->
    <div class="section-title">PID — Yaw 轴 (水平旋转)</div>
    <div class="slider-grid">
        <div class="slider-group">
            <label>Kp <span class="val" id="v_yaw_kp">4.00</span></label>
            <input type="range" id="yaw_kp" min="0" max="10" step="0.1" value="4.0"
                   oninput="setP('yaw_kp',this.value)">
        </div>
        <div class="slider-group">
            <label>Ki <span class="val" id="v_yaw_ki">0.80</span></label>
            <input type="range" id="yaw_ki" min="0" max="5" step="0.05" value="0.80"
                   oninput="setP('yaw_ki',this.value)">
        </div>
        <div class="slider-group">
            <label>Kd <span class="val" id="v_yaw_kd">0.08</span></label>
            <input type="range" id="yaw_kd" min="0" max="1" step="0.01" value="0.08"
                   oninput="setP('yaw_kd',this.value)">
        </div>
        <div class="slider-group">
            <label>积分限幅 <span class="val" id="v_yaw_integral_limit">0.20</span></label>
            <input type="range" id="yaw_integral_limit" min="0.05" max="2" step="0.05" value="0.20"
                   oninput="setP('yaw_integral_limit',this.value)">
        </div>
        <div class="slider-group">
            <label>输出限幅 <span class="val" id="v_yaw_output_limit">1.00</span></label>
            <input type="range" id="yaw_output_limit" min="0.1" max="1.0" step="0.05" value="1.0"
                   oninput="setP('yaw_output_limit',this.value)">
        </div>
    </div>

    <!-- PID Pitch -->
    <div class="section-title">PID — Pitch 轴 (俯仰)</div>
    <div class="slider-grid">
        <div class="slider-group">
            <label>Kp <span class="val" id="v_pitch_kp">3.00</span></label>
            <input type="range" id="pitch_kp" min="0" max="10" step="0.1" value="3.0"
                   oninput="setP('pitch_kp',this.value)">
        </div>
        <div class="slider-group">
            <label>Ki <span class="val" id="v_pitch_ki">0.60</span></label>
            <input type="range" id="pitch_ki" min="0" max="5" step="0.05" value="0.60"
                   oninput="setP('pitch_ki',this.value)">
        </div>
        <div class="slider-group">
            <label>Kd <span class="val" id="v_pitch_kd">0.06</span></label>
            <input type="range" id="pitch_kd" min="0" max="1" step="0.01" value="0.06"
                   oninput="setP('pitch_kd',this.value)">
        </div>
        <div class="slider-group">
            <label>积分限幅 <span class="val" id="v_pitch_integral_limit">0.20</span></label>
            <input type="range" id="pitch_integral_limit" min="0.05" max="2" step="0.05" value="0.20"
                   oninput="setP('pitch_integral_limit',this.value)">
        </div>
        <div class="slider-group">
            <label>输出限幅 <span class="val" id="v_pitch_output_limit">1.00</span></label>
            <input type="range" id="pitch_output_limit" min="0.1" max="1.0" step="0.05" value="1.0"
                   oninput="setP('pitch_output_limit',this.value)">
        </div>
    </div>

    <!-- 控制参数 -->
    <div class="section-title">追踪控制参数</div>
    <div class="slider-grid">
        <div class="slider-group">
            <label>启用控制 <span class="val" id="v_control_enabled">1</span></label>
            <input type="range" id="control_enabled" min="0" max="1" step="1" value="1"
                   oninput="setP('control_enabled',this.value)">
        </div>
        <div class="slider-group">
            <label>像素死区 <span class="val" id="v_deadband_px">0.0</span></label>
            <input type="range" id="deadband_px" min="0" max="100" step="1" value="0"
                   oninput="setP('deadband_px',this.value)">
        </div>
        <div class="slider-group">
            <label>丢目标超时 (s) <span class="val" id="v_lost_timeout_s">0.40</span></label>
            <input type="range" id="lost_timeout_s" min="0.05" max="2.0" step="0.05" value="0.4"
                   oninput="setP('lost_timeout_s',this.value)">
        </div>
        <div class="slider-group">
            <label>Yaw 最大 RPM <span class="val" id="v_max_rpm_yaw">20.0</span></label>
            <input type="range" id="max_rpm_yaw" min="1" max="200" step="1" value="20"
                   oninput="setP('max_rpm_yaw',this.value)">
        </div>
        <div class="slider-group">
            <label>Pitch 最大 RPM <span class="val" id="v_max_rpm_pitch">20.0</span></label>
            <input type="range" id="max_rpm_pitch" min="1" max="200" step="1" value="20"
                   oninput="setP('max_rpm_pitch',this.value)">
        </div>
        <div class="slider-group">
            <label>反转 Yaw (0=正常 1=反转) <span class="val" id="v_invert_yaw">1</span></label>
            <input type="range" id="invert_yaw" min="0" max="1" step="1" value="1"
                   oninput="setP('invert_yaw',this.value)">
        </div>
        <div class="slider-group">
            <label>反转 Pitch <span class="val" id="v_invert_pitch">0</span></label>
            <input type="range" id="invert_pitch" min="0" max="1" step="1" value="0"
                   oninput="setP('invert_pitch',this.value)">
        </div>
    </div>

    <!-- 串口 -->
    <div class="section-title">串口通信</div>
    <div class="serial-row">
        <label>端口:</label>
        <input type="text" id="serial_port" placeholder="COM3 或留空" value="">
        <label>波特率:</label>
        <input type="number" id="serial_baud" value="115200" style="width:100px;">
        <button class="btn btn-serial" onclick="reconnectSerial()">重连串口</button>
    </div>

    <div class="btn-row">
        <button class="btn btn-apply" onclick="applyPID()">应用 PID / 控制参数</button>
        <button class="btn btn-reset" onclick="resetAll()">重置所有为默认值</button>
    </div>
</div>

<div class="info-bar">
    <span>QGimbal-Vision v2.0</span>
    <span>OpenCV + PID + STM32</span>
    <span>Flask 实时调参</span>
    <span id="param_count">28 参数</span>
</div>

<script>
// ── 需要浮点格式化的键 ──
const FLOAT_KEYS = new Set([
    'approx_eps','angle_tol','persp_min','ratio_min','ratio_max',
    'min_area_ratio','max_area_ratio','deadband_px','lost_timeout_s',
    'yaw_kp','yaw_ki','yaw_kd','yaw_integral_limit','yaw_output_limit',
    'pitch_kp','pitch_ki','pitch_kd','pitch_integral_limit','pitch_output_limit',
    'max_rpm_yaw','max_rpm_pitch'
]);

function fmt(key, val) {
    if (FLOAT_KEYS.has(key)) return parseFloat(val).toFixed(3);
    return val;
}

function setP(name, value) {
    document.getElementById('v_' + name).textContent = fmt(name, value);
    fetch('/set_param', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({name:name, value:parseFloat(value)})
    }).catch(e=>console.error(e));
}

function applyPID() {
    fetch('/update_tracker', {method:'POST'})
        .then(r=>r.json())
        .then(d=>console.log('PID updated:', d))
        .catch(e=>console.error(e));
}

function reconnectSerial() {
    const port = document.getElementById('serial_port').value || null;
    const baud = parseInt(document.getElementById('serial_baud').value) || 115200;
    fetch('/reconnect_serial', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({port:port, baud:baud})
    }).then(r=>r.json()).then(d=>console.log('Serial:', d)).catch(e=>console.error(e));
}

function resetAll() {
    fetch('/reset_params', {method:'POST'})
        .then(r=>r.json())
        .then(d=>{
            for(const [k,v] of Object.entries(d.params)) {
                const el = document.getElementById(k);
                if(el) {
                    el.value = v;
                    document.getElementById('v_'+k).textContent = fmt(k, v);
                }
            }
        })
        .then(()=>console.log('Reset done'))
        .catch(e=>console.error(e));
}

function updateData() {
    fetch('/data').then(r=>r.json()).then(d=>{
        const s=document.getElementById('status');
        const dot=document.getElementById('status_dot');
        document.getElementById('fps_disp').textContent = d.fps || '--';

        if(d.found && d.center) {
            s.innerHTML='<span class="status-badge found">已检测到</span>';
            dot.style.background='#4ade80';
            document.getElementById('score').textContent=d.score;
            document.getElementById('cx').textContent=d.center.x;
            document.getElementById('cy').textContent=d.center.y;
            document.getElementById('area').textContent=d.area_px;
            document.getElementById('num_candidates').textContent=d.num_candidates;
            document.getElementById('hratio').textContent=d.h_ratio;
            document.getElementById('vratio').textContent=d.v_ratio;
            if(d.edges_px) {
                document.getElementById('edges_h').textContent=d.edges_px[0]+' / '+d.edges_px[2];
                document.getElementById('edges_v').textContent=d.edges_px[3]+' / '+d.edges_px[1];
            }
            if(d.corners) {
                const g=document.getElementById('corners_grid');
                g.innerHTML=['TL','TR','BR','BL'].map((l,i)=>
                    '<div class="coord-cell"><div class="cl">'+l+'</div><div class="cv">('+d.corners[i].x+', '+d.corners[i].y+')</div></div>'
                ).join('');
            }
        } else {
            s.innerHTML='<span class="status-badge lost">未检测到</span>';
            dot.style.background='#ef4444';
            ['score','cx','cy','area','num_candidates','hratio','vratio','edges_h','edges_v'].forEach(
                id=>document.getElementById(id).textContent='--');
        }
        document.getElementById('yaw_rpm').textContent = d.yaw_rpm || '0';
        document.getElementById('pitch_rpm').textContent = d.pitch_rpm || '0';
        document.getElementById('err_x_px').textContent = d.err_x_px || '0';
        document.getElementById('err_y_px').textContent = d.err_y_px || '0';
    }).catch(e=>console.error(e));
}

// ── 遥测轮询 ──
function updateTelemetry() {
    fetch('/telemetry').then(r=>r.json()).then(d=>{
        const toDeg = function(rad) { return (rad * 180 / Math.PI).toFixed(1); };
        if (d.connected) {
            document.getElementById('t_imu_yaw').textContent = toDeg(d.imu_yaw);
            document.getElementById('t_imu_pitch').textContent = toDeg(d.imu_pitch);
            document.getElementById('t_imu_roll').textContent = toDeg(d.imu_roll);
            document.getElementById('t_motor_yaw').textContent = toDeg(d.yaw_motor_angle);
            document.getElementById('t_motor_pitch').textContent = toDeg(d.pitch_motor_angle);
            document.getElementById('t_enabled').textContent = d.enabled;
            document.getElementById('t_enabled').className = d.enabled ? 't-val' : 't-off';
            document.getElementById('t_stability').textContent = d.stability;
            document.getElementById('t_stability').className = d.stability ? 't-val' : 't-warn';
        }
    }).catch(e=>{});
}

// ── 方向控制 ──
(function() {
    const dpadBtns = document.querySelectorAll('#dpad .btn-dir');
    let activeDir = null;

    function sendDir(dir) {
        if (activeDir === dir) return;
        activeDir = dir;
        fetch('/manual_control', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({direction:dir})
        }).catch(e=>console.error(e));

        // 视觉反馈
        dpadBtns.forEach(b => b.classList.remove('active'));
        const btn = document.querySelector('[data-dir="'+dir+'"]');
        if (btn) btn.classList.add('active');
    }

    dpadBtns.forEach(btn => {
        const dir = btn.getAttribute('data-dir');
        if (dir === 'stop') return;  // stop 按钮不同处理

        btn.addEventListener('pointerdown', function(e) {
            e.preventDefault();
            sendDir(dir);
        });
        btn.addEventListener('pointerleave', function(e) {
            sendDir('stop');
        });
        btn.addEventListener('pointerup', function(e) {
            sendDir('stop');
        });
    });

    // stop 按钮点击
    document.getElementById('btn-stop-dpad').addEventListener('click', function() {
        sendDir('stop');
        dpadBtns.forEach(b => b.classList.remove('active'));
    });
})();

// ── 模式切换 ──
async function setMode(mode) {
    const r = await fetch('/set_mode', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({mode:mode})
    });
    const d = await r.json();
    document.getElementById('mode_disp').textContent = d.mode;
    document.getElementById('btn-track').classList.toggle('active', d.mode === 'track');
    document.getElementById('btn-idle').classList.toggle('active', d.mode === 'idle');
    console.log('Mode:', d.mode);
}

// ── 测试信号 ──
async function startTest(signal) {
    const r = await fetch('/test_signal', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({signal:signal})
    });
    const d = await r.json();
    document.getElementById('mode_disp').textContent = d.mode;
    document.getElementById('btn-circle').classList.toggle('running', signal==='circle');
    document.getElementById('btn-nod').classList.toggle('running', signal==='nod');
    console.log('Test signal:', signal);
}

async function stopTest() {
    const r = await fetch('/test_signal', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({signal:'stop'})
    });
    const d = await r.json();
    document.getElementById('mode_disp').textContent = d.mode;
    document.getElementById('btn-circle').classList.remove('running');
    document.getElementById('btn-nod').classList.remove('running');
    console.log('Test stopped');
}

// ── 复位 ──
async function resetGimbal() {
    await fetch('/gimbal_reset', {method:'POST'});
    document.getElementById('mode_disp').textContent = 'idle';
    document.getElementById('btn-track').classList.remove('active');
    document.getElementById('btn-idle').classList.add('active');
    document.getElementById('btn-circle').classList.remove('running');
    document.getElementById('btn-nod').classList.remove('running');
    console.log('Gimbal reset');
}

setInterval(updateData,200);
setInterval(updateTelemetry,200);
updateData();
updateTelemetry();
</script>
</body>
</html>
'''


# ══════════════════════════════════════════════════════════════════
# 启动入口
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="QGimbal-Vision — 云台视觉追踪 + Flask 调参")
    p.add_argument('--camera', type=int, default=0, help='摄像头索引（默认 0）')
    p.add_argument('--port', type=int, default=5000, help='Flask 端口（默认 5000）')
    p.add_argument('--no-flip', action='store_true', help='不翻转画面')
    p.add_argument('--serial-port', type=str, default="/dev/ttyS2", help='串口端口号（默认 Orange Pi 5 Max UART2: /dev/ttyS2）')
    p.add_argument('--serial-baud', type=int, default=115200, help='串口波特率（默认 115200）')
    return p.parse_args()


def main():
    global tracker, gimbal_serial

    args = parse_args()

    # ── 初始化 PID 追踪器 ──
    with params_lock:
        p = dict(params)

    with tracker_lock:
        tracker = GimbalTracker(_build_control_config(p))

    # ── 初始化串口 ──
    if args.serial_port:
        try:
            gimbal_serial = GimbalSerial(port=args.serial_port, baudrate=args.serial_baud)
            print(f"  串口已连接: {args.serial_port} @ {args.serial_baud}")
            # 上电立即发送安全指令: 速度归零, 禁用电机 (防毛刺误触发)
            time.sleep(0.05)
            gimbal_serial.send(CommandPacket(
                yaw_speed=0.0, pitch_speed=0.0,
                enabled=0, stability_enabled=0,
            ))
            gimbal_serial.drain()
            print(f"  已发送安全指令 (disable + zero speed)")
        except Exception as e:
            print(f"  ⚠ 串口打开失败: {e}")
            gimbal_serial = None
    else:
        gimbal_serial = None

    # ── 启动遥测读取线程 ──
    telemetry_thread = threading.Thread(target=_telemetry_reader, daemon=True)
    telemetry_thread.start()

    # ── 信号处理 ──
    def shutdown(_sig, _frame):
        print("\nShutting down...")
        stop_event.set()
        test_stop.set()
        if gimbal_serial:
            with tracker_lock:
                try:
                    gimbal_serial.send(CommandPacket(enabled=0))
                    gimbal_serial.close()
                except Exception:
                    pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── 后台采集线程 ──
    t = threading.Thread(
        target=capture_loop,
        args=(args.camera, not args.no_flip),
        daemon=True,
    )
    t.start()
    time.sleep(1)  # 等待摄像头初始化

    print("=" * 60)
    print("  QGimbal-Vision 启动成功")
    print(f"  摄像头:       {args.camera}")
    print(f"  串口:         {args.serial_port or '(未连接)'}")
    print(f"  控制面板:     http://0.0.0.0:{args.port}")
    print(f"  主视频流:     http://0.0.0.0:{args.port}/video_feed")
    print(f"  调试视频流:   http://0.0.0.0:{args.port}/debug_feed")
    print(f"  数据接口:     http://0.0.0.0:{args.port}/data")
    print(f"  遥测接口:     http://0.0.0.0:{args.port}/telemetry")
    print("=" * 60)

    app.run(host='0.0.0.0', port=args.port, threaded=True)


if __name__ == '__main__':
    main()
