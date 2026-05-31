from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectedRect:
    """A detected (possibly rotated) rectangle in image coordinates."""

    center: Tuple[float, float]
    box: np.ndarray  # shape: (4, 2), dtype: float32
    area: float


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the angle (in degrees) between 2 vectors.

    If either vector is near-zero, returns 0.0.
    """
    dot = float(v1.dot(v2))
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 * n2 == 0:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos))


def detect_rectangles(
    frame: np.ndarray,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.5,
    angle_tol: float = 25.0,
) -> List[DetectedRect]:
    """
    在输入 BGR 图像中检测矩形（包括旋转矩形）。返回矩形的 box points 和相关信息。
    - min_area_ratio: 与图像面积的最小比率（过小的轮廓会被丢弃）
    - angle_tol: 角度容忍度（判断为矩形时，四个角接近 90 度的容差）
    Returns rectangles sorted by area (descending).
    """
    h, w = frame.shape[:2]
    img_area = h * w
    min_area = img_area * float(min_area_ratio)
    max_area = img_area * float(max_area_ratio)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # 大津法二值化
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 形态学核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    opened = cv2.erode(opened, kernel, iterations=1)
    # 边缘检测
    edges = cv2.Canny(opened, 25, 75)
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects: List[DetectedRect] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        # 尝试多边形逼近
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype(np.float32)
            # 验证角度接近直角
            angles = []
            for i in range(4):
                p0 = pts[i]
                p1 = pts[(i + 1) % 4]
                p2 = pts[(i + 2) % 4]
                angles.append(angle_between(p0 - p1, p2 - p1))
            # 确保每个角都在容差范围内或矩形足够规则
            if all(abs(a - 90) < angle_tol for a in angles):
                # 计算最短边与最长边比率，确保不是过于扭曲的矩形
                dists = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)]
                min_dist = min(dists)
                max_dist = max(dists)
                if min_dist <= 0:
                    continue
                if max_dist / min_dist > 3:
                    continue
                # 计算中心点
                center_arr = np.mean(pts, axis=0)
                rects.append(
                    DetectedRect(
                        center=(float(center_arr[0]), float(center_arr[1])),
                        box=pts,
                        area=area,
                    )
                )
    # 按面积降序返回（优先较大的矩形）
    rects.sort(key=lambda r: r.area, reverse=True)
    return rects


def draw_detected_rect(
    frame: np.ndarray,
    rect: Optional[DetectedRect],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    draw_center: bool = True,
    draw_text: bool = True,
) -> None:
    """Draw a detected rectangle in-place on a BGR frame."""
    if rect is None:
        return

    box = rect.box.astype(np.int32)
    cv2.polylines(frame, [box], isClosed=True, color=color, thickness=thickness)

    cx, cy = rect.center

    if draw_text:
        label = f"A:{int(rect.area)}"
        label_coord = f"X:{int(cx)} Y:{int(cy)}"
        cv2.putText(
            frame,
            label,
            (int(cx) - 80, int(cy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        cv2.putText(
            frame,
            label_coord,
            (int(cx) - 80, int(cy) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    if draw_center:
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
