"""Vision modules for this project.

This package contains small, reusable computer vision utilities used by `main.py`.
"""

from .rect_detect import DetectedRect, detect_rectangles, draw_detected_rect

__all__ = [
    "DetectedRect",
    "detect_rectangles",
    "draw_detected_rect",
]
