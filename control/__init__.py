"""Control modules (PID, gimbal tracking, serial I/O stubs).

This package converts vision detections (target center in image coordinates) into
control outputs (e.g., gimbal yaw/pitch speed in RPM).
"""

from .config import ControlConfig
from .tracker_control import GimbalTracker

__all__ = [
    "ControlConfig",
    "GimbalTracker",
]
