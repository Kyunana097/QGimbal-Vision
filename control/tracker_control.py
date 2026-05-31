from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

from .config import ControlConfig, PIDConfig
from .pid import PID


def _apply_deadband(err_px: float, deadband_px: float) -> float:
    if abs(err_px) <= deadband_px:
        return 0.0
    return err_px


@dataclass(slots=True)
class ControlOutput:
    yaw_rpm: float
    pitch_rpm: float
    err_x_px: float
    err_y_px: float


class GimbalTracker:
    """Convert image target center to gimbal yaw/pitch RPM commands."""

    def __init__(self, cfg: ControlConfig) -> None:
        self.cfg = cfg
        self._yaw_pid = _pid_from_cfg(cfg.yaw_pid)
        self._pitch_pid = _pid_from_cfg(cfg.pitch_pid)
        self._last_seen_ts = 0.0

    def reset(self) -> None:
        self._yaw_pid.reset()
        self._pitch_pid.reset()
        self._last_seen_ts = 0.0

    def update(
        self,
        frame_w: int,
        frame_h: int,
        target_center: Tuple[float, float] | None,
        dt: float,
        now: float | None = None,
    ) -> tuple[bool, ControlOutput]:
        """Compute RPM commands.

        Args:
            frame_w/frame_h: image size.
            target_center: (cx, cy) in image pixels, or None when target not found.
            dt: seconds since last call.
            now: pass in `time.time()` from caller to avoid recomputing.
        """
        if now is None:
            now = time.time()

        if not self.cfg.enabled:
            return False,ControlOutput(0.0, 0.0, 0.0, 0.0)

        # Lost target handling
        if target_center is None:
            if self._last_seen_ts > 0 and (now - self._last_seen_ts) <= self.cfg.lost_timeout_s:
                # within grace period: keep trying with zero error (hold still).
                return False,ControlOutput(0.0, 0.0, 0.0, 0.0)
            self.reset()
            return True,ControlOutput(0.0, 0.0, 0.0, 0.0)

        self._last_seen_ts = now

        cx, cy = target_center
        center_x = frame_w * 0.5
        center_y = frame_h * 0.5

        err_x_px = _apply_deadband(cx - center_x, self.cfg.deadband_px)
        err_y_px = _apply_deadband(cy - center_y, self.cfg.deadband_px)

        # Normalize error to [-1, 1] w.r.t. half image size.
        half_w = max(1.0, center_x)
        half_h = max(1.0, center_y)
        err_x = err_x_px / half_w
        err_y = err_y_px / half_h

        yaw_u = self._yaw_pid.update(err_x, dt)
        pitch_u = self._pitch_pid.update(err_y, dt)

        yaw_rpm = yaw_u * self.cfg.max_rpm_yaw
        pitch_rpm = pitch_u * self.cfg.max_rpm_pitch

        if self.cfg.invert_yaw:
            yaw_rpm = -yaw_rpm
        if self.cfg.invert_pitch:
            pitch_rpm = -pitch_rpm

        return True,ControlOutput(yaw_rpm, pitch_rpm, err_x_px, err_y_px)


def _pid_from_cfg(cfg: PIDConfig) -> PID:
    return PID(
        kp=cfg.kp,
        ki=cfg.ki,
        kd=cfg.kd,
        integral_limit=cfg.integral_limit,
        output_limit=cfg.output_limit,
    )

