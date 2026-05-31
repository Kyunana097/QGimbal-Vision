from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PIDConfig:
    """PID parameters.

    Notes:
        - Units of input/output are defined by the caller.
        - Integral is clamped to avoid wind-up.
    """

    kp: float = 0.8
    ki: float = 0.0
    kd: float = 0.08
    integral_limit: float = 0.8
    output_limit: float = 1.0


@dataclass(slots=True)
class ControlConfig:
    """High-level control configuration.

    Conventions:
        - Image coordinates: x right, y down.
        - We define error as (target - image_center). So target to the right => +err_x.
        - RPM sign and axis direction for your gimbal may differ; use invert_*.
    """

    enabled: bool = True

    # Error preprocessing
    deadband_px: float = 6.0
    lost_timeout_s: float = 0.25

    # Map PID output (normalized) -> RPM
    max_rpm_yaw: float = 120.0
    max_rpm_pitch: float = 120.0

    # If you don't know your gimbal's axis sign yet, keep these False and adjust later.
    invert_yaw: bool = True
    invert_pitch: bool = False

    # PID per axis. Input is normalized error (err_px / (w/2 or h/2)).
    yaw_pid: PIDConfig = field(
        default_factory=lambda: PIDConfig(kp=4.0, ki=0.80, kd=0.08, integral_limit=0.2, output_limit=1.0)
    )
    pitch_pid: PIDConfig = field(
        default_factory=lambda: PIDConfig(kp=3.0, ki=0.6, kd=0.06, integral_limit=0.2, output_limit=1.0)
    )
