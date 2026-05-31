from __future__ import annotations

from dataclasses import dataclass


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass(slots=True)
class PID:
    """A small PID controller.

    This PID is meant for real-time control where dt varies. It includes:
      - integral clamping (anti-windup)
      - derivative on measurement via error derivative (simple form)

    Inputs/outputs are unitless; choose a consistent scale in your application.
    """

    kp: float
    ki: float
    kd: float
    integral_limit: float
    output_limit: float

    _integral: float = 0.0
    _prev_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def update(self, error: float, dt: float) -> float:
        """Step the controller and return the control output.

        Args:
            error: setpoint - measurement (or any defined error)
            dt: seconds since last update (must be > 0)
        """
        if dt <= 0.0:
            return 0.0

        # P
        p = self.kp * error

        # I
        self._integral += error * dt
        self._integral = _clamp(self._integral, -self.integral_limit, self.integral_limit)
        i = self.ki * self._integral

        # D
        if self._prev_error is None:
            d = 0.0
        else:
            de = (error - self._prev_error) / dt
            d = self.kd * de
        self._prev_error = error

        out = p + i + d
        return _clamp(out, -self.output_limit, self.output_limit)

