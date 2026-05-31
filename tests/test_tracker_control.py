from control.config import ControlConfig
from control.tracker_control import GimbalTracker


def test_tracker_outputs_zero_when_disabled() -> None:
    cfg = ControlConfig(enabled=False)
    t = GimbalTracker(cfg)
    out = t.update(640, 480, (320.0, 240.0), dt=0.01)
    assert out.yaw_rpm == 0.0
    assert out.pitch_rpm == 0.0


def test_tracker_deadband() -> None:
    cfg = ControlConfig(deadband_px=10.0)
    t = GimbalTracker(cfg)
    out = t.update(640, 480, (325.0, 245.0), dt=0.02)
    assert out.err_x_px == 0.0
    assert out.err_y_px == 0.0

