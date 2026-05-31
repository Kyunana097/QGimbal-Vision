from control.pid import PID


def test_pid_zero_error() -> None:
    pid = PID(kp=1.0, ki=0.5, kd=0.1, integral_limit=1.0, output_limit=10.0)
    out = pid.update(0.0, 0.01)
    assert out == 0.0


def test_pid_clamps_output_and_integral() -> None:
    pid = PID(kp=0.0, ki=10.0, kd=0.0, integral_limit=0.2, output_limit=0.5)
    # big constant error should push integral to limit
    for _ in range(100):
        pid.update(1.0, 0.01)
    assert abs(pid.update(1.0, 0.01)) <= 0.5

