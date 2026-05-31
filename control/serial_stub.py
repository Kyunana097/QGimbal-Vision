from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional


def _checksum_mod_256(payload: bytes) -> int:
    """Checksum: sum(all bytes) % 256."""
    return sum(payload) & 0xFF


@dataclass(slots=True)
class GimbalSerialStub:
    """Serial sender for STM32 gimbal.

    Packet format (little-endian, packed):
        uint8  laser_enabled
        uint8  enabled
        uint8  stability_enabled
        float  yaw_speed   (rpm)
        float  pitch_speed (rpm)
        uint8  check_sum   (sum of previous bytes % 256)

    Notes:
        - This file is intentionally self-contained. Replace/extend as you like.
        - Floats are IEEE754 32-bit; STM32 is little-endian.
    """

    port: Optional[str] = None
    baudrate: int = 115200

    # Enable fields: 0 disable, 1 enable, other no action
    laser_enabled: int = 2
    enabled: int = 1
    stability_enabled: int = 2

    _ser: object | None = field(default=None, init=False, repr=False)

    def open(self) -> None:
        """Open serial port.

        If `port` is None, this remains a no-op (safe default).
        """
        if self.port is None:
            self._ser = None
            return

        # Import lazily so the project still runs without pyserial installed
        # when you don't use real serial.
        import serial  # type: ignore

        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=0,
            write_timeout=0,
        )

    def close(self) -> None:
        ser = self._ser
        self._ser = None
        if ser is None:
            return
        close = getattr(ser, "close", None)
        if callable(close):
            close()

    def build_packet(self, yaw_rpm: float, pitch_rpm: float) -> bytes:
        """Build a binary packet matching `ReceivePackage` (little-endian)."""
        header = struct.pack(
            "<ffBBB",
            float(yaw_rpm),
            float(pitch_rpm),
            int(self.laser_enabled) & 0xFF,
            int(self.enabled) & 0xFF,
            int(self.stability_enabled) & 0xFF,
        )
        chk = _checksum_mod_256(header)
        pkt = header + struct.pack("<B", chk)
        return pkt

    def send_rpm(self, yaw_rpm: float, pitch_rpm: float) -> None:
        """Send RPM command.

        Safe behavior:
            - If serial isn't opened (or port is None), does nothing.
        """
        ser = self._ser
        if ser is None:
            return

        pkt = self.build_packet(yaw_rpm, pitch_rpm)
        write = getattr(ser, "write", None)
        if callable(write):
            write(pkt)
