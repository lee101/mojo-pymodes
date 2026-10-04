"""pyModeS.decoder.bds._helpers — shared bit-math helpers."""

from __future__ import annotations

from ._lib import lib


def signed(value: int, width: int, sign: int) -> int:
    """Combine an unsigned magnitude and a separate sign bit into a signed int."""
    return lib().mp_signed(value, width, sign)


def normalise_angle(deg: float) -> float:
    """Wrap an angle into the half-open interval [0, 360)."""
    return lib().mp_normalise_angle(deg)


def wrong_status(
    payload: int, status_bit: int, value_start: int, value_width: int
) -> bool:
    """True if a status-gated value field is inconsistent."""
    return bool(lib().mp_wrong_status(payload, status_bit, value_start, value_width))
