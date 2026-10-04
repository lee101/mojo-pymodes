"""BDS 0,6 -- ADS-B surface position (TC 5-8).

No altitude (the aircraft is on the ground). Decodes movement (packed speed),
track with its validity flag, and the raw CPR lat/lon fields.
"""

from __future__ import annotations

from ._base import _decode, _one


def decode_bds06(payload: int) -> dict:
    """Decode a BDS 0,6 payload (ADS-B surface position)."""
    return _one("bds06", payload)


def decode_bds06_batch(payloads) -> list[dict]:
    """`decode_bds06` over many payloads, one kernel call."""
    return _decode("bds06", payloads)
