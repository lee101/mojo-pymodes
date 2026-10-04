"""BDS 0,5 -- ADS-B airborne position.

Two typecode ranges:
- TC 9-18: barometric altitude (from the 12-bit altcode)
- TC 20-22: GNSS altitude (12-bit integer meters converted to feet)

The 12-bit AC field is turned back into a 13-bit altcode with a zero M bit
before it goes to `altcode_to_altitude`, exactly as upstream does.
"""

from __future__ import annotations

from ._base import _one, _run


def decode_bds05(payload: int, *, tc: int) -> dict:
    """Decode a BDS 0,5 payload (ADS-B airborne position)."""
    return _one("bds05", payload, tc)


def decode_bds05_batch(payloads, tcs) -> list[dict]:
    """`decode_bds05` over many payloads, one kernel call."""
    return _run("bds05", payloads, tcs)
