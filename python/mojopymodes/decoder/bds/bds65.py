"""BDS 6,5 -- ADS-B Aircraft Operational Status (TC=31).

Subtype 0 is airborne and subtype 1 is surface, and the field breakdown
depends on the ADS-B version as well. This decoder extracts the reliably
positioned fields and leaves capability_class / operational_mode as raw
16-bit values, as upstream does.
"""

from __future__ import annotations

from ._base import _decode, _one


def decode_bds65(payload: int) -> dict:
    """Decode a BDS 6,5 payload (ADS-B operational status)."""
    return _one("bds65", payload)


def decode_bds65_batch(payloads) -> list[dict]:
    """`decode_bds65` over many payloads, one kernel call."""
    return _decode("bds65", payloads)
