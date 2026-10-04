"""BDS 6,2 -- ADS-B Target State and Status (TC=29).

One format is defined (DO-260B section 2.2.3.2.7.1). The mode status bit at
payload bit 46 gates the five autopilot/navigation flags; TCAS operational
(bit 52) is always valid.
"""

from __future__ import annotations

from ._base import _decode, _one


def decode_bds62(payload: int) -> dict:
    """Decode a BDS 6,2 payload (ADS-B target state and status)."""
    return _one("bds62", payload)


def decode_bds62_batch(payloads) -> list[dict]:
    """`decode_bds62` over many payloads, one kernel call."""
    return _decode("bds62", payloads)
