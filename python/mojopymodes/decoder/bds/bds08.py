"""BDS 0,8 -- ADS-B aircraft identification and category (TC=1-4).

The wake-vortex table is data rather than computation: the Mojo kernel does
the lookup and returns an index into `_WAKE_VORTEX_NAMES`, which is the
upstream dict in upstream's order.
"""

from __future__ import annotations

from ._base import _decode, _one


def decode_bds08(payload: int) -> dict:
    """Decode a BDS 0,8 payload (ADS-B identification)."""
    return _one("bds08", payload)


def decode_bds08_batch(payloads) -> list[dict]:
    """`decode_bds08` over many payloads, one kernel call."""
    return _decode("bds08", payloads)
