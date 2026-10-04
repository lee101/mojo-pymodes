"""BDS 6,1 -- ADS-B aircraft status (TC=28).

Subtype 1 carries an emergency state and a squawk; subtype 2 embeds an ACAS
RA report in payload bits 8-55, which is decoded as a synthetic BDS 3,0.
"""

from __future__ import annotations

from ._base import _decode, _one


def decode_bds61(payload: int) -> dict:
    """Decode a BDS 6,1 payload (ADS-B aircraft status)."""
    return _one("bds61", payload)


def decode_bds61_batch(payloads) -> list[dict]:
    """`decode_bds61` over many payloads, one kernel call."""
    return _decode("bds61", payloads)
