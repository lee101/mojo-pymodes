"""BDS 3,0 -- ACAS Active Resolution Advisory.

Used directly by Comm-B and, through BDS 6,1 subtype 2, by ADS-B TC=28.
TTI=0 means no threat identity, TTI=1 carries a 24-bit ICAO address and TTI=2
carries a 13-bit altitude code plus range and bearing.
"""

from __future__ import annotations

from ..._lib import lib
from ._base import _decode, _one


def is_bds30(payload: int) -> bool:
    """True if `payload` is a plausible BDS 3,0 ACAS RA report."""
    return bool(lib().mp_is_bds30(payload))


def decode_bds30(payload: int) -> dict:
    """Decode a BDS 3,0 ACAS Active RA report."""
    return _one("bds30", payload)


def decode_bds30_batch(payloads) -> list[dict]:
    """`decode_bds30` over many payloads, one kernel call."""
    return _decode("bds30", payloads)
