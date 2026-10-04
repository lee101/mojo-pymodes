"""BDS 0,9 -- ADS-B airborne velocity (TC=19).

Four subtypes share a common header and trailer but differ in the middle
22 bits: 1 and 2 are sub/supersonic ground speed (in knots, from N/S and
E/W velocity components), 3 and 4 are sub/supersonic airspeed (heading plus
IAS/TAS). The trailer carries the vertical rate and the GNSS-minus-baro
altitude difference.
"""

from __future__ import annotations

from ._base import _decode, _one


def decode_bds09(payload: int) -> dict:
    """Decode a BDS 0,9 payload (ADS-B airborne velocity, TC=19)."""
    return _one("bds09", payload)


def decode_bds09_batch(payloads) -> list[dict]:
    """`decode_bds09` over many payloads, one kernel call."""
    return _decode("bds09", payloads)
