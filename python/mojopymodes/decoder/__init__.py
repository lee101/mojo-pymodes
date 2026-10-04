"""Decoders for the BDS register families, keyed by ADS-B type code.

The dispatch table mirrors pyModeS.decoder.adsb._ADSB_RANGES so that a
payload can be routed to the right decoder with the same rules upstream uses.
"""

from .bds import (
    decode_bds05,
    decode_bds05_batch,
    decode_bds06,
    decode_bds06_batch,
    decode_bds08,
    decode_bds08_batch,
    decode_bds09,
    decode_bds09_batch,
    decode_bds30,
    decode_bds30_batch,
    decode_bds61,
    decode_bds61_batch,
    decode_bds62,
    decode_bds62_batch,
    decode_bds65,
    decode_bds65_batch,
)

_ADSB_RANGES = (
    (range(1, 5), "0,8", decode_bds08),
    (range(5, 9), "0,6", decode_bds06),
    (range(9, 19), "0,5", decode_bds05),
    ({19}, "0,9", decode_bds09),
    (range(20, 23), "0,5", decode_bds05),
    ({28}, "6,1", decode_bds61),
    ({29}, "6,2", decode_bds62),
    ({31}, "6,5", decode_bds65),
)

ADSB_DISPATCH = {
    tc: (bds, fn) for tc_set, bds, fn in _ADSB_RANGES for tc in tc_set
}

__all__ = [
    "ADSB_DISPATCH",
    "decode_bds05",
    "decode_bds05_batch",
    "decode_bds06",
    "decode_bds06_batch",
    "decode_bds08",
    "decode_bds08_batch",
    "decode_bds09",
    "decode_bds09_batch",
    "decode_bds30",
    "decode_bds30_batch",
    "decode_bds61",
    "decode_bds61_batch",
    "decode_bds62",
    "decode_bds62_batch",
    "decode_bds65",
    "decode_bds65_batch",
]
