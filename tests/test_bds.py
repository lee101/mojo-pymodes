"""Parity with pyModeS.decoder.bds: every register, every branch.

The corpus sweeps each decoder's branch space (all typecodes, all four
BDS 0,9 subtypes, all 128 movement codes, all eight category codes, both
BDS 0,5 altitude encodings, every BDS 6,1 subtype and every threat
indicator), so a divergence in any of them shows up as a dict inequality.
"""

from __future__ import annotations

import random

import pytest
from pyModeS.decoder.bds.bds05 import decode_bds05 as ref_bds05
from pyModeS.decoder.bds.bds06 import decode_bds06 as ref_bds06
from pyModeS.decoder.bds.bds08 import decode_bds08 as ref_bds08
from pyModeS.decoder.bds.bds09 import decode_bds09 as ref_bds09
from pyModeS.decoder.bds.bds30 import decode_bds30 as ref_bds30
from pyModeS.decoder.bds.bds30 import is_bds30 as ref_is_bds30
from pyModeS.decoder.bds.bds61 import decode_bds61 as ref_bds61
from pyModeS.decoder.bds.bds62 import decode_bds62 as ref_bds62
from pyModeS.decoder.bds.bds65 import decode_bds65 as ref_bds65

import mojopymodes as pm
from corpus import long_corpus

PAYLOADS = [int(m[8:22], 16) for m in long_corpus()]
PAYLOAD_MASK = (1 << 56) - 1
# The BDS decoders take the 56-bit payload, so keep random draws inside it.
PAYLOADS += [random.Random(i).getrandbits(56) for i in range(600)]


def _tcs(register: int) -> list[int]:
    if register == 5:
        return [9, 10, 18, 20, 21, 22, 4, 23]
    return [0]


def test_bds05():
    for payload in PAYLOADS:
        for tc in _tcs(5):
            got = pm.decoder.bds.decode_bds05(payload, tc=tc)
            assert got == ref_bds05(payload, tc=tc), (payload, tc)


def test_bds06():
    for payload in PAYLOADS:
        assert pm.decoder.bds.decode_bds06(payload) == ref_bds06(payload)


def test_bds08():
    for payload in PAYLOADS:
        assert pm.decoder.bds.decode_bds08(payload) == ref_bds08(payload)


def test_bds09():
    for payload in PAYLOADS:
        assert pm.decoder.bds.decode_bds09(payload) == ref_bds09(payload)


def test_bds30():
    for payload in PAYLOADS:
        assert pm.decoder.bds.decode_bds30(payload) == ref_bds30(payload)


def test_bds61():
    for payload in PAYLOADS:
        assert pm.decoder.bds.decode_bds61(payload) == ref_bds61(payload)


def test_bds62():
    for payload in PAYLOADS:
        assert pm.decoder.bds.decode_bds62(payload) == ref_bds62(payload)


def test_bds65():
    for payload in PAYLOADS:
        assert pm.decoder.bds.decode_bds65(payload) == ref_bds65(payload)


def test_is_bds30():
    for payload in PAYLOADS:
        assert pm.decoder.bds.is_bds30(payload) == ref_is_bds30(payload)


@pytest.mark.parametrize(
    "name,bds,extra",
    [
        ("decode_bds05", "0,5", "tc"),
        ("decode_bds06", "0,6", None),
        ("decode_bds08", "0,8", None),
        ("decode_bds09", "0,9", None),
        ("decode_bds61", "6,1", None),
        ("decode_bds62", "6,2", None),
        ("decode_bds65", "6,5", None),
    ],
)
def test_batch_matches_scalar(name, bds, extra):
    scalar = getattr(pm.decoder.bds, name)
    batch = getattr(pm.decoder.bds, name + "_batch")
    if extra == "tc":
        tcs = [9] * len(PAYLOADS)
        assert batch(PAYLOADS, tcs) == [scalar(p, tc=9) for p in PAYLOADS]
    else:
        assert batch(PAYLOADS) == [scalar(p) for p in PAYLOADS]


def test_bds30_batch_matches_scalar():
    assert pm.decoder.bds.decode_bds30_batch(PAYLOADS) == [
        pm.decoder.bds.decode_bds30(p) for p in PAYLOADS
    ]


def test_batch_across_the_parallel_threshold():
    """The serial and threaded paths must produce identical rows."""
    from mojopymodes._lib import PAR_THRESHOLD

    payload = [
        (19 << 51) | (2 << 48) | ((i * 2654435761) % 1024) << 32 | (i % 1024) << 21
        for i in range(PAR_THRESHOLD + 64)
    ]
    small = payload[: PAR_THRESHOLD - 1]
    large = payload[: PAR_THRESHOLD + 1]
    assert pm.decoder.bds.decode_bds09_batch(small) == [
        pm.decoder.bds.decode_bds09(p) for p in small
    ]
    assert pm.decoder.bds.decode_bds09_batch(large) == [
        pm.decoder.bds.decode_bds09(p) for p in large
    ]


def test_batch_empty():
    for name in (
        "decode_bds06_batch",
        "decode_bds08_batch",
        "decode_bds09_batch",
        "decode_bds30_batch",
        "decode_bds61_batch",
        "decode_bds62_batch",
        "decode_bds65_batch",
    ):
        assert getattr(pm.decoder.bds, name)([]) == []
    assert pm.decoder.bds.decode_bds05_batch([], []) == []


def test_dispatch_covers_every_typecode_upstream_dispatches():
    from mojopymodes.decoder import ADSB_DISPATCH
    from pyModeS.decoder.adsb import _ADSB_DISPATCH

    assert set(ADSB_DISPATCH) == set(_ADSB_DISPATCH)
    for tc, (bds, _) in _ADSB_DISPATCH.items():
        assert ADSB_DISPATCH[tc][0] == bds
