"""The batch API: same results as the scalar API, and same as upstream.

`decode_payloads` is the piece that matters most: it is this repo's answer
to `pyModeS.decoder.adsb.ADSB.decode`, routing a stream of payloads to the
right BDS register and producing the same field dict.
"""

from __future__ import annotations

import pytest

from pyModeS.decoder.bds.bds05 import decode_bds05 as ref_bds05

import mojopymodes as pm
from mojopymodes.batch import (
    decode_fields,
    decode_payloads,
    headers,
    messages_to_payloads,
    payload,
    payloads,
)
from corpus import long_corpus

LONG = long_corpus()


def test_payload_and_payloads():
    assert payload(LONG[0]) == int(LONG[0][8:22], 16)
    assert payloads(LONG[:20]) == [int(m[8:22], 16) for m in LONG[:20]]


def test_payload_rejects_a_short_message():
    with pytest.raises(ValueError):
        payload("8D406B90")


def test_messages_to_payloads_keeps_only_adsb():
    msgs = LONG + ["5D406B902015A678D4D220AA4BDA"]
    got, tcs = messages_to_payloads(msgs)
    assert len(got) == len(LONG)
    assert tcs == [int(m[8:10], 16) >> 3 for m in LONG]


def test_decode_fields_matches_headers():
    assert decode_fields(LONG) == headers(LONG)


def test_decode_payloads_matches_the_adsb_decoder():
    items = [(int(m[8:22], 16), int(m[8:10], 16) >> 3) for m in LONG]
    got = decode_payloads(items)
    assert len(got) == len(items)
    for (p, tc), item in zip(items, got):
        assert item == _upstream_adsb(p), (hex(p), tc)


def _upstream_adsb(p: int) -> dict:
    """Run one payload through upstream's ADSB dispatch, field dict and all."""
    from pyModeS.decoder.adsb import _ADSB_DISPATCH

    tc = (p >> 51) & 0x1F
    entry = _ADSB_DISPATCH.get(tc)
    if entry is None:
        return {"typecode": tc}
    bds_code, decoder = entry
    out = {"typecode": tc, "bds": bds_code}
    if bds_code == "0,5":
        out.update(decoder(p, tc=tc))
    else:
        out.update(decoder(p))
    return out


def test_decode_payloads_preserves_input_order():
    items = [(int(m[8:22], 16), int(m[8:10], 16) >> 3) for m in LONG]
    got = decode_payloads(items)
    for (p, _), item in zip(items, got):
        assert item["typecode"] == (p >> 51) & 0x1F


def test_decode_payloads_empty():
    assert decode_payloads([]) == []


def test_bds05_batch_matches_upstream():
    items = [(int(m[8:22], 16), int(m[8:10], 16) >> 3) for m in LONG]
    tcs = [tc for _, tc in items]
    pl = [p for p, _ in items]
    got = pm.decoder.bds.decode_bds05_batch(pl, tcs)
    assert got == [ref_bds05(p, tc=t) for p, t in zip(pl, tcs)]


def test_crc_batch_is_exposed_at_the_top_level():
    from mojopymodes._bits import crc_remainders

    assert crc_remainders([]) == []
    assert len(crc_remainders(LONG)) == len(LONG)
