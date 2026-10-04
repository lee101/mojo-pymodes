"""Parity with pyModeS._altcode, _callsign and _idcode."""

from __future__ import annotations

import random

import pytest
from pyModeS._altcode import altcode_to_altitude as ref_altcode
from pyModeS._callsign import decode_callsign as ref_callsign
from pyModeS._callsign import is_valid_callsign_char as ref_valid_char
from pyModeS._idcode import idcode_to_squawk as ref_squawk

from mojopymodes import altcode_to_altitude, decode_callsign, idcode_to_squawk
from mojopymodes._altcode import altcodes_to_altitudes
from mojopymodes._callsign import decode_callsigns, is_valid_callsign_char
from mojopymodes._idcode import idcodes_to_squawks

ALL_ALTCODES = list(range(8192))
ALL_IDCODES = list(range(8192))


def test_altcode_every_13_bit_value():
    got = [altcode_to_altitude(ac) for ac in ALL_ALTCODES]
    assert got == [ref_altcode(ac) for ac in ALL_ALTCODES]


def test_altcode_batch_every_13_bit_value():
    assert altcodes_to_altitudes(ALL_ALTCODES) == [
        ref_altcode(ac) for ac in ALL_ALTCODES
    ]


def test_altcode_known_values():
    assert altcode_to_altitude(0) is None
    # 25-foot linear encoding, N = 2000 -> 49000 ft. The AC field is
    # C1 A1 C2 A2 C4 A4 M B1 Q B2 D2 B4 D4, so N occupies bits 12..1 with
    # M clear and Q set.
    assert altcode_to_altitude((62 << 7) | (1 << 5) | (1 << 4)) == 49000
    # M = 1 is the reserved meter-based encoding
    assert altcode_to_altitude(0xFFF) is None
    # Q = 0 with M = 1 is equally reserved.
    assert altcode_to_altitude(1 << 6) is None


def test_altcode_batch_empty():
    assert altcodes_to_altitudes([]) == []


def test_idcode_every_13_bit_value():
    got = [idcode_to_squawk(c) for c in ALL_IDCODES]
    assert got == [ref_squawk(c) for c in ALL_IDCODES]


def test_idcode_batch_every_13_bit_value():
    assert idcodes_to_squawks(ALL_IDCODES) == [ref_squawk(c) for c in ALL_IDCODES]


def test_idcode_known_values():
    assert idcode_to_squawk(0x0808) == "1200"


def test_callsign_random():
    rng = random.Random(3)
    values = [rng.getrandbits(48) for _ in range(5000)]
    assert [decode_callsign(v) for v in values] == [ref_callsign(v) for v in values]


def test_callsign_batch_matches_scalar():
    rng = random.Random(4)
    values = [rng.getrandbits(48) for _ in range(5000)]
    assert decode_callsigns(values) == [ref_callsign(v) for v in values]


def test_callsign_padding_and_strip():
    def pack(text: str) -> int:
        return int("".join(f"{ord(c) & 0x3F:06b}" for c in text), 2)

    for text, expected in (
        ("EZY85MH ", "EZY85MH"),
        ("  KLM12 ", "KLM12"),
        ("        ", ""),
        ("########", "########"),
        ("A B C D ", "A B C D"),
    ):
        assert decode_callsign(pack(text)) == expected == ref_callsign(pack(text))


@pytest.mark.parametrize("idx", list(range(64)))
def test_is_valid_callsign_char(idx):
    assert is_valid_callsign_char(idx) == ref_valid_char(idx)


def test_callsign_batch_empty():
    assert decode_callsigns([]) == []
    assert idcodes_to_squawks([]) == []
