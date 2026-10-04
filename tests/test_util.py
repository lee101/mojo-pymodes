"""Parity with pyModeS.util over long (DF17/18) and short messages.

The short messages matter as much as the long ones: they are the only way
to reach the `icao` branch that recovers the address from the CRC remainder,
and the only way to reach `altcode` and `idcode` at all.
"""

from __future__ import annotations

import pytest
from pyModeS.util import altcode as ref_altcode
from pyModeS.util import bin2hex as ref_bin2hex
from pyModeS.util import bin2int as ref_bin2int
from pyModeS.util import crc as ref_crc
from pyModeS.util import df as ref_df
from pyModeS.util import hex2bin as ref_hex2bin
from pyModeS.util import hex2int as ref_hex2int
from pyModeS.util import icao as ref_icao
from pyModeS.util import idcode as ref_idcode
from pyModeS.util import typecode as ref_typecode

import mojopymodes as pm
from mojopymodes.util import (
    altcode,
    bin2hex,
    bin2int,
    crc,
    df,
    hex2bin,
    hex2int,
    icao,
    idcode,
    typecode,
)
from corpus import all_messages

MESSAGES = all_messages()
LONG = [m for m in MESSAGES if len(m) == 28]
SHORT = [m for m in MESSAGES if len(m) == 14]


@pytest.mark.parametrize("msg", MESSAGES)
def test_df(msg):
    assert df(msg) == ref_df(msg)


@pytest.mark.parametrize("msg", MESSAGES)
def test_icao(msg):
    assert icao(msg) == ref_icao(msg)


@pytest.mark.parametrize("msg", MESSAGES)
def test_typecode(msg):
    assert typecode(msg) == ref_typecode(msg)


@pytest.mark.parametrize("msg", MESSAGES)
def test_crc(msg):
    assert crc(msg) == ref_crc(msg)


@pytest.mark.parametrize("msg", SHORT)
def test_altcode(msg):
    assert altcode(msg) == ref_altcode(msg)


@pytest.mark.parametrize("msg", SHORT)
def test_idcode(msg):
    assert idcode(msg) == ref_idcode(msg)


def test_altcode_is_none_on_long_messages():
    for msg in LONG[:50]:
        assert altcode(msg) is None == ref_altcode(msg)


def test_idcode_is_none_on_long_messages():
    for msg in LONG[:50]:
        assert idcode(msg) is None == ref_idcode(msg)


def test_icao_is_none_for_unaddressed_downlink_formats():
    """DF1/2/3/6/22-24 carry no address, so upstream returns None."""
    for df in (1, 2, 3, 6, 22, 23):
        msg = corpus_short(df)
        assert icao(msg) == ref_icao(msg) is None


def corpus_short(df: int) -> str:
    from corpus import short_message

    return short_message(df, 0x406B90, 0x1FFF)


@pytest.mark.parametrize("s", ["0", "00", "8D", "F", "8D406B902015A678D4D220AA4BDA", "0F", "1A2b"])
def test_hex2bin(s):
    assert hex2bin(s) == ref_hex2bin(s)


@pytest.mark.parametrize("s", ["1010", "0", "111"])
def test_bin2int(s):
    assert bin2int(s) == ref_bin2int(s)


@pytest.mark.parametrize("s", ["8D40", "0", "abc", "8D406B902015A678D4D220AA4BDA"])
def test_hex2int(s):
    assert hex2int(s) == ref_hex2int(s)


@pytest.mark.parametrize("s", ["1010", "1", "111", "1010101"])
def test_bin2hex(s):
    assert bin2hex(s) == ref_bin2hex(s)


@pytest.mark.parametrize(
    "fn,ref_fn", [(hex2bin, ref_hex2bin), (bin2int, ref_bin2int),
                 (hex2int, ref_hex2int), (bin2hex, ref_bin2hex)]
)
def test_empty_string_raises_the_same_way(fn, ref_fn):
    """Upstream raises ValueError on ''; so must this, not return something."""
    with pytest.raises(ValueError):
        ref_fn("")
    with pytest.raises(ValueError):
        fn("")


def test_headers_batch_matches_the_scalar_helpers():
    batch = pm.util.headers(MESSAGES)
    assert [h["df"] for h in batch] == [df(m) for m in MESSAGES]
    assert [h["icao"] for h in batch] == [icao(m) for m in MESSAGES]
    assert [h["crc"] for h in batch] == [crc(m) for m in MESSAGES]


def test_headers_batch_empty():
    assert pm.util.headers([]) == []
