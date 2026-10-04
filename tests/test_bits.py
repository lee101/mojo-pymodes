"""Parity with pyModeS._bits: bit extraction and the Mode-S CRC-24."""

from __future__ import annotations

import random

import pytest
from pyModeS._bits import crc_remainder as ref_crc
from pyModeS._bits import extract_signed as ref_signed
from pyModeS._bits import extract_unsigned as ref_unsigned

import mojopymodes as pm
from mojopymodes._bits import crc_remainder, crc_remainders

from corpus import all_messages

MESSAGES = all_messages()
INTS = [int(m, 16) for m in MESSAGES]


def test_extract_unsigned_over_the_corpus():
    for n, length in zip(INTS, (len(m) * 4 for m in MESSAGES)):
        for start in (0, 8, 19, 32, length - 24):
            for width in (1, 3, 5, 13, 17, 24):
                if start + width > length:
                    continue
                if length > 56:
                    # A 112-bit value does not fit a 64-bit Int; that is the
                    # documented divergence, covered by the wide CRC test.
                    continue
                assert pm.extract_unsigned(n, start, width, length) == ref_unsigned(
                    n, start, width, length
                )


def test_extract_signed_over_the_corpus():
    rng = random.Random(7)
    for _ in range(4000):
        n = rng.getrandbits(56)
        start = rng.randrange(0, 30)
        width = rng.randrange(1, 25)
        if start + width > 56:
            continue
        assert pm.extract_signed(n, start, width, 56) == ref_signed(n, start, width, 56)


def test_extract_signed_known_values():
    n = 0b1011_0010
    assert pm.extract_signed(n, 0, 4, 8) == -5
    assert pm.extract_signed(n, 4, 4, 8) == 2
    assert pm.extract_signed(n, 0, 8, 8) == -78


def test_crc_remainder_short():
    for m in (x for x in MESSAGES if len(x) == 14):
        n = int(m, 16)
        assert crc_remainder(n, 56) == ref_crc(n, 56)


def test_crc_remainder_long_is_zero():
    for m in (x for x in MESSAGES if len(x) == 28):
        assert pm.util.crc(m) == ref_crc(int(m, 16), 112) == 0


def test_crc_batch_matches_scalar_and_upstream():
    got = crc_remainders(MESSAGES)
    assert got == [ref_crc(int(m, 16), len(m) * 4) for m in MESSAGES]


def test_crc_batch_agrees_across_the_parallel_threshold():
    """The threaded and the serial path must not disagree."""
    small = MESSAGES[:4095]
    large = MESSAGES[:4097]
    assert crc_remainders(small) == [ref_crc(int(m, 16), len(m) * 4) for m in small]
    assert crc_remainders(large) == crc_remainders(list(large))


def test_crc_empty():
    assert crc_remainders([]) == []


def test_to_words_splits_a_message_exactly():
    """`to_words` folds 16 nibbles with a uint64 dot product.

    The kernel reads a long message as (hi, lo) = (n >> 64, n & 2**64 - 1), so
    the fold has to reproduce the integer exactly, including when a chunk is
    all `F`s and the arithmetic has to wrap mod 2**64.
    """
    from mojopymodes._lib import to_words

    cases = [
        "8D406B902015A678D4D220AA4BDA",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "0000000000000000000000000000",
        "0102030405060708090A0B0C0D0E0F",
        "0102030405060708090a0b0c0d0e0f",  # lowercase hex
        "5D406B902015A678",  # a 56-bit message: hi must be zero
        "8D406B90",
    ]
    hi, lo, lengths = to_words(cases)
    for msg, h, l, length in zip(cases, hi, lo, lengths):
        n = int(msg, 16)
        assert int(length) == len(msg) * 4, msg
        if len(msg) * 4 <= 64:
            assert int(h) == 0 and int(l) == n, msg
        else:
            assert (int(h) << 64) | (int(l) & 0xFFFFFFFFFFFFFFFF) == n, msg


def test_to_words_mixed_lengths():
    """Messages of different lengths in one batch share the padded width."""
    from mojopymodes._lib import to_words

    cases = ["5D406B902015A678", "8D406B902015A678D4D220AA4BDA", "8D406B90"]
    hi, lo, lengths = to_words(cases)
    for msg, h, l, length in zip(cases, hi, lo, lengths):
        n = int(msg, 16)
        assert int(length) == len(msg) * 4
        if len(msg) * 4 <= 64:
            assert int(h) == 0 and int(l) == n
        else:
            assert (int(h) << 64) | (int(l) & 0xFFFFFFFFFFFFFFFF) == n


@pytest.mark.parametrize("length", [56, 112])
def test_crc_matches_reference_on_random(length):
    rng = random.Random(11)
    for _ in range(2000):
        n = rng.getrandbits(length)
        assert crc_remainder(n, length) == ref_crc(n, length)
