"""Message corpus for the parity tests.

Two published Mode-S messages plus a generated one for every branch the
decoders can take. The generated messages are real DF17/DF18 extended
squitters: the 88 data bits are filled from the BDS layouts in the ICAO
DO-260B tables and the 24 parity bits are the CRC that makes
`crc_remainder` return 0, so `pyModeS.decode` accepts them exactly as it
accepts a captured frame.

Deterministic: every draw comes from a fixed seed, so a failure is
reproducible.
"""

from __future__ import annotations

import random

from pyModeS._bits import crc_remainder

# Published in the pyModeS docstrings and on mode-s.org.
PUBLISHED = [
    "8D406B902015A678D4D220AA4BDA",
    "8D40058B58C901375147EFD09357",
]

DF_LONG = (17, 18)
DF_ADDRESSED = (0, 4, 5, 11, 16, 20, 21)


def long_message(df: int, icao: int, payload: int) -> str:
    """A 112-bit DF17/18 message with a correct CRC parity field."""
    n = ((df & 0x1F) << 3 | 5) << 80 | (icao & 0xFFFFFF) << 56 | payload
    parity = crc_remainder(n << 24, 112)
    return f"{(n << 24 | parity):028X}"


def _crc24_basis() -> list[int]:
    """`crc_remainder` of each single low bit of a 56-bit message.

    The CRC-24 is GF(2)-linear in the message and the generator's constant
    term is 1, so the map from the 24 low bits to the 24-bit remainder is a
    bijection. Inverting it lets `_solve_low` hit any target remainder.
    """
    return [crc_remainder(1 << k, 56) for k in range(24)]


_BASIS = _crc24_basis()


def _solve_low(target: int) -> int:
    """The 24 low bits whose CRC remainder is `target`."""
    rows = []
    for bit in range(24):
        acc = 0
        for k, col in enumerate(_BASIS):
            acc |= ((col >> bit) & 1) << k
        rows.append(acc)
    rhs = [(target >> bit) & 1 for bit in range(24)]
    for col in range(24):
        pivot = next((r for r in range(col, 24) if (rows[r] >> col) & 1), None)
        if pivot is None:
            raise ValueError("CRC remainder is not reachable")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        for r in range(24):
            if r != col and (rows[r] >> col) & 1:
                rows[r] ^= rows[col]
                rhs[r] ^= rhs[col]
    return sum(rhs[k] << k for k in range(24))


def short_message(df: int, icao: int, field13: int) -> str:
    """A 56-bit message that decodes back to `icao` and to `field13`.

    DF11 carries the address in the AA field outright. For DF0/4/5/16/20/21
    the address is recovered from the CRC remainder, so the 24 low bits are
    solved to make the remainder come out as the address -- which is exactly
    what an interrogated transponder's reply does.
    """
    top = (df & 0x1F) << 51 | 5 << 48 | (icao & 0xFFFFFF) << 24
    if df == 11:
        return f"{(top | 0x5A5A5):014X}"
    low = _solve_low((icao & 0xFFFFFF) ^ crc_remainder(top, 56))
    return f"{(top | low):014X}"


def _payload(tc: int, body: int) -> int:
    return ((tc & 0x1F) << 51) | (body & ((1 << 51) - 1))


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def long_corpus(seed: int = 20260929) -> list[str]:
    """One message per typecode, then the branch sweeps on top."""
    rng = _rng(seed)
    out = list(PUBLISHED)
    icao = 0x406B90

    for tc in range(32):
        for df in DF_LONG:
            out.append(long_message(df, icao, _payload(tc, rng.getrandbits(51))))

    # BDS 0,9 subtypes, with and without the "not available" magnitudes.
    for subtype in range(8):
        for _ in range(4):
            body = (rng.getrandbits(2) << 45) | (rng.getrandbits(3) << 42)
            out.append(long_message(17, icao, _payload(19, body | (subtype << 48))))
    for subtype in range(1, 5):
        for mag in (0, 1, 1023):
            body = (subtype << 48) | (rng.getrandbits(3) << 42) | (mag << 32)
            body |= (rng.getrandbits(1) << 31) | (mag << 21)
            out.append(long_message(17, icao, _payload(19, body)))

    # BDS 0,6 movement: every value the 7-bit field can take, both track states.
    for mov in range(128):
        for status in (0, 1):
            body = (mov << 44) | (status << 43) | (rng.getrandbits(21) << 21)
            out.append(long_message(17, icao, _payload(5, body)))

    # BDS 0,8: every category value, plus callsigns with padding and junk.
    for category in range(8):
        for tc in (1, 2, 3, 4, 31):
            cs = rng.getrandbits(48)
            out.append(long_message(17, icao, _payload(tc, (category << 48) | cs)))
    for text in ("        ", "  EZY85MH", "EZY85MH ", "########", "A1       ", "  1234  "):
        cs = int("".join(f"{ord(c) & 0x3F:06b}" for c in text), 2)
        out.append(long_message(17, icao, _payload(4, (3 << 48) | cs)))

    # BDS 0,5: both altitude encodings, both CPR formats.
    for tc in list(range(9, 23)):
        for ac in (0, 0x001, 0x555, 0xFFF, rng.getrandbits(12)):
            body = (rng.getrandbits(2) << 49) | (rng.getrandbits(1) << 48)
            body |= (ac & 0xFFF) << 36 | (rng.getrandbits(1) << 34)
            body |= rng.getrandbits(17) << 17
            body |= rng.getrandbits(17)
            out.append(long_message(17, icao, _payload(tc, body)))

    # BDS 6,1 subtypes, BDS 6,2, BDS 6,5, and BDS 3,0 threat indicators.
    for subtype in range(8):
        body = (subtype << 48) | rng.getrandbits(48)
        out.append(long_message(17, icao, _payload(28, body)))
    for _ in range(64):
        out.append(long_message(17, icao, _payload(29, rng.getrandbits(51))))
        out.append(long_message(17, icao, _payload(31, rng.getrandbits(51))))
    for tti in range(4):
        ra = (0x30 << 48) | (rng.getrandbits(19) << 29) | (tti << 26)
        ra |= rng.getrandbits(26)
        out.append(long_message(17, icao, _payload(28, (2 << 48) | (ra & ((1 << 48) - 1)))))

    return out


def short_corpus(seed: int = 20260929) -> list[str]:
    """DF0/4/5/11/16/20/21 messages carrying altitude, squawk and address."""
    rng = _rng(seed)
    out = []
    for k in range(120):
        icao = rng.getrandbits(24)
        for df in DF_ADDRESSED:
            out.append(short_message(df, icao, rng.getrandbits(13)))
    return out


def all_messages() -> list[str]:
    return long_corpus() + short_corpus()


def payloads_of(msgs: list[str]) -> list[int]:
    return [int(m[8:22], 16) for m in msgs if len(m) == 28]


def typecodes_of(msgs: list[str]) -> list[int]:
    return [int(m[8:10], 16) >> 3 for m in msgs if len(m) == 28]
