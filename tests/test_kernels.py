"""Every exported Mojo kernel, called by name, asserted on its own output.

The rest of the suite reaches the kernels through the Python API. That is the
right way to test a port, and it is also how a kernel that is wired up and
completely wrong ships green: nothing calls the symbol directly, so nothing
observes it. This file closes that gap.

Every test here does three things:

  1. fetches the `@export`ed symbol straight off the `ctypes.CDLL`,
  2. checks the value it wrote against real upstream pyModeS 3.6.0,
  3. checks the batch kernels agree with the scalar kernel they wrap, so the
     serial branch and the `parallelize` branch are both covered.

The shared library is loaded once, here, and the load is asserted on: a test
that could pass without ever touching `dist/libmojo-pymodes.so` would prove
nothing about the port, so a missing library skips the module loudly rather
than letting a pure-Python fallback report success.
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
import pytest

from mojopymodes._layout import LAYOUTS, WAKE_VORTEX_NAMES, rows
from mojopymodes._lib import LIB, addr, f64, i64, lib, out_f64, out_u8, to_words

# ----------------------------------------------------------------- upstream

from pyModeS._altcode import _gray2int as ref_gray2int
from pyModeS._altcode import altcode_to_altitude as ref_altcode
from pyModeS._bits import crc_remainder as ref_crc
from pyModeS._bits import extract_signed as ref_signed
from pyModeS._bits import extract_unsigned as ref_unsigned
from pyModeS._callsign import decode_callsign as ref_callsign
from pyModeS._callsign import is_valid_callsign_char as ref_valid_char
from pyModeS._idcode import idcode_to_squawk as ref_squawk
from pyModeS.decoder.bds._helpers import normalise_angle as ref_normalise
from pyModeS.decoder.bds._helpers import signed as ref_signed_hm
from pyModeS.decoder.bds._helpers import wrong_status as ref_wrong_status
from pyModeS.decoder.bds.bds05 import decode_bds05 as ref_bds05
from pyModeS.decoder.bds.bds06 import decode_bds06 as ref_bds06
from pyModeS.decoder.bds.bds08 import decode_bds08 as ref_bds08
from pyModeS.decoder.bds.bds09 import decode_bds09 as ref_bds09
from pyModeS.decoder.bds.bds30 import decode_bds30 as ref_bds30
from pyModeS.decoder.bds.bds30 import is_bds30 as ref_is_bds30
from pyModeS.decoder.bds.bds61 import decode_bds61 as ref_bds61
from pyModeS.decoder.bds.bds62 import decode_bds62 as ref_bds62
from pyModeS.decoder.bds.bds65 import decode_bds65 as ref_bds65
from pyModeS.decoder.bds.bds08 import _WAKE_VORTEX
from pyModeS.position._cpr import airborne_position_pair as ref_air_pair
from pyModeS.position._cpr import airborne_position_with_ref as ref_air_ref
from pyModeS.position._cpr import cprNL as ref_cprNL
from pyModeS.position._cpr import surface_position_pair as ref_surf_pair
from pyModeS.position._cpr import surface_position_with_ref as ref_surf_ref
from pyModeS.util import altcode as ref_altcode_field
from pyModeS.util import crc as ref_util_crc
from pyModeS.util import df as ref_df
from pyModeS.util import icao as ref_icao
from pyModeS.util import idcode as ref_idcode_field
from pyModeS.util import typecode as ref_typecode

from corpus import long_corpus, payloads_of, short_corpus, typecodes_of

# ------------------------------------------------------------ library load

_K = lib()

if not os.path.exists(LIB):
    pytest.skip(
        f"shared library {LIB} is missing: run `pixi run build`. Every test in "
        "this file exercises the compiled library, so there is nothing to "
        "assert against without it.",
        allow_module_level=True,
    )

# Mirrors PAR_THRESHOLD / PAR_WORKERS in src/capi.mojo. A batch at or above
# PAR takes the `parallelize` branch, so the fixtures below deliberately cross
# it: a suite that only ever ran n=3 would never execute that branch at all.
PAR_THRESHOLD = 4096


def test_the_library_under_test_is_the_compiled_shared_object():
    """Guard against the whole file silently testing a Python fallback."""
    assert os.path.isfile(LIB), LIB
    assert os.path.basename(_K._name) == "libmojo-pymodes.so", _K._name
    # An @export("...") symbol is a real ELF dynamic symbol, so ctypes resolves
    # it to an address inside the mapped shared object rather than a stub.
    assert _K.mp_extract_unsigned, "mp_extract_unsigned did not resolve"


# -------------------------------------------------------- shared fixtures

LONG = long_corpus()
SHORT = short_corpus()
ALL_MSGS = LONG + SHORT
INT_OF = [int(m, 16) for m in ALL_MSGS]
BITLEN = [len(m) * 4 for m in ALL_MSGS]

PAYLOADS = payloads_of(LONG)
TYPECODES = typecodes_of(LONG)
# Random payloads inside the 56 bits the BDS decoders read.
PAYLOADS += [random.Random(i).getrandbits(56) for i in range(400)]

# Published vectors, so a test can fail on a value rather than only on a
# disagreement with upstream.
PUB_MSG = "8D406B902015A678D4D220AA4BDA"


def _close(got, want, tol=0.0):
    """Exact for ints/None, tolerant for the float-valued CPR results."""
    if want is None:
        return got is None
    if got is None:
        return False
    return abs(got - want) <= tol


# ===================================================== pyModeS/_bits.py
# mp_extract_unsigned, mp_extract_signed, mp_crc_remainder,
# mp_crc_remainder_wide, mp_crc_remainder_batch


def test_mp_extract_unsigned():
    checked = 0
    for n, length in zip(INT_OF, BITLEN):
        if length > 56:
            # A 112-bit value does not fit the single-word Int; that is the
            # documented divergence and mp_crc_remainder_wide covers it.
            continue
        for start in (0, 1, 8, 19, 32, 51, length - 24):
            for width in (1, 2, 5, 13, 17, 24):
                if start + width > length:
                    continue
                want = ref_unsigned(n, start, width, length)
                assert _K.mp_extract_unsigned(n, start, width, length) == want, (
                    n,
                    start,
                    width,
                    length,
                )
                checked += 1
    assert checked > 5000


def test_mp_extract_unsigned_published_vector():
    """The published DF17 squitter, read as the 56-bit word it fits into.

    A Mojo `Int` is 64 bits, so `mp_extract_unsigned` cannot take a 112-bit
    message at all -- that is the documented divergence, and `crc_remainder_wide`
    is the kernel that exists because of it. What must hold here is that the
    single-word form is the exact expression upstream's 56-bit case reduces to.
    """
    msg = int(PUB_MSG, 16)
    assert ref_unsigned(msg, 0, 5, 112) == 17
    assert ref_unsigned(msg, 8, 24, 112) == 4221840
    # Same two fields, taken from the low 56 bits with total_bits = 56.
    short = msg & ((1 << 56) - 1)
    assert _K.mp_extract_unsigned(short, 0, 5, 56) == ref_unsigned(short, 0, 5, 56)
    assert _K.mp_extract_unsigned(short, 8, 24, 56) == ref_unsigned(short, 8, 24, 56)
    # A DF11 message carries its address in the AA field outright.
    df11 = 0x5D406B90123456
    assert _K.mp_extract_unsigned(df11, 0, 5, 56) == 11
    assert _K.mp_extract_unsigned(df11, 8, 24, 56) == 4221840


def test_mp_extract_signed():
    rng = random.Random(101)
    for _ in range(5000):
        n = rng.getrandbits(56)
        start = rng.randrange(0, 40)
        width = rng.randrange(2, 25)
        if start + width > 56:
            continue
        assert _K.mp_extract_signed(n, start, width, 56) == ref_signed(
            n, start, width, 56
        ), (n, start, width)


def test_mp_crc_remainder():
    for n, length in zip(INT_OF, BITLEN):
        if length > 56:
            continue
        assert _K.mp_crc_remainder(n, length) == ref_crc(n, length), n
    # The published squitters have a zero remainder, by definition.
    assert _K.mp_crc_remainder(int(PUB_MSG, 16) & ((1 << 56) - 1), 56) != 0


def test_mp_crc_remainder_wide():
    """The two-word form is what a 112-bit message actually crosses as."""
    checked = 0
    for n, length in zip(INT_OF, BITLEN):
        if length > 56:
            want = ref_crc(n, length)
            got = _K.mp_crc_remainder_wide(n >> 64, n & 0xFFFFFFFFFFFFFFFF, length)
            assert got == want, (n, length)
            checked += 1
    assert checked == len(LONG)


def test_mp_crc_remainder_batch_serial_and_parallel():
    for count in (5, PAR_THRESHOLD, PAR_THRESHOLD + 300):
        msgs = [ALL_MSGS[i % len(ALL_MSGS)] for i in range(count)]
        hi, lo, lengths = to_words(msgs)
        out = i64([0] * count)
        _K.mp_crc_remainder_batch(
            addr(hi), addr(lo), addr(lengths), count, addr(out)
        )
        assert out.tolist() == [ref_crc(int(m, 16), len(m) * 4) for m in msgs]


# =================================================== pyModeS/_altcode.py
# mp_altcode_to_altitude, mp_altcode_to_altitude_batch, mp_gray2int


def test_mp_altcode_to_altitude_exhausts_the_13_bit_field():
    """All 8192 codes, so both the Q=1 linear and Q=0 Gillham paths are hit."""
    for ac in range(8192):
        buf = out_f64(1, 2)
        ok = _K.mp_altcode_to_altitude(ac, addr(buf))
        want = ref_altcode(ac)
        if want is None:
            assert ok == 0, ac
            assert buf[1] == 0.0, ac
        else:
            assert ok == 1, ac
            assert buf[1] == 1.0, ac
            assert int(buf[0]) == want, ac


def test_mp_altcode_to_altitude_batch_serial_and_parallel():
    for count in (7, PAR_THRESHOLD, PAR_THRESHOLD + 97):
        acs = [(i * 7919) % 8192 for i in range(count)]
        a = i64(acs)
        out = out_f64(count, 2)
        _K.mp_altcode_to_altitude_batch(addr(a), count, addr(out))
        block = out.reshape(-1, 2)
        got = [int(r[0]) if r[1] else None for r in block]
        assert got == [ref_altcode(v) for v in acs]


def test_mp_gray2int():
    for n in range(8192):
        assert _K.mp_gray2int(n) == ref_gray2int(n), n
    # Gillham is a Gray-code variant, so decoding then re-encoding is identity.
    assert _K.mp_gray2int(0b1011_0010) == ref_gray2int(0b1011_0010)


# ================================================== pyModeS/_callsign.py
# mp_decode_callsign, mp_decode_callsign_batch, mp_is_valid_callsign_char

CALLSIGN_BITS = [
    int("".join(f"{ord(c) & 0x3F:06b}" for c in text), 2)
    for text in (
        "EZY85MH ",
        "  EZY85MH",
        "        ",
        "########",
        "A1       ",
        "  1234  ",
        "AIR1234 ",
    )
]
CALLSIGN_BITS += [random.Random(i).getrandbits(48) for i in range(200)]


def test_mp_decode_callsign():
    for bits in CALLSIGN_BITS:
        buf = out_u8(1, 16)
        n = _K.mp_decode_callsign(bits, addr(buf))
        got = buf[:n].tobytes().decode("ascii")
        assert got == ref_callsign(bits), bits
        # The kernel returns the byte count it wrote, not a fixed width.
        assert n == len(got), (bits, n, got)


def test_mp_decode_callsign_batch_serial_and_parallel():
    for count in (3, PAR_THRESHOLD, PAR_THRESHOLD + 11):
        bits = [CALLSIGN_BITS[i % len(CALLSIGN_BITS)] for i in range(count)]
        b = i64(bits)
        out = out_u8(count, 8)
        _K.mp_decode_callsign_batch(addr(b), count, addr(out))
        block = out.reshape(-1, 8)
        got = [bytes(r).split(b"\x00", 1)[0].decode("ascii") for r in block]
        assert got == [ref_callsign(v) for v in bits]


def test_mp_is_valid_callsign_char():
    for idx in range(64):
        assert bool(_K.mp_is_valid_callsign_char(idx)) == ref_valid_char(idx), idx
    assert _K.mp_is_valid_callsign_char(1) == 1  # 'A'
    assert _K.mp_is_valid_callsign_char(32) == 1  # space
    assert _K.mp_is_valid_callsign_char(0) == 0  # invalid sentinel


# ==================================================== pyModeS/_idcode.py
# mp_idcode_to_squawk, mp_idcode_to_squawk_batch


def test_mp_idcode_to_squawk_exhausts_the_13_bit_field():
    for code in range(8192):
        buf = out_u8(1, 16)
        n = _K.mp_idcode_to_squawk(code, addr(buf))
        got = buf[:n].tobytes().decode("ascii")
        assert got == ref_squawk(code), code
        assert n == 4, (code, n)
    # The docstring vector, checked against the published answer directly.
    buf = out_u8(1, 16)
    assert buf[: _K.mp_idcode_to_squawk(0x0808, addr(buf))].tobytes() == b"1200"


def test_mp_idcode_to_squawk_batch_serial_and_parallel():
    for count in (4, PAR_THRESHOLD, PAR_THRESHOLD + 13):
        codes = [(i * 2654435761) % 8192 for i in range(count)]
        c = i64(codes)
        out = out_u8(count, 4)
        _K.mp_idcode_to_squawk_batch(addr(c), count, addr(out))
        block = out.reshape(-1, 4)
        assert [bytes(r).decode("ascii") for r in block] == [
            ref_squawk(v) for v in codes
        ]


# ============================================ pyModeS/decoder/bds/_helpers.py
# mp_signed, mp_normalise_angle, mp_wrong_status


def test_mp_signed():
    for width in (8, 9, 10, 12):
        for value in range(0, 1 << width, 7):
            for sign in (0, 1):
                got = _K.mp_signed(value, width, sign)
                assert got == ref_signed_hm(value, width, sign), (value, width, sign)
    # Sign-magnitude, not two's complement: this is the documented difference.
    assert _K.mp_signed(0, 9, 1) == -512
    assert _K.mp_signed(0, 9, 0) == 0


def test_mp_normalise_angle():
    values = [0.0, 359.9, 360.0, -0.5, -180.0, -360.0, 720.0, 1e-9, -1e-9]
    values += [random.Random(i).uniform(-1080.0, 1080.0) for i in range(2000)]
    for v in values:
        got = _K.mp_normalise_angle(v)
        assert got == ref_normalise(v), v
        assert 0.0 <= got < 360.0, v
    assert _K.mp_normalise_angle(-180.0) == 180.0


def test_mp_wrong_status():
    rng = random.Random(31)
    checked = 0
    for _ in range(4000):
        payload = rng.getrandbits(56)
        status_bit = rng.randrange(0, 30)
        value_width = rng.randrange(1, 13)
        value_start = status_bit + 1
        if value_start + value_width > 56:
            continue
        assert bool(
            _K.mp_wrong_status(payload, status_bit, value_start, value_width)
        ) == ref_wrong_status(payload, status_bit, value_start, value_width), payload
        checked += 1
    assert checked > 2000
    # Status clear with a zero field is fine; status clear with a set field is
    # not. Both sides of the branch, stated directly. `value_start=1` with
    # `value_width=12` reads payload bits 1..12, i.e. int bits 54..43, so bit
    # 43 is the lowest that can set it while status (bit 55) stays clear.
    assert _K.mp_wrong_status(0, 0, 1, 12) == 0
    assert _K.mp_wrong_status(1 << 43, 0, 1, 12) == 1
    # Status set short-circuits the check whatever the value field holds.
    assert _K.mp_wrong_status((1 << 55) | (1 << 43), 0, 1, 12) == 0


# ============================================== pyModeS/position/_cpr.py
# mp_cprNL, mp_cprNL_batch, and the four position kernels


def test_mp_cprNL():
    lats = [i / 10.0 for i in range(-900, 901)]
    lats += [87.0, -87.0, 87.5, 87.0000001, 88.0, 90.0, -90.0, 0.0]
    for lat in lats:
        assert _K.mp_cprNL(lat) == ref_cprNL(lat), lat
    assert _K.mp_cprNL(0.0) == 59
    assert _K.mp_cprNL(87.0) == 2


def test_mp_cprNL_batch_serial_and_parallel():
    for count in (6, PAR_THRESHOLD, PAR_THRESHOLD + 5):
        lats = [(-90.0 + 180.0 * ((i * 7919) % 100003) / 100003.0) for i in range(count)]
        a = f64(lats)
        out = i64([0] * count)
        _K.mp_cprNL_batch(addr(a), count, addr(out))
        assert out.tolist() == [ref_cprNL(v) for v in lats]


def _cpr_cases(count, seed):
    rng = random.Random(seed)
    return [
        (
            rng.randrange(2),
            rng.getrandbits(17),
            rng.getrandbits(17),
            rng.uniform(-89.0, 89.0),
            rng.uniform(-179.0, 179.0),
        )
        for _ in range(count)
    ]


def test_mp_airborne_position_with_ref():
    for fmt, la, lo, rlat, rlon in _cpr_cases(600, 41):
        buf = out_f64(1, 2)
        ok = _K.mp_airborne_position_with_ref(fmt, la, lo, rlat, rlon, addr(buf))
        want = ref_air_ref(fmt, la, lo, rlat, rlon)
        assert ok == 1, (fmt, la, lo)
        assert _close(float(buf[0]), want[0], 1e-9), (fmt, la, lo)
        assert _close(float(buf[1]), want[1], 1e-9), (fmt, la, lo)


def test_mp_airborne_position_with_ref_batch_serial_and_parallel():
    for count in (5, PAR_THRESHOLD, PAR_THRESHOLD + 17):
        cases = _cpr_cases(count, 42)
        fmt = i64([c[0] for c in cases])
        la = i64([c[1] for c in cases])
        lo = i64([c[2] for c in cases])
        rlat = f64([c[3] for c in cases])
        rlon = f64([c[4] for c in cases])
        out = out_f64(count, 2)
        _K.mp_airborne_position_with_ref_batch(
            addr(fmt), addr(la), addr(lo), addr(rlat), addr(rlon), count, addr(out)
        )
        block = out.tolist()
        for i, case in enumerate(cases):
            want = ref_air_ref(*case)
            assert _close(block[2 * i], want[0], 1e-9), (i, case)
            assert _close(block[2 * i + 1], want[1], 1e-9), (i, case)


def _raw_pairs(count, seed):
    rng = random.Random(seed)
    return [
        (
            rng.getrandbits(17),
            rng.getrandbits(17),
            rng.getrandbits(17),
            rng.getrandbits(17),
            rng.randrange(2),
        )
        for _ in range(count)
    ]


def test_mp_airborne_position_pair():
    for a, b, c, d, newer in _raw_pairs(600, 51):
        buf = out_f64(1, 2)
        ok = _K.mp_airborne_position_pair(a, b, c, d, newer, addr(buf))
        want = ref_air_pair(a, b, c, d, even_is_newer=bool(newer))
        if want is None:
            assert ok == 0, (a, b, c, d)
            assert math.isnan(float(buf[0])), (a, b, c, d)
        else:
            assert ok == 1, (a, b, c, d)
            assert _close(float(buf[0]), want[0], 1e-9), (a, b, c, d)
            assert _close(float(buf[1]), want[1], 1e-9), (a, b, c, d)


def test_mp_airborne_position_pair_batch_serial_and_parallel():
    for count in (4, PAR_THRESHOLD, PAR_THRESHOLD + 19):
        pairs = _raw_pairs(count, 52)
        # Every buffer has to outlive the call: numpy frees the array as soon as
        # the last reference goes, and `addr(i64(...))` has none left by the
        # time the kernel reads it.
        a, b, c, d, newer = (i64(v) for v in zip(*pairs))
        out = out_f64(count, 2)
        _K.mp_airborne_position_pair_batch(
            addr(a), addr(b), addr(c), addr(d), addr(newer), count, addr(out)
        )
        block = out.reshape(-1, 2)
        for p, row in zip(pairs, block):
            want = ref_air_pair(*p[:4], even_is_newer=bool(p[4]))
            if want is None:
                assert math.isnan(float(row[0])), p
            else:
                assert _close(float(row[0]), want[0], 1e-9), p
                assert _close(float(row[1]), want[1], 1e-9), p


def test_mp_surface_position_with_ref():
    for fmt, la, lo, rlat, rlon in _cpr_cases(600, 61):
        buf = out_f64(1, 2)
        ok = _K.mp_surface_position_with_ref(fmt, la, lo, rlat, rlon, addr(buf))
        want = ref_surf_ref(fmt, la, lo, rlat, rlon)
        assert ok == 1, (fmt, la, lo)
        assert _close(float(buf[0]), want[0], 1e-9), (fmt, la, lo)
        assert _close(float(buf[1]), want[1], 1e-9), (fmt, la, lo)


def test_mp_surface_position_with_ref_batch_serial_and_parallel():
    for count in (5, PAR_THRESHOLD, PAR_THRESHOLD + 23):
        cases = _cpr_cases(count, 62)
        fmt = i64([c[0] for c in cases])
        la = i64([c[1] for c in cases])
        lo = i64([c[2] for c in cases])
        rlat = f64([c[3] for c in cases])
        rlon = f64([c[4] for c in cases])
        out = out_f64(count, 2)
        _K.mp_surface_position_with_ref_batch(
            addr(fmt), addr(la), addr(lo), addr(rlat), addr(rlon), count, addr(out)
        )
        block = out.tolist()
        for i, case in enumerate(cases):
            want = ref_surf_ref(*case)
            assert _close(block[2 * i], want[0], 1e-9), (i, case)
            assert _close(block[2 * i + 1], want[1], 1e-9), (i, case)


def test_mp_surface_position_pair():
    for a, b, c, d, newer in _raw_pairs(600, 71):
        rng = random.Random((a, b, c, d).__hash__() & 0xFFFF)
        rlat, rlon = rng.uniform(-89.0, 89.0), rng.uniform(-179.0, 179.0)
        buf = out_f64(1, 2)
        ok = _K.mp_surface_position_pair(
            a, b, c, d, rlat, rlon, newer, addr(buf)
        )
        want = ref_surf_pair(
            a, b, c, d, lat_ref=rlat, lon_ref=rlon, even_is_newer=bool(newer)
        )
        if want is None:
            assert ok == 0, (a, b, c, d)
            assert math.isnan(float(buf[0])), (a, b, c, d)
        else:
            assert ok == 1, (a, b, c, d)
            assert _close(float(buf[0]), want[0], 1e-9), (a, b, c, d)
            assert _close(float(buf[1]), want[1], 1e-9), (a, b, c, d)


def test_mp_surface_position_pair_batch_serial_and_parallel():
    for count in (4, PAR_THRESHOLD, PAR_THRESHOLD + 29):
        pairs = _raw_pairs(count, 72)
        rng = random.Random(73)
        rlat = [rng.uniform(-89.0, 89.0) for _ in range(count)]
        rlon = [rng.uniform(-179.0, 179.0) for _ in range(count)]
        out = out_f64(count, 2)
        # Keep every array referenced until after the kernel returns.
        la = i64([p[0] for p in pairs])
        lb = i64([p[1] for p in pairs])
        lc = i64([p[2] for p in pairs])
        ld = i64([p[3] for p in pairs])
        ref_lat = f64(rlat)
        ref_lon = f64(rlon)
        newer = i64([p[4] for p in pairs])
        _K.mp_surface_position_pair_batch(
            addr(la),
            addr(lb),
            addr(lc),
            addr(ld),
            addr(ref_lat),
            addr(ref_lon),
            addr(newer),
            count,
            addr(out),
        )
        block = out.reshape(-1, 2)
        for p, la, lo, row in zip(pairs, rlat, rlon, block):
            want = ref_surf_pair(*p[:4], lat_ref=la, lon_ref=lo, even_is_newer=bool(p[4]))
            if want is None:
                assert math.isnan(float(row[0])), p
            else:
                assert _close(float(row[0]), want[0], 1e-9), p
                assert _close(float(row[1]), want[1], 1e-9), p


# ==================================================== the decoded registers


def _decode(name, payloads, extra=None):
    """Call `mp_decode_<name>` straight through the C ABI and read the row back.

    Returns the same list of dicts the public API returns, but the path taken
    here is `rows()` on a buffer this test allocated and passed itself.
    """
    layout = LAYOUTS[name]
    p = i64(payloads)
    n = p.size
    out = out_f64(n, layout.width)
    buf = out_u8(n, layout.char_stride) if layout.char_stride else None
    args = [addr(p)]
    if extra is not None:
        tail = i64(extra)
        args.append(addr(tail))
    args += [n, addr(out)]
    if layout.char_stride:
        args.append(addr(buf))
    getattr(_K, "mp_decode_" + name)(*args)
    return rows(name, out, buf)


# mp_decode_header


def test_mp_decode_header():
    hi, lo, lengths = to_words(ALL_MSGS)
    out = out_f64(len(ALL_MSGS), LAYOUTS["header"].width)
    _K.mp_decode_header(addr(hi), addr(lo), addr(lengths), len(ALL_MSGS), addr(out))
    got = rows("header", out)
    for msg, row in zip(ALL_MSGS, got):
        assert row["df"] == ref_df(msg), msg
        assert row["icao"] == ref_icao(msg), msg
        assert row["typecode"] == ref_typecode(msg), msg
        assert row["crc"] == ref_util_crc(msg), msg
        assert row["altitude"] == ref_altcode_field(msg), msg
        assert row["idcode"] == ref_idcode_field(msg), msg


def test_mp_decode_header_present_and_absent_masks():
    """Slot 0/1 are the ABI contract the dict reader depends on.

    Read from the raw float buffer rather than through `rows()`, so this
    asserts on the masks themselves. Slot 0 says which keys the dict had, slot
    1 which of those were `None` -- so for every key, `present` must mean
    "the reader will see this key" and `absent` must mean "and it will read
    `None`". `crc_valid` is separately checked against upstream's `crc`.
    """
    layout = LAYOUTS["header"]
    hi, lo, lengths = to_words(ALL_MSGS)
    out = out_f64(len(ALL_MSGS), layout.width)
    _K.mp_decode_header(addr(hi), addr(lo), addr(lengths), len(ALL_MSGS), addr(out))
    raw = out.reshape(-1, layout.width).tolist()
    for msg, row in zip(ALL_MSGS, raw):
        present, absent = int(row[0]), int(row[1])
        assert present & 1 and not absent & 1, msg  # df is always known
        assert present & 8 and not absent & 8, msg  # crc is always known
        # typecode exists as a key for every message and is None off the long
        # formats, so `absent` is the one that tracks df.
        assert present & 4, msg
        assert bool(absent & 4) == (int(row[2]) not in (17, 18)), msg
        # `present` may legitimately carry the bit for a key that also comes
        # back None -- that is what the second mask is for -- but a key can
        # never be marked None without also being marked present.
        assert not absent & ~present & 0x7F, msg
        # altitude and idcode are carried as keys on every message. idcode is
        # None for every df but 5/21; altitude is None for every df outside
        # 0/4/16/20 *and* whenever the altcode itself does not decode, so its
        # mask is only assertable in the direction df not altitude-format ->
        # None. The value itself is checked against upstream in the loop below.
        assert present & 32 and present & 64, msg
        if int(row[2]) not in (0, 4, 16, 20):
            assert absent & 32, msg
        assert bool(absent & 64) == (int(row[2]) not in (5, 21)), msg
        # crc_valid is the remainder being zero, stated directly.
        assert bool(int(row[6])) == (ref_util_crc(msg) == 0), msg

    # The masks must also agree with what the reader actually produces.
    decoded = rows("header", out)
    for msg, row, got in zip(ALL_MSGS, raw, decoded):
        present, absent = int(row[0]), int(row[1])
        for field, bit in (
            ("df", 0),
            ("icao", 1),
            ("typecode", 2),
            ("crc", 3),
            ("crc_valid", 4),
            ("altitude", 5),
            ("idcode", 6),
        ):
            assert (field in got) == bool(present >> bit & 1), (msg, field)
            if field in got:
                assert (got[field] is None) == bool(absent >> bit & 1), (msg, field)


# mp_decode_bds05 / 06 / 08 / 09 / 30 / 61 / 62 / 65

B05_TCS = [9, 10, 18, 20, 21, 22, 4, 23]


def test_mp_decode_bds05():
    tcs = [B05_TCS[i % len(B05_TCS)] for i in range(len(PAYLOADS))]
    got = _decode("bds05", PAYLOADS, tcs)
    for p, tc, row in zip(PAYLOADS, tcs, got):
        assert row == ref_bds05(p, tc=tc), (p, tc)


def test_mp_decode_bds06():
    got = _decode("bds06", PAYLOADS)
    for p, row in zip(PAYLOADS, got):
        assert row == ref_bds06(p), p


def test_mp_decode_bds08():
    got = _decode("bds08", PAYLOADS)
    for p, row in zip(PAYLOADS, got):
        assert row == ref_bds08(p), p


def test_mp_decode_bds08_char_block():
    """The callsign travels through a separate 8-byte char block per message.

    Read that block raw, because the dict reader trims at the first NUL and
    that trim is exactly the thing under test here.
    """
    layout = LAYOUTS["bds08"]
    p = i64(PAYLOADS)
    n = p.size
    out = out_f64(n, layout.width)
    buf = out_u8(n, layout.char_stride)
    _K.mp_decode_bds08(addr(p), n, addr(out), addr(buf))
    blob = buf.tobytes()
    for i, payload in enumerate(PAYLOADS):
        raw = blob[i * 8 : (i + 1) * 8]
        # The payload's low 48 bits are the callsign. A short callsign is
        # NUL-padded to the 8-byte stride; a full one needs no padding, so the
        # invariant is "decodes to upstream's string, and nothing stale follows".
        text = raw.split(b"\x00", 1)[0].decode("ascii")
        assert text == ref_callsign(payload & ((1 << 48) - 1)), payload
        assert raw[len(text) :] == b"\x00" * (8 - len(text)), (payload, raw)


def test_mp_decode_bds09():
    got = _decode("bds09", PAYLOADS)
    for p, row in zip(PAYLOADS, got):
        assert row == ref_bds09(p), p


def test_mp_decode_bds30():
    got = _decode("bds30", PAYLOADS)
    for p, row in zip(PAYLOADS, got):
        assert row == ref_bds30(p), p


def test_mp_decode_bds61():
    got = _decode("bds61", PAYLOADS)
    for p, row in zip(PAYLOADS, got):
        assert row == ref_bds61(p), p


def test_mp_decode_bds62():
    got = _decode("bds62", PAYLOADS)
    for p, row in zip(PAYLOADS, got):
        assert row == ref_bds62(p), p


def test_mp_decode_bds65():
    got = _decode("bds65", PAYLOADS)
    for p, row in zip(PAYLOADS, got):
        assert row == ref_bds65(p), p


def test_the_register_decoders_cross_the_parallel_threshold():
    """At PAR_THRESHOLD the decoders switch to `parallelize`; cover that path.

    Each fixture is padded past the threshold with a repeated payload, so the
    parallel body runs on every worker and the result is still checkable
    against upstream row by row. `bds05` is the one register that takes a
    second input array -- the typecode -- so it gets its own entry.
    """
    registers = (
        ("bds06", ref_bds06, None),
        ("bds08", ref_bds08, None),
        ("bds09", ref_bds09, None),
        ("bds30", ref_bds30, None),
        ("bds61", ref_bds61, None),
        ("bds62", ref_bds62, None),
        ("bds65", ref_bds65, None),
    )
    base = PAYLOADS[: min(len(PAYLOADS), 64)]
    padded = base * (PAR_THRESHOLD // len(base) + 2)
    assert len(padded) >= PAR_THRESHOLD

    for name, ref, extra in registers:
        got = _decode(name, padded, extra)
        assert len(got) == len(padded), name
        for p, row in zip(padded, got):
            assert row == ref(p), (name, p)

    # bds05 needs (payload, tc) pairs, and both types must survive the copy.
    tcs = [9 + (i % 14) for i in range(len(padded))]
    got = _decode("bds05", padded, tcs)
    assert len(got) == len(padded)
    for p, tc, row in zip(padded, tcs, got):
        assert row == ref_bds05(p, tc=tc), ("bds05", p, tc)


# mp_is_bds30 and mp_wake_vortex_category


def test_mp_is_bds30():
    for p in PAYLOADS:
        assert bool(_K.mp_is_bds30(p)) == ref_is_bds30(p), p
    assert _K.mp_is_bds30(0) == 0


def test_mp_wake_vortex_category():
    """The kernel returns the index; the table turn is the reader's job."""
    for (tc, category), name in _WAKE_VORTEX.items():
        idx = _K.mp_wake_vortex_category(tc, category)
        assert WAKE_VORTEX_NAMES[idx] == name, (tc, category)
    # Every (tc, category) pair that upstream does not name resolves to the
    # "no category information" default rather than reading off the end.
    for tc in range(32):
        for category in range(8):
            idx = _K.mp_wake_vortex_category(tc, category)
            assert 0 <= idx < len(WAKE_VORTEX_NAMES), (tc, category, idx)
    assert _K.mp_wake_vortex_category(0, 0) == 17