"""Parity with pyModeS.position._cpr: cprNL and the four position decoders."""

from __future__ import annotations

import random

from pyModeS.position._cpr import airborne_position_pair as ref_pair
from pyModeS.position._cpr import airborne_position_with_ref as ref_with_ref
from pyModeS.position._cpr import cprNL as ref_nl
from pyModeS.position._cpr import surface_position_pair as ref_surface_pair
from pyModeS.position._cpr import surface_position_with_ref as ref_surface_ref

from mojopymodes import (
    airborne_position_pair,
    airborne_position_pair_batch,
    airborne_position_with_ref,
    airborne_position_with_ref_batch,
    cprNL,
    cprNL_batch,
    surface_position_pair,
    surface_position_pair_batch,
    surface_position_with_ref,
    surface_position_with_ref_batch,
)

LATS = [i / 10.0 for i in range(-900, 901)]
LATS += [87.0, -87.0, 87.0000001, -87.0000001, 88.0, 90.0, 0.0]


def test_cprnl_every_tenth_degree():
    assert [cprNL(v) for v in LATS] == [ref_nl(v) for v in LATS]


def test_cprnl_batch_matches_scalar_and_upstream():
    assert cprNL_batch(LATS) == [ref_nl(v) for v in LATS]


def test_cprnl_endpoints():
    assert cprNL(0.0) == 59
    assert cprNL(87.0) == 2
    assert cprNL(87.5) == 1
    assert cprNL(-52.0) == ref_nl(-52.0)


def _pairs(n, seed):
    rng = random.Random(seed)
    return [
        (
            rng.getrandbits(17),
            rng.getrandbits(17),
            rng.getrandbits(17),
            rng.getrandbits(17),
        )
        for _ in range(n)
    ]


def test_airborne_position_pair_random():
    for a, b, c, d in _pairs(2000, 21):
        for newer in (True, False):
            got = airborne_position_pair(a, b, c, d, even_is_newer=newer)
            assert got == ref_pair(a, b, c, d, even_is_newer=newer), (a, b, c, d)


def test_airborne_position_pair_batch_matches_scalar():
    pairs = _pairs(3000, 22)
    newer = [k % 2 == 0 for k in range(len(pairs))]
    got = airborne_position_pair_batch(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        [p[2] for p in pairs],
        [p[3] for p in pairs],
        newer,
    )
    assert got == [
        ref_pair(*p, even_is_newer=n) for p, n in zip(pairs, newer)
    ]


def test_airborne_position_with_ref_random():
    rng = random.Random(23)
    for _ in range(2000):
        fmt = rng.randrange(2)
        la = rng.getrandbits(17)
        lo = rng.getrandbits(17)
        ref = (rng.uniform(-89.0, 89.0), rng.uniform(-179.0, 179.0))
        got = airborne_position_with_ref(fmt, la, lo, *ref)
        assert got == ref_with_ref(fmt, la, lo, *ref), (fmt, la, lo, ref)


def test_airborne_position_with_ref_batch():
    rng = random.Random(24)
    cases = [
        (
            rng.randrange(2),
            rng.getrandbits(17),
            rng.getrandbits(17),
            rng.uniform(-89.0, 89.0),
            rng.uniform(-179.0, 179.0),
        )
        for _ in range(3000)
    ]
    got = airborne_position_with_ref_batch(
        [c[0] for c in cases],
        [c[1] for c in cases],
        [c[2] for c in cases],
        [c[3] for c in cases],
        [c[4] for c in cases],
    )
    assert got == [ref_with_ref(*c) for c in cases]


def test_surface_position_with_ref_random():
    rng = random.Random(25)
    for _ in range(2000):
        fmt = rng.randrange(2)
        la = rng.getrandbits(17)
        lo = rng.getrandbits(17)
        ref = (rng.uniform(-89.0, 89.0), rng.uniform(-179.0, 179.0))
        got = surface_position_with_ref(fmt, la, lo, *ref)
        assert got == ref_surface_ref(fmt, la, lo, *ref)


def test_surface_position_with_ref_batch():
    rng = random.Random(26)
    cases = [
        (
            rng.randrange(2),
            rng.getrandbits(17),
            rng.getrandbits(17),
            rng.uniform(-89.0, 89.0),
            rng.uniform(-179.0, 179.0),
        )
        for _ in range(2000)
    ]
    got = surface_position_with_ref_batch(
        [c[0] for c in cases],
        [c[1] for c in cases],
        [c[2] for c in cases],
        [c[3] for c in cases],
        [c[4] for c in cases],
    )
    assert got == [ref_surface_ref(*c) for c in cases]


def test_surface_position_pair_random():
    for a, b, c, d in _pairs(2000, 27):
        rng = random.Random((a, b, c, d).__hash__() & 0xFFFF)
        ref = (rng.uniform(-89.0, 89.0), rng.uniform(-179.0, 179.0))
        for newer in (True, False):
            got = surface_position_pair(
                a, b, c, d, lat_ref=ref[0], lon_ref=ref[1], even_is_newer=newer
            )
            assert got == ref_surface_pair(
                a, b, c, d, lat_ref=ref[0], lon_ref=ref[1], even_is_newer=newer
            )


def test_surface_position_pair_batch():
    pairs = _pairs(1500, 28)
    rng = random.Random(29)
    refs = [(rng.uniform(-89.0, 89.0), rng.uniform(-179.0, 179.0)) for _ in pairs]
    newer = [k % 2 == 0 for k in range(len(pairs))]
    got = surface_position_pair_batch(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        [p[2] for p in pairs],
        [p[3] for p in pairs],
        [r[0] for r in refs],
        [r[1] for r in refs],
        newer,
    )
    assert got == [
        ref_surface_pair(*p, lat_ref=r[0], lon_ref=r[1], even_is_newer=n)
        for p, r, n in zip(pairs, refs, newer)
    ]


def test_position_round_trip_through_a_known_pair():
    """A decoded position fed back as a reference must stay put."""
    lat_even, lon_even = 93000, 100000
    lat_odd, lon_odd = 93200, 101000
    pos = airborne_position_pair(
        lat_even, lon_even, lat_odd, lon_odd, even_is_newer=True
    )
    assert pos is not None
    again = airborne_position_with_ref(0, lat_even, lon_even, *pos)
    assert abs(again[0] - pos[0]) < 1e-9
    assert abs(again[1] - pos[1]) < 1e-9


def test_batch_position_decoders_refuse_columns_of_different_length():
    """Every column a batch kernel reads must be one value per message.

    The kernels read `n` elements from each column, so a scalar reference or a
    short list would be read past its end; the wrappers reject that instead.
    """
    import pytest

    n = 4
    lat = [93000, 93001, 93002, 93003]
    lon = [100000, 100001, 100002, 100003]
    newer = [1, 0, 1, 0]
    cases = [
        (airborne_position_pair_batch, (lat, lon, lat, lon[:3], newer)),
        (airborne_position_pair_batch, (lat, lon, lat, lon, 1)),
        (airborne_position_with_ref_batch, ([0] * n, lat, lon, 51.0, [0.0] * n)),
        (surface_position_with_ref_batch, ([0] * n, lat, lon, [51.0] * (n - 1), [0.0] * n)),
        (surface_position_pair_batch, (lat, lon, lat, lon, 51.0, 0.0, newer[:2])),
    ]
    for fn, columns in cases:
        with pytest.raises(ValueError):
            fn(*columns)
    # Equal lengths still decode, and agree with the scalar function.
    got = airborne_position_pair_batch(lat, lon, lat, lon, newer)
    want = [
        ref_pair(a, b, c, d, even_is_newer=bool(k))
        for a, b, c, d, k in zip(lat, lon, lat, lon, newer)
    ]
    assert got == want
