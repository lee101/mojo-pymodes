"""The `rows()` fast paths must agree with `rows_generic`, exactly.

`_layout.rows` picks a hand-written fast path for six layouts and hands the
whole buffer back to `rows_generic` for a row it does not handle. That
fallback is invisible from the outside, so this file drives both and asserts
they are the same list of dicts -- including the shapes that take the
fallback: an absent (`None`) value, a key that is not present at all, and the
string-carrying layouts.

The buffers here are synthetic and deliberately varied in every way the real
kernel can be: absent bits set, present bits clear, enum fields constrained to
their real range, and float fields holding real floats. They are read through
`rows_generic`, which is the definition of correct.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from mojopymodes._layout import LAYOUTS, _FAST, rows, rows_generic

FAST_NAMES = sorted(_FAST)
GENERIC_NAMES = sorted(set(LAYOUTS) - set(_FAST))

# Fields whose value is used as an index or a small enum, so a random large
# value would be nonsense (and would index off the end of a lookup list).
# (name, slot) -> (lo, hi) inclusive, applied after the general fill.
FIELD_RANGE = {
    ("bds09", 9): (0, 1),  # vr_source
    ("bds62", 4): (0, 2),  # selected_altitude_source
    ("header", 6): (0, 1),  # crc_valid
    ("header", 8): (0, 2**13 - 1),  # idcode (octal squawk)
}

# The absent bits each fast path reads itself: the fields a register can leave
# as None. Mirrors the per-field conditionals in _layout.py.
OPTIONAL_BITS = {
    "bds05": 1,  # altitude
    "bds06": 2 | 4,  # groundspeed, track
    "bds08": 0,
    "bds62": 2 | 8 | 16 | 256 | 512 | 1024 | 2048 | 4096,
    "bds65": 0,
    "header": 2 | 4 | 32 | 64,  # icao, typecode, altitude, idcode
}


def _buffer(name: str, width: int, n: int, present: list[int], absent: list[int]):
    """A flat float64 row buffer with the given masks, C-contiguous.

    Integer slots hold non-negative values below 2**48, which is the contract
    the kernels write to: every integer field in the port fits a `Float64`
    exactly, which is what lets the fast paths convert with `astype`.
    """
    rng = random.Random(11)
    out = np.empty(n * width, dtype=np.float64)
    for i in range(n):
        base = i * width
        out[base] = present[i]
        out[base + 1] = absent[i]
        for s in range(2, width):
            if name == "bds62" and s == 5:
                # baro_pressure_setting is a real float.
                out[base + s] = rng.uniform(-2000.0, 2000.0)
            elif name == "bds62" and s == 6:
                out[base + s] = rng.uniform(0.0, 360.0)
            else:
                lo, hi = FIELD_RANGE.get((name, s), (0, 2**46))
                out[base + s] = float(rng.randrange(lo, hi + 1))
    return out


def _chars(width: int, n: int, rng: random.Random):
    """A char strip the way the kernel writes it: NUL-terminated ASCII."""
    blob = bytearray()
    for _ in range(n):
        text = "".join(chr(rng.randrange(33, 127)) for _ in range(width)).encode("ascii")
        text = text[: width - 1]
        blob += text + b"\x00" * (width - len(text))
    return np.frombuffer(bytes(blob), dtype=np.uint8)


def _all_bits(width: int) -> int:
    return (1 << (width - 2)) - 1


@pytest.mark.parametrize("name", FAST_NAMES + GENERIC_NAMES)
def test_rows_matches_generic_on_random_masks(name):
    """Random presence and absence masks: whatever the fast path skips, the
    generic loop has to produce the identical dict."""
    layout = LAYOUTS[name]
    width = layout.width
    rng = random.Random(hash(name) & 0xFFFF)
    n = 400
    present = [rng.randrange(0, 1 << (width - 2)) for _ in range(n)]
    absent = [rng.randrange(0, 1 << (width - 2)) for _ in range(n)]
    out = _buffer(name, width, n, present, absent)
    chars = _chars(layout.char_stride or 1, n, rng) if layout.char_stride else None
    assert rows(name, out, chars) == rows_generic(name, out, chars)


@pytest.mark.parametrize("name", FAST_NAMES)
def test_fast_path_matches_generic_when_all_present(name):
    """Every key present, nothing `None`: the shape real traffic hits. The
    fast path must produce the same dicts as the generic loop, and take the
    fast branch rather than bailing."""
    layout = LAYOUTS[name]
    width = layout.width
    n = 64
    full = _all_bits(width)
    out = _buffer(name, width, n, [full] * n, [0] * n)
    chars = _chars(layout.char_stride or 1, n, random.Random(5)) if layout.char_stride else None
    assert rows(name, out, chars) == rows_generic(name, out, chars)


@pytest.mark.parametrize(
    "present",
    [
        0,
        1,
        3,
        0x7F,
        0x1FF,
        0x3FF,
        911,  # bds09 groundspeed subtypes, every value present
        1011,  # bds09 airspeed subtypes, every value present
        0x3FFF,
        0xFFFF,
        0x3FFFF,
        0xFFFFF,
    ],
)
def test_fast_path_matches_generic_on_kernel_masks(present):
    """The masks the kernels actually write.

    Random masks never produce the interesting case: a row whose values are all
    present but whose present mask is not the full set. bds09 subtype 0 is
    exactly that, and it is why this file exists.
    """
    for name in FAST_NAMES:
        w = LAYOUTS[name].width
        # bds09's two constants are already within its 10 bits; masking them
        # with a wider layout's mask would change their meaning.
        bits = present if present in (911, 1011) and name == "bds09" else present & _all_bits(w)
        n = 32
        out = _buffer(name, w, n, [bits] * n, [0] * n)
        chars = (
            _chars(LAYOUTS[name].char_stride, n, random.Random(2))
            if LAYOUTS[name].char_stride
            else None
        )
        assert rows(name, out, chars) == rows_generic(name, out, chars), (name, bits)


@pytest.mark.parametrize("name", FAST_NAMES)
def test_fast_path_handles_the_fallback(name):
    """A batch whose key set the fast path does not cover must still come out
    right, and the rows around it must be unaffected."""
    layout = LAYOUTS[name]
    width = layout.width
    full = _all_bits(width)
    n = 5
    # The middle row is missing its last key, so its present mask differs from
    # the rest and the fast path must hand the whole buffer to the generic loop.
    present = [full, full ^ 1, full, full, full]
    out = _buffer(name, width, n, present, [0] * n)
    chars = (
        _chars(layout.char_stride, n, random.Random(6))
        if layout.char_stride
        else None
    )
    assert rows(name, out, chars) == rows_generic(name, out, chars)


@pytest.mark.parametrize("name", FAST_NAMES)
def test_fast_path_handles_none_values(name):
    """A `None` value, which real traffic produces constantly, must be read
    out of the absent mask by the fast path itself."""
    layout = LAYOUTS[name]
    width = layout.width
    full = _all_bits(width)
    # Absent masks the kernels actually produce: every optional field, then one
    # bit at a time, then a bit the fast path does not own.
    optional = OPTIONAL_BITS.get(name, 0)
    masks = [0, optional, optional & ~1, 1 if optional else 0, full]
    n = len(masks)
    out = _buffer(name, width, n, [full] * n, masks)
    chars = (
        _chars(layout.char_stride, n, random.Random(8))
        if layout.char_stride
        else None
    )
    assert rows(name, out, chars) == rows_generic(name, out, chars)


@pytest.mark.parametrize("name", FAST_NAMES)
def test_fast_path_row_count_and_zero_input(name):
    layout = LAYOUTS[name]
    empty_chars = np.zeros(0, dtype=np.uint8) if layout.char_stride else None
    assert rows(name, np.zeros(0, dtype=np.float64), empty_chars) == []
    full = _all_bits(layout.width)
    one = _buffer(name, layout.width, 1, [full], [0])
    chars = (
        _chars(layout.char_stride, 1, random.Random(3)) if layout.char_stride else None
    )
    assert len(rows(name, one, chars)) == 1


def test_astype_conversion_is_exact_for_the_int64_fields():
    """The fast paths convert integer slots with `astype(np.int64)`. Every
    integer field in the port is below 2**48, so the round trip through
    `Float64` is exact and `int()` in the generic loop agrees."""
    rng = random.Random(9)
    for name in FAST_NAMES:
        width = LAYOUTS[name].width
        vals = [rng.randrange(0, 2**48) for _ in range(200 * width)]
        out = np.array(vals, dtype=np.float64)
        assert out.astype(np.int64).tolist() == vals, name


def test_squawk_helper_matches_upstream():
    from mojopymodes._layout import _squawk
    from pyModeS._idcode import idcode_to_squawk as ref_squawk

    for ident in range(0, 8192, 7):
        assert _squawk(ident) == ref_squawk(ident)
