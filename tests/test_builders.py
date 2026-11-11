"""Builder tests: every fused builder against the generic reference.

`rows()` and `rows_generic()` are two public entry points, but the builders
inside `rows()` also fall back to `_ordered`. These tests call the builders
directly so every branch of every builder is compared with `rows_generic`,
including the mixed per-row shapes a capture actually produces (BDS 0,9's
subtype-dependent key set, BDS 6,5's two registers) which a uniform-mask test
can never reach.
"""

from __future__ import annotations

import random
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojopymodes import _layout as L  # noqa: E402


def _columns(name: str, floats: np.ndarray, blob: bytes = b""):
    ints = floats.astype(np.int64)
    n = len(floats)
    ever = int(ints[:, 1].max()) if n else 0
    present = int(ints[0, 0]) if n else 0
    return ints, blob, ever, present


def _run(name: str, floats: np.ndarray, blob: bytes = b"") -> list[dict]:
    ints, blob, ever, present = _columns(name, floats, blob)
    return L._BUILDERS[name](L.LAYOUTS[name].width, floats, [], blob, ever, present)


def _reference(name: str, floats: np.ndarray, blob: bytes = b"") -> list[dict]:
    chars = np.frombuffer(blob, dtype=np.uint8) if blob else None
    # `rows()` applies `_squawks` after the builder, and a builder leaves the
    # ID field raw; both sides are compared with that step applied.
    rows = L.rows_generic(name, floats.reshape(-1), chars)
    return L._squawks(rows) if name == "header" else rows


def _built(name: str, floats: np.ndarray, blob: bytes = b"") -> list[dict]:
    rows = _run(name, floats, blob)
    return L._squawks(rows) if name == "header" else rows


@pytest.mark.parametrize("name", sorted(L._BUILDERS))
def test_builder_matches_generic_on_random_rows(name):
    """Random presence, absence and value per row, every mask independent."""
    layout = L.LAYOUTS[name]
    width = layout.width
    rng = random.Random(hash(name) & 0xFFFF)
    n = 300
    out = np.zeros((n, width), dtype=np.float64)
    for i in range(n):
        out[i, 0] = rng.randrange(0, 1 << (width - 2))
        out[i, 1] = rng.randrange(0, 1 << (width - 2))
        for s in range(2, width):
            out[i, s] = float(rng.randrange(0, 2**40))
    # Fields that index a table need to stay inside it for the generic loop to
    # be able to run at all; the builder is not allowed to be more forgiving.
    for i in range(n):
        if name == "bds62":
            out[i, 4] = rng.randrange(0, 3)
        if name == "bds08":
            out[i, 3] = rng.randrange(0, 18)
        if name in ("bds09", "bds10"):
            out[i, 8] = rng.randrange(0, 2)
            out[i, 9] = rng.randrange(0, 2)
        if name == "header":
            out[i, 3] = float(rng.randrange(0, 1 << 24))
    blob = bytes(rng.randrange(32, 127) for _ in range(n * width)) if layout.char_stride else b""
    assert _built(name, out, blob) == _reference(name, out, blob)


@pytest.mark.parametrize("name", sorted(L._BUILDERS))
def test_builder_matches_generic_on_a_uniform_batch(name):
    layout = L.LAYOUTS[name]
    width = layout.width
    n = 64
    full = (1 << (width - 2)) - 1
    out = np.zeros((n, width), dtype=np.float64)
    out[:, 0] = full
    for s in range(2, width):
        out[:, s] = float(1000 + s)
    for i in range(n):
        if name == "bds62":
            out[i, 4] = i % 3
        if name == "bds08":
            out[i, 3] = i % 18
    assert _built(name, out) == _reference(name, out)


def test_builder_handles_zero_rows():
    for name in L._BUILDERS:
        width = L.LAYOUTS[name].width
        assert _run(name, np.zeros((0, width))) == []
