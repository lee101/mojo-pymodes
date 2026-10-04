"""pyModeS._altcode — the 13-bit Mode-S altitude code."""

from __future__ import annotations

from ._lib import addr, f64, i64, lib, out_f64


def altcode_to_altitude(ac: int) -> int | None:
    """Decode a 13-bit Mode-S altitude code field to feet."""
    buf = out_f64(1, 2)
    ok = lib().mp_altcode_to_altitude(ac, addr(buf))
    return int(buf[0]) if ok else None


def altcodes_to_altitudes(acs) -> list[int | None]:
    """`altcode_to_altitude` over many codes, one kernel call."""
    a = i64(acs)
    n = a.size
    if n == 0:
        return []
    out = out_f64(n, 2)
    lib().mp_altcode_to_altitude_batch(addr(a), n, addr(out))
    # The kernel writes the value and the presence flag as two adjacent slots.
    # One tolist() and two list slices beat iterating the rows: a numpy row is
    # a fresh object and a scalar read per field, and this runs a million times.
    flat = out.tolist()
    return [int(v) if p else None for v, p in zip(flat[0::2], flat[1::2])]
