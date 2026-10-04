"""pyModeS._callsign — the 6-bit ICAO callsign encoding."""

from __future__ import annotations

from ._lib import addr, i64, lib, out_u8, scratch_16u8


def decode_callsign(bits: int) -> str:
    """Decode an 8x6-bit ICAO callsign from a 48-bit integer."""
    buf = scratch_16u8()
    n = lib().mp_decode_callsign(bits, addr(buf))
    return buf[:n].tobytes().decode("ascii")


def decode_callsigns(bits) -> list[str]:
    """`decode_callsign` over many payloads, one kernel call."""
    b = i64(bits)
    n = b.size
    if n == 0:
        return []
    out = out_u8(n, 8)
    lib().mp_decode_callsign_batch(addr(b), n, addr(out))
    # The callsigns are variable length inside a fixed slot, so the strip cannot
    # be split in one pass: a short callsign leaves padding NULs behind and the
    # split would report them as entries. One bytes copy, sliced per row.
    blob = out.tobytes()
    return [
        blob[i * 8 : i * 8 + 8].split(b"\x00", 1)[0].decode("ascii") for i in range(n)
    ]


def is_valid_callsign_char(idx: int) -> bool:
    """True if a 6-bit index maps to a valid callsign character."""
    return bool(lib().mp_is_valid_callsign_char(idx))
