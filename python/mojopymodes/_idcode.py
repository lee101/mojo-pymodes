"""pyModeS._idcode — the 13-bit identity code (squawk)."""

from __future__ import annotations

from ._lib import addr, i64, lib, out_u8, scratch_16u8


def idcode_to_squawk(idcode: int) -> str:
    """Decode a 13-bit identity code to a 4-digit octal squawk string."""
    buf = scratch_16u8()
    n = lib().mp_idcode_to_squawk(idcode, addr(buf))
    return buf[:n].tobytes().decode("ascii")


def idcodes_to_squawks(codes) -> list[str]:
    """`idcode_to_squawk` over many codes, one kernel call."""
    c = i64(codes)
    n = c.size
    if n == 0:
        return []
    out = out_u8(n, 4)
    lib().mp_idcode_to_squawk_batch(addr(c), n, addr(out))
    block = out.reshape(-1, 4)
    return [bytes(r).decode("ascii") for r in block]
