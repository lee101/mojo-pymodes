"""pyModeS.util — public low-level helpers for inspecting a single message.

The hex/binary string converters are one-liners over the stdlib and stay
here in Python; everything that touches bits goes through a Mojo kernel.
"""

from __future__ import annotations

from ._bits import crc_remainder
from ._layout import LAYOUTS, one, rows
from ._lib import addr, lib, out_f64, to_words
from .position._cpr import cprNL

__all__ = [
    "altcode",
    "bin2hex",
    "bin2int",
    "cprNL",
    "crc",
    "df",
    "hex2bin",
    "hex2int",
    "icao",
    "idcode",
    "typecode",
]


def hex2bin(hexstr: str) -> str:
    """Convert a hexadecimal string to a zero-padded binary string."""
    return f"{int(hexstr, 16):0{4 * len(hexstr)}b}"


def bin2int(binstr: str) -> int:
    """Parse a binary ('0'/'1') string to an integer."""
    return int(binstr, 2)


def hex2int(hexstr: str) -> int:
    """Parse a hexadecimal string to an integer."""
    return int(hexstr, 16)


def bin2hex(binstr: str) -> str:
    """Convert a binary string to an uppercase hexadecimal string."""
    width = (len(binstr) + 3) // 4
    return f"{int(binstr, 2):0{width}X}"


def headers(msgs: list[str]) -> list[dict]:
    """`df`, `icao`, `typecode`, `crc`, `altitude` and `idcode` for many messages.

    One kernel call for the whole list; the per-message helpers below are
    this with a one-element list.
    """
    if not msgs:
        return []
    hi, lo, lengths = to_words(msgs)
    out = out_f64(len(msgs), LAYOUTS["header"].width)
    lib().mp_decode_header(
        addr(hi), addr(lo), addr(lengths), len(msgs), addr(out)
    )
    return rows("header", out)


def _header(msg: str) -> dict:
    return headers([msg])[0]


def df(msg: str) -> int:
    """Mode-S downlink format (bits 0-4), clamped at 24."""
    return _header(msg)["df"]


def icao(msg: str) -> str | None:
    """The ICAO 24-bit address carried by `msg`, or None."""
    return _header(msg)["icao"]


def typecode(msg: str) -> int | None:
    """ADS-B type code (ME bits 32-36) for DF17/18, else None."""
    return _header(msg)["typecode"]


def crc(msg: str) -> int:
    """Mode-S CRC-24 remainder of a hex message."""
    return crc_remainder(int(msg, 16), len(msg) * 4)


def altcode(msg: str) -> int | None:
    """Altitude in feet from the 13-bit AC field (DF0/4/16/20)."""
    return _header(msg)["altitude"]


def idcode(msg: str) -> str | None:
    """4-digit octal squawk from the 13-bit ID field (DF5/21)."""
    return _header(msg)["idcode"]
