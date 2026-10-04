"""pyModeS._bits — bit extraction and the Mode-S CRC-24."""

from __future__ import annotations

from ._lib import i64, lib, to_words


def extract_unsigned(n: int, start: int, width: int, total_bits: int) -> int:
    """Extract `width` bits starting at bit `start` (MSB-first) from `n`."""
    return lib().mp_extract_unsigned(n, start, width, total_bits)


def extract_signed(n: int, start: int, width: int, total_bits: int) -> int:
    """Extract `width` bits as a two's-complement integer."""
    return lib().mp_extract_signed(n, start, width, total_bits)


def crc_remainder(n: int, length: int) -> int:
    """Mode-S CRC-24 remainder of a `length`-bit integer.

    A 56-bit message fits in the single-argument form. A 112-bit message
    arrives from a hex string and is routed through the two-word kernel,
    which is the same algorithm over (hi, lo).
    """
    if length <= 56:
        return lib().mp_crc_remainder(n, length)
    return lib().mp_crc_remainder_wide(n >> 64, n & 0xFFFFFFFFFFFFFFFF, length)


def crc_remainders(msgs: list[str]) -> list[int]:
    """`crc_remainder` over many hex messages, one kernel call."""
    if not msgs:
        return []
    hi, lo, lengths = to_words(msgs)
    out = i64([0] * len(msgs))
    lib().mp_crc_remainder_batch(hi.ctypes.data, lo.ctypes.data, lengths.ctypes.data, len(msgs), out.ctypes.data)
    return out.tolist()
