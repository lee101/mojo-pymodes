"""Shared plumbing for the BDS register decoders.

Every `decode_bdsXX(payload)` below is a one-element call into the matching
`decode_bdsXX_batch`, so there is exactly one code path and the parity tests
exercise the same kernel the batch API uses.
"""

from __future__ import annotations

from ..._layout import LAYOUTS, rows
from ..._lib import addr, i64, lib, out_f64, out_u8


def _run(name: str, payloads, extra=None) -> list[dict]:
    layout = LAYOUTS[name]
    p = i64(payloads)
    n = p.size
    if n == 0:
        return []
    out = out_f64(n, layout.width)
    buf = out_u8(n, layout.char_stride) if layout.char_stride else None
    fn = getattr(lib(), "mp_decode_" + name)
    args = [addr(p)]
    if extra is not None:
        # The array must outlive the call: an inline `addr(i64(extra))` lets
        # numpy free it before the kernel reads it.
        tail = i64(extra)
        args.append(addr(tail))
    args.append(n)
    args.append(addr(out))
    if layout.char_stride:
        args.append(addr(buf))
    fn(*args)
    return rows(name, out, buf)


def _decode(name: str, payloads) -> list[dict]:
    return _run(name, payloads)


def _one(name: str, payload: int, extra=None) -> dict:
    return _run(name, [payload], None if extra is None else [extra])[0]


