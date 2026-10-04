"""Batched decoding over a whole capture, in one kernel call.

Upstream pyModeS decodes one message at a time, so this module is an
addition rather than a port: it runs the same Mojo kernels over arrays. The
per-message functions in `util`, `position` and `decoder.bds` are one-element
calls into exactly these paths, so nothing is decoded two different ways.
"""

from __future__ import annotations

from ._altcode import altcodes_to_altitudes
from ._callsign import decode_callsigns
from ._idcode import idcodes_to_squawks
from ._layout import LAYOUTS, rows
from ._lib import addr, lib, out_f64, to_words
from .util import headers
from .decoder import ADSB_DISPATCH
from .decoder.bds import (
    decode_bds05_batch,
    decode_bds06_batch,
    decode_bds08_batch,
    decode_bds09_batch,
    decode_bds61_batch,
    decode_bds62_batch,
    decode_bds65_batch,
)

BATCH_BY_BDS = {
    "0,6": decode_bds06_batch,
    "0,8": decode_bds08_batch,
    "0,9": decode_bds09_batch,
    "6,1": decode_bds61_batch,
    "6,2": decode_bds62_batch,
    "6,5": decode_bds65_batch,
}

__all__ = [
    "altcodes_to_altitudes",
    "decode_callsigns",
    "decode_fields",
    "decode_payloads",
    "headers",
    "idcodes_to_squawks",
    "messages_to_payloads",
    "payload",
    "payloads",
]


def messages_to_payloads(msgs: list[str]) -> tuple[list[int], list[int]]:
    """Split DF17/18 hex messages into (payload, typecode) lists."""
    hi, lo, lengths = to_words(msgs)
    out = out_f64(len(msgs), LAYOUTS["header"].width)
    lib().mp_decode_header(addr(hi), addr(lo), addr(lengths), len(msgs), addr(out))
    head = rows("header", out)
    payloads = []
    tcs = []
    for msg, h in zip(msgs, head):
        if h["df"] in (17, 18):
            payloads.append(int(msg[8:22], 16))
            tcs.append(h["typecode"])
    return payloads, tcs


def payload(msg: str) -> int:
    """The 56-bit ME field of a DF17/18 message, as an int."""
    if len(msg) != 28:
        raise ValueError("a long message has 28 hex characters")
    return int(msg[8:22], 16)


def payloads(msgs: list[str]) -> list[int]:
    """`payload` over many messages."""
    return [payload(m) for m in msgs]


def decode_fields(msgs: list[str]) -> list[dict]:
    """df, icao, typecode, crc, crc_valid, altitude and idcode per message."""
    return headers(msgs)


def decode_payloads(items) -> list[dict]:
    """Route 56-bit payloads to their BDS decoder, as ADSB.decode does.

    `items` is a sequence of `(payload, typecode)` pairs. A reserved
    typecode comes back as `{"typecode": tc}` and nothing else, which is what
    upstream returns for it.
    """
    if not len(items):
        return []
    result = [{"typecode": it[1]} for it in items]
    groups: dict[str, tuple[list[int], list[int]]] = {}
    for k, it in enumerate(items):
        entry = ADSB_DISPATCH.get(it[1])
        if entry is None:
            continue
        idx, vals = groups.setdefault(entry[0], ([], []))
        idx.append(k)
        vals.append(it[0])
    for bds, (idx, vals) in groups.items():
        if bds == "0,5":
            decoded = decode_bds05_batch(vals, [items[k][1] for k in idx])
        else:
            decoded = BATCH_BY_BDS[bds](vals)
        for k, d in zip(idx, decoded):
            result[k] = {"typecode": items[k][1], "bds": bds, **d}
    return result
