"""Split the losing cases into kernel time, input-prep time and output-assembly time.

A diagnostic, like kernel_vs_wrapper.py: run it under the same flock as the
benchmark when the numbers are going in the README.

    flock /tmp/mojo-bench.lock pixi run python bench/profile_split.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import corpus  # noqa: E402

import mojopymodes as pm  # noqa: E402
from mojopymodes._altcode import altcodes_to_altitudes  # noqa: E402
from mojopymodes._callsign import decode_callsigns  # noqa: E402
from mojopymodes._layout import LAYOUTS, rows  # noqa: E402
from mojopymodes._lib import addr, f64, i64, lib, out_f64, out_u8  # noqa: E402
from mojopymodes.batch import decode_payloads  # noqa: E402

N = 1_000_000
REPEAT = 3


def best_of(fn, repeat=REPEAT):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def row(label, value):
    print(f"{label:34s}: {value * 1e3:9.1f} ms")


def main() -> None:
    base = corpus.long_corpus()
    msgs = [base[i % len(base)] for i in range(N)]
    payloads = [int(m[8:22], 16) for m in msgs]
    tcs = [int(m[8:10], 16) >> 3 for m in msgs]
    k = lib()

    cases = [
        ("crc_remainder", "mp_crc_remainder_batch", None),
        ("altcode_to_altitude", "mp_altcode_to_altitude_batch", None),
        ("bds05", "mp_decode_bds05", "bds05"),
        ("bds08", "mp_decode_bds08", "bds08"),
        ("bds09", "mp_decode_bds09", "bds09"),
        ("bds62", "mp_decode_bds62", "bds62"),
    ]
    src = {"crc_remainder": None, "altcode_to_altitude": [i % 8192 for i in range(N)]}
    src["bds05"] = (payloads, tcs)
    src["bds08"] = (payloads,)
    src["bds09"] = (payloads,)
    src["bds62"] = (payloads,)

    for name, sym, layout_name in cases:
        print(f"--- {name} ---")
        data = src[name]
        if name == "crc_remainder":
            from mojopymodes._bits import crc_remainders

            inputs = list(zip(*_words(msgs)))
            p_i = i64(list(inputs[0]))
            p_l = i64(list(inputs[1]))
            p_g = i64(list(inputs[2]))
            out = out_f64(N, 1).astype(np.int64) if False else np.zeros(N, dtype=np.int64)
            row("  kernel", best_of(lambda: k.mp_crc_remainder_batch(addr(p_i), addr(p_l), addr(p_g), N, addr(out))))
            row("  full api", best_of(lambda: crc_remainders(msgs)))
        elif name == "altcode_to_altitude":
            codes = src[name]
            c = i64(codes)
            out = out_f64(N, 2)
            row("  kernel", best_of(lambda: k.mp_altcode_to_altitude_batch(addr(c), N, addr(out))))
            row("  full api", best_of(lambda: altcodes_to_altitudes(codes)))
        else:
            layout = LAYOUTS[layout_name]
            ins = [i64(d) for d in data]
            out = out_f64(N, layout.width)
            buf = out_u8(N, layout.char_stride) if layout.char_stride else None
            fn = getattr(k, sym)
            if layout_name == "bds05":
                call = lambda: fn(addr(ins[0]), addr(ins[1]), N, addr(out))  # noqa: E731
            elif layout.char_stride:
                call = lambda: fn(addr(ins[0]), N, addr(out), addr(buf))  # noqa: E731
            else:
                call = lambda: fn(addr(ins[0]), N, addr(out))  # noqa: E731
            row("  kernel", best_of(call))
            row("  rows()", best_of(lambda: rows(layout_name, out, buf)))
            pyfn = getattr(pm.decoder.bds, "decode_" + layout_name + "_batch")
            row("  full api", best_of(lambda: pyfn(*data)))
            print()

    print("--- full ADS-B decode ---")
    items = list(zip(payloads[:200_000], tcs[:200_000]))
    row("  full api", best_of(lambda: decode_payloads(items)))

    print("--- callsign ---")
    bits = [(i * 0x9E3779B97F4A7C15) % (1 << 48) for i in range(N)]
    b = i64(bits)
    buf = out_u8(N, 8)
    row("  kernel", best_of(lambda: k.mp_decode_callsign_batch(addr(b), N, addr(buf))))
    row("  full api", best_of(lambda: decode_callsigns(bits)))

    print("--- cprNL ---")
    lats = [(-90.0 + 180.0 * ((i * 7919) % 100003) / 100003.0) for i in range(N)]
    f = f64(lats)
    out = np.zeros(N, dtype=np.int64)
    row("  kernel", best_of(lambda: k.mp_cprNL_batch(addr(f), N, addr(out))))
    row("  full api", best_of(lambda: pm.position._cpr.cprNL_batch(lats)))


def _words(msgs):
    from mojopymodes._lib import to_words

    return to_words(msgs)


if __name__ == "__main__":
    main()
