"""A/B the fast paths against `rows_generic` on identical kernel output.

Both are timed in the same process on the same buffer, so the machine load
that moves the absolute numbers between runs cancels out.

    flock /tmp/mojo-bench.lock pixi run python bench/ab_rows.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import corpus  # noqa: E402

from mojopymodes._layout import LAYOUTS, rows, rows_generic  # noqa: E402
from mojopymodes._lib import addr, i64, lib, out_f64, out_u8  # noqa: E402

N = 1_000_000
REPEAT = 5


def best_of(fn, repeat=REPEAT):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    base = corpus.long_corpus()
    payloads = [int(m[8:22], 16) for m in (base[i % len(base)] for i in range(N))]
    tcs = [int(m[8:10], 16) >> 3 for m in base]
    tcs = [tcs[i % len(tcs)] for i in range(N)]
    p = i64(payloads)
    t = i64(tcs)
    k = lib()

    cases = [("bds05", "mp_decode_bds05", True), ("bds08", "mp_decode_bds08", False),
             ("bds09", "mp_decode_bds09", False), ("bds62", "mp_decode_bds62", False)]
    print(f"{'case':8s} {'generic':>12s} {'fast':>12s} {'speedup':>9s}  equal")
    for name, sym, extra in cases:
        layout = LAYOUTS[name]
        out = out_f64(N, layout.width)
        buf = out_u8(N, layout.char_stride) if layout.char_stride else None
        fn = getattr(k, sym)
        if extra:
            fn(addr(p), addr(t), N, addr(out))
        elif layout.char_stride:
            fn(addr(p), N, addr(out), addr(buf))
        else:
            fn(addr(p), N, addr(out))
        a = best_of(lambda: rows_generic(name, out, buf))
        b = best_of(lambda: rows(name, out, buf))
        same = rows(name, out, buf) == rows_generic(name, out, buf)
        print(
            f"{name:8s} {a / N * 1e9:9.0f} ns {b / N * 1e9:9.0f} ns {a / b:8.2f}x  {same}",
            flush=True,
        )
        del out, buf


if __name__ == "__main__":
    main()
