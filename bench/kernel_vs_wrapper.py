"""Kernel-only vs wrapper-inclusive timing for one register decoder.

The README attributes the BDS 0,5 loss to Python dict building in
`_layout.rows` rather than to the kernel. That is a claim about where the time
goes, so it needs a number that is actually measured.

Run through `pixi run bench`? No -- this is a diagnostic, not a benchmark, and
the bench task's flock exists to keep other jobs off the machine. Run it on an
otherwise idle machine:

    pixi run python bench/kernel_vs_wrapper.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import corpus  # noqa: E402

import mojopymodes as pm  # noqa: E402
from mojopymodes._layout import LAYOUTS  # noqa: E402
from mojopymodes._lib import addr, i64, lib, out_f64, out_u8  # noqa: E402

REPEAT = 3
N = 1_000_000


def best_of(fn, repeat=REPEAT):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    base = corpus.long_corpus()
    payloads = ([int(m[8:22], 16) for m in base] * 700)[:N]
    tcs = ([int(m[8:10], 16) >> 3 for m in base] * 700)[:N]
    n = len(payloads)

    p = i64(payloads)  # kept referenced for the whole run
    t = i64(tcs)
    k = lib()
    layout = LAYOUTS["bds05"]

    def decode_one():
        """One record through the kernel: the fixed cost of a decode."""
        out = out_f64(1, layout.width)
        k.mp_decode_bds05(addr(p), addr(t), 1, addr(out))
        return out

    def kernel_only():
        """Exactly the FFI call: one kernel, no Python-side row assembly."""
        out = out_f64(n, layout.width)
        k.mp_decode_bds05(addr(p), addr(t), n, addr(out))
        return out

    def kernel_plus_dicts():
        """The public batch API, which is the kernel plus `rows()`."""
        return pm.decoder.bds.decode_bds05_batch(payloads, tcs)

    a = best_of(kernel_only)
    b = best_of(kernel_plus_dicts)
    one = best_of(decode_one)
    print(f"n = {n}")
    print(f"bds05 kernel only       : {a * 1e3:8.1f} ms  ({a / n * 1e9:6.1f} ns/msg)")
    print(f"bds05 through the API   : {b * 1e3:8.1f} ms  ({b / n * 1e9:6.1f} ns/msg)")
    print(
        f"dict assembly share     : {(b - a) * 1e3:8.1f} ms "
        f"({100 * (b - a) / b:.1f}% of the API time)"
    )
    print(f"kernel is {b / a:.0f}x faster than the API that wraps it")
    # The one-record path is what upstream has to compete with, and its cost is
    # a fixed ctypes call, not a share of the kernel: the benchmark's "single
    # cprNL call" case is the same effect on a cheaper kernel.
    print(f"one bds05 record        : {one * 1e3:8.3f} ms  ({one * 1e6:6.1f} us/record)")


if __name__ == "__main__":
    main()