"""A/B the `rows()` builders against the generic reference, same kernel output.

Both sides read one buffer that the kernel filled once, so this measures the
Python-side assembly only. Run it under the bench lock, like `bench.py`:

    flock /tmp/mojo-bench.lock pixi run python bench/ab_rows.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import corpus  # noqa: E402

from mojopymodes._layout import LAYOUTS, rows, rows_generic  # noqa: E402
from mojopymodes._lib import addr, i64, lib, out_f64, out_u8  # noqa: E402

N = 1_000_000
NAMES = ("bds08", "bds05", "bds62")


def best_of(fn, repeat=3) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def fill(name: str, payloads) -> tuple[object, object]:
    """Run one kernel over `payloads`; return (float rows, char block or None)."""
    layout = LAYOUTS[name]
    p = i64(payloads)
    n = p.size
    out = out_f64(n, layout.width)
    buf = out_u8(n, layout.char_stride) if layout.char_stride else None
    args = [addr(p)]
    tcs = None
    if name == "bds05":
        # Bound to a local, not built inline: `addr(i64([...]))` lets numpy
        # free the column before the kernel reads it.
        tcs = i64([9 + (i % 14) for i in range(n)])
        args.append(addr(tcs))
    args += [n, addr(out)]
    if buf is not None:
        args.append(addr(buf))
    getattr(lib(), "mp_decode_" + name)(*args)
    # Both buffers outlive the call and the timed functions below.
    return out, buf


def main() -> None:
    base = corpus.long_corpus()
    payloads = [int(base[i % len(base)][8:22], 16) for i in range(N)]
    for name in NAMES:
        out, buf = fill(name, payloads)
        fast = best_of(lambda: rows(name, out, buf))
        slow = best_of(lambda: rows_generic(name, out, buf))
        rows(name, out, buf)  # sanity: the two must agree before timing is quoted
        assert rows(name, out, buf) == rows_generic(name, out, buf), name
        print(
            f"{name:6s} builder {fast / N * 1e9:7.0f} ns/row   "
            f"generic {slow / N * 1e9:7.0f} ns/row   {slow / fast:.2f}x"
        )


if __name__ == "__main__":
    main()
