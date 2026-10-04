"""Where the losing cases spend their time: kernel vs Python-side assembly.

A diagnostic, like kernel_vs_wrapper.py. Run it under the bench flock when the
numbers are going anywhere:

    flock /tmp/mojo-bench.lock pixi run python bench/split.py
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
from mojopymodes._bits import crc_remainders  # noqa: E402
from mojopymodes._callsign import decode_callsigns  # noqa: E402
from mojopymodes._layout import LAYOUTS, rows  # noqa: E402
from mojopymodes._lib import addr, f64, i64, lib, out_f64, out_u8, to_words  # noqa: E402

N = 1_000_000
REPEAT = 3
out_lines: list[str] = []


def say(text: str) -> None:
    out_lines.append(text)
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


def best_of(fn, repeat=REPEAT):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def row(label, seconds, n=N):
    say(f"{label:38s}: {seconds * 1e3:9.1f} ms  ({seconds / n * 1e9:7.1f} ns/item)")


def main() -> None:
    base = corpus.long_corpus()
    msgs = [base[i % len(base)] for i in range(N)]
    payloads = [int(m[8:22], 16) for m in msgs]
    tcs = [int(m[8:10], 16) >> 3 for m in msgs]
    k = lib()

    # ---- crc_remainder
    say("--- crc_remainder (1M) ---")
    hi, lo, ln = to_words(msgs)
    crc_out = np.zeros(N, dtype=np.int64)
    row("kernel only", best_of(lambda: k.mp_crc_remainder_batch(hi.ctypes.data, lo.ctypes.data, ln.ctypes.data, N, crc_out.ctypes.data)))
    row("+ to_words (hex parse)", best_of(lambda: to_words(msgs)))
    row("full api", best_of(lambda: crc_remainders(msgs)))
    del hi, lo, ln

    # ---- altcode
    say("--- altcode_to_altitude (1M) ---")
    codes = [(i * 7919) % 8192 for i in range(N)]
    c = i64(codes)
    alt_out = out_f64(N, 2)
    row("kernel only", best_of(lambda: k.mp_altcode_to_altitude_batch(addr(c), N, addr(alt_out))))
    rows2d = alt_out.reshape(-1, 2)
    row("python list build", best_of(lambda: [int(r[0]) if r[1] else None for r in rows2d]))
    flat = alt_out.tolist()
    row("tolist + zip build", best_of(lambda: [int(v) if p else None for v, p in zip(flat[0::2], flat[1::2])]))
    row("full api", best_of(lambda: altcodes_to_altitudes(codes)))

    # ---- callsign
    say("--- decode_callsign (1M) ---")
    bits = [(i * 0x9E3779B97F4A7C15) % (1 << 48) for i in range(N)]
    b = i64(bits)
    cs_buf = out_u8(N, 8)
    row("kernel only", best_of(lambda: k.mp_decode_callsign_batch(addr(b), N, addr(cs_buf))))
    block = cs_buf.reshape(-1, 8)
    row("python list build", best_of(lambda: [bytes(r).split(b"\x00", 1)[0].decode("ascii") for r in block]))
    blob = cs_buf.tobytes()
    row("bytes blob build", best_of(lambda: [blob[i * 8 : i * 8 + 8].split(b"\x00", 1)[0].decode("ascii") for i in range(N)]))
    row("full api", best_of(lambda: decode_callsigns(bits)))

    # ---- bds
    p_arr = i64(payloads)
    t_arr = i64(tcs)
    for name, sym, extra in (("bds05", "mp_decode_bds05", True), ("bds08", "mp_decode_bds08", False), ("bds09", "mp_decode_bds09", False), ("bds62", "mp_decode_bds62", False)):
        say(f"--- {name} (1M) ---")
        layout = LAYOUTS[name]
        o = out_f64(N, layout.width)
        buf = out_u8(N, layout.char_stride) if layout.char_stride else None
        fn = getattr(k, sym)
        if extra:
            call = lambda fn=fn, o=o: fn(addr(p_arr), addr(t_arr), N, addr(o))
        elif layout.char_stride:
            call = lambda fn=fn, o=o, buf=buf: fn(addr(p_arr), N, addr(o), addr(buf))
        else:
            call = lambda fn=fn, o=o: fn(addr(p_arr), N, addr(o))
        row("kernel only", best_of(call))
        row("rows()", best_of(lambda name=name, o=o, buf=buf: rows(name, o, buf)))
        pyfn = getattr(pm.decoder.bds, "decode_" + name + "_batch")
        row("full api", best_of(lambda pyfn=pyfn: pyfn(payloads, tcs) if extra else pyfn(payloads)))
        row("i64() prep of the python list", best_of(lambda: i64(payloads)))
        del o, buf

    # ---- cprNL
    say("--- cprNL (1M) ---")
    lats = [(-90.0 + 180.0 * ((i * 7919) % 100003) / 100003.0) for i in range(N)]
    f = f64(lats)
    nl_out = np.zeros(N, dtype=np.int64)
    row("kernel only", best_of(lambda: k.mp_cprNL_batch(addr(f), N, addr(nl_out))))
    row("f64() prep", best_of(lambda: f64(lats)))
    row("out.tolist()", best_of(lambda: nl_out.tolist()))
    row("full api", best_of(lambda: pm.position._cpr.cprNL_batch(lats)))

    say("")
    say("done")


if __name__ == "__main__":
    main()
