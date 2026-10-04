"""Correct integer-slot extraction: astype(int64).tolist() vs per-element int().

The `.view(np.int64)` trick is wrong -- a float64 holding 5.0 has the bit
pattern 0x4014000000000000, so reinterpreting gives 4617315517961601024. The
conversion has to be `astype`, and this measures what that costs against the
per-element `int()` calls it replaces.

    flock /tmp/mojo-bench.lock pixi run python bench/micro_rows2.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

N = 1_000_000
REPEAT = 5
B05_NAMES = ("altitude", "surveillance_status", "nic_b", "cpr_format", "cpr_lat", "cpr_lon", "nuc_p")


def best_of(fn, repeat=REPEAT):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def row(label, seconds):
    print(f"{label:46s}: {seconds * 1e3:9.1f} ms  ({seconds / N * 1e9:6.0f} ns/row)", flush=True)


def b05_float_int(out):
    t = out.tolist()
    result = []
    append = result.append
    for i in range(0, len(t), 9):
        r = t[i + 2 : i + 9]
        append(
            {
                "altitude": int(r[0]),
                "surveillance_status": int(r[1]),
                "nic_b": int(r[2]),
                "cpr_format": int(r[3]),
                "cpr_lat": int(r[4]),
                "cpr_lon": int(r[5]),
                "nuc_p": int(r[6]),
            }
        )
    return result


def b05_astype(out):
    """Convert integer slots with one astype, then slice Python ints."""
    it = out.astype(np.int64).tolist()
    result = []
    append = result.append
    for i in range(0, len(it), 9):
        r = it[i + 2 : i + 9]
        append(
            {
                "altitude": r[0],
                "surveillance_status": r[1],
                "nic_b": r[2],
                "cpr_format": r[3],
                "cpr_lat": r[4],
                "cpr_lon": r[5],
                "nuc_p": r[6],
            }
        )
    return result


def b05_astype_slice_cols(out):
    """Convert only the integer columns, as a (n, width-2) block."""
    f = out.reshape(-1, 9)
    it = f[:, 2:].astype(np.int64).tolist()
    result = []
    append = result.append
    for r in it:
        append(
            {
                "altitude": r[0],
                "surveillance_status": r[1],
                "nic_b": r[2],
                "cpr_format": r[3],
                "cpr_lat": r[4],
                "cpr_lon": r[5],
                "nuc_p": r[6],
            }
        )
    return result


def b05_float_int_slice(out):
    """tolist() then index without slicing the mask columns."""
    t = out.reshape(-1, 9).tolist()
    result = []
    append = result.append
    for r in t:
        append(
            {
                "altitude": int(r[2]),
                "surveillance_status": int(r[3]),
                "nic_b": int(r[4]),
                "cpr_format": int(r[5]),
                "cpr_lat": int(r[6]),
                "cpr_lon": int(r[7]),
                "nuc_p": int(r[8]),
            }
        )
    return result


def masks_astype(out):
    """The masks too, so the per-row test is an int compare, not int()."""
    return out.reshape(-1, 9)[:, 0:2].astype(np.int64).tolist()


def masks_float(out):
    return out.reshape(-1, 9)[:, 0:2].tolist()


def main() -> None:
    rng = np.random.default_rng(3)
    f = rng.integers(0, 100000, size=(N, 9)).astype(np.float64)
    f[:, 0] = 0x7F
    f[:, 1] = 0
    flat = np.ascontiguousarray(f).reshape(-1)

    print("bds05 (7 int fields, all present, nothing None)")
    row("tolist + int() per field", best_of(lambda: b05_float_int(flat)))
    row("tolist reshape + int() per field", best_of(lambda: b05_float_int_slice(flat)))
    row("astype(int64).tolist() whole buffer", best_of(lambda: b05_astype(flat)))
    row("astype(int64) on the int columns only", best_of(lambda: b05_astype_slice_cols(flat)))

    print("mask extraction")
    row("masks via float tolist", best_of(lambda: masks_float(flat)))
    row("masks via astype(int64).tolist", best_of(lambda: masks_astype(flat)))

    print("all-present precheck")
    row("any(x for x in flat[1::9])", best_of(lambda: any(x for x in flat[1::9].tolist())))
    row("flat[:,1].any() (numpy)", best_of(lambda: bool((flat.reshape(-1, 9)[:, 1] != 0).any())))


if __name__ == "__main__":
    main()
