"""Which Python row-assembly shape is fastest?

Compares the strategies for turning one flat float64 row into the dict upstream
returns:

  generic    the current `rows()`: per field, a presence bit test, a None bit
             test, a coercion call and an optional transform call
  const_map  a hand-written dict literal (BUILD_CONST_KEY_MAP, no hashing)
  zip_map    dict(zip(NAMES, values)) over a prebuilt value list
  intview    as const_map, but integer slots are read through an int64 view of
             the same buffer and tolist()'d in one go, so no per-row int()

Run under the bench flock.

    flock /tmp/mojo-bench.lock pixi run python bench/micro_rows.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojopymodes._layout import LAYOUTS, rows  # noqa: E402

N = 1_000_000
REPEAT = 5
_SRC = ("N/A", "MCP/FCU", "FMS")


def best_of(fn, repeat=REPEAT):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def row(label, seconds):
    print(f"{label:44s}: {seconds * 1e3:9.1f} ms  ({seconds / N * 1e9:6.0f} ns/row)", flush=True)


B05_NAMES = ("altitude", "surveillance_status", "nic_b", "cpr_format", "cpr_lat", "cpr_lon", "nuc_p")


def b05_const_map(out):
    t = out.tolist()
    result = []
    append = result.append
    for i in range(0, len(t), 9):
        r = t[i : i + 9]
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


def b05_zip_map(out):
    t = out.tolist()
    result = []
    append = result.append
    for i in range(0, len(t), 9):
        append(dict(zip(B05_NAMES, map(int, t[i + 2 : i + 9]))))
    return result


def b05_intview(out):
    """Integer slots via an int64 view, so int() is not called per element."""
    ints = out.view(np.int64)
    n = len(ints) // 9
    # one tolist over the whole int64 buffer, then slice
    t = ints.tolist()
    result = []
    append = result.append
    for i in range(0, n * 9, 9):
        r = t[i : i + 9]
        append(
            {
                "altitude": r[2],
                "surveillance_status": r[3],
                "nic_b": r[4],
                "cpr_format": r[5],
                "cpr_lat": r[6],
                "cpr_lon": r[7],
                "nuc_p": r[8],
            }
        )
    return result


def b05_intview_zip(out):
    t = out.view(np.int64).tolist()
    result = []
    append = result.append
    for i in range(0, len(t), 9):
        append(dict(zip(B05_NAMES, t[i + 2 : i + 9])))
    return result


def b62_const_map(out):
    t = out.tolist()
    result = []
    append = result.append
    for i in range(0, len(t), 16):
        r = t[i : i + 16]
        append(
            {
                "subtype": int(r[2]),
                "selected_altitude": int(r[3]),
                "selected_altitude_source": _SRC[int(r[4])],
                "baro_pressure_setting": float(r[5]),
                "selected_heading": float(r[6]),
                "nac_p": int(r[7]),
                "nic_baro": int(r[8]),
                "sil": int(r[9]),
                "autopilot": bool(r[10]),
                "vnav_mode": bool(r[11]),
                "altitude_hold_mode": bool(r[12]),
                "approach_mode": bool(r[13]),
                "lnav_mode": bool(r[14]),
                "tcas_operational": bool(r[15]),
            }
        )
    return result


def b62_split(out):
    """Int slots read as ints, float slots read as floats, one buffer each."""
    ints = out.view(np.int64)
    w = 16
    n = len(ints) // w
    it = ints.tolist()
    result = []
    append = result.append
    for i in range(n):
        b = i * w
        f = it[b]
        r = it[b + 2 : b + 16]
        append(
            {
                "subtype": r[0],
                "selected_altitude": r[1],
                "selected_altitude_source": _SRC[r[2]],
                "baro_pressure_setting": f,
                "selected_heading": out[out.__class__].__class__ and 0.0,
                "nac_p": r[4],
                "nic_baro": r[5],
                "sil": r[6],
                "autopilot": r[7] is not None,
                "vnav_mode": r[8] is not None,
                "altitude_hold_mode": r[9] is not None,
                "approach_mode": r[10] is not None,
                "lnav_mode": r[11] is not None,
                "tcas_operational": r[12] is not None,
            }
        )
    return result


def b62_zip_mixed(out):
    t = out.tolist()
    result = []
    append = result.append
    for i in range(0, len(t), 16):
        r = t[i + 2 : i + 16]
        append(
            dict(
                zip(
                    B62_NAMES,
                    (
                        int(r[0]),
                        int(r[1]),
                        _SRC[int(r[2])],
                        r[3],
                        r[4],
                        int(r[5]),
                        int(r[6]),
                        int(r[7]),
                        r[8] is not None,
                        r[9] is not None,
                        r[10] is not None,
                        r[11] is not None,
                        r[12] is not None,
                        r[13] is not None,
                    ),
                )
            )
        )
    return result


B62_NAMES = (
    "subtype",
    "selected_altitude",
    "selected_altitude_source",
    "baro_pressure_setting",
    "selected_heading",
    "nac_p",
    "nic_baro",
    "sil",
    "autopilot",
    "vnav_mode",
    "altitude_hold_mode",
    "approach_mode",
    "lnav_mode",
    "tcas_operational",
)


def main() -> None:
    rng = np.random.default_rng(7)

    def fixture(width, present):
        a = rng.integers(0, 100000, size=(N, width)).astype(np.float64)
        a[:, 0] = present
        a[:, 1] = 0
        return np.ascontiguousarray(a).reshape(-1)

    o05 = fixture(LAYOUTS["bds05"].width, 0x7F)
    o62 = fixture(LAYOUTS["bds62"].width, 0x3FFF)
    o62.reshape(-1, 16)[:, 4] = rng.integers(0, 3, size=N)
    o62.reshape(-1, 16)[:, 5] = 1000.25
    o62.reshape(-1, 16)[:, 6] = 270.5

    print("bds05 (7 int fields)")
    row("existing generic", best_of(lambda: rows("bds05", o05)))
    row("const-map literal + int()", best_of(lambda: b05_const_map(o05)))
    row("dict(zip(names, map(int, slice)))", best_of(lambda: b05_zip_map(o05)))
    row("int64 view, const-map, no int()", best_of(lambda: b05_intview(o05)))
    row("int64 view, dict(zip(...))", best_of(lambda: b05_intview_zip(o05)))

    print("bds62 (8 int, 3 float-source, 6 bool)")
    row("existing generic", best_of(lambda: rows("bds62", o62)))
    row("const-map literal + int()/float()/bool()", best_of(lambda: b62_const_map(o62)))
    row("dict(zip(names, tuple))", best_of(lambda: b62_zip_mixed(o62)))


if __name__ == "__main__":
    main()
