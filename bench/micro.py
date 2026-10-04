"""Microbenchmarks for the candidate Python-side list builders.

Run under the bench flock; these are pure CPython numbers.

    flock /tmp/mojo-bench.lock pixi run python bench/micro.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojopymodes._layout import LAYOUTS  # noqa: E402

N = 1_000_000
REPEAT = 5


def best_of(fn, repeat=REPEAT):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def row(label, seconds):
    print(f"{label:46s}: {seconds * 1e3:9.1f} ms", flush=True)


# --------------------------------------------------------------------- rows()

_nucp = LAYOUTS["bds05"].fields[6].name


def rows_bitwise(name, out):
    layout = LAYOUTS[name]
    width = layout.width
    table = out.tolist()
    result = []
    append = result.append
    for i in range(0, len(table), width):
        row_ = table[i : i + width]
        present = int(row_[0])
        if present == 0x7F:
            v = row_[2]
            if v:
                append(
                    {
                        "altitude": int(v),
                        "surveillance_status": int(row_[3]),
                        "nic_b": int(row_[4]),
                        "cpr_format": int(row_[5]),
                        "cpr_lat": int(row_[6]),
                        "cpr_lon": int(row_[7]),
                        "nuc_p": int(row_[8]),
                    }
                )
            else:
                append(
                    {
                        "altitude": None,
                        "surveillance_status": int(row_[3]),
                        "nic_b": int(row_[4]),
                        "cpr_format": int(row_[5]),
                        "cpr_lat": int(row_[6]),
                        "cpr_lon": int(row_[7]),
                        "nuc_p": int(row_[8]),
                    }
                )
        else:
            item = {}
            if present & 1:
                item["altitude"] = int(row_[2])
            if present & 2:
                item["surveillance_status"] = int(row_[3])
            if present & 4:
                item["nic_b"] = int(row_[4])
            if present & 8:
                item["cpr_format"] = int(row_[5])
            if present & 16:
                item["cpr_lat"] = int(row_[6])
            if present & 32:
                item["cpr_lon"] = int(row_[7])
            if present & 64:
                item["nuc_p"] = int(row_[8])
            append(item)
    return result


def rows_generic_fast(name, out):
    """Generic loop, but hoisting the mask test and using tuples."""
    layout = LAYOUTS[name]
    width = layout.width
    table = out.tolist()
    plan = []
    for f in layout.fields:
        if f.kind == "i":
            plan.append((f.name, int, f.slot, f.bit))
        elif f.kind == "f":
            plan.append((f.name, float, f.slot, f.bit))
        else:
            plan.append((f.name, bool, f.slot, f.bit))
    result = []
    append = result.append
    for i in range(0, len(table), width):
        r = table[i : i + width]
        present = int(r[0])
        absent = int(r[1])
        if absent:
            item = {}
            for field, coerce, slot, bit in plan:
                if (present >> bit) & 1:
                    item[field] = coerce(r[slot]) if not (absent >> bit) & 1 else None
            append(item)
        else:
            item = {}
            for field, coerce, slot, bit in plan:
                if (present >> bit) & 1:
                    item[field] = coerce(r[slot])
            append(item)
    return result


def rows_existing(name, out):
    from mojopymodes._layout import rows as _rows

    return _rows(name, out)


_SRC = ("N/A", "MCP/FCU", "FMS")


def rows_bitwise62(out):
    """bds62 all-present fast path, sources mapped by index rather than a lambda."""
    w = LAYOUTS["bds62"].width
    t = out.tolist()
    result = []
    append = result.append
    for i in range(0, len(t), w):
        r = t[i : i + w]
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


def rows_bitwise62_none(out):
    """Same, but with an `is not None` test instead of `bool()`."""
    w = LAYOUTS["bds62"].width
    t = out.tolist()
    result = []
    append = result.append
    for i in range(0, len(t), w):
        r = t[i : i + w]
        a = r[10]
        b = r[11]
        c = r[12]
        d = r[13]
        e = r[14]
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
                "autopilot": a is not None,
                "vnav_mode": b is not None,
                "altitude_hold_mode": c is not None,
                "approach_mode": d is not None,
                "lnav_mode": e is not None,
                "tcas_operational": bool(r[15]),
            }
        )
    return result


def main() -> None:
    rng = np.random.default_rng(7)

    def flat(width, present, absent=0):
        a = rng.integers(0, 1000, size=(N, width)).astype(np.float64)
        a[:, 0] = present
        a[:, 1] = absent
        return np.ascontiguousarray(a).reshape(-1)

    o05 = flat(LAYOUTS["bds05"].width, 0x7F)
    o09 = flat(LAYOUTS["bds09"].width, 0x7FF)
    o62 = flat(LAYOUTS["bds62"].width, 0x3FFF)
    o62.reshape(-1, LAYOUTS["bds62"].width)[:, 4] = rng.integers(0, 3, size=N)  # selected_altitude_source
    o62.reshape(-1, LAYOUTS["bds62"].width)[:, 5] = 1000.0  # baro_pressure_setting (float)

    print("bds05 rows() variants")
    row("existing generic", best_of(lambda: rows_existing("bds05", o05)))
    row("per-layout bitwise, all-present fast path", best_of(lambda: rows_bitwise("bds05", o05)))
    row("generic with absent fast path", best_of(lambda: rows_generic_fast("bds05", o05)))

    print("bds09 rows() variants (4 lamdas + 3 index tests)")
    row("existing generic", best_of(lambda: rows_existing("bds09", o09)))
    row("generic with absent fast path", best_of(lambda: rows_generic_fast("bds09", o09)))

    print("bds62 rows() variants")
    row("existing generic", best_of(lambda: rows_existing("bds62", o62)))
    row("generic with absent fast path", best_of(lambda: rows_generic_fast("bds62", o62)))
    row("per-layout bitwise, all-present fast path", best_of(lambda: rows_bitwise62(o62)))
    row("per-layout bitwise, `is not None` bools", best_of(lambda: rows_bitwise62_none(o62)))

    # ------------------------------------------------------------ altcode
    print("altcode list build")
    flat = np.empty(N * 2, dtype=np.float64)
    flat[0::2] = rng.integers(-1000, 100000, size=N)
    flat[1::2] = 1.0
    f = flat.tolist()
    row("reshape rows2d (current)", best_of(lambda: [int(r[0]) if r[1] else None for r in np.asarray(flat).reshape(-1, 2)]))
    row("tolist + slices", best_of(lambda: [int(v) if p else None for v, p in zip(f[0::2], f[1::2])]))

    # ------------------------------------------------------------ callsign
    print("callsign list build")
    big = np.tile(np.arange(8, dtype=np.uint8), N // 8 + 1)[: N * 8].reshape(N, 8)
    block = big
    blob = big.tobytes()
    row("reshape rows (current)", best_of(lambda: [bytes(r).split(b"\x00", 1)[0].decode("ascii") for r in block]))
    row("blob slice per row", best_of(lambda: [blob[i : i + 8].split(b"\x00", 1)[0].decode("ascii") for i in range(N)]))

    # ------------------------------------------------------------ to_words
    print("to_words")
    import corpus  # noqa: E402

    base = corpus.long_corpus()
    msgs = [base[i % len(base)] for i in range(N)]
    from mojopymodes._lib import to_words

    row("current", best_of(lambda: to_words(msgs), 3))
    row("current, no astype", best_of(lambda: to_words_noastype(msgs), 3))


def to_words_noastype(msgs):
    from mojopymodes._lib import i64

    lengths = np.array([len(m) for m in msgs], dtype=np.int64)
    width = int(lengths.max())
    padded = (width + 15) // 16 * 16
    blob = "".join(m.rjust(padded, "0") for m in msgs).encode("ascii")
    nibbles = np.frombuffer(blob, dtype=np.uint8).reshape(len(msgs), padded)
    values = np.where(nibbles <= 57, nibbles - 48, nibbles - 55).astype(np.uint64)
    words = []
    for c in range(padded // 16):
        chunk = values[:, c * 16 : (c + 1) * 16]
        word = np.zeros(len(msgs), dtype=np.uint64)
        for j in range(16):
            word = (word << np.uint64(4)) | chunk[:, j]
        words.append(word)
    hi = words[0] if len(words) > 1 else np.zeros(len(msgs), dtype=np.uint64)
    return np.ascontiguousarray(hi, dtype=np.int64), np.ascontiguousarray(words[-1], dtype=np.int64), lengths * 4


if __name__ == "__main__":
    main()
