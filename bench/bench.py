"""mojo-pymodes against pyModeS 3.6.0 on the same data.

Upstream decodes one message at a time, so every case here is "N things,
give me N results" on both sides: upstream gets its own public function in a
Python loop, this repo gets the batch call. The inputs are built once, before
the clock starts, on both sides, so neither is charged for the other's setup.

Run it through `pixi run bench`: the task takes a machine-wide flock so a
concurrent factory job cannot distort the numbers.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import pyModeS  # noqa: E402
from pyModeS import util as ref_util  # noqa: E402
from pyModeS._altcode import altcode_to_altitude as ref_altcode  # noqa: E402
from pyModeS._bits import crc_remainder as ref_crc  # noqa: E402
from pyModeS._callsign import decode_callsign as ref_callsign  # noqa: E402
from pyModeS.decoder.bds.bds05 import decode_bds05 as ref_bds05  # noqa: E402
from pyModeS.decoder.bds.bds08 import decode_bds08 as ref_bds08  # noqa: E402
from pyModeS.decoder.bds.bds09 import decode_bds09 as ref_bds09  # noqa: E402
from pyModeS.decoder.bds.bds62 import decode_bds62 as ref_bds62  # noqa: E402
from pyModeS.position._cpr import airborne_position_pair as ref_pair  # noqa: E402
from pyModeS.position._cpr import airborne_position_with_ref as ref_with_ref  # noqa: E402
from pyModeS.position._cpr import cprNL as ref_nl  # noqa: E402

import mojopymodes as pm  # noqa: E402
from mojopymodes.batch import decode_payloads  # noqa: E402
from mojopymodes._bits import crc_remainders  # noqa: E402
from mojopymodes._altcode import altcodes_to_altitudes  # noqa: E402
from mojopymodes._callsign import decode_callsigns  # noqa: E402
from mojopymodes.position._cpr import (  # noqa: E402
    airborne_position_pair_batch,
    airborne_position_with_ref_batch,
    cprNL_batch,
)

import corpus  # noqa: E402

REPEAT = 3
N_BIG = 1_000_000
N_MED = 200_000
N_SMALL = 100_000


def timeit(fn, repeat: int = REPEAT) -> float:
    """Best-of-`repeat` wall time in seconds."""
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ----------------------------------------------------------------- fixtures


def message_stream(n: int) -> list[str]:
    """`n` distinct valid DF17/18 messages."""
    base = corpus.long_corpus()
    return [base[i % len(base)] for i in range(n)]


def payload_stream(n: int) -> list[int]:
    msgs = message_stream(n)
    return [int(m[8:22], 16) for m in msgs]


def typecode_stream(n: int) -> list[int]:
    msgs = message_stream(n)
    return [int(m[8:10], 16) >> 3 for m in msgs]


MSGS = message_stream(N_BIG)
PAYLOADS = payload_stream(N_BIG)
TYPECODES = typecode_stream(N_BIG)
ALTCODES = [(i * 7919) % 8192 for i in range(N_BIG)]
CALLSIGN_BITS = [(i * 0x9E3779B97F4A7C15) % (1 << 48) for i in range(N_BIG)]
LATS = [(-90.0 + 180.0 * ((i * 7919) % 100003) / 100003.0) for i in range(N_BIG)]
REF_LATS = [(-85.0 + 170.0 * ((i * 104729) % 99991) / 99991.0) for i in range(N_BIG)]
REF_LONS = [(-180.0 + 360.0 * ((i * 15485863) % 99989) / 99989.0) for i in range(N_BIG)]
CPR_EVEN_LAT = [(i * 2654435761) % 131072 for i in range(N_BIG)]
CPR_EVEN_LON = [(i * 40503 * 7919) % 131072 for i in range(N_BIG)]
CPR_ODD_LAT = [(i * 22695477 + 17) % 131072 for i in range(N_BIG)]
CPR_ODD_LON = [(i * 69069 + 31) % 131072 for i in range(N_BIG)]
CPR_FMT = [i & 1 for i in range(N_BIG)]

# Upstream's ADSB dispatch, applied one message at a time.
from pyModeS.decoder.adsb import _ADSB_DISPATCH  # noqa: E402


def upstream_adsb_batch(items):
    out = []
    for p, tc in items:
        entry = _ADSB_DISPATCH.get(tc)
        if entry is None:
            out.append({"typecode": tc})
            continue
        bds, decoder = entry
        row = {"typecode": tc, "bds": bds}
        if bds == "0,5":
            row.update(decoder(p, tc=tc))
        else:
            row.update(decoder(p))
        out.append(row)
    return out


# -------------------------------------------------------------------- cases

CASES = []


def case(name, note=""):
    def deco(fn):
        CASES.append((name, note, fn))
        return fn

    return deco


@case("crc_remainder (1M messages)")
def _():
    return (
        lambda: crc_remainders(MSGS),
        lambda: [ref_crc(int(m, 16), len(m) * 4) for m in MSGS],
    )


@case("df + icao + typecode + crc (1M messages)", "header decode, incl. hex parse")
def _():
    return (
        lambda: pm.util.headers(MSGS),
        lambda: [
            (ref_util.df(m), ref_util.icao(m), ref_util.typecode(m), ref_util.crc(m))
            for m in MSGS
        ],
    )


@case("altcode_to_altitude (1M codes)")
def _():
    return (
        lambda: altcodes_to_altitudes(ALTCODES),
        lambda: [ref_altcode(a) for a in ALTCODES],
    )


@case("decode_callsign (1M payloads)")
def _():
    return (
        lambda: decode_callsigns(CALLSIGN_BITS),
        lambda: [ref_callsign(b) for b in CALLSIGN_BITS],
    )


@case("cprNL (1M latitudes)")
def _():
    return (
        lambda: cprNL_batch(LATS),
        lambda: [ref_nl(v) for v in LATS],
    )


@case("airborne_position_with_ref (1M frames)")
def _():
    return (
        lambda: airborne_position_with_ref_batch(
            CPR_FMT, CPR_EVEN_LAT, CPR_EVEN_LON, REF_LATS, REF_LONS
        ),
        lambda: [
            ref_with_ref(f, a, b, la, lo)
            for f, a, b, la, lo in zip(
                CPR_FMT, CPR_EVEN_LAT, CPR_EVEN_LON, REF_LATS, REF_LONS
            )
        ],
    )


@case("airborne_position_pair (1M pairs)", "even/odd CPR, the real position fix")
def _():
    return (
        lambda: airborne_position_pair_batch(
            CPR_EVEN_LAT, CPR_EVEN_LON, CPR_ODD_LAT, CPR_ODD_LON, CPR_FMT
        ),
        lambda: [
            ref_pair(a, b, c, d, even_is_newer=bool(f))
            for a, b, c, d, f in zip(
                CPR_EVEN_LAT, CPR_EVEN_LON, CPR_ODD_LAT, CPR_ODD_LON, CPR_FMT
            )
        ],
    )


@case("BDS 0,5 airborne position (1M payloads)")
def _():
    return (
        lambda: pm.decoder.bds.decode_bds05_batch(PAYLOADS, TYPECODES),
        lambda: [ref_bds05(p, tc=t) for p, t in zip(PAYLOADS, TYPECODES)],
    )


@case("BDS 0,8 identification (1M payloads)")
def _():
    return (
        lambda: pm.decoder.bds.decode_bds08_batch(PAYLOADS),
        lambda: [ref_bds08(p) for p in PAYLOADS],
    )


@case("BDS 0,9 airborne velocity (1M payloads)")
def _():
    return (
        lambda: pm.decoder.bds.decode_bds09_batch(PAYLOADS),
        lambda: [ref_bds09(p) for p in PAYLOADS],
    )


@case("BDS 6,2 target state (1M payloads)")
def _():
    return (
        lambda: pm.decoder.bds.decode_bds62_batch(PAYLOADS),
        lambda: [ref_bds62(p) for p in PAYLOADS],
    )


@case("full ADS-B decode (200k messages)", "typecode dispatch to field dict")
def _():
    items = list(zip(PAYLOADS[:N_MED], TYPECODES[:N_MED]))
    return (
        lambda: decode_payloads(items),
        lambda: upstream_adsb_batch(items),
    )


@case("single cprNL call (100k calls)", "ctypes round trip, one message at a time")
def _():
    lats = LATS[:N_SMALL]
    return (
        lambda: [pm.cprNL(v) for v in lats],
        lambda: [ref_nl(v) for v in lats],
    )


# --------------------------------------------------------------------- main


def main() -> None:
    print(f"pyModeS {pyModeS.__version__} vs mojo-pymodes {pm.__version__}")
    print(
        f"{platform.platform()} | {platform.processor() or platform.machine()} | "
        f"python {platform.python_version()} | numpy {np.__version__}"
    )
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    print(f"cpu: {line.split(':', 1)[1].strip()}")
                    break
    except OSError:
        pass
    print()
    print("| case | mojo-pymodes | pyModeS | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for name, _note, fn in CASES:
        ours, theirs = fn()
        a = timeit(ours)
        b = timeit(theirs)
        flag = ""
        if b / a < 1.0:
            flag = " slower"
        print(f"| {name} | {a * 1e3:.1f} ms | {b * 1e3:.1f} ms | {b / a:.2f}x{flag} |")
    print()
    print("ratio is pyModeS / mojo-pymodes, so above 1.0 means this repo is faster.")


if __name__ == "__main__":
    main()
