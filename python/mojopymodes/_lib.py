"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOPYMODES_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pymodes.so"
)

# Mirrors the PAR_THRESHOLD / PAR_WORKERS constants in src/capi.mojo: below the
# threshold a batch runs on one thread, at or above it the kernel fans out
# across PAR_WORKERS workers.
PAR_THRESHOLD = 4096
PAR_WORKERS = 8

I = ctypes.c_int64
F = ctypes.c_double
V = ctypes.c_void_p

# name -> (argtypes, restype)
_SIGNATURES = {
    "mp_extract_unsigned": ([I, I, I, I], I),
    "mp_extract_signed": ([I, I, I, I], I),
    "mp_crc_remainder": ([I, I], I),
    "mp_crc_remainder_wide": ([I, I, I], I),
    "mp_crc_remainder_batch": ([I, I, I, I, I], None),
    "mp_altcode_to_altitude": ([I, I], I),
    "mp_altcode_to_altitude_batch": ([I, I, I], None),
    "mp_gray2int": ([I], I),
    "mp_decode_callsign": ([I, I], I),
    "mp_decode_callsign_batch": ([I, I, I], I),
    "mp_is_valid_callsign_char": ([I], I),
    "mp_idcode_to_squawk": ([I, I], I),
    "mp_idcode_to_squawk_batch": ([I, I, I], I),
    "mp_cprNL": ([F], I),
    "mp_cprNL_batch": ([I, I, I], None),
    "mp_airborne_position_with_ref": ([I, I, I, F, F, I], I),
    "mp_airborne_position_with_ref_batch": ([I, I, I, I, I, I, I], None),
    "mp_airborne_position_pair": ([I, I, I, I, I, I], I),
    "mp_airborne_position_pair_batch": ([I, I, I, I, I, I, I], None),
    "mp_surface_position_with_ref": ([I, I, I, F, F, I], I),
    "mp_surface_position_with_ref_batch": ([I, I, I, I, I, I, I], None),
    "mp_surface_position_pair": ([I, I, I, I, F, F, I, I], I),
    "mp_surface_position_pair_batch": ([I] * 9, None),
    "mp_signed": ([I, I, I], I),
    "mp_normalise_angle": ([F], F),
    "mp_wrong_status": ([I, I, I, I], I),
    "mp_is_bds30": ([I], I),
    "mp_wake_vortex_category": ([I, I], I),
    "mp_decode_header": ([I, I, I, I, I], None),
    "mp_decode_bds05": ([I, I, I, I], None),
    "mp_decode_bds06": ([I, I, I], None),
    "mp_decode_bds08": ([I, I, I, I], I),
    "mp_decode_bds09": ([I, I, I], None),
    "mp_decode_bds30": ([I, I, I, I], I),
    "mp_decode_bds61": ([I, I, I, I], I),
    "mp_decode_bds62": ([I, I, I], None),
    "mp_decode_bds65": ([I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOPYMODES_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [
            pixi,
            "run",
            "--manifest-path",
            os.path.join(ROOT, "pixi.toml"),
            "mojo",
        ]
    raise BuildError("mojo not found; set MOJOPYMODES_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-pymodes.so` if it is stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        "-I",
        SRC,
        os.path.join(SRC, "capi.mojo"),
        "-o",
        LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


# --------------------------------------------------------------- buffers


def i64(a) -> np.ndarray:
    """A C-contiguous int64 view of `a`, copying only when it has to."""
    return np.ascontiguousarray(a, dtype=np.int64)


def f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def out_f64(n: int, width: int) -> np.ndarray:
    return np.zeros(n * width, dtype=np.float64)


def out_u8(n: int, width: int) -> np.ndarray:
    return np.zeros(n * width, dtype=np.uint8)


def scratch_2f() -> np.ndarray:
    return np.zeros(2, dtype=np.float64)


def scratch_16u8() -> np.ndarray:
    return np.zeros(16, dtype=np.uint8)

# A 256-entry table turning a hex ASCII byte into its nibble. Covers 0-9, A-F
# and a-f; the remaining byte values map to 0, which is what a padded (and
# therefore meaningless) position needs anyway.
_HEX_LUT = np.frombuffer(
    bytes(
        (c - 48) if 48 <= c <= 57 else ((c - 55) if 65 <= c <= 70 else ((c - 87) & 0xFF))
        for c in range(256)
    ),
    dtype=np.uint8,
)

# 16^15 .. 16^0, so a dot product against a row of 16 nibbles folds them into a
# word. uint64 arithmetic wraps mod 2**64, which is exactly what the shift-or
# loop computes, so the two agree bit for bit.
_POW16 = (np.uint64(16) ** np.arange(15, -1, -1)).astype(np.uint64)


def to_words(msgs: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hex messages to (hi, lo, length) int64 arrays.

    Vectorised with numpy: a per-message `int(x, 16)` in Python would cost
    more than the decode it feeds, and it would flatter neither side of the
    benchmark.

    Each 16-nibble chunk of the padded message is folded into a word with one
    uint64 dot product against the powers of sixteen rather than sixteen
    shift-or passes over the array; on a million messages that is the
    difference between about 840 ms and about 100 ms for the fold.
    """
    if not msgs:
        return i64([]), i64([]), i64([])
    lengths = np.array([len(m) for m in msgs], dtype=np.int64)
    width = int(lengths.max())
    padded = (width + 15) // 16 * 16
    blob = "".join(m.rjust(padded, "0") for m in msgs).encode("ascii")
    nibbles = _HEX_LUT[np.frombuffer(blob, dtype=np.uint8).reshape(len(msgs), padded)]

    words = []
    for c in range(padded // 16):
        words.append(nibbles[:, c * 16 : (c + 1) * 16].astype(np.uint64) @ _POW16)

    # words[0] holds the most significant 64 bits; a 56-bit message has
    # only one chunk and therefore hi == 0.
    hi = words[0] if len(words) > 1 else np.zeros(len(msgs), dtype=np.uint64)
    return (
        hi.astype(np.int64),
        words[-1].astype(np.int64),
        lengths * 4,
    )


def main() -> int:
    """`python -m mojopymodes._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
