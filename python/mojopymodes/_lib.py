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
_DEFAULT_LIB = os.path.join(ROOT, "dist", "libmojo-pymodes.so")
# The library path is read at call time, so MOJOPYMODES_LIB works whether it is
# set before import (the usual case) or by a test afterwards.
LIB = os.environ.get("MOJOPYMODES_LIB") or _DEFAULT_LIB


def library_path() -> str:
    """Where the shared library is expected: the override, or `dist/`."""
    return os.environ.get("MOJOPYMODES_LIB") or LIB

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


def _toolchain_home(mojo_bin: str) -> str | None:
    """The package root (`share/max`) for a `mojo` from a conda environment.

    The conda package's activation script exports
    `MODULAR_HOME=$CONDA_PREFIX/share/max`, which is where the compiler looks
    for `std` and `max`. `mojo` lives in `<prefix>/bin`, so the prefix is its
    grandparent directory. A prefix can be usable even before the first run
    creates `share/max`, so its existence is not the test; a missing
    `MODULAR_HOME` or one pointing at a path that does not exist is.
    """
    prefix = os.path.dirname(os.path.dirname(os.path.realpath(mojo_bin)))
    if not os.path.isfile(os.path.join(prefix, "lib", "mojo", "algorithm.mojoc")):
        return None
    return os.path.join(prefix, "share", "max")


def _toolchain_env(mojo_bin: str) -> dict[str, str]:
    """`os.environ` with `MODULAR_HOME` pointed at `mojo_bin`'s toolchain."""
    env = dict(os.environ)
    home = _toolchain_home(mojo_bin)
    if home is not None:
        env["MODULAR_HOME"] = home
    return env


def mojo_command() -> list[str]:
    """The command prefix that compiles one Mojo file.

    `mojo build` on its own has no `std` outside the environment it shipped
    in, so a `mojo` from a conda prefix is run with that prefix's
    `MODULAR_HOME`, and `pixi run` is the fallback that sets it up.
    """
    override = os.environ.get("MOJOPYMODES_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found and _toolchain_home(found) is not None:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        # `--` keeps pixi's own flag parsing out of the mojo arguments.
        return [
            pixi,
            "run",
            "--manifest-path",
            os.path.join(ROOT, "pixi.toml"),
            "--",
            "mojo",
        ]
    if found:
        return [found]
    raise BuildError("mojo not found; set MOJOPYMODES_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-pymodes.so` if it is stale.

    An explicit `MOJOPYMODES_LIB` is used as given: it is a library the caller
    already built, so there is nothing to rebuild (unless `force`), and a
    packaged install has no `src/` next to it to rebuild from.
    """
    target = library_path()
    if os.environ.get("MOJOPYMODES_LIB") and not force:
        if not os.path.isfile(target):
            raise BuildError(
                f"MOJOPYMODES_LIB points at {target}, which does not exist"
            )
        return target
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not sources:
        raise BuildError(
            f"no Mojo sources under {SRC}; build the library first and point "
            "MOJOPYMODES_LIB at it"
        )
    if not force and os.path.exists(target):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(target) >= newest:
            return target
    os.makedirs(os.path.dirname(target), exist_ok=True)
    cmd = mojo_command()
    env = _toolchain_env(cmd[0]) if os.path.basename(cmd[0]).startswith("mojo") else None
    cmd = cmd + [
        "build",
        "--emit",
        "shared-lib",
        "-I",
        SRC,
        os.path.join(SRC, "capi.mojo"),
        "-o",
        target,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    if proc.returncode != 0 or not os.path.exists(target):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return target


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


_ARRAY = (list, tuple, np.ndarray)


def _count(value) -> int:
    """Elements of `value` that the batch wrappers are about to read."""
    return len(value) if isinstance(value, _ARRAY) else 1


def same_length(name: str, *columns) -> None:
    """Refuse batch columns that are not all the same length.

    The kernels read `n` elements from every column. A scalar or a shorter
    column would be read past its end, silently, so it is rejected here.
    """
    counts = {_count(c) for c in columns}
    if len(counts) > 1:
        raise ValueError(
            f"{name}: columns of different length {sorted(counts)}; "
            "pass one value per message"
        )

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
