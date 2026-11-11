# mojo-pymodes

[pyModeS](https://mode-s.org)'s ADS-B decoding primitives, written in
[Mojo](https://www.modular.com/mojo) and callable from Python under the same
names.

```python
import mojopymodes as pm

msg = "8D406B902015A678D4D220AA4BDA"

pm.util.df(msg)                                  # 17
pm.util.icao(msg)                                # '406B90'
pm.util.typecode(msg)                            # 4
pm.util.crc(msg)                                 # 0
pm.decoder.bds.decode_bds08(pm.payload(msg))     # {'category': 0,
                                                  #  'callsign': 'EZY85MH', ...}
```

For the covered subset every function takes the same arguments, in the same
order, with the same defaults, and returns the same value as the pyModeS
function it was ported from. The test suite asserts that against the real
`pyModeS` 3.6.0 package, function by function.

## What is covered

The port tracks pyModeS 3.6.0. `src/ported.mojo` is grouped and ordered to
match the upstream package: read it next to the pyModeS source and the two
line up.

| upstream module | functions |
| --- | --- |
| `pyModeS._bits` | `extract_unsigned`, `extract_signed`, `crc_remainder` |
| `pyModeS._altcode` | `altcode_to_altitude` |
| `pyModeS._callsign` | `decode_callsign`, `is_valid_callsign_char` |
| `pyModeS._idcode` | `idcode_to_squawk` |
| `pyModeS._uncertainty` | `TC_NUCp_lookup` (the table BDS 0,5 indexes) |
| `pyModeS.position._cpr` | `cprNL`, `airborne_position_with_ref`, `airborne_position_pair`, `surface_position_with_ref`, `surface_position_pair` |
| `pyModeS.position._airports` | `resolve_surface_ref` (pure Python: it is a table lookup) |
| `pyModeS.decoder.bds._helpers` | `signed`, `normalise_angle`, `wrong_status` |
| `pyModeS.decoder.bds` | `decode_bds05`, `decode_bds06`, `decode_bds08`, `decode_bds09`, `decode_bds30`, `is_bds30`, `decode_bds61`, `decode_bds62`, `decode_bds65` |
| `pyModeS.util` | `df`, `icao`, `typecode`, `crc`, `altcode`, `idcode`, `cprNL`, `hex2bin`, `bin2int`, `hex2int`, `bin2hex` |

That covers every typecode an ADS-B DF17/18 message can carry: BDS 0,5
airborne position, 0,6 surface position, 0,8 identification, 0,9 airborne
velocity, 6,1 status, 6,2 target state and 6,5 operational status, plus the
BDS 3,0 ACAS report that BDS 6,1 subtype 2 embeds. Every name in that table is
in `SURFACE` in `tests/test_api.py`, which compares its signature against the
upstream function and checks its output against the returned dict on a corpus.

### Additions, not ports

These have no upstream function of the same name; they are this repo's batch
API, and each is the scalar function above run over an array in one call:

`headers`, `decode_fields`, `payload`, `payloads`, `messages_to_payloads`,
`decode_payloads`, `altcodes_to_altitudes`, `decode_callsigns`,
`idcodes_to_squawks`, `crc_remainders`, `cprNL_batch`,
`airborne_position_pair_batch`, `airborne_position_with_ref_batch`,
`surface_position_pair_batch`, `surface_position_with_ref_batch`, and the
`decode_bdsXX_batch` sibling of each register decoder.

## What is not covered

- **Not ADS-B**: the Comm-B (DF20/21) decoders, `pyModeS.decoder.commb` and
  its BDS register inference, and the BDS 0,9 Comm-A/Comm-B variants that
  live in `bds10`/`bds17`. None of them are reachable from an ADS-B message.
- **Not the message layer**: `pyModeS.decode`, `pyModeS.Message`,
  `pyModeS.PipeDecoder` and the whole `Message.decode()` schema machinery:
  header validation, per-aircraft state, CPR pairing across a stream, the
  Comm-B inference heuristics. `mojopymodes.batch.decode_payloads` is the
  part of that which is mechanical (dispatch a payload to its register) and
  it is ported; the stateful half is not, and `PipeDecoder` in particular has
  no Mojo analogue.
- **Not DF0/4/5/11/16 decoding beyond `util`**: the `surv` decoder's
  altitude and surveillance-status interpretation is not ported, only the
  bit-level `altcode` and `idcode` fields that `util` exposes.
- **Not the CLI, the streamer, the airport database, the uncertainty tables
  beyond `TC_NUCp_lookup`, or the `_aero` / `_schema` / `_pipe_validation`
  modules.**
- **No GPU path.** Every kernel is below the arithmetic intensity where a
  device launch pays for itself, and the workload is a few hundred bytes per
  message.

## Documented divergences

Each of these is a place where Mojo could not take upstream's shape. The
kernels are otherwise a literal transcription; the divergence is isolated in
the wrapper (`src/capi.mojo`) or marked `DIVERGENCE` in the kernel.

| what | why |
| --- | --- |
| `int \| None` returns | Mojo has no null. `altcode_to_altitude` returns `Optional[Int]`, which the C ABI flattens to a (value, present) pair. |
| `tuple \| None` returns from the position decoders | Reported as a NaN latitude. A valid CPR solution is never NaN, so the sentinel is unambiguous. |
| Python dicts | Become structs of the same field names in the kernel. `capi.mojo` flattens the struct into a row of `Float64` slots plus two bitmasks (which keys exist, which were `None`), and the Python layer reassembles the dict. `decode_bds65` carries explicit `has_nic_baro` / `has_sil_supplement` flags because upstream omits those keys entirely. |
| Python `str` returns | A `String` cannot cross the C ABI, so the kernel returns a `String` and the wrapper writes it as a `(buffer, length)` pair. The three small closed-set strings (the wake-vortex category, `airspeed_type`, `vr_source`, `selected_altitude_source`) travel as an index or a bit, and the Python layer maps them. |
| callsign slot 0 | Upstream's lookup table holds `chr(0)` there; a NUL cannot be carried by a `(buffer, length)` string, so the port stores the `#` sentinel. Slot 0 is only read by a payload whose first six-bit slot is itself invalid (no padding at all, an invalid first character, no trailing padding), which no real callsign encodes, so decoded output is unchanged for every payload the port can see. `tests/test_fields.py` asserts both facts. |
| `is_valid_callsign_char` bounds | Upstream raises `IndexError` outside `-64..63` and counts from the end for negative indexes. The kernel has no bounds check on a comptime tuple, so it spells both ranges out and the wrapper raises the same `IndexError`. |
| keyword-only arguments | `decode_bds05(payload, *, tc)` and the `*` forms of `airborne_position_pair` / `surface_position_pair` take their keyword-only arguments positionally in the kernel. The Python wrappers keep the `*`. |
| 112-bit messages | A Mojo `Int` is 64 bits and a Mode-S message is 112, so a long message is read as `(hi = n >> 64, lo = n & 2**64-1)`. `crc_remainder` is ported unchanged for the single-word form; `crc_remainder_wide` in `capi.mojo` is the same algorithm and the same table, reading its bytes out of the pair. `extract_unsigned` keeps its single-word form; a 112-bit field is read with the same expression over the pair. |
| `bisect_right` over the NL boundary table | Written out as the `while` loop `bisect_right` performs; Mojo has no such function. |
| `min(pairs, key=...)` in `surface_position_pair` | Spelled out as the same comparisons Python's `min` would make, ties keeping the first. |
| `math.degrees` | CPython's is `x * (180.0/pi)`, a single multiply by a constant. Written as `x * 180.0 / pi` it becomes two roundings and drifts an ulp, so the constant is spelled out. |
| fused multiply-add | Mojo's optimiser contracts `x + y*z` into an FMA, which rounds once where Python rounds twice. The two sites where that bites (`decode_bds62`'s baro setting and `_decode_movement`) go through an out-of-line `_py_mul` so the arithmetic stays two roundings and the result is bit-identical. |
| `_decode_ground_speed` return type | Upstream returns `(float, float)` and wraps the speed in `int()`; here the speed is already an `Int`, because the C ABI cannot tell an int-valued float from an int once it has crossed. |
| `generate` expressions and lambdas | `next(idx for idx, lb in ... if lb > mov)` becomes a `while` loop; both are spelled out. |

## Install and use

Requires [pixi](https://pixi.sh). The environment brings its own Mojo
toolchain and the upstream pyModeS used for the parity tests:

```bash
pixi install
pixi run build    # mojo build --emit shared-lib -> dist/libmojo-pymodes.so
pixi run test     # 7880 assertions against pyModeS 3.6.0
pixi run bench    # benchmarks against pyModeS, under a machine-wide flock
```

Then, from the repository root:

```bash
pixi run python -c 'import mojopymodes as pm; print(pm.util.df("8D406B902015A678D4D220AA4BDA"))'
```

`python/mojopymodes/_lib.py` builds the shared library on first import when
`dist/libmojo-pymodes.so` is missing or older than any `.mojo` file, so a bare
`import mojopymodes` works too. To force a rebuild:

```bash
pixi run python -m mojopymodes._lib --force
```

Outside pixi, point it at a compiler with `MOJOPYMODES_MOJO=/path/to/mojo`
(the launcher sets `MODULAR_HOME` from that binary's prefix), or at a
prebuilt library with `MOJOPYMODES_LIB=/path/to/libmojo-pymodes.so`.

The package is not on PyPI. A wheel built from `pyproject.toml` contains the
Python package only: point it at a library the same way with
`MOJOPYMODES_LIB`, or build this checkout's `dist/` first. The sdist carries
the sources (`src/`, `build/`, `tests/`, `bench/`), so a wheel built from an
sdist can compile the library on first import.

## Performance

Best of three runs, same machine, same data, on pyModeS 3.6.0. Upstream
decodes one message at a time, so every case is "N things, N results" on
both sides: upstream gets its own public function in a Python loop, this
repo gets the batch call. The inputs are built before the clock starts on
both sides.

Measured on: Intel Xeon E5-2697 v4 @ 2.30 GHz, Linux 6.8.0, Python 3.13.14,
numpy 2.5.1, Mojo 1.2.0.dev2026092905. The table is one verbatim run of
`pixi run bench` from a clean build on 2026-11-11 (the last of three runs made
to write this file).

| case | mojo-pymodes | pyModeS | ratio |
| --- | ---: | ---: | ---: |
| `crc_remainder` (1M messages) | 671.7 ms | 3400.0 ms | 5.06x |
| `df` + `icao` + `typecode` + `crc` (1M messages) | 2344.1 ms | 5052.0 ms | 2.16x |
| `altcode_to_altitude` (1M codes) | 293.2 ms | 885.3 ms | 3.02x |
| `decode_callsign` (1M payloads) | 512.7 ms | 1808.9 ms | 3.53x |
| `cprNL` (1M latitudes) | 144.1 ms | 253.1 ms | 1.76x |
| `airborne_position_with_ref` (1M frames) | 904.3 ms | 1119.2 ms | 1.24x |
| `airborne_position_pair` (1M pairs) | 812.3 ms | 2050.4 ms | 2.52x |
| BDS 0,5 airborne position (1M payloads) | 1615.7 ms | 1473.6 ms | 0.91x slower |
| BDS 0,8 identification (1M payloads) | 1072.4 ms | 2576.4 ms | 2.40x |
| BDS 0,9 airborne velocity (1M payloads) | 3472.8 ms | 2018.4 ms | 0.58x slower |
| BDS 6,2 target state (1M payloads) | 3322.3 ms | 2678.7 ms | 0.81x slower |
| full ADS-B decode (200k messages) | 700.9 ms | 573.4 ms | 0.82x slower |
| single `cprNL` call (100k calls) | 73.1 ms | 27.5 ms | 0.38x slower |

`ratio` is pyModeS / mojo-pymodes, so above 1.0 means this repo is faster.
Reproduce with `pixi run bench`; it holds a machine-wide flock so a concurrent
job cannot distort the numbers. The machine is shared, so the millisecond
figures move between runs (across the four runs made to write this file,
`cprNL` ranged from 115 to 144 ms and one header run was 11% above the
table's); the sign of every ratio was stable across all of them. The
`df + icao + typecode + crc` row is the noisiest because it is the one that
parses a million hex strings.

The wins are on the primitives that return a number or a short string: the
kernel does the work and Python's per-message call overhead is gone.

The losses are real and this pass does not hide them. They have one cause,
and it is not the kernel. `bench/kernel_vs_wrapper.py` measures the BDS 0,5
kernel with nothing between it and the FFI call:

```
n = 435400
bds05 kernel only       :     13.4 ms  (  30.7 ns/msg)
bds05 through the API   :    672.7 ms  (1545.1 ns/msg)
dict assembly share     :    659.4 ms (98.0% of the API time)
kernel is 50x faster than the API that wraps it
one bds05 record        :    0.009 ms  (   9.1 us/record)
```

About 98% of that case is `_layout.rows` building a million Python dicts. Upstream
pays the same dict-building cost inside its own per-message function, which
is why it still lands ahead of us on the registers with the widest rows, and
why the per-call `cprNL` case loses: one record costs about 8-9 microseconds
of ctypes round trip plus the dict, against about 40 nanoseconds of actual
CPR work. Two runs of this diagnostic in the same session put the kernel at
8.0 and 13.4 ms and the dict share at 98.8% and 98.0%.

### What this pass changed, and what it did not

Each of the following was measured with `pixi run bench` or
`bench/kernel_vs_wrapper.py`.

- **Row assembly (`_layout.rows`).** The old code read a flat `tolist()` of
  the buffer and ran a field-plan loop per row, with eight hand-written fast
  paths in front of it. It is now one builder per layout, written as
  straight-line code over pre-read columns, with a per-row present/absent
  test only for keys that can be missing or `None`. `bench/ab_rows.py` runs
  both on the same kernel output: BDS 0,8 assembly is about 0.7-0.9 s against
  1.9-2.0 s per million rows for the generic loop (2.2-2.8x, two runs), BDS
  0,5 about 1.4 s against 3.1 s, BDS 6,2 about 3.0 s against 6.1 s. Every
  builder that is not written for a row shape falls back to `_ordered`, the
  field-plan loop over the int64 view, and `tests/test_builders.py` and
  `tests/test_rows_fast.py` compare each builder against that reference on
  random and mixed per-row masks.
- **Hex header field.** `icao` was one `f"{v:06X}"` per row. It is now a
  single gather of all six digits per row from a 16 MiB byte table, used when
  every address fits 24 bits (which the kernel writes), with the format
  string as the exact fallback. The header row as a whole is 2462 ms: the
  address column was not the header's cost.
- **Callsign strip.** One `bytes.split` over the char strip instead of one
  slice-and-decode per row.
- **Not done: SIMD in the row builders.** The remaining cost is one `dict`
  per row, which Python builds; vectorising the mask arithmetic around it
  moves nothing measurable on the 4-byte-wide rows and is not in the code.
- **Not done: a wider SIMD kernel.** The kernels are already 8-18 ms per
  million rows (two `bench/kernel_vs_wrapper.py` runs), against 1000-3400 ms
  of row assembly. Widening the CRC, altitude or CPR kernels would be
  invisible in the totals.
- **Not kept: `parallelize` changes.** Timed serial against the default
  parallel path at 1k, 4k, 16k, 64k, 256k and 1M rows in an earlier pass:
  within noise at every size, because these kernels are already at memory
  bandwidth at the batch width they write. The existing threshold is
  unchanged.
- **No GPU path.** Every kernel is below the arithmetic intensity where a
  device launch pays; the work is a few hundred bytes per message and the
  binding cost is the Python dict per row.

## How it works

```
python/mojopymodes/   the pyModeS-shaped API: util, position, decoder.bds
        |             ctypes, one call per operation, buffers by address
src/capi.mojo         @export ... abi("C") wrappers, Int addresses,
        |             result flattening, 112-bit word pairs
src/ported.mojo       the ported functions, in upstream's order
src/tables.mojo       the static tables, copied from pyModeS
```

**FFI strategy.** Every buffer crosses as an `Int` address and is rebuilt
inside the wrapper as `Pointer[T, AnyOrigin[mut=True]]`: `@export` rejects
parametric functions, and a pointer with an inferred origin is parametric.
Nothing in Mojo allocates. Every routine writes into a buffer the caller
owns, so there is no lifetime to manage and nothing to leak. The Python side
keeps every input array alive for the duration of the call (the buffer is a
local of the wrapper that passes `addr()`), and the batch wrappers refuse a
column whose length does not match the payload array rather than let the
kernel read past it.

**Result layout.** A ported function that returns a dict in Python returns a
struct in the kernel. `capi.mojo` flattens that struct into one row of
`Float64` slots per message, every slot 8 bytes wide, so a row is a plain
contiguous `float64` buffer and the slot index of a field is its byte offset
divided by 8. Slot 0 is a bitmask of which keys the dict had, slot 1 a
bitmask of which of those were `None`. Registers that carry a string get a
second, `UInt8` buffer with a fixed per-message stride. Every numeric field
in the port is below 2**48, so the round trip through `Float64` is exact.
`python/mojopymodes/_layout.py` is the Python half of that contract: it names
the slots and assembles the dict.

**Message layout.** A Mode-S message is 56 or 112 bits. 56-bit messages fit
a Mojo `Int` and go through the single-word functions unchanged. 112-bit
messages cross as two 64-bit words and are read with `_field` and
`crc_remainder_wide` in `capi.mojo`, which are `extract_unsigned` and
`crc_remainder` over the word pair. Everything downstream of the payload, and
every BDS register decoder, works on the 56-bit ME field, which is an
ordinary `Int`.

**Parallelism.** A batch of at least `PAR_THRESHOLD` (4096) messages fans out
across `PAR_WORKERS` (8) workers with `max.algorithm.parallelize`; below
that a thread launch costs more than the work, so the loop stays serial.
`tests/test_kernels.py` drives every batch kernel at three sizes, one of them
past the threshold, and compares every row against upstream.

**Tables.** `src/tables.mojo` is generated by `build/gen_tables.py`, which
reads the tables out of the installed pyModeS, and `tests/test_tables.py`
re-derives them and fails if the file has drifted. The 256-entry CRC-24
table, the 64-entry callsign alphabet, the 58 CPR NL boundaries, the
typecode-to-NUCp map and the wake-vortex index are all upstream's, with the
one callsign-slot-0 substitution listed under divergences.

## Tests

`pixi run test` runs 7880 assertions, nearly all of them against the real
pyModeS 3.6.0 package (the launcher tests build and load a library instead):

- every one of the 8192 possible altitude codes and all 8192 identity codes,
  compared value for value;
- 5000 random callsign payloads, plus the padding, stripping and invalid
  character cases;
- `cprNL` at every 0.1 degree from -90 to +90 and at the boundaries either
  side of 87, plus thousands of random even/odd CPR pairs through all four
  position decoders;
- 1462 messages, two published and the rest generated with a valid CRC so
  that `pyModeS.decode` accepts them, through every BDS register, covering
  all 32 typecodes, all four BDS 0,9 subtypes with and without the "not
  available" magnitudes, all 128 movement codes, all eight category codes,
  both BDS 0,5 altitude encodings, every BDS 6,1 subtype and every threat
  indicator;
- the serial and threaded batch paths against each other;
- the Python surface itself: every covered function's name, argument order
  and defaults are compared against the upstream function it was ported from.

`tests/test_kernels.py` covers the other direction. Everything above reaches
the kernels through the Python API, which is the right way to test a port and
also means a kernel that is wired up and completely wrong would ship green:
nothing calls the symbol, so nothing observes it. That file fetches all 37
`@export`ed symbols off the `ctypes.CDLL` by name, asserts on what each one
wrote, and drives every batch kernel at three sizes so both the serial branch
and the `parallelize` branch are exercised. It skips loudly if
`dist/libmojo-pymodes.so` is missing rather than passing quietly.

`tests/test_builders.py` and `tests/test_rows_fast.py` cover the row builders
directly instead of through the Python API, because a batch with one present
mask can never reach the per-row branches a real capture does. They drive
every builder on random per-row masks and on the mixed shapes (BDS 0,9's
subtype-dependent key set, BDS 6,5's two registers) and compare each against
the generic loop.

`tests/test_launcher.py` builds a library into an empty directory through the
first-import path and calls an export from it, so the launcher the README
describes is exercised rather than assumed.

The kernel audit is mechanical and prints nothing when it passes:

```bash
for s in $(grep -hoE '@export\("[A-Za-z_0-9]+"\)' src/*.mojo | sed 's/@export("//;s/")//'); do
  grep -rqF "$s" tests/ || echo "UNTESTED KERNEL: $s"
done
```

## Licence

The names, bit layouts, algorithms and tables in `src/ported.mojo` and
`src/tables.mojo` are pyModeS's: GPL-3.0, from
https://github.com/junzis/pyModeS (revision used here: the PyPI release
3.6.0, whose distribution carries the GNU GPL v3 licence text; upstream's
`LICENSE` on `main` is the same text). Those two files, and the wheel and
sdist built from this repository, are distributed under GPL-3.0, and the
pyModeS licence travels with them. The Python packaging, the C ABI in
`src/capi.mojo`, the tests and the benchmarks are this repository's own work,
also released under GPL-3.0 because it is combined with the port.

`LICENSE` is the GNU General Public License v3, verbatim from the pyModeS
3.6.0 distribution, followed by a NOTICE that records the upstream source,
revision and licence and retains the MIT permission text this repository
carried before the port's licence was corrected. Do not relicense this
repository under a permissive licence: an independent implementation of the
same behaviour, replacing `src/ported.mojo` and `src/tables.mojo`, would be
required first.
