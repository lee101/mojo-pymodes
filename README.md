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

Every function in the covered subset takes the same arguments, in the same
order, with the same defaults, and returns the same value as the pyModeS
function it was ported from. The test suite asserts that against the real
`pyModeS` 3.6.0 package on every call, not just on a sample.

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
BDS 3,0 ACAS report that BDS 6,1 subtype 2 embeds.

## What is not covered

- **Not ADS-B**: the Comm-B (DF20/21) decoders, `pyModeS.decoder.commb` and
  its BDS register inference, and the BDS 0,9 Comm-A/Comm-B variants that
  live in `bds10`/`bds17`. None of them are reachable from an ADS-B message.
- **Not the message layer**: `pyModeS.decode`, `pyModeS.Message`,
  `pyModeS.PipeDecoder` and the whole `Message.decode()` schema machinery —
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
kernels themselves stay a literal transcription; the divergence is isolated
in the wrapper (`src/capi.mojo`) or marked `DIVERGENCE` in the kernel.

| what | why |
| --- | --- |
| `int \| None` returns | Mojo has no null. `altcode_to_altitude` returns `Optional[Int]`, which the C ABI flattens to a (value, present) pair. |
| `tuple \| None` returns from the position decoders | Reported as a NaN latitude. A valid CPR solution is never NaN, so the sentinel is unambiguous. |
| Python dicts | Become structs of the same field names in the kernel. `capi.mojo` flattens the struct into a row of `Float64` slots plus two bitmasks (which keys exist, which were `None`), and the Python layer reassembles the dict. `decode_bds65` carries explicit `has_nic_baro` / `has_sil_supplement` flags because upstream omits those keys entirely. |
| Python `str` returns | A `String` cannot cross the C ABI, so `decode_callsign` and `idcode_to_squawk` return a `String` from the kernel and `(buffer, length)` from the wrapper. The three small closed-set strings — the wake-vortex category, `airspeed_type`, `vr_source`, `selected_altitude_source` — travel as an index or a bit, and the Python layer maps them. |
| keyword-only arguments | `decode_bds05(payload, *, tc)` and `airborne_position_pair(..., *, even_is_newer)` take their keyword-only argument positionally in the kernel. The Python wrappers keep the `*`. |
| 112-bit messages | A Mojo `Int` is 64 bits and a Mode-S message is 112, so a long message crosses as `(hi = n >> 64, lo = n & 2**64-1)`. `crc_remainder` is ported unchanged for the single-word form; `crc_remainder_wide` in `capi.mojo` is the same algorithm and the same table, reading its bytes out of the pair. |
| `bisect_right` over the NL boundary table | Written out as the `while` loop `bisect_right` performs; Mojo has no such function. |
| `min(pairs, key=...)` in `surface_position_pair` | Spelled out as the two comparisons Python's `min` would make, ties keeping the first. |
| `math.degrees` | CPython's is `x * (180.0/pi)`, a single multiply by a constant. Written as `x * 180.0 / pi` it becomes two roundings and drifts an ulp, so the constant is spelled out. |
| fused multiply-add | Mojo's optimiser contracts `x + y*z` into an FMA, which rounds once where Python rounds twice. The two sites where that bites (`decode_bds62`'s baro setting and `_decode_movement`) go through an out-of-line `_py_mul` so the arithmetic stays two roundings and the result is bit-identical. |
| `_decode_ground_speed` return type | Upstream returns `(float, float)` and wraps the speed in `int()`; here the speed is already an `Int`, because the C ABI cannot tell an int-valued float from an int once it has crossed. |
| `generate` expressions and lambdas | `next(idx for idx, lb in ... if lb > mov)` becomes a `while` loop; both are spelled out. |

## Install

```bash
pixi install     # brings its own Mojo toolchain and the upstream pyModeS
pixi run build   # mojo build --emit shared-lib -> dist/libmojo-pymodes.so
pixi run test    # 7801 parity assertions against pyModeS 3.6.0
pixi run bench   # benchmarks against pyModeS, under a machine-wide flock
```

The shared library is built by `pixi run build`. `python/mojopymodes/_lib.py`
also builds it on first import if it is missing or older than any `.mojo`
file, so a bare `python -c "import mojopymodes"` works too. To force a
rebuild:

```bash
python -m mojopymodes._lib --force
```

Outside pixi, point it at a compiler with `MOJOPYMODES_MOJO=/path/to/mojo`,
or at a prebuilt library with `MOJOPYMODES_LIB=/path/to/libmojo-pymodes.so`.

## Performance

Best of three runs, same machine, same data, on pyModeS 3.6.0. Upstream
decodes one message at a time, so every case is "N things, N results" on
both sides: upstream gets its own public function in a Python loop, this
repo gets the batch call. The inputs are built before the clock starts on
both sides.

Measured on: Intel Xeon E5-2697 v4 @ 2.30 GHz, Linux 6.8.0, Python 3.13.14,
numpy 2.5.1, Mojo 1.2.0.dev2026092605.

| case | mojo-pymodes | pyModeS | ratio |
| --- | ---: | ---: | ---: |
| `crc_remainder` (1M messages) | 2942.3 ms | 9078.8 ms | 3.09x faster |
| `df` + `icao` + `typecode` + `crc` (1M messages) | 12498.4 ms | 13281.6 ms | 1.06x faster |
| `altcode_to_altitude` (1M codes) | 939.4 ms | 1914.7 ms | 2.04x faster |
| `decode_callsign` (1M payloads) | 1704.2 ms | 4399.5 ms | 2.58x faster |
| `cprNL` (1M latitudes) | 284.9 ms | 536.7 ms | 1.88x faster |
| `airborne_position_with_ref` (1M frames) | 2083.5 ms | 2527.4 ms | 1.21x faster |
| `airborne_position_pair` (1M pairs) | 2204.8 ms | 4952.4 ms | 2.25x faster |
| BDS 0,8 identification (1M payloads) | 4672.4 ms | 6500.1 ms | 1.39x faster |
| BDS 0,5 airborne position (1M payloads) | 6763.2 ms | 3519.3 ms | 0.52x slower |
| BDS 0,9 airborne velocity (1M payloads) | 8134.4 ms | 4400.6 ms | 0.54x slower |
| BDS 6,2 target state (1M payloads) | 13320.0 ms | 6216.2 ms | 0.47x slower |
| full ADS-B decode (200k messages) | 2302.4 ms | 1321.0 ms | 0.57x slower |
| single `cprNL` call (100k calls) | 157.7 ms | 38.8 ms | 0.25x slower |

`ratio` is pyModeS / mojo-pymodes, so above 1.0 means this repo is faster.
Reproduce with `pixi run bench`; it holds a machine-wide flock so a concurrent
job cannot distort the numbers. The table above is one verbatim run of it. The
machine is shared, so the millisecond figures move by 20-30% between runs —
the sign of every ratio is stable, the `df + icao + typecode + crc` row is the
noisiest because it is the one that parses a million hex strings.

The wins are on the primitives that return a number or a short string: the
kernel does the work and Python's per-message call overhead is gone.

The losses are real and this pass does not hide them. They have one cause, and
it is not the kernel. `bench/kernel_vs_wrapper.py` measures the BDS 0,5 kernel
with nothing between it and the FFI call:

```
n = 435400
bds05 kernel only       :     26.5 ms  (  60.8 ns/msg)
bds05 through the API   :   2995.5 ms  (6880.0 ns/msg)
dict assembly share     :   2969.1 ms (99.1% of the API time)
kernel is 113x faster than the API that wraps it
```

99.1% of that case is `_layout.rows` rebuilding a million Python dicts field by
field. Upstream pays the same dict-building cost inside its own per-message
function, which is why it still lands ahead of us overall. The kernels are
unchanged by that measurement and the batch structure is intact, so removing
it is a change to the wrapper's row layout, not a rewrite.

The single-message row is the one case with no way to win: a ctypes round trip
costs about 0.7 microseconds, and `cprNL` on one latitude is about 40
nanoseconds of actual work.

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
owns, so there is no lifetime to manage and nothing to leak.

**Result layout.** A ported function that returns a dict in Python returns a
struct in the kernel. `capi.mojo` flattens that struct into one row of
`Float64` slots per message, every slot 8 bytes wide, so a row is a plain
contiguous `float64` buffer and the slot index of a field is its byte offset
divided by 8. Slot 0 is a bitmask of which keys the dict had, slot 1 a
bitmask of which of those were `None`. Registers that carry a string get a
second, `UInt8` buffer with a fixed per-message stride. Every numeric field
in the whole port is below 2**48, so the round trip through `Float64` is
exact. `python/mojopymodes/_layout.py` is the Python half of that contract:
it names the slots and assembles the dict.

**Message layout.** A Mode-S message is 56 or 112 bits. 56-bit messages fit
a Mojo `Int` and go through the single-word functions unchanged. 112-bit
messages cross as two 64-bit words and are read with `_field` and
`crc_remainder_wide` in `capi.mojo`, which are `extract_unsigned` and
`crc_remainder` over the word pair. Everything downstream of the payload —
and every BDS register decoder — works on the 56-bit ME field, which is an
ordinary `Int`.

**Parallelism.** A batch of at least `PAR_THRESHOLD` (4096) messages fans out
across `PAR_WORKERS` (8) workers with `max.algorithm.parallelize`; below
that a thread launch costs more than the work, so the loop stays serial. The
serial and threaded paths are tested to produce identical output on both
sides of the threshold.

**Tables.** `src/tables.mojo` is generated by `build/gen_tables.py`, which
reads the tables out of the installed pyModeS, and `tests/test_tables.py`
re-derives them and fails if the file has drifted. The 256-entry CRC-24
table, the 64-entry callsign alphabet, the 58 CPR NL boundaries, the
typecode-to-NUCp map and the wake-vortex index are all upstream's, not
transcribed by hand.

## Tests

`pixi run test` runs 7801 assertions, all of them against the real pyModeS
3.6.0 package:

- every one of the 8192 possible altitude codes and all 8192 identity codes,
  compared value for value;
- 5000 random callsign payloads, plus the padding, stripping and invalid
  character cases;
- `cprNL` at every 0.1 degree from -90 to +90 and at the boundaries either
  side of 87, plus thousands of random even/odd CPR pairs through all four
  position decoders;
- 1462 messages — two published, the rest generated with a valid CRC so that
  `pyModeS.decode` accepts them — through every BDS register, covering all 32
  typecodes, all four BDS 0,9 subtypes with and without the "not available"
  magnitudes, all 128 movement codes, all eight category codes, both BDS 0,5
  altitude encodings, every BDS 6,1 subtype and every threat indicator;
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

The audit is mechanical and prints nothing when it passes:

```bash
for s in $(grep -hoE '@export\("[A-Za-z_0-9]+"\)' src/*.mojo | sed 's/@export("//;s/")//'); do
  grep -rqF "$s" tests/ || echo "UNTESTED KERNEL: $s"
done
```

## Licence

MIT, like the rest of this family of ports. The ported algorithms come from
pyModeS, which is GPL-3.0; `src/ported.mojo` and `src/tables.mojo` are
derived from it and carry that obligation. See LICENSE.
