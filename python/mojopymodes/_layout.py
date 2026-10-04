"""Field layouts for the flattened decode rows the Mojo kernels write.

`src/capi.mojo` flattens each ported register's result struct into a row of
`Float64` slots. This module is the Python half of that contract: it names
the slots, says how to read them back, and assembles the dict upstream
returns. Slot 0 of a row holds a bitmask of the keys the dict had, slot 1 a
bitmask of the keys whose value was `None`.

A `Field` is `(name, kind, slot, bit, transform, char_offset, char_width)`:

  kind      "i" int, "f" float, "b" bool, "c" chars from the char block,
            "h" 6 uppercase hex digits from the char block
  bit       this field's bit in the present / absent masks
  transform optional value mapping applied after the numeric read
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import numpy as np


class Field(NamedTuple):
    name: str
    kind: str
    slot: int
    bit: int
    transform: Callable[[Any], Any] | None = None
    char_offset: int = 0
    char_width: int = 0


class Layout(NamedTuple):
    width: int
    char_stride: int
    fields: tuple[Field, ...]


def _squawk(value: float) -> str:
    """13-bit ID field to a 4-digit octal squawk, as pyModeS._idcode does.

    Each digit is three interleaved pulse bits: A = A4 A2 A1 at payload
    positions 5, 3, 1; B = B4 B2 B1 at 11, 9, 7; C = C4 C2 C1 at 4, 2, 0;
    D = D4 D2 D1 at 12, 10, 8. Position p is LSB bit 12 - p.
    """
    ident = int(value)
    digits = []
    for hi, mid, lo in ((7, 9, 11), (1, 3, 5), (8, 10, 12), (0, 2, 4)):
        digit = ((ident >> hi) & 1) << 2 | ((ident >> mid) & 1) << 1 | ((ident >> lo) & 1)
        digits.append(chr(48 + digit))
    return "".join(digits)


def _icao(value: float) -> str:
    """24-bit address to 6 uppercase hex digits, as pyModeS.util.icao does."""
    return f"{int(value):06X}"


def _bool(value: float) -> bool:
    return bool(int(value))


# Mirrors decode_bds62's _BDS62_NA / _BDS62_MCP / _BDS62_FMS codes.
BDS62_SOURCE_LIST = ("N/A", "MCP/FCU", "FMS")


# Mirrors bds08._WAKE_VORTEX, ordered as build/gen_tables.py expects: index
# 17 is the "no entry" default and the rest follow the upstream dict.
WAKE_VORTEX_NAMES = [
    "Light",
    "Medium 1",
    "Medium 2",
    "High vortex aircraft",
    "Heavy",
    "High performance",
    "Rotorcraft",
    "Glider / sailplane",
    "Lighter-than-air",
    "Parachutist / skydiver",
    "Ultralight / hang-glider / paraglider",
    "Reserved",
    "UAV",
    "Space / transatmospheric vehicle",
    "Surface emergency vehicle",
    "Surface service vehicle",
    "Ground obstacle",
    "No category information",
]

# Byte width of each string-valued field, for the char-block layouts.
CHARS = {"callsign": 8, "squawk": 8, "threat_icao": 6}

_BDS30_TAIL = (
    ("threat_icao", "h", 16, 14, None, 0, 6),
    ("threat_altitude", "i", 17, 15, None, 0, 0),
    ("threat_range", "f", 18, 16, None, 0, 0),
    ("threat_bearing", "f", 19, 17, None, 0, 0),
)

_BDS30_HEAD = (
    ("threat_type_indicator", "i", 2, 0),
    ("issued_ra", "b", 3, 1),
    ("corrective", "b", 4, 2),
    ("downward_sense", "b", 5, 3),
    ("increased_rate", "b", 6, 4),
    ("sense_reversal", "b", 7, 5),
    ("altitude_crossing", "b", 8, 6),
    ("positive", "b", 9, 7),
    ("no_below", "b", 10, 8),
    ("no_above", "b", 11, 9),
    ("no_left", "b", 12, 10),
    ("no_right", "b", 13, 11),
    ("ra_terminated", "b", 14, 12),
    ("multiple_threat", "b", 15, 13),
)


def _bds30_fields(slot_shift: int, bit_shift: int) -> tuple[Field, ...]:
    head = tuple(
        Field(name, kind, slot + slot_shift, bit + bit_shift, None, 0, 0)
        for name, kind, slot, bit in _BDS30_HEAD
    )
    tail = tuple(
        Field(name, kind, slot + slot_shift, bit + bit_shift, None, off, width)
        for name, kind, slot, bit, _, off, width in _BDS30_TAIL
    )
    return head + tail


LAYOUTS: dict[str, Layout] = {
    "header": Layout(
        9,
        0,
        (
            Field("df", "i", 2, 0),
            Field("icao", "i", 3, 1, _icao),
            Field("typecode", "i", 4, 2),
            Field("crc", "i", 5, 3),
            Field("crc_valid", "b", 6, 4, _bool),
            Field("altitude", "i", 7, 5),
            Field("idcode", "i", 8, 6, _squawk),
        ),
    ),
    "bds05": Layout(
        9,
        0,
        (
            Field("altitude", "i", 2, 0),
            Field("surveillance_status", "i", 3, 1),
            Field("nic_b", "i", 4, 2),
            Field("cpr_format", "i", 5, 3),
            Field("cpr_lat", "i", 6, 4),
            Field("cpr_lon", "i", 7, 5),
            Field("nuc_p", "i", 8, 6),
        ),
    ),
    "bds06": Layout(
        9,
        0,
        (
            Field("movement", "i", 2, 0),
            Field("groundspeed", "f", 3, 1),
            Field("track", "f", 4, 2),
            Field("track_status", "i", 5, 3),
            Field("cpr_format", "i", 6, 4),
            Field("cpr_lat", "i", 7, 5),
            Field("cpr_lon", "i", 8, 6),
        ),
    ),
    "bds08": Layout(
        4,
        8,
        (
            Field("category", "i", 2, 0),
            Field("callsign", "c", 0, 1, None, 0, 8),
            Field("wake_vortex", "i", 3, 2, WAKE_VORTEX_NAMES.__getitem__),
        ),
    ),
    "bds09": Layout(
        12,
        0,
        (
            Field("subtype", "i", 2, 0),
            Field("nac_v", "i", 3, 1),
            Field("groundspeed", "i", 4, 2),
            Field("track", "f", 5, 3),
            Field("heading", "f", 6, 4),
            Field("airspeed", "i", 7, 5),
            Field(
                "airspeed_type",
                "i",
                8,
                6,
                lambda v: "TAS" if int(v) == 1 else "IAS",
            ),
            Field("vr_source", "i", 9, 7, lambda v: "BARO" if int(v) == 1 else "GNSS"),
            Field("vertical_rate", "i", 10, 8),
            Field("geo_minus_baro", "i", 11, 9),
        ),
    ),
    "bds30": Layout(20, 6, _bds30_fields(0, 0)),
    "bds61": Layout(
        22,
        8,
        (
            Field("subtype", "i", 2, 0),
            Field("emergency_state", "i", 3, 1),
            Field("squawk", "c", 0, 2, None, 0, 8),
        )
        + _bds30_fields(2, 3),
    ),
    "bds62": Layout(
        16,
        0,
        (
            Field("subtype", "i", 2, 0),
            Field("selected_altitude", "i", 3, 1),
            Field(
                "selected_altitude_source",
                "i",
                4,
                2,
                lambda v: BDS62_SOURCE_LIST[int(v)],
            ),
            Field("baro_pressure_setting", "f", 5, 3),
            Field("selected_heading", "f", 6, 4),
            Field("nac_p", "i", 7, 5),
            Field("nic_baro", "i", 8, 6),
            Field("sil", "i", 9, 7),
            Field("autopilot", "b", 10, 8, _bool),
            Field("vnav_mode", "b", 11, 9, _bool),
            Field("altitude_hold_mode", "b", 12, 10, _bool),
            Field("approach_mode", "b", 13, 11, _bool),
            Field("lnav_mode", "b", 14, 12, _bool),
            Field("tcas_operational", "b", 15, 13, _bool),
        ),
    ),
    "bds65": Layout(
        12,
        0,
        (
            Field("subtype", "i", 2, 0),
            Field("version", "i", 3, 1),
            Field("nic_supplement_a", "i", 4, 2),
            Field("nac_p", "i", 5, 3),
            Field("sil", "i", 6, 4),
            Field("hrd", "i", 7, 5),
            Field("capability_class", "i", 8, 6),
            Field("operational_mode", "i", 9, 7),
            Field("nic_baro", "i", 10, 8),
            Field("sil_supplement", "i", 11, 9),
        ),
    ),
}

for _name, _layout in LAYOUTS.items():
    _slots = [f.slot for f in _layout.fields if f.kind not in ("c", "h")]
    assert len(set(_slots)) == len(_slots), (_name, "duplicate slot")
    assert max(_slots) < _layout.width, (_name, "slot past the row")
    _bits = [f.bit for f in _layout.fields]
    assert len(set(_bits)) == len(_bits), (_name, "duplicate bit")


_NUMERIC = {"i": int, "f": float, "b": _bool}


def _plan(layout: Layout) -> tuple:
    """Field tuples with the coercion resolved once, not once per field per row.

    Reading a million rows is the dominant cost of the whole batch path, so
    the inner loop does no attribute lookups and no dict indexing.
    """
    out = []
    for f in layout.fields:
        out.append(
            (
                f.name,
                None if f.kind in ("c", "h") else _NUMERIC[f.kind],
                f.slot,
                f.bit,
                f.transform,
            )
        )
    return tuple(out)


_PLANS = {name: _plan(layout) for name, layout in LAYOUTS.items()}


def rows(name: str, out: np.ndarray, chars: np.ndarray | None = None) -> list[dict]:
    """Turn the kernel's flat output into the list of dicts upstream returns.

    A layout that has a hand-written fast path goes through it; the fast path
    handles the shapes real traffic produces (every key present, every value
    non-`None`) in one dict literal and falls back to `rows_generic` for the
    rest, so the result is identical either way.
    """
    fast = _FAST.get(name)
    if fast is not None:
        return fast(out, chars)
    return rows_generic(name, out, chars)


def rows_generic(name: str, out: np.ndarray, chars: np.ndarray | None = None) -> list[dict]:
    """The field-plan loop: one branch per field per row, any shape of result."""
    layout = LAYOUTS[name]
    plan = _PLANS[name]
    # One tolist() over the flat buffer beats numpy scalar indexing by an
    # order of magnitude, and the rows become Python dicts either way.
    table = out.tolist()
    # One bytes copy of the whole char strip, sliced per row: taking a numpy
    # view per row instead costs more than the decode it is reading.
    blob = chars.tobytes() if (layout.char_stride and chars is not None) else b""
    stride = layout.char_stride
    result: list[dict[str, Any]] = []
    append = result.append
    width = layout.width
    for i in range(0, len(table), width):
        row = table[i : i + width]
        present = int(row[0])
        absent = int(row[1])
        item: dict[str, Any] = {}
        for field, coerce, slot, bit, transform in plan:
            if not (present >> bit) & 1:
                continue
            if (absent >> bit) & 1:
                item[field] = None
            elif coerce is None:
                base = (i // width) * stride
                item[field] = blob[base : base + CHARS[field]].split(b"\x00", 1)[0].decode("ascii")
            else:
                value = coerce(row[slot])
                item[field] = transform(value) if transform else value
        append(item)
    return result


# ----------------------------------------------------------------- fast paths
#
# `rows_generic` above costs a presence test, a `None` test, a coercion call
# and a transform call for every field of every row. Real traffic does not hit
# that: a decoded message normally has every key present and every value
# non-`None`, so one dict literal per row does the same work as a C-level
# constant-key map with no per-field branching.
#
# Two more things fall out of that shape:
#
#   * Integer slots are converted once, in numpy, with `astype(np.int64)`, so
#     Python builds the ints during one `tolist()` instead of `int()` being
#     called once per field per row. It is a conversion, not a `.view()`: the
#     kernel stores an integer as a `Float64`, so its bit pattern is the
#     float's, and reinterpreting would read garbage.
#   * Whether any key was `None` anywhere in the batch is one numpy reduction
#     up front, not a test per field per row.
#
# Every fast path hands the whole buffer back to `rows_generic` as soon as it
# meets a row it does not handle, so the two can never disagree about a value.
# `tests/test_rows_fast.py` asserts that on random masks.


def _ints(out: np.ndarray) -> list:
    """The whole buffer as a flat list of Python ints, in one numpy pass."""
    return out.astype(np.int64).tolist()


def _strings(chars: np.ndarray, n: int, stride: int) -> list[str]:
    """The char strip as a list of n strings.

    Variable-length entries cannot be recovered by splitting the strip once: a
    short string leaves padding NULs behind and the split would report them as
    empty entries. So each slot is sliced and trimmed at its own terminator.
    """
    if n == 0:
        return []
    blob = chars.tobytes()
    return [
        blob[i * stride : (i + 1) * stride].split(b"\x00", 1)[0].decode("ascii")
        for i in range(n)
    ]


def _uniform_masks(out: np.ndarray, width: int):
    """The present mask when every row of the batch carries the same one.

    The kernels write a present mask that is a constant of the register --
    bds09 is the only one that varies, and it has its own fast path for each of
    its two shapes -- so a batch whose rows disagree about which keys exist has
    no single dict shape and goes to the generic loop.
    """
    frame = out.reshape(-1, width)
    if frame.shape[0] == 0:
        return 0
    present = frame[:, 0].astype(np.int64)
    if bool((present != present[0]).any()):
        return None
    return int(present[0])


def _no_unexpected_none(out: np.ndarray, width: int, optional: int) -> bool:
    """True when no row marks a field `None` that the fast path does not read.

    A fast path handles the register's optional fields by testing their absent
    bits inline. If a row ever set an absent bit outside that set, the fast path
    would read a value where the generic loop returns `None`. One numpy pass
    over the absent column rules that out for the whole batch, which makes the
    fast path equivalent to `rows_generic` for any input rather than only for
    the masks the kernels happen to produce.
    """
    absent = out.reshape(-1, width)[:, 1].astype(np.int64)
    return not bool((absent & ~optional).any())


def _fast_bds05(out, chars=None):
    if _uniform_masks(out, 9) != 0x7F or not _no_unexpected_none(out, 9, 1):
        return rows_generic("bds05", out, chars)
    # One 2-D conversion so the loop walks per-row lists rather than striding a
    # nine-million-element flat list. Only `altitude` can be None on a BDS 0,5
    # row (bit 0), which is why the rest need no test.
    ints = out.reshape(-1, 9).astype(np.int64).tolist()
    result = []
    append = result.append
    for r in ints:
        m = r[1]
        append(
            {
                "altitude": None if m & 1 else r[2],
                "surveillance_status": r[3],
                "nic_b": r[4],
                "cpr_format": r[5],
                "cpr_lat": r[6],
                "cpr_lon": r[7],
                "nuc_p": r[8],
            }
        )
    return result


def _fast_bds06(out, chars=None):
    if _uniform_masks(out, 9) != 0x7F or not _no_unexpected_none(out, 9, 2 | 4):
        return rows_generic("bds06", out, chars)
    frame = out.reshape(-1, 9)
    # One 2-D conversion so the loop walks per-row lists; groundspeed and track
    # are the two float fields and the two that can be None (bits 1 and 2).
    ints = frame.astype(np.int64).tolist()
    floats = frame[:, 3:5].tolist()
    result = []
    append = result.append
    for k, r in enumerate(ints):
        m = r[1]
        fb = floats[k]
        append(
            {
                "movement": r[2],
                "groundspeed": None if m & 2 else fb[0],
                "track": None if m & 4 else fb[1],
                "track_status": r[5],
                "cpr_format": r[6],
                "cpr_lat": r[7],
                "cpr_lon": r[8],
            }
        )
    return result


def _fast_bds62(out, chars=None):
    optional = 2 | 8 | 16 | 256 | 512 | 1024 | 2048 | 4096
    if _uniform_masks(out, 16) != 0x3FFF or not _no_unexpected_none(out, 16, optional):
        return rows_generic("bds62", out, chars)
    frame = out.reshape(-1, 16)
    # One 2-D conversion, so the loop walks per-row lists rather than striding a
    # sixteen-million-element flat list. The two float fields come from their
    # own columns; everything else rides the integer conversion.
    ints = frame.astype(np.int64).tolist()
    floats = frame[:, 5:7].tolist()
    result = []
    append = result.append
    for k, r in enumerate(ints):
        m = r[1]
        fb = floats[k]
        append(
            {
                "subtype": r[2],
                "selected_altitude": None if m & 2 else r[3],
                "selected_altitude_source": BDS62_SOURCE_LIST[r[4]],
                "baro_pressure_setting": None if m & 8 else fb[0],
                "selected_heading": None if m & 16 else fb[1],
                "nac_p": r[7],
                "nic_baro": r[8],
                "sil": r[9],
                "autopilot": None if m & 256 else r[10] != 0,
                "vnav_mode": None if m & 512 else r[11] != 0,
                "altitude_hold_mode": None if m & 1024 else r[12] != 0,
                "approach_mode": None if m & 2048 else r[13] != 0,
                "lnav_mode": None if m & 4096 else r[14] != 0,
                "tcas_operational": r[15] != 0,
            }
        )
    return result


def _fast_bds65(out, chars=None):
    # nic_baro and sil_supplement are the only two conditional keys, so the
    # present mask is 0xFF when neither is carried and 0x3FF when both are.
    present = _uniform_masks(out, 12)
    if present not in (0xFF, 0x3FF) or not _no_unexpected_none(out, 12, 0):
        return rows_generic("bds65", out, chars)
    table = _ints(out)
    result = []
    append = result.append
    if present == 0x3FF:
        for i in range(2, len(table), 12):
            append(
                {
                    "subtype": table[i],
                    "version": table[i + 1],
                    "nic_supplement_a": table[i + 2],
                    "nac_p": table[i + 3],
                    "sil": table[i + 4],
                    "hrd": table[i + 5],
                    "capability_class": table[i + 6],
                    "operational_mode": table[i + 7],
                    "nic_baro": table[i + 8],
                    "sil_supplement": table[i + 9],
                }
            )
        return result
    for i in range(2, len(table), 12):
        append(
            {
                "subtype": table[i],
                "version": table[i + 1],
                "nic_supplement_a": table[i + 2],
                "nac_p": table[i + 3],
                "sil": table[i + 4],
                "hrd": table[i + 5],
                "capability_class": table[i + 6],
                "operational_mode": table[i + 7],
            }
        )
    return result


def _fast_header(out, chars=None):
    """`util.headers`: df, icao, typecode, crc, crc_valid, altitude, idcode.

    `icao` is an int in the kernel and a hex string here and `idcode` is an int
    that becomes an octal squawk, so those two are formatted per row. `icao`,
    `typecode`, `altitude` and `idcode` are all carried as keys on every
    message, but their value is None for the downlink formats that do not carry
    them, which the kernel signals with the absent mask.
    """
    if _uniform_masks(out, 9) != 0x7F or not _no_unexpected_none(out, 9, 2 | 4 | 32 | 64):
        return rows_generic("header", out, chars)
    table = _ints(out)
    result = []
    append = result.append
    for i in range(2, len(table), 9):
        m = table[i - 1]
        append(
            {
                "df": table[i],
                "icao": None if m & 2 else f"{table[i + 1]:06X}",
                "typecode": None if m & 4 else table[i + 2],
                "crc": table[i + 3],
                "crc_valid": table[i + 4] != 0,
                "altitude": None if m & 32 else table[i + 5],
                "idcode": None if m & 64 else _squawk(table[i + 6]),
            }
        )
    return result


def _fast_bds08(out, chars=None):
    """BDS 0,8 always carries all three keys and never a None."""
    if _uniform_masks(out, 4) != 7 or chars is None or not _no_unexpected_none(out, 4, 0):
        return rows_generic("bds08", out, chars)
    table = _ints(out)
    names = _strings(chars, len(table) // 4, LAYOUTS["bds08"].char_stride)
    result = []
    append = result.append
    for k, r in enumerate(out.reshape(-1, 4).astype(np.int64).tolist()):
        append(
            {
                "callsign": names[k],
                "category": r[2],
                "wake_vortex": WAKE_VORTEX_NAMES[r[3]],
            }
        )
    return result


_FAST = {
    "bds05": _fast_bds05,
    "bds06": _fast_bds06,
    "bds08": _fast_bds08,
    "bds62": _fast_bds62,
    "bds65": _fast_bds65,
    "header": _fast_header,
}


def one(name: str, out: np.ndarray, chars: np.ndarray | None = None) -> dict:
    return rows(name, out, chars)[0]
