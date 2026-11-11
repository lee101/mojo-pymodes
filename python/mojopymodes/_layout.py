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

The idcode field of the header has no transform: its octal squawk comes from
the ported `idcode_to_squawk` kernel, run once over a whole batch by
`_squawks`, rather than from a second Python implementation of it.
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


def _squawks(rows_in: list[dict]) -> list[dict]:
    """Turn the header rows' raw 13-bit ID fields into octal squawk strings.

    The conversion is the ported `idcode_to_squawk` kernel, run once over the
    whole batch rather than reimplemented here: a Python copy of it would be a
    second implementation of an algorithm that already lives in Mojo, and
    nothing would observe the two drifting apart.
    """
    from ._idcode import idcodes_to_squawks

    # A raw field is an int (or a float straight out of the buffer); a squawk
    # that has already been through this function is the octal string the
    # kernel wrote. Only the former is converted, so calling this twice on the
    # same list leaves the squawks alone instead of decoding them again.
    def _raw(value: Any) -> bool:
        return not isinstance(value, str)

    at = [
        i
        for i, r in enumerate(rows_in)
        if r.get("idcode") is not None and _raw(r["idcode"])
    ]
    if not at:
        return rows_in
    codes = idcodes_to_squawks([rows_in[i]["idcode"] for i in at])
    for i, code in zip(at, codes):
        rows_in[i]["idcode"] = code
    return rows_in


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
            Field("idcode", "i", 8, 6),
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


# ------------------------------------------------------------ column planning
#
# A row is not built by testing every field of every row. The batch is read
# once in numpy first, so the common shapes are already resolved:
#
#   floats  the buffer's own float view, shaped rows by slots
#   hexes   the layout's address slot as six ASCII digits per row, one gather
#           into a 16 MiB byte table for the whole batch, used only when every
#           address fits 24 bits (which the kernels write; the format string
#           is the exact fallback)
#   blob    the char strip, split once rather than sliced per entry
#   ever    the OR of every row's absent mask, so a builder can skip a `None`
#           branch that no row in the batch reaches
#
# Then one loop per layout walks the rows. The loop reads each row's own
# present and absent masks for the keys that can be missing or `None`, and
# emits the keys in the layout's field order, which is what `rows_generic`
# does. `rows_generic` is the reference implementation, written the obvious
# way over a flat `tolist()`; `tests/test_rows_fast.py` and
# `tests/test_builders.py` compare the builders against it on every mask.

_DIGITS = np.frombuffer(b"0123456789ABCDEF", dtype=np.uint8)
# _HEX_TABLE[v] is the six ASCII digits of v, so the addresses of a whole batch
# become six bytes each in one gather and one copy.
_HEX_TABLE = np.stack(
    [_DIGITS[((np.arange(1 << 24, dtype=np.uint32) >> (4 * (5 - k))) & 0xF)] for k in range(6)],
    axis=1,
)
_HEX_TABLE.setflags(write=False)


_COERCE = {"i": int, "f": float, "b": _bool}

# field name -> its kind, for the plans that carry a transform on an int field
_KINDS = {name: {f.name: f.kind for f in layout.fields} for name, layout in LAYOUTS.items()}


def _kind(field: Field) -> str:
    """How a row loop reads one field.

    `rows_generic` applies a field's transform only after it has coerced the
    value, so a float or bool field with a transform has one; an int field's
    transform is dead code there. The plan records which, so both loops agree.
    """
    if field.kind in ("c", "h"):
        return "c"
    if field.kind in _COERCE and field.transform is not None:
        return "t"
    if field.transform is _icao:
        return "x"
    return field.kind


def _plan(layout: Layout) -> tuple:
    """(name, kind, slot, bit, char width, transform) per field."""
    return tuple(
        (f.name, _kind(f), f.slot, f.bit, CHARS.get(f.name, 0), f.transform)
        for f in layout.fields
    )


_PLANS = {name: _plan(layout) for name, layout in LAYOUTS.items()}

# The layouts a fused builder below covers. It is empty until the builders are
# defined, at the end of the file.
_BUILDERS: dict[str, Callable] = {}


def _missing(floats: np.ndarray, bits: int) -> bool:
    """True when some row does not carry every key in `bits`.

    A key absent from a row is left out of its dict, which is a different
    answer from a key present with a `None` value, so a builder has to see it
    too. It is one AND-and-compare over the present column, and for a batch
    where every row carries the keys -- what the kernels produce for any real
    capture -- the result is `False` and the row loop has no branch for it.
    """
    if not len(floats):
        return False
    return bool(np.any((floats[:, 0].astype(np.int64) & bits) != bits))


def _strings(blob: bytes, n: int, stride: int) -> list[str]:
    """`n` NUL-terminated entries of `stride` bytes each, as strings.

    One `split` over the whole strip, not one slice per entry: every entry the
    kernels write is terminated, so with `maxsplit = n` the result has exactly
    `n` elements whatever follows the last terminator. A strip with no
    terminator at all -- only a synthetic buffer in the tests -- is read on the
    stride instead.
    """
    if not n:
        return []
    parts = blob.split(b"\x00", n)
    if len(parts) != n:
        return [
            blob[i * stride : (i + 1) * stride].split(b"\x00", 1)[0].decode("latin-1")
            for i in range(n)
        ]
    return [p.decode("latin-1") for p in parts]


def _columns(name: str, out: np.ndarray, chars: np.ndarray | None):
    """(width, floats, hexes, blob, ever, present) for one result buffer.

    `hexes` and `blob` are the two batch-level formats, left empty when the
    layout has no such field (or, for `hexes`, when an address does not fit
    the six-digit table). `present` is the first row's mask; a builder only
    uses it where the whole batch is known to share one.
    """
    layout = LAYOUTS[name]
    width = layout.width
    n = out.size // width
    # The float view, not a rounded copy: a builder that reads a slot with
    # `int()` has to see the fraction to know whether there is one, and the
    # rounded column is taken per slot inside the builder.
    floats = out.reshape(n, width)
    ever = int(floats[:, 1].max()) if n else 0
    hexes = _hexes(name, floats) if n and _fits_hex(name, floats) else []
    blob = chars.tobytes() if (layout.char_stride and chars is not None and n) else b""
    present = int(floats[0, 0]) if n else 0
    return width, floats, hexes, blob, ever, present


def _fits_hex(name: str, floats: np.ndarray) -> bool:
    """True when every address in the batch fits the table and the format."""
    for _, kind, slot, _, _, _ in _PLANS[name]:
        if kind == "x":
            column = floats[:, slot]
            return bool(column.min() >= 0 and column.max() < 1 << 24)
    return False


def _hexes(name: str, floats: np.ndarray) -> list[str]:
    """The address column as one six-digit string per row, or `[]`.

    `ndarray.tolist()` on a fixed-width byte array yields one `bytes` per row
    in C, which is the cheapest way to cut a six-byte-per-row strip into
    strings; a Python slicing loop over the flattened blob is 4x slower.
    """
    for _, kind, slot, _, _, _ in _PLANS[name]:
        if kind == "x":
            # `& 0xFFFFFF` rather than a test: the kernels write a 24-bit
            # address, and a synthetic buffer with a wider one would otherwise
            # read past the table. The last six digits are what `f"{v:06X}"`
            # of the masked value gives, and the generic loop reads the value
            # only on a row whose absent mask clears this bit.
            rows = _HEX_TABLE[floats[:, slot].astype(np.int64) & 0xFFFFFF]
            return [b.decode("ascii") for b in rows.tolist()]
    return []


def rows(name: str, out: np.ndarray, chars: np.ndarray | None = None) -> list[dict]:
    """Turn the kernel's flat output into the list of dicts upstream returns.

    A planned layout goes through its builder, which sends any row shape it
    does not cover to `_ordered`; anything else goes through `rows_generic`.
    Both produce exactly the rows `rows_generic` produces, which
    `tests/test_rows_fast.py` and `tests/test_builders.py` assert on every mask.
    """
    build = _BUILDERS.get(name)
    if build is None:
        return rows_generic(name, out, chars)
    result = build(*_columns(name, out, chars))
    return _squawks(result) if name == "header" else result


def rows_generic(name: str, out: np.ndarray, chars: np.ndarray | None = None) -> list[dict]:
    """The reference builder: a field-plan loop over a flat buffer."""
    layout = LAYOUTS[name]
    plan = _PLANS[name]
    # One tolist() over the flat buffer beats numpy scalar indexing by an
    # order of magnitude, and the rows become Python dicts either way.
    table = out.tolist()
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
        for field, kind, slot, bit, char_width, transform in plan:
            if not (present >> bit) & 1:
                continue
            if (absent >> bit) & 1:
                item[field] = None
            elif kind == "c":
                base = (i // width) * stride
                item[field] = (
                    blob[base : base + char_width].split(b"\x00", 1)[0].decode("latin-1")
                )
            elif kind == "x":
                item[field] = f"{int(row[slot]):06X}"
            elif kind == "b":
                item[field] = bool(row[slot])
            elif kind == "i":
                item[field] = int(row[slot])
            elif kind == "t":
                item[field] = transform(_COERCE[_KINDS[name][field]](row[slot]))
            else:
                item[field] = row[slot]
        append(item)
    return _squawks(result) if name == "header" else result


# ------------------------------------------------------------ fused builders
#
# Each builder is the field plan of one layout, unrolled: straight-line code
# over columns read once, with a test per key only where the kernels can leave
# the key out or set it to `None`. Shapes a builder is not written for -- a
# key set that differs from row to row in a way the merged loop cannot express
# -- go to `_ordered`, the same field plan over the int64 view, which is
# `rows_generic` for every mask a kernel or a test can produce.


def _ordered(floats, width, blob, hexes, name):
    """The rows of a buffer that a builder's own loop does not cover.

    `rows_generic`'s field plan, over the int64 view instead of a flat float
    list, so keys come out in the layout's order and float fields keep the
    fraction a round trip through int64 would drop.
    """
    n = floats.size // width
    rows_: list[dict] = []
    append = rows_.append
    plan = _PLANS[name]
    stride = LAYOUTS[name].char_stride
    block = floats.astype(np.int64).reshape(n, width).tolist()
    # Float fields are read from the float view: rounding them through the
    # int64 block would drop the fraction before `rows_generic` ever saw it.
    real = floats.reshape(n, width).tolist()
    for k in range(n):
        row = block[k]
        real_row = real[k]
        p = row[0]
        m = row[1]
        item: dict = {}
        for field, kind, slot, bit, char_width, transform in plan:
            if not (p >> bit) & 1:
                continue
            if (m >> bit) & 1:
                item[field] = None
            elif kind not in ("c", "x", "t", "i", "b"):
                item[field] = real_row[slot]
            elif kind == "c":
                at = k * stride
                item[field] = blob[at : at + char_width].split(b"\x00", 1)[0].decode("latin-1")
            elif kind == "x":
                item[field] = f"{row[slot]:06X}"
            elif kind == "t":
                item[field] = transform(_COERCE[_KINDS[name][field]](row[slot]))
            elif kind == "i":
                item[field] = row[slot]
            elif kind == "b":
                item[field] = bool(row[slot])
            else:
                item[field] = float(row[slot])
        append(item)
    return rows_


def _build_header(width, floats, hexes, blob, ever, present):
    """util.headers. `idcode` stays the raw 13-bit field; `rows` squawks it.

    Every key except `crc` and `crc_valid` can be missing on a row or `None`,
    so the loop reads each row's own present and absent masks. `icao` is six
    hex digits: the batch-level table when every address fits 24 bits, and
    `f"{v:06X}"` otherwise, which is what the generic loop does.
    """
    ints = floats.astype(np.int64)
    present_at = ints[:, 0].tolist()
    absent_at = ints[:, 1].tolist()
    df = ints[:, 2].tolist()
    tc = ints[:, 4].tolist()
    crc = ints[:, 5].tolist()
    valid = ints[:, 6].tolist()
    alt = ints[:, 7].tolist()
    sid = ints[:, 8].tolist()
    masked = ever & (2 | 4 | 32 | 64)
    if masked == 0 and hexes and not _missing(ints, 0x7F):
        return [
            {
                "df": d,
                "icao": ic,
                "typecode": t,
                "crc": c,
                "crc_valid": v != 0,
                "altitude": a,
                "idcode": s,
            }
            for d, ic, t, c, v, a, s in zip(df, hexes, tc, crc, valid, alt, sid)
        ]
    rows_: list[dict] = []
    append = rows_.append
    for k in range(len(df)):
        p = present_at[k]
        m = absent_at[k]
        item = {}
        if p & 1:
            item["df"] = None if m & 1 else df[k]
        if p & 2:
            item["icao"] = None if m & 2 else (
                hexes[k] if hexes else f"{int(ints[k, 3]):06X}"
            )
        if p & 4:
            item["typecode"] = None if m & 4 else tc[k]
        if p & 8:
            item["crc"] = None if m & 8 else crc[k]
        if p & 16:
            item["crc_valid"] = None if m & 16 else valid[k] != 0
        if p & 32:
            item["altitude"] = None if m & 32 else alt[k]
        if p & 64:
            item["idcode"] = None if m & 64 else sid[k]
        append(item)
    return rows_


def _build_bds05(width, floats, hexes, blob, ever, present):
    """BDS 0,5. `altitude` is the only key that can be `None`.

    Both key facts are read from the row: `present` says whether the key is in
    the dict at all and `absent` whether its value is `None`. That is two mask
    reads per row, which cost about a tenth of the dict this loop builds, and
    it is what makes the literal exactly the generic loop for any masks.
    """
    ints = floats.astype(np.int64)
    present_at = ints[:, 0].tolist()
    absent_at = ints[:, 1].tolist()
    alt = ints[:, 2].tolist()
    ss = ints[:, 3].tolist()
    nb = ints[:, 4].tolist()
    cf = ints[:, 5].tolist()
    la = ints[:, 6].tolist()
    lo = ints[:, 7].tolist()
    np_ = ints[:, 8].tolist()
    rows_: list[dict] = []
    append = rows_.append
    for k in range(len(alt)):
        p = present_at[k]
        m = absent_at[k]
        item = {}
        if p & 1:
            item["altitude"] = None if m & 1 else alt[k]
        if p & 2:
            item["surveillance_status"] = None if m & 2 else ss[k]
        if p & 4:
            item["nic_b"] = None if m & 4 else nb[k]
        if p & 8:
            item["cpr_format"] = None if m & 8 else cf[k]
        if p & 16:
            item["cpr_lat"] = None if m & 16 else la[k]
        if p & 32:
            item["cpr_lon"] = None if m & 32 else lo[k]
        if p & 64:
            item["nuc_p"] = None if m & 64 else np_[k]
        append(item)
    return rows_


def _build_bds06(width, floats, hexes, blob, ever, present):
    """BDS 0,6. `groundspeed` and `track` are the two that can be `None`."""
    ints = floats.astype(np.int64)
    present_at = ints[:, 0].tolist()
    absent_at = ints[:, 1].tolist()
    moves = ints[:, 2].tolist()
    gs = floats[:, 3].tolist()
    trk = floats[:, 4].tolist()
    ts = ints[:, 5].tolist()
    cf = ints[:, 6].tolist()
    la = ints[:, 7].tolist()
    lo = ints[:, 8].tolist()
    rows_: list[dict] = []
    append = rows_.append
    for k in range(len(moves)):
        p = present_at[k]
        m = absent_at[k]
        item = {}
        if p & 1:
            item["movement"] = None if m & 1 else moves[k]
        if p & 2:
            item["groundspeed"] = None if m & 2 else gs[k]
        if p & 4:
            item["track"] = None if m & 4 else trk[k]
        if p & 8:
            item["track_status"] = None if m & 8 else ts[k]
        if p & 16:
            item["cpr_format"] = None if m & 16 else cf[k]
        if p & 32:
            item["cpr_lat"] = None if m & 32 else la[k]
        if p & 64:
            item["cpr_lon"] = None if m & 64 else lo[k]
        append(item)
    return rows_


def _build_bds08(width, floats, hexes, blob, ever, present):
    """BDS 0,8 always carries all three keys and never a value that is `None`.

    A wake-vortex index outside the table cannot come out of the kernel, and a
    synthetic buffer that holds one raises here exactly as the generic loop
    raises, which is the property the tests need.
    """
    if ever == 0 and not _missing(floats, 7):
        ints = floats.astype(np.int64)
        return [
            {"category": c, "callsign": cs, "wake_vortex": WAKE_VORTEX_NAMES[w]}
            for c, cs, w in zip(
                ints[:, 2].tolist(), _strings(blob, len(ints), 8), ints[:, 3].tolist()
            )
        ]
    return _ordered(floats, width, blob, hexes, "bds08")


def _build_bds09(width, floats, hexes, blob, ever, present):
    """BDS 0,9, whose key set is whatever the row's payload carries.

    A ground-speed row carries `groundspeed` and `track` between `nac_v` and
    `vertical_rate`; an air-speed row carries heading, airspeed, type and
    source there; a row of a subtype that carries neither block has only the
    two common keys. Those three are the feature's shapes, and each is one
    contiguous run of the field order, so one loop builds all three.

    A row that sets `airspeed_type` or `vr_source` together with a ground key
    is not any payload's shape: those two keys sit on either side of
    `vertical_rate`. Such rows are built by the ordered loop, in order, from
    their own copies of the row.
    """
    ints = floats.astype(np.int64)
    present_at = ints[:, 0].tolist()
    absent_at = ints[:, 1].tolist()
    subtype = ints[:, 2].tolist()
    nac = ints[:, 3].tolist()
    gs = ints[:, 4].tolist()
    trk = floats[:, 5].tolist()
    hdg = floats[:, 6].tolist()
    spd = ints[:, 7].tolist()
    typ = ints[:, 8].tolist()
    src = ints[:, 9].tolist()
    vr = ints[:, 10].tolist()
    gmb = ints[:, 11].tolist()
    rows_: list[dict] = [{}] * len(present_at)
    slow: list[int] = []
    for k in range(len(present_at)):
        p = present_at[k]
        # A row with bits from both blocks, or with both of the ground keys
        # plus any air key, cannot be written as one contiguous run here.
        air = p & 0x38C
        ground = p & 0x0C
        if (air and ground) or ((p & 0x04 or p & 0x08) and p & 0x380):
            slow.append(k)
            continue
        m = absent_at[k]
        item: dict = {}
        if p & 1:
            item["subtype"] = None if m & 1 else subtype[k]
        if p & 2:
            item["nac_v"] = None if m & 2 else nac[k]
        if p & 0x0F0 and not p & 0x0C:
            if p & 16:
                item["heading"] = None if m & 16 else hdg[k]
            if p & 32:
                item["airspeed"] = None if m & 32 else spd[k]
            if p & 64:
                item["airspeed_type"] = None if m & 64 else ("TAS" if typ[k] == 1 else "IAS")
            if p & 128:
                item["vr_source"] = None if m & 128 else ("BARO" if src[k] == 1 else "GNSS")
        elif p & 0x0C and not p & 0x38C:
            if p & 4:
                item["groundspeed"] = None if m & 4 else gs[k]
            if p & 8:
                item["track"] = None if m & 8 else trk[k]
        if p & 256:
            item["vertical_rate"] = None if m & 256 else vr[k]
        if p & 512:
            item["geo_minus_baro"] = None if m & 512 else gmb[k]
        rows_[k] = item
    if slow:
        block = floats[slow]
        for k, item in zip(slow, _ordered(block, width, blob, hexes, "bds09")):
            rows_[k] = item
    return rows_


def _build_bds62(width, floats, hexes, blob, ever, present):
    """BDS 6,2, whose source column indexes a three-entry name table."""
    ints = floats.astype(np.int64)
    present_at = ints[:, 0].tolist()
    absent_at = ints[:, 1].tolist()
    st = ints[:, 2].tolist()
    sas = ints[:, 3].tolist()
    src = ints[:, 4].tolist()
    baro = floats[:, 5].tolist()
    hdg = floats[:, 6].tolist()
    nac = ints[:, 7].tolist()
    nicb = ints[:, 8].tolist()
    sil = ints[:, 9].tolist()
    ap = ints[:, 10].tolist()
    vn = ints[:, 11].tolist()
    ahm = ints[:, 12].tolist()
    app = ints[:, 13].tolist()
    ln = ints[:, 14].tolist()
    tcas = ints[:, 15].tolist()
    rows_: list[dict] = []
    append = rows_.append
    for k in range(len(st)):
        p = present_at[k]
        m = absent_at[k]
        item = {}
        if p & 1:
            item["subtype"] = None if m & 1 else st[k]
        if p & 2:
            item["selected_altitude"] = None if m & 2 else sas[k]
        if p & 4:
            item["selected_altitude_source"] = None if m & 4 else BDS62_SOURCE_LIST[src[k]]
        if p & 8:
            item["baro_pressure_setting"] = None if m & 8 else baro[k]
        if p & 16:
            item["selected_heading"] = None if m & 16 else hdg[k]
        if p & 32:
            item["nac_p"] = None if m & 32 else nac[k]
        if p & 64:
            item["nic_baro"] = None if m & 64 else nicb[k]
        if p & 128:
            item["sil"] = None if m & 128 else sil[k]
        if p & 256:
            item["autopilot"] = None if m & 256 else ap[k] != 0
        if p & 512:
            item["vnav_mode"] = None if m & 512 else vn[k] != 0
        if p & 1024:
            item["altitude_hold_mode"] = None if m & 1024 else ahm[k] != 0
        if p & 2048:
            item["approach_mode"] = None if m & 2048 else app[k] != 0
        if p & 4096:
            item["lnav_mode"] = None if m & 4096 else ln[k] != 0
        if p & 8192:
            item["tcas_operational"] = None if m & 8192 else tcas[k] != 0
        append(item)
    return rows_


def _build_bds65(width, floats, hexes, blob, ever, present):
    """BDS 6,5: eight keys always, `nic_baro` and `sil_supplement` sometimes.

    The kernel writes the same present mask on every row of a batch (0xFF,
    0x3FF, or the shapes a subtype adds up to), so the key set is one fact for
    the batch; a row's own absent mask still decides which value is `None`.
    """
    ints = floats.astype(np.int64)
    present_at = ints[:, 0].tolist()
    absent_at = ints[:, 1].tolist()
    st = ints[:, 2].tolist()
    ver = ints[:, 3].tolist()
    nsa = ints[:, 4].tolist()
    nac = ints[:, 5].tolist()
    sil = ints[:, 6].tolist()
    hrd = ints[:, 7].tolist()
    cc = ints[:, 8].tolist()
    om = ints[:, 9].tolist()
    nicb = ints[:, 10].tolist()
    sils = ints[:, 11].tolist()
    rows_: list[dict] = []
    append = rows_.append
    for k in range(len(st)):
        p = present_at[k]
        m = absent_at[k]
        item = {}
        if p & 1:
            item["subtype"] = None if m & 1 else st[k]
        if p & 2:
            item["version"] = None if m & 2 else ver[k]
        if p & 4:
            item["nic_supplement_a"] = None if m & 4 else nsa[k]
        if p & 8:
            item["nac_p"] = None if m & 8 else nac[k]
        if p & 16:
            item["sil"] = None if m & 16 else sil[k]
        if p & 32:
            item["hrd"] = None if m & 32 else hrd[k]
        if p & 64:
            item["capability_class"] = None if m & 64 else cc[k]
        if p & 128:
            item["operational_mode"] = None if m & 128 else om[k]
        if p & 256:
            item["nic_baro"] = None if m & 256 else nicb[k]
        if p & 512:
            item["sil_supplement"] = None if m & 512 else sils[k]
        append(item)
    return rows_


_BUILDERS.update(
    {
        "header": _build_header,
        "bds05": _build_bds05,
        "bds06": _build_bds06,
        "bds08": _build_bds08,
        "bds09": _build_bds09,
        "bds62": _build_bds62,
        "bds65": _build_bds65,
    }
)


def one(name: str, out: np.ndarray, chars: np.ndarray | None = None) -> dict:
    return rows(name, out, chars)[0]
