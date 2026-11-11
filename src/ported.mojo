"""pyModeS 3.6.0's ADS-B decoding primitives, ported to Mojo.

The file is grouped and ordered to match the upstream package it came from:
each section header names the upstream module, and the functions inside a
section appear in the order they appear upstream. Read it next to the pyModeS
source and they line up.

Names, argument order and branch structure are upstream's. The places where
Mojo forced a different shape are marked `DIVERGENCE`, and each one is also
listed in the coverage section of README.md:

  * Python `int | None` / `tuple | None` returns become `Optional[...]` or an
    explicit NaN, because Mojo has no null.
  * Python dicts become structs of the same field names; the ctypes layer
    reassembles the dict, and a field that upstream only emits under some
    branch carries a presence flag.
  * Python `str` returns stay `String` inside the kernel, but the C ABI
    carries them as a (buffer, length) pair, and small closed-set strings
    (the wake-vortex category, the airspeed type) travel as an index.
  * `even_is_newer` and the other keyword-only arguments are positional here;
    Mojo has no keyword-only parameters.

Upstream is GPL-3.0. See LICENSE.
"""

from std.math import atan2, floor, isnan, nan, pi, sqrt
from std.traits import Copyable
from std.utils import StaticTuple
from tables import (
    _CALLSIGN_TABLE,
    _CRC_TABLE,
    _NL_BOUNDARIES,
    _WAKE_VORTEX,
    TC_NUCp_lookup,
)

# ===================================================================
# pyModeS/_bits.py
# ===================================================================


@no_inline
def _py_mul(a: Float64, b: Float64) -> Float64:
    """A single multiply, kept out of line.

    DIVERGENCE, forced by the toolchain: Mojo's optimiser contracts
    `x + y * z` into a fused multiply-add, which rounds once where Python
    rounds twice, so a decoded float can land one ulp from upstream's.
    Upstream's arithmetic is two roundings and stays that way; the call
    boundary is what stops the contraction. Only the two sites below need
    it, one multiply per decoded message each.
    """
    return a * b

# ===================================================================
# pyModeS/_bits.py
# ===================================================================


def extract_unsigned(n: Int, start: Int, width: Int, total_bits: Int) -> Int:
    """Extract `width` bits starting at bit `start` (MSB-first).

    Positions are 0-indexed from the message MSB. DIVERGENCE: `n` is a
    64-bit Mojo Int, so a 112-bit message has to arrive as a word pair; see
    `_field` in capi.mojo, which is this same expression over (hi, lo).
    """
    var shift = total_bits - start - width
    return (n >> shift) & ((1 << width) - 1)


def extract_signed(n: Int, start: Int, width: Int, total_bits: Int) -> Int:
    """Extract `width` bits as a two's-complement integer."""
    var raw = extract_unsigned(n, start, width, total_bits)
    var sign_bit = 1 << (width - 1)
    if (raw & sign_bit) != 0:
        raw -= 1 << width
    return raw


def crc_remainder(n: Int, length: Int) -> Int:
    """Mode-S CRC-24 remainder of a `length`-bit message.

    Byte-at-a-time long division against the 256-entry table upstream builds
    in `_build_crc_table`. Kept as a loop, not unrolled.
    """
    var n_data_bytes = (length - 24) // 8
    var crc = 0
    var shift = length - 8
    for _ in range(n_data_bytes):
        var byte = (n >> shift) & 0xFF
        crc = ((crc << 8) & 0xFFFFFF) ^ _CRC_TABLE[((crc >> 16) ^ byte) & 0xFF]
        shift -= 8
    var parity = n & 0xFFFFFF
    return (crc ^ parity) & 0xFFFFFF


# ===================================================================
# pyModeS/_altcode.py
# ===================================================================


def _gray2int(n: Int) -> Int:
    """Gillham code to plain binary, by the standard XOR-with-shifts fold."""
    var v = n
    v ^= v >> 8
    v ^= v >> 4
    v ^= v >> 2
    v ^= v >> 1
    return v


def altcode_to_altitude(ac: Int) -> Optional[Int]:
    """Decode a 13-bit Mode-S altitude code to feet.

    DIVERGENCE: upstream returns `int | None`; `Optional[Int]` carries the
    same information.
    """
    if ac == 0:
        return None

    var m_bit = (ac >> 6) & 0x1
    var q_bit = (ac >> 4) & 0x1

    if m_bit == 0 and q_bit == 1:
        # 25-foot interval, linear encoding.
        var n = ((ac >> 2) & 0x7E0) | ((ac >> 1) & 0x10) | (ac & 0xF)
        return n * 25 - 1000

    if m_bit == 0 and q_bit == 0:
        # 100-foot interval, Gillham code.
        def b(pos: Int) {ac} -> Int:
            return (ac >> (12 - pos)) & 0x1

        var c1 = b(0)
        var a1 = b(1)
        var c2 = b(2)
        var a2 = b(3)
        var c4 = b(4)
        var a4 = b(5)
        var b1 = b(7)
        var b2 = b(9)
        var d2 = b(10)
        var b4 = b(11)
        var d4 = b(12)

        var gc500 = (
            (d2 << 7)
            | (d4 << 6)
            | (a1 << 5)
            | (a2 << 4)
            | (a4 << 3)
            | (b1 << 2)
            | (b2 << 1)
            | b4
        )
        var gc100 = (c1 << 2) | (c2 << 1) | c4

        var n500 = _gray2int(gc500)
        var n100 = _gray2int(gc100)

        if n100 == 0 or n100 == 5 or n100 == 6:
            return None
        if n100 == 7:
            n100 = 5
        if n500 % 2 != 0:
            n100 = 6 - n100

        return n500 * 500 + n100 * 100 - 1300

    # M = 1: meter-based encoding (rare, non-standard).
    return None


# ===================================================================
# pyModeS/_callsign.py
# ===================================================================


def decode_callsign(bits: Int) -> String:
    """Decode an 8x6-bit ICAO callsign from a 48-bit integer.

    Characters come from `_CALLSIGN_TABLE` in the order upstream's generator
    expression walks the slots; the string is then stripped, as upstream does.
    """
    var out = String()
    for i in range(8):
        var idx = (bits >> (42 - 6 * i)) & 0x3F
        out += chr(Int(_CALLSIGN_TABLE[idx]))
    return String(out.strip())


def is_valid_callsign_char(idx: Int) -> Bool:
    """True if a 6-bit index maps to a valid callsign character.

    DIVERGENCE: upstream's `_CALLSIGN_TABLE[idx] != "#"` raises `IndexError`
    for an index outside -64..63 and indexes from the end for -64..-1. Mojo
    has no bounds check on a comptime tuple, so both ranges are spelled out
    here and the Python wrapper raises the same `IndexError`, so the two
    agree for every input.
    """
    if idx > 63:
        return False
    if idx < 0:
        # Python indexes from the end: -1 is the last entry and every index
        # below -64 is a tuple `IndexError` (re-raised by the wrapper). Every
        # in-range negative entry here is '#', so the answer is False.
        if idx < -64:
            return False
        return _CALLSIGN_TABLE[64 + idx] != UInt8(ord("#"))
    return _CALLSIGN_TABLE[idx] != UInt8(ord("#"))


# ===================================================================
# pyModeS/_idcode.py
# ===================================================================


def idcode_to_squawk(idcode: Int) -> String:
    """Decode a 13-bit identity code to a 4-digit octal squawk."""

    def bit(pos: Int) {idcode} -> Int:
        return (idcode >> (12 - pos)) & 0x1

    var a = (bit(5) << 2) | (bit(3) << 1) | bit(1)
    var b = (bit(11) << 2) | (bit(9) << 1) | bit(7)
    var c = (bit(4) << 2) | (bit(2) << 1) | bit(0)
    var d = (bit(12) << 2) | (bit(10) << 1) | bit(8)

    return chr(48 + a) + chr(48 + b) + chr(48 + c) + chr(48 + d)


# ===================================================================
# pyModeS/position/_cpr.py
# ===================================================================

comptime _CPR_DENOM = 131072.0  # 2**17


def cprNL(lat: Float64) -> Int:
    """Number of CPR longitude zones at `lat`, per DO-260B A.1.7.2.

    The `while` loop is `bisect_right(_NL_BOUNDARIES, abs_lat)`: the first
    index whose boundary is strictly greater than the value.
    """
    var abs_lat = abs(lat)
    if abs_lat > 87.0:
        return 1
    if abs_lat == 87.0:
        return 2
    var lo = 0
    var hi = 58
    while lo < hi:
        var mid = (lo + hi) // 2
        if abs_lat < _NL_BOUNDARIES[mid]:
            hi = mid
        else:
            lo = mid + 1
    return 59 - lo


def airborne_position_with_ref(
    cpr_format: Int,
    cpr_lat_raw: Int,
    cpr_lon_raw: Int,
    lat_ref: Float64,
    lon_ref: Float64,
) -> Tuple[Float64, Float64]:
    """Resolve an airborne CPR frame against a nearby reference."""
    var cpr_lat = Float64(cpr_lat_raw) / _CPR_DENOM
    var cpr_lon = Float64(cpr_lon_raw) / _CPR_DENOM
    var d_lat = 360.0 / 59.0 if cpr_format != 0 else 360.0 / 60.0

    var j = floor(0.5 + lat_ref / d_lat - cpr_lat)
    var lat = d_lat * (j + cpr_lat)

    var ni = cprNL(lat) - cpr_format
    var d_lon = 360.0 / Float64(ni) if ni > 0 else 360.0

    var m = floor(0.5 + lon_ref / d_lon - cpr_lon)
    var lon = d_lon * (m + cpr_lon)

    return Tuple[Float64, Float64](lat, lon)


def airborne_position_pair(
    cpr_lat_even_raw: Int,
    cpr_lon_even_raw: Int,
    cpr_lat_odd_raw: Int,
    cpr_lon_odd_raw: Int,
    even_is_newer: Bool,
) -> Tuple[Float64, Float64]:
    """Resolve absolute lat/lon from an even/odd CPR pair.

    DIVERGENCE: upstream returns `tuple[float, float] | None` and takes
    `even_is_newer` keyword-only. The absence case is reported as a NaN
    latitude, which no valid CPR solution can produce.
    """
    var cprlat_even = Float64(cpr_lat_even_raw) / _CPR_DENOM
    var cprlon_even = Float64(cpr_lon_even_raw) / _CPR_DENOM
    var cprlat_odd = Float64(cpr_lat_odd_raw) / _CPR_DENOM
    var cprlon_odd = Float64(cpr_lon_odd_raw) / _CPR_DENOM

    var j = floor(59.0 * cprlat_even - 60.0 * cprlat_odd + 0.5)

    var lat_even = (360.0 / 60.0) * (j % 60.0 + cprlat_even)
    var lat_odd = (360.0 / 59.0) * (j % 59.0 + cprlat_odd)

    if lat_even >= 270.0:
        lat_even -= 360.0
    if lat_odd >= 270.0:
        lat_odd -= 360.0

    if cprNL(lat_even) != cprNL(lat_odd):
        return Tuple[Float64, Float64](nan[DType.float64](), nan[DType.float64]())

    var lat = lat_odd
    var lon = 0.0
    if even_is_newer:
        lat = lat_even
        var nl = cprNL(lat)
        var ni = max(nl, 1)
        var m = floor(cprlon_even * Float64(nl - 1) - cprlon_odd * Float64(nl) + 0.5)
        lon = (360.0 / Float64(ni)) * (m % Float64(ni) + cprlon_even)
    else:
        lat = lat_odd
        var nl = cprNL(lat)
        var ni = max(nl - 1, 1)
        var m = floor(cprlon_even * Float64(nl - 1) - cprlon_odd * Float64(nl) + 0.5)
        lon = (360.0 / Float64(ni)) * (m % Float64(ni) + cprlon_odd)

    if lon > 180.0:
        lon -= 360.0

    if abs(lat) > 90.0 or abs(lon) > 180.0:
        return Tuple[Float64, Float64](nan[DType.float64](), nan[DType.float64]())

    return Tuple[Float64, Float64](lat, lon)


def surface_position_with_ref(
    cpr_format: Int,
    cpr_lat_raw: Int,
    cpr_lon_raw: Int,
    lat_ref: Float64,
    lon_ref: Float64,
) -> Tuple[Float64, Float64]:
    """Resolve a surface CPR frame against a nearby reference."""
    var cpr_lat = Float64(cpr_lat_raw) / _CPR_DENOM
    var cpr_lon = Float64(cpr_lon_raw) / _CPR_DENOM
    var d_lat = 90.0 / 59.0 if cpr_format != 0 else 90.0 / 60.0

    var j = floor(0.5 + lat_ref / d_lat - cpr_lat)
    var lat = d_lat * (j + cpr_lat)

    var ni = cprNL(lat) - cpr_format
    var d_lon = 90.0 / Float64(ni) if ni > 0 else 90.0

    var m = floor(0.5 + lon_ref / d_lon - cpr_lon)
    var lon = d_lon * (m + cpr_lon)

    return Tuple[Float64, Float64](lat, lon)


def surface_position_pair(
    cpr_lat_even_raw: Int,
    cpr_lon_even_raw: Int,
    cpr_lat_odd_raw: Int,
    cpr_lon_odd_raw: Int,
    lat_ref: Float64,
    lon_ref: Float64,
    even_is_newer: Bool,
) -> Tuple[Float64, Float64]:
    """Resolve an even/odd surface CPR pair against a receiver location.

    DIVERGENCE: as `airborne_position_pair`, plus upstream's
    `min(latitude_pairs, key=...)` over a two-element tuple of pairs, which is
    spelled out below as the same two comparisons.
    """
    var cprlat_even = Float64(cpr_lat_even_raw) / _CPR_DENOM
    var cprlon_even = Float64(cpr_lon_even_raw) / _CPR_DENOM
    var cprlat_odd = Float64(cpr_lat_odd_raw) / _CPR_DENOM
    var cprlon_odd = Float64(cpr_lon_odd_raw) / _CPR_DENOM

    var j = floor(59.0 * cprlat_even - 60.0 * cprlat_odd + 0.5)

    var lat_even_n = (90.0 / 60.0) * (j % 60.0 + cprlat_even)
    var lat_odd_n = (90.0 / 59.0) * (j % 59.0 + cprlat_odd)
    var lat_even_s = lat_even_n - 90.0
    var lat_odd_s = lat_odd_n - 90.0

    # min(latitude_pairs, key=lambda pair: abs(lat_ref - pair[newer_index]))
    var lat_even = lat_even_n
    var lat_odd = lat_odd_n
    if even_is_newer:
        if abs(lat_ref - lat_even_s) < abs(lat_ref - lat_even_n):
            lat_even = lat_even_s
            lat_odd = lat_odd_s
    else:
        if abs(lat_ref - lat_odd_s) < abs(lat_ref - lat_odd_n):
            lat_even = lat_even_s
            lat_odd = lat_odd_s

    if cprNL(lat_even) != cprNL(lat_odd):
        return Tuple[Float64, Float64](nan[DType.float64](), nan[DType.float64]())

    var lat = lat_odd
    var lon_base = 0.0
    if even_is_newer:
        lat = lat_even
        var nl = cprNL(lat)
        var ni = max(nl, 1)
        var m = floor(cprlon_even * Float64(nl - 1) - cprlon_odd * Float64(nl) + 0.5)
        lon_base = (90.0 / Float64(ni)) * (m % Float64(ni) + cprlon_even)
    else:
        lat = lat_odd
        var nl = cprNL(lat)
        var ni = max(nl - 1, 1)
        var m = floor(cprlon_even * Float64(nl - 1) - cprlon_odd * Float64(nl) + 0.5)
        lon_base = (90.0 / Float64(ni)) * (m % Float64(ni) + cprlon_odd)

    # Four quadrant candidates; longitude is circular, so the distance to the
    # reference is itself wrapped. Python's min keeps the first on a tie.
    var lon = _wrap180(lon_base)
    var best_d = abs(_wrap180(lon - lon_ref))
    var quadrants = StaticTuple[Float64, 3](90.0, 180.0, 270.0)
    for k in range(3):
        var c = _wrap180(lon_base + quadrants[k])
        var d = abs(_wrap180(c - lon_ref))
        if d < best_d:
            lon = c
            best_d = d
    return Tuple[Float64, Float64](lat, lon)


def _wrap180(x: Float64) -> Float64:
    return ((x + 180.0) % 360.0) - 180.0


def _is_nan(x: Float64) -> Bool:
    return isnan(x)


# ===================================================================
# pyModeS/decoder/bds/_helpers.py
# ===================================================================


def signed(value: Int, width: Int, sign: Int) -> Int:
    """Sign-magnitude to signed int. Not two's complement, by design."""
    if sign != 0:
        return value - (1 << width)
    return value


def normalise_angle(deg: Float64) -> Float64:
    """Wrap an angle into [0, 360)."""
    return deg % 360.0


def wrong_status(
    payload: Int, status_bit: Int, value_start: Int, value_width: Int
) -> Bool:
    """True if a status-gated value field is inconsistent."""
    var status = (payload >> (55 - status_bit)) & 0x1
    if status != 0:
        return False
    var value_shift = 55 - (value_start + value_width - 1)
    var value = (payload >> value_shift) & ((1 << value_width) - 1)
    return value != 0


# ===================================================================
# pyModeS/decoder/bds/bds05.py
# ===================================================================


struct Bds05:
    var altitude: Optional[Int]
    var surveillance_status: Int
    var nic_b: Int
    var cpr_format: Int
    var cpr_lat: Int
    var cpr_lon: Int
    var nuc_p: Int

    def __init__(out self,
        altitude: Optional[Int],
        surveillance_status: Int,
        nic_b: Int,
        cpr_format: Int,
        cpr_lat: Int,
        cpr_lon: Int,
        nuc_p: Int,
    ):
        self.altitude = altitude
        self.surveillance_status = surveillance_status
        self.nic_b = nic_b
        self.cpr_format = cpr_format
        self.cpr_lat = cpr_lat
        self.cpr_lon = cpr_lon
        self.nuc_p = nuc_p


def decode_bds05(payload: Int, tc: Int) -> Bds05:
    """Decode a BDS 0,5 payload (ADS-B airborne position).

    DIVERGENCE: `tc` is positional (Mojo has no keyword-only parameters) and
    the dict becomes a struct of the same field names.
    """
    var ss = (payload >> 49) & 0x3
    var nic_b = (payload >> 48) & 0x1
    var ac = (payload >> 36) & 0xFFF
    var cpr_format = (payload >> 34) & 0x1
    var cpr_lat = (payload >> 17) & 0x1FFFF
    var cpr_lon = payload & 0x1FFFF

    var altitude: Optional[Int] = None
    if 9 <= tc <= 18:
        # Barometric altitude: insert a zero M bit at position 6 of the
        # 13-bit altcode.
        var altcode = ((ac >> 6) << 7) | (ac & 0x3F)
        altitude = altcode_to_altitude(altcode)
    elif 20 <= tc <= 22:
        # GNSS altitude: 12-bit integer meters converted to feet.
        altitude = Int(Float64(ac) * 3.28084)
    else:
        altitude = None

    return Bds05(
        altitude, ss, nic_b, cpr_format, cpr_lat, cpr_lon, TC_NUCp_lookup[tc]
    )


# ===================================================================
# pyModeS/decoder/bds/bds06.py
# ===================================================================

comptime _MOV_LB = StaticTuple[Int, 7](2, 9, 13, 39, 94, 109, 124)
comptime _KTS_LB = StaticTuple[Float64, 7](
    0.125, 1.0, 2.0, 15.0, 70.0, 100.0, 175.0
)
comptime _STEP = StaticTuple[Float64, 6](0.125, 0.25, 0.5, 1.0, 2.0, 5.0)


def _decode_movement(mov: Int) -> Optional[Float64]:
    """Decode the 7-bit movement field to ground speed in knots."""
    if mov == 0 or mov > 124:
        return None
    if mov == 1:
        return 0.0
    if mov == 124:
        return 175.0
    # Smallest bin index whose lower bound is > mov.
    var i = 0
    while i < 7:
        if _MOV_LB[i] > mov:
            return _KTS_LB[i - 1] + _py_mul(
                Float64(mov - _MOV_LB[i - 1]), _STEP[i - 1]
            )
        i += 1
    return None


struct Bds06:
    var movement: Int
    var groundspeed: Optional[Float64]
    var track: Optional[Float64]
    var track_status: Int
    var cpr_format: Int
    var cpr_lat: Int
    var cpr_lon: Int

    def __init__(out self,
        movement: Int,
        groundspeed: Optional[Float64],
        track: Optional[Float64],
        track_status: Int,
        cpr_format: Int,
        cpr_lat: Int,
        cpr_lon: Int,
    ):
        self.movement = movement
        self.groundspeed = groundspeed
        self.track = track
        self.track_status = track_status
        self.cpr_format = cpr_format
        self.cpr_lat = cpr_lat
        self.cpr_lon = cpr_lon


def decode_bds06(payload: Int) -> Bds06:
    """Decode a BDS 0,6 payload (ADS-B surface position)."""
    var mov = (payload >> 44) & 0x7F
    var track_status = (payload >> 43) & 0x1
    var track_raw = (payload >> 36) & 0x7F
    var cpr_format = (payload >> 34) & 0x1
    var cpr_lat = (payload >> 17) & 0x1FFFF
    var cpr_lon = payload & 0x1FFFF

    var track: Optional[Float64] = None
    if track_status == 1:
        track = Float64(track_raw) * 360.0 / 128.0

    return Bds06(
        mov,
        _decode_movement(mov),
        track,
        track_status,
        cpr_format,
        cpr_lat,
        cpr_lon,
    )


# ===================================================================
# pyModeS/decoder/bds/bds08.py
# ===================================================================


def wake_vortex_category(tc: Int, category: Int) -> Int:
    """Index into `_WAKE_VORTEX_NAMES` for a (tc, category) pair.

    DIVERGENCE: upstream's `_WAKE_VORTEX` is a dict of strings and a string
    cannot cross the C ABI, so the lookup yields an index into the same table
    kept in python/mojopymodes/decoder/bds/bds08.py.
    """
    if tc < 0 or tc > 4 or category < 0 or category > 7:
        return 17
    return _WAKE_VORTEX[tc * 8 + category]


struct Bds08:
    var category: Int
    var callsign: String
    var wake_vortex: Int

    def __init__(out self,
        category: Int,
        callsign: String,
        wake_vortex: Int,
    ):
        self.category = category
        self.callsign = callsign
        self.wake_vortex = wake_vortex


def decode_bds08(payload: Int) -> Bds08:
    """Decode a BDS 0,8 payload (ADS-B identification)."""
    var tc = (payload >> 51) & 0x1F
    var category = (payload >> 48) & 0x7
    var cs_bits = payload & ((1 << 48) - 1)

    var callsign = decode_callsign(cs_bits)
    var wake_vortex = 17
    if category != 0 and tc != 1:
        wake_vortex = wake_vortex_category(tc, category)

    return Bds08(category, callsign, wake_vortex)


# ===================================================================
# pyModeS/decoder/bds/bds09.py
# ===================================================================


comptime _RAD_TO_DEG = 180.0 / pi  # what math.degrees multiplies by


def _decode_ground_speed(payload: Int, subtype: Int) -> Tuple[Int, Float64]:
    """Subtype 1 (subsonic) or 2 (supersonic) ground-speed decoding.

    DIVERGENCE: upstream's pair is `(float, float)` and wraps the speed in
    `int()`; here the speed is already an Int, because the C ABI cannot tell
    an int-valued float from an int once it has crossed.
    """
    var v_ew_sign = (payload >> 42) & 0x1
    var v_ew_mag = (payload >> 32) & 0x3FF
    var v_ns_sign = (payload >> 31) & 0x1
    var v_ns_mag = (payload >> 21) & 0x3FF

    var v_ew = Float64(v_ew_mag - 1)
    var v_ns = Float64(v_ns_mag - 1)
    if subtype == 2:
        v_ew *= 4.0
        v_ns *= 4.0

    var v_we = -v_ew if v_ew_sign == 1 else v_ew
    var v_sn = -v_ns if v_ns_sign == 1 else v_ns

    var spd = Int(floor(sqrt(v_we * v_we + v_sn * v_sn)))
    var trk = atan2(v_we, v_sn) * _RAD_TO_DEG
    if trk < 0.0:
        trk += 360.0

    return Tuple[Int, Float64](spd, trk)


def _decode_air_speed(payload: Int, subtype: Int) -> Tuple[Float64, Float64]:
    """Subtype 3 (subsonic) or 4 (supersonic) airspeed decoding.

    DIVERGENCE: returns (heading, airspeed) with NaN heading for "status
    clear" and airspeed -1 for "not available", because a Mojo Tuple cannot
    hold `None`. `decode_bds09` below turns both back into `Optional`.
    """
    var hdg_status = (payload >> 42) & 0x1
    var hdg_raw = (payload >> 32) & 0x3FF
    var as_mag = (payload >> 21) & 0x3FF

    var heading = nan[DType.float64]()
    if hdg_status != 0:
        heading = Float64(hdg_raw) / 1024.0 * 360.0

    var airspeed = -1.0
    if as_mag != 0:
        if subtype == 4:
            airspeed = Float64((as_mag - 1) * 4)
        else:
            airspeed = Float64(as_mag - 1)

    return Tuple[Float64, Float64](heading, airspeed)


struct Bds09:
    var subtype: Int
    var nac_v: Int
    var groundspeed: Optional[Int]
    var track: Optional[Float64]
    var heading: Optional[Float64]
    var airspeed: Optional[Int]
    var airspeed_type: Int
    var vr_source: Int
    var vertical_rate: Optional[Int]
    var geo_minus_baro: Optional[Int]

    def __init__(out self,
        subtype: Int,
        nac_v: Int,
        groundspeed: Optional[Int],
        track: Optional[Float64],
        heading: Optional[Float64],
        airspeed: Optional[Int],
        airspeed_type: Int,
        vr_source: Int,
        vertical_rate: Optional[Int],
        geo_minus_baro: Optional[Int],
    ):
        self.subtype = subtype
        self.nac_v = nac_v
        self.groundspeed = groundspeed
        self.track = track
        self.heading = heading
        self.airspeed = airspeed
        self.airspeed_type = airspeed_type
        self.vr_source = vr_source
        self.vertical_rate = vertical_rate
        self.geo_minus_baro = geo_minus_baro


def decode_bds09(payload: Int) -> Bds09:
    """Decode a BDS 0,9 payload (ADS-B airborne velocity, TC=19)."""
    var subtype = (payload >> 48) & 0x7
    var nac_v = (payload >> 43) & 0x7

    var groundspeed: Optional[Int] = None
    var track: Optional[Float64] = None
    var heading: Optional[Float64] = None
    var airspeed: Optional[Int] = None
    var airspeed_type = 0

    if subtype == 1 or subtype == 2:
        # Mirrors the "not available" early return inside
        # _decode_ground_speed, kept here so the branch order is unchanged.
        var v_ew_mag = (payload >> 32) & 0x3FF
        var v_ns_mag = (payload >> 21) & 0x3FF
        if v_ew_mag != 0 and v_ns_mag != 0:
            var sp = _decode_ground_speed(payload, subtype)
            groundspeed = sp[0]
            track = sp[1]
    elif subtype == 3 or subtype == 4:
        var as_mag = (payload >> 21) & 0x3FF
        airspeed_type = (payload >> 31) & 0x1
        var hs = _decode_air_speed(payload, subtype)
        if not _is_nan(hs[0]):
            heading = hs[0]
        if as_mag != 0:
            airspeed = Int(hs[1])

    var vr_source = (payload >> 20) & 0x1
    var vr_sign = (payload >> 19) & 0x1
    var vr_mag = (payload >> 10) & 0x1FF
    var vertical_rate: Optional[Int] = None
    if vr_mag != 0:
        var sign = -1 if vr_sign == 1 else 1
        vertical_rate = sign * (vr_mag - 1) * 64

    var diff_sign = (payload >> 7) & 0x1
    var diff_mag = payload & 0x7F
    var geo_minus_baro: Optional[Int] = None
    if diff_mag != 0 and diff_mag != 127:
        var sign = -1 if diff_sign == 1 else 1
        geo_minus_baro = sign * (diff_mag - 1) * 25

    return Bds09(
        subtype,
        nac_v,
        groundspeed,
        track,
        heading,
        airspeed,
        airspeed_type,
        vr_source,
        vertical_rate,
        geo_minus_baro,
    )


# ===================================================================
# pyModeS/decoder/bds/bds30.py
# ===================================================================


def is_bds30(payload: Int) -> Bool:
    """True if `payload` is a plausible BDS 3,0 ACAS RA report."""
    if payload == 0:
        return False
    if (payload >> 48) & 0xFF != 0x30:
        return False
    var ara_reserved = (payload >> (55 - 21)) & 0x7F
    if ara_reserved >= 48:
        return False
    var tti = (payload >> (55 - 29)) & 0x3
    return tti != 0b11


struct Bds30(Copyable):
    var threat_type_indicator: Int
    var issued_ra: Bool
    var corrective: Bool
    var downward_sense: Bool
    var increased_rate: Bool
    var sense_reversal: Bool
    var altitude_crossing: Bool
    var positive: Bool
    var no_below: Bool
    var no_above: Bool
    var no_left: Bool
    var no_right: Bool
    var ra_terminated: Bool
    var multiple_threat: Bool
    var threat_icao: Int
    var threat_altitude: Optional[Int]
    var threat_range: Optional[Float64]
    var threat_bearing: Optional[Float64]

    def __init__(out self,
        threat_type_indicator: Int,
        issued_ra: Bool,
        corrective: Bool,
        downward_sense: Bool,
        increased_rate: Bool,
        sense_reversal: Bool,
        altitude_crossing: Bool,
        positive: Bool,
        no_below: Bool,
        no_above: Bool,
        no_left: Bool,
        no_right: Bool,
        ra_terminated: Bool,
        multiple_threat: Bool,
        threat_icao: Int,
        threat_altitude: Optional[Int],
        threat_range: Optional[Float64],
        threat_bearing: Optional[Float64],
    ):
        self.threat_type_indicator = threat_type_indicator
        self.issued_ra = issued_ra
        self.corrective = corrective
        self.downward_sense = downward_sense
        self.increased_rate = increased_rate
        self.sense_reversal = sense_reversal
        self.altitude_crossing = altitude_crossing
        self.positive = positive
        self.no_below = no_below
        self.no_above = no_above
        self.no_left = no_left
        self.no_right = no_right
        self.ra_terminated = ra_terminated
        self.multiple_threat = multiple_threat
        self.threat_icao = threat_icao
        self.threat_altitude = threat_altitude
        self.threat_range = threat_range
        self.threat_bearing = threat_bearing

    def __init__(out self, *, copy: Self):
        self.threat_type_indicator = copy.threat_type_indicator
        self.issued_ra = copy.issued_ra
        self.corrective = copy.corrective
        self.downward_sense = copy.downward_sense
        self.increased_rate = copy.increased_rate
        self.sense_reversal = copy.sense_reversal
        self.altitude_crossing = copy.altitude_crossing
        self.positive = copy.positive
        self.no_below = copy.no_below
        self.no_above = copy.no_above
        self.no_left = copy.no_left
        self.no_right = copy.no_right
        self.ra_terminated = copy.ra_terminated
        self.multiple_threat = copy.multiple_threat
        self.threat_icao = copy.threat_icao
        self.threat_altitude = copy.threat_altitude
        self.threat_range = copy.threat_range
        self.threat_bearing = copy.threat_bearing


def decode_bds30(payload: Int) -> Bds30:
    """Decode a BDS 3,0 ACAS Active RA report.

    DIVERGENCE: `threat_icao` is an int here and is formatted as 6 hex digits
    by the Python layer; every other field is as upstream.
    """
    var tti = (payload >> (55 - 29)) & 0x3

    var threat_icao = 0
    var threat_altitude: Optional[Int] = None
    var threat_range: Optional[Float64] = None
    var threat_bearing: Optional[Float64] = None

    if tti == 1:
        threat_icao = (payload >> 2) & 0xFFFFFF
    elif tti == 2:
        threat_altitude = altcode_to_altitude((payload >> 13) & 0x1FFF)
        var range_raw = (payload >> 6) & 0x7F
        if range_raw > 0:
            threat_range = Float64(range_raw - 1) / 10.0
        var bearing_raw = payload & 0x3F
        if bearing_raw > 0:
            threat_bearing = Float64(6 * (bearing_raw - 1) + 3)

    return Bds30(
        tti,
        (payload >> (55 - 8)) & 0x1 != 0,
        (payload >> (55 - 9)) & 0x1 != 0,
        (payload >> (55 - 10)) & 0x1 != 0,
        (payload >> (55 - 11)) & 0x1 != 0,
        (payload >> (55 - 12)) & 0x1 != 0,
        (payload >> (55 - 13)) & 0x1 != 0,
        (payload >> (55 - 14)) & 0x1 != 0,
        (payload >> (55 - 22)) & 0x1 != 0,
        (payload >> (55 - 23)) & 0x1 != 0,
        (payload >> (55 - 24)) & 0x1 != 0,
        (payload >> (55 - 25)) & 0x1 != 0,
        (payload >> (55 - 26)) & 0x1 != 0,
        (payload >> (55 - 27)) & 0x1 != 0,
        threat_icao,
        threat_altitude,
        threat_range,
        threat_bearing,
    )


# ===================================================================
# pyModeS/decoder/bds/bds61.py
# ===================================================================


struct Bds61:
    var subtype: Int
    var emergency_state: Int
    var squawk: String
    var ra: Bds30

    def __init__(out self,
        subtype: Int,
        emergency_state: Int,
        squawk: String,
        ra: Bds30,
    ):
        self.subtype = subtype
        self.emergency_state = emergency_state
        self.squawk = squawk
        self.ra = ra.copy()


def decode_bds61(payload: Int) -> Bds61:
    """Decode a BDS 6,1 payload (ADS-B aircraft status)."""
    var subtype = (payload >> 48) & 0x7

    if subtype == 1:
        var emergency_state = (payload >> 45) & 0x7
        var idcode = (payload >> 32) & 0x1FFF
        var ra = decode_bds30(0)
        return Bds61(
            subtype, emergency_state, idcode_to_squawk(idcode), ra^
        )

    if subtype == 2:
        # Build a synthetic BDS 3,0 payload: 0x30 in bits 0-7, ACAS RA
        # payload in bits 8-55 (copied verbatim).
        var ra_payload = payload & ((1 << 48) - 1)
        var synthetic_bds30 = (0x30 << 48) | ra_payload
        var ra = decode_bds30(synthetic_bds30)
        return Bds61(subtype, 0, String(""), ra^)

    var ra = decode_bds30(0)
    return Bds61(subtype, 0, String(""), ra^)


# ===================================================================
# pyModeS/decoder/bds/bds62.py
# ===================================================================

comptime _BDS62_NA = 0
comptime _BDS62_MCP = 1
comptime _BDS62_FMS = 2


struct Bds62:
    var subtype: Int
    var selected_altitude: Optional[Int]
    var selected_altitude_source: Int
    var baro_pressure_setting: Optional[Float64]
    var selected_heading: Optional[Float64]
    var nac_p: Int
    var nic_baro: Int
    var sil: Int
    var autopilot: Optional[Bool]
    var vnav_mode: Optional[Bool]
    var altitude_hold_mode: Optional[Bool]
    var approach_mode: Optional[Bool]
    var lnav_mode: Optional[Bool]
    var tcas_operational: Bool

    def __init__(out self,
        subtype: Int,
        selected_altitude: Optional[Int],
        selected_altitude_source: Int,
        baro_pressure_setting: Optional[Float64],
        selected_heading: Optional[Float64],
        nac_p: Int,
        nic_baro: Int,
        sil: Int,
        autopilot: Optional[Bool],
        vnav_mode: Optional[Bool],
        altitude_hold_mode: Optional[Bool],
        approach_mode: Optional[Bool],
        lnav_mode: Optional[Bool],
        tcas_operational: Bool,
    ):
        self.subtype = subtype
        self.selected_altitude = selected_altitude
        self.selected_altitude_source = selected_altitude_source
        self.baro_pressure_setting = baro_pressure_setting
        self.selected_heading = selected_heading
        self.nac_p = nac_p
        self.nic_baro = nic_baro
        self.sil = sil
        self.autopilot = autopilot
        self.vnav_mode = vnav_mode
        self.altitude_hold_mode = altitude_hold_mode
        self.approach_mode = approach_mode
        self.lnav_mode = lnav_mode
        self.tcas_operational = tcas_operational


def decode_bds62(payload: Int) -> Bds62:
    """Decode a BDS 6,2 payload (ADS-B target state and status).

    DIVERGENCE: the three string-valued fields are returned as small integer
    codes (`_BDS62_*`), because a String cannot cross the C ABI. The mapping
    back to "N/A" / "MCP/FCU" / "FMS" lives in the Python layer.
    """
    var subtype = (payload >> 49) & 0x3

    var alt_source_bit = (payload >> 47) & 0x1
    var alt_raw = (payload >> 36) & 0x7FF
    var selected_altitude: Optional[Int] = None
    var selected_altitude_source = _BDS62_NA
    if alt_raw != 0:
        selected_altitude = (alt_raw - 1) * 32
        selected_altitude_source = (
            _BDS62_FMS if alt_source_bit == 1 else _BDS62_MCP
        )

    var baro_raw = (payload >> 27) & 0x1FF
    var baro_pressure_setting: Optional[Float64] = None
    if baro_raw != 0:
        baro_pressure_setting = 800.0 + _py_mul(Float64(baro_raw - 1), 0.8)

    var hdg_status = (payload >> 26) & 0x1
    var hdg_raw = (payload >> 17) & 0x1FF
    var selected_heading: Optional[Float64] = None
    if hdg_status != 0:
        selected_heading = Float64(hdg_raw) * 360.0 / 512.0

    var nac_p = (payload >> 13) & 0xF
    var nic_baro = (payload >> 12) & 0x1
    var sil = (payload >> 10) & 0x3

    var mode_status = (payload >> 9) & 0x1
    var autopilot: Optional[Bool] = None
    var vnav_mode: Optional[Bool] = None
    var altitude_hold_mode: Optional[Bool] = None
    var approach_mode: Optional[Bool] = None
    var lnav_mode: Optional[Bool] = None
    if mode_status == 0:
        autopilot = None
        vnav_mode = None
        altitude_hold_mode = None
        approach_mode = None
        lnav_mode = None
    else:
        autopilot = (payload >> 8) & 0x1 != 0
        vnav_mode = (payload >> 7) & 0x1 != 0
        altitude_hold_mode = (payload >> 6) & 0x1 != 0
        approach_mode = (payload >> 4) & 0x1 != 0
        lnav_mode = (payload >> 2) & 0x1 != 0

    return Bds62(
        subtype,
        selected_altitude,
        selected_altitude_source,
        baro_pressure_setting,
        selected_heading,
        nac_p,
        nic_baro,
        sil,
        autopilot,
        vnav_mode,
        altitude_hold_mode,
        approach_mode,
        lnav_mode,
        (payload >> 3) & 0x1 != 0,
    )


# ===================================================================
# pyModeS/decoder/bds/bds65.py
# ===================================================================


struct Bds65:
    var subtype: Int
    var version: Int
    var nic_supplement_a: Int
    var nac_p: Int
    var sil: Int
    var hrd: Int
    var capability_class: Int
    var operational_mode: Int
    var nic_baro: Int
    var has_nic_baro: Bool
    var sil_supplement: Int
    var has_sil_supplement: Bool

    def __init__(out self,
        subtype: Int,
        version: Int,
        nic_supplement_a: Int,
        nac_p: Int,
        sil: Int,
        hrd: Int,
        capability_class: Int,
        operational_mode: Int,
        nic_baro: Int,
        has_nic_baro: Bool,
        sil_supplement: Int,
        has_sil_supplement: Bool,
    ):
        self.subtype = subtype
        self.version = version
        self.nic_supplement_a = nic_supplement_a
        self.nac_p = nac_p
        self.sil = sil
        self.hrd = hrd
        self.capability_class = capability_class
        self.operational_mode = operational_mode
        self.nic_baro = nic_baro
        self.has_nic_baro = has_nic_baro
        self.sil_supplement = sil_supplement
        self.has_sil_supplement = has_sil_supplement


def decode_bds65(payload: Int) -> Bds65:
    """Decode a BDS 6,5 payload (ADS-B operational status).

    DIVERGENCE: `nic_baro` and `sil_supplement` are only present for some
    subtypes, and upstream expresses that by omitting the key. Mojo structs
    have no optional fields, so each carries an explicit presence flag; the
    Python layer uses the flags to decide which keys to build.
    """
    var subtype = (payload >> 48) & 0x7

    var capability_class = (payload >> 32) & 0xFFFF
    var operational_mode = (payload >> 16) & 0xFFFF
    var version = (payload >> 13) & 0x7
    var nic_supplement_a = (payload >> 12) & 0x1
    var nac_p = (payload >> 8) & 0xF
    var sil = (payload >> 4) & 0x3
    var hrd = (payload >> 2) & 0x1

    var has_nic_baro = False
    var nic_baro = 0
    if subtype == 0 and version >= 1:
        nic_baro = (payload >> 3) & 0x1
        has_nic_baro = True

    var has_sil_supplement = False
    var sil_supplement = 0
    if version == 2:
        sil_supplement = (payload >> 1) & 0x1
        has_sil_supplement = True

    return Bds65(
        subtype,
        version,
        nic_supplement_a,
        nac_p,
        sil,
        hrd,
        capability_class,
        operational_mode,
        nic_baro,
        has_nic_baro,
        sil_supplement,
        has_sil_supplement,
    )
