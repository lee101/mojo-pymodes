"""The C ABI that python/mojopymodes calls through.

Every buffer crosses as an `Int` address and is rebuilt inside the wrapper,
because `@export` rejects parametric functions and an inferred pointer origin
is parametric.

Two shapes of divergence live here and nowhere else, so `ported.mojo` can stay
a literal transcription of pyModeS:

  * Results. A ported function that returns a dict in Python returns a struct
    here. The struct is flattened into one row of `Float64` slots per message,
    every slot 8 bytes wide, plus a `UInt8` strip for the registers that carry
    a string. Slot 0 of a row is a bitmask of which keys the dict had and
    slot 1 is a bitmask of which of those keys were not `None`, so the Python
    side never re-derives a branch Mojo already took.

  * 112-bit messages. Python ints are unbounded and a Mojo `Int` is 64 bits,
    so a long message crosses as two words (`hi` = n >> 64, `lo` = n & 2**64-1)
    and `_field` / `crc_remainder_wide` read it out of the pair. Every
    extracted field in this file stays below 2**48, so the Float64 slots hold
    integer fields exactly.
"""

from max.algorithm import parallelize
from ported import (
    _gray2int,
    _is_nan,
    airborne_position_pair,
    airborne_position_with_ref,
    altcode_to_altitude,
    crc_remainder,
    cprNL,
    decode_bds05,
    decode_bds06,
    decode_bds08,
    decode_bds09,
    decode_bds30,
    decode_bds61,
    decode_bds62,
    decode_bds65,
    decode_callsign,
    extract_signed,
    extract_unsigned,
    idcode_to_squawk,
    is_bds30,
    is_valid_callsign_char,
    normalise_angle,
    signed,
    surface_position_pair,
    surface_position_with_ref,
    wake_vortex_category,
    wrong_status,
)
from tables import _CRC_TABLE

comptime PtrI = Pointer[Int, AnyOrigin[mut=True]]
comptime PtrF = Pointer[Float64, AnyOrigin[mut=True]]
comptime PtrU8 = Pointer[UInt8, AnyOrigin[mut=True]]

# Below this many messages a thread launch costs more than the work itself.
comptime PAR_THRESHOLD = 4096
comptime PAR_WORKERS = 8

# A decoded string is never longer than this, so one loop bound covers every
# string pyModeS produces from a 56-bit payload.
comptime STR_CAP = 16


@always_inline
def _str(dst: PtrU8, base: Int, s: String) -> Int:
    var i = 0
    for b in s.as_bytes():
        if i >= STR_CAP:
            break
        dst.unsafe_store(base + i, b)
        i += 1
    return i


@always_inline
def _hex6(dst: PtrU8, base: Int, value: Int):
    # Same characters as f"{value:06X}".
    var digits = "0123456789ABCDEF"
    for k in range(6):
        var nib = (value >> (4 * (5 - k))) & 0xF
        dst.unsafe_store(base + k, UInt8(ord(digits[byte=nib])))


# ------------------------------------------------------- pyModeS/_bits.py


@always_inline
def _field(hi: Int, lo: Int, total: Int, start: Int, width: Int) -> Int:
    """`extract_unsigned` over a (hi, lo) word pair."""
    var shift = total - start - width
    var mask = (1 << width) - 1
    if shift >= 64:
        return (hi >> (shift - 64)) & mask
    return (lo >> shift) & mask


@always_inline
def _byte_at(hi: Int, lo: Int, shift: Int) -> Int:
    if shift >= 64:
        return (hi >> (shift - 64)) & 0xFF
    return (lo >> shift) & 0xFF


def crc_remainder_wide(hi: Int, lo: Int, length: Int) -> Int:
    """`crc_remainder` for a message that does not fit in one 64-bit word.

    DIVERGENCE: no upstream counterpart, because upstream's Python int holds
    all 112 bits in one value. The algorithm and the table are the ones in
    `crc_remainder`; only the byte fetch reads a word pair.
    """
    var n_data_bytes = (length - 24) // 8
    var crc = 0
    var shift = length - 8
    for _ in range(n_data_bytes):
        var byte = _byte_at(hi, lo, shift)
        crc = ((crc << 8) & 0xFFFFFF) ^ _CRC_TABLE[((crc >> 16) ^ byte) & 0xFF]
        shift -= 8
    var parity = lo & 0xFFFFFF
    return (crc ^ parity) & 0xFFFFFF


@export("mp_extract_unsigned")
def mp_extract_unsigned(n: Int, start: Int, width: Int, total_bits: Int) abi("C") -> Int:
    return extract_unsigned(n, start, width, total_bits)


@export("mp_extract_signed")
def mp_extract_signed(n: Int, start: Int, width: Int, total_bits: Int) abi("C") -> Int:
    return extract_signed(n, start, width, total_bits)


@export("mp_crc_remainder")
def mp_crc_remainder(n: Int, length: Int) abi("C") -> Int:
    return crc_remainder(n, length)


@export("mp_crc_remainder_wide")
def mp_crc_remainder_wide(hi: Int, lo: Int, length: Int) abi("C") -> Int:
    return crc_remainder_wide(hi, lo, length)


@export("mp_crc_remainder_batch")
def mp_crc_remainder_batch(hi: Int, lo: Int, lengths: Int, n: Int, dst: Int) abi("C"):
    var hp = PtrI(unsafe_from_address=hi)
    var lp = PtrI(unsafe_from_address=lo)
    var gp = PtrI(unsafe_from_address=lengths)
    var dp = PtrI(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        dp.unsafe_store(
            i,
            crc_remainder_wide(
                hp.unsafe_load(i), lp.unsafe_load(i), gp.unsafe_load(i)
            ),
        )

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)


# ----------------------------------------------------- pyModeS/_altcode.py


@export("mp_altcode_to_altitude")
def mp_altcode_to_altitude(ac: Int, dst: Int) abi("C") -> Int:
    var p = PtrF(unsafe_from_address=dst)
    var a = altcode_to_altitude(ac)
    if a is None:
        p.unsafe_store(1, 0.0)
        return 0
    p.unsafe_store(0, Float64(a.value()))
    p.unsafe_store(1, 1.0)
    return 1


@export("mp_altcode_to_altitude_batch")
def mp_altcode_to_altitude_batch(acs: Int, n: Int, dst: Int) abi("C"):
    var ap = PtrI(unsafe_from_address=acs)
    var dp = PtrF(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        var a = altcode_to_altitude(ap.unsafe_load(i))
        var row = dp.unsafe_offset(2 * i)
        if a is None:
            row.unsafe_store(1, 0.0)
        else:
            row.unsafe_store(0, Float64(a.value()))
            row.unsafe_store(1, 1.0)

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)


@export("mp_gray2int")
def mp_gray2int(n: Int) abi("C") -> Int:
    return _gray2int(n)


# ---------------------------------------------------- pyModeS/_callsign.py


@export("mp_decode_callsign")
def mp_decode_callsign(bits: Int, dst: Int) abi("C") -> Int:
    var p = PtrU8(unsafe_from_address=dst)
    return _str(p, 0, decode_callsign(bits))


@export("mp_decode_callsign_batch")
def mp_decode_callsign_batch(bits: Int, n: Int, dst: Int) abi("C") -> Int:
    var bp = PtrI(unsafe_from_address=bits)
    var dp = PtrU8(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        _ = _str(dp, 8 * i, decode_callsign(bp.unsafe_load(i)))

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)
    return 0


@export("mp_is_valid_callsign_char")
def mp_is_valid_callsign_char(idx: Int) abi("C") -> Int:
    return 1 if is_valid_callsign_char(idx) else 0


# ------------------------------------------------------- pyModeS/_idcode.py


@export("mp_idcode_to_squawk")
def mp_idcode_to_squawk(idcode: Int, dst: Int) abi("C") -> Int:
    var p = PtrU8(unsafe_from_address=dst)
    return _str(p, 0, idcode_to_squawk(idcode))


@export("mp_idcode_to_squawk_batch")
def mp_idcode_to_squawk_batch(codes: Int, n: Int, dst: Int) abi("C") -> Int:
    var cp = PtrI(unsafe_from_address=codes)
    var dp = PtrU8(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        _ = _str(dp, 4 * i, idcode_to_squawk(cp.unsafe_load(i)))

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)
    return 0


# ------------------------------------------------- pyModeS/position/_cpr.py


@export("mp_cprNL")
def mp_cprNL(lat: Float64) abi("C") -> Int:
    return cprNL(lat)


@export("mp_cprNL_batch")
def mp_cprNL_batch(lats: Int, n: Int, dst: Int) abi("C"):
    var lp = PtrF(unsafe_from_address=lats)
    var dp = PtrI(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        dp.unsafe_store(i, cprNL(lp.unsafe_load(i)))

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)


@export("mp_airborne_position_with_ref")
def mp_airborne_position_with_ref(
    cpr_format: Int,
    cpr_lat_raw: Int,
    cpr_lon_raw: Int,
    lat_ref: Float64,
    lon_ref: Float64,
    dst: Int,
) abi("C") -> Int:
    var p = PtrF(unsafe_from_address=dst)
    var t = airborne_position_with_ref(
        cpr_format, cpr_lat_raw, cpr_lon_raw, lat_ref, lon_ref
    )
    p.unsafe_store(0, t[0])
    p.unsafe_store(1, t[1])
    return 1


@export("mp_airborne_position_with_ref_batch")
def mp_airborne_position_with_ref_batch(
    formats: Int, lat_raw: Int, lon_raw: Int, lat_ref: Int, lon_ref: Int, n: Int, dst: Int
) abi("C"):
    var fp = PtrI(unsafe_from_address=formats)
    var ap = PtrI(unsafe_from_address=lat_raw)
    var bp = PtrI(unsafe_from_address=lon_raw)
    var rp = PtrF(unsafe_from_address=lat_ref)
    var sp = PtrF(unsafe_from_address=lon_ref)
    var dp = PtrF(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        var t = airborne_position_with_ref(
            fp.unsafe_load(i),
            ap.unsafe_load(i),
            bp.unsafe_load(i),
            rp.unsafe_load(i),
            sp.unsafe_load(i),
        )
        dp.unsafe_store(2 * i, t[0])
        dp.unsafe_store(2 * i + 1, t[1])

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)


@export("mp_airborne_position_pair")
def mp_airborne_position_pair(
    lat_even: Int,
    lon_even: Int,
    lat_odd: Int,
    lon_odd: Int,
    even_is_newer: Int,
    dst: Int,
) abi("C") -> Int:
    var p = PtrF(unsafe_from_address=dst)
    var t = airborne_position_pair(
        lat_even, lon_even, lat_odd, lon_odd, even_is_newer != 0
    )
    p.unsafe_store(0, t[0])
    p.unsafe_store(1, t[1])
    return 0 if _is_nan(t[0]) else 1


@export("mp_airborne_position_pair_batch")
def mp_airborne_position_pair_batch(
    lat_even: Int, lon_even: Int, lat_odd: Int, lon_odd: Int, newer: Int, n: Int, dst: Int
) abi("C"):
    var ep = PtrI(unsafe_from_address=lat_even)
    var gp = PtrI(unsafe_from_address=lon_even)
    var op = PtrI(unsafe_from_address=lat_odd)
    var qp = PtrI(unsafe_from_address=lon_odd)
    var np = PtrI(unsafe_from_address=newer)
    var dp = PtrF(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        var t = airborne_position_pair(
            ep.unsafe_load(i),
            gp.unsafe_load(i),
            op.unsafe_load(i),
            qp.unsafe_load(i),
            np.unsafe_load(i) != 0,
        )
        dp.unsafe_store(2 * i, t[0])
        dp.unsafe_store(2 * i + 1, t[1])

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)


@export("mp_surface_position_with_ref")
def mp_surface_position_with_ref(
    cpr_format: Int,
    cpr_lat_raw: Int,
    cpr_lon_raw: Int,
    lat_ref: Float64,
    lon_ref: Float64,
    dst: Int,
) abi("C") -> Int:
    var p = PtrF(unsafe_from_address=dst)
    var t = surface_position_with_ref(
        cpr_format, cpr_lat_raw, cpr_lon_raw, lat_ref, lon_ref
    )
    p.unsafe_store(0, t[0])
    p.unsafe_store(1, t[1])
    return 1


@export("mp_surface_position_with_ref_batch")
def mp_surface_position_with_ref_batch(
    formats: Int, lat_raw: Int, lon_raw: Int, lat_ref: Int, lon_ref: Int, n: Int, dst: Int
) abi("C"):
    var fp = PtrI(unsafe_from_address=formats)
    var ap = PtrI(unsafe_from_address=lat_raw)
    var bp = PtrI(unsafe_from_address=lon_raw)
    var rp = PtrF(unsafe_from_address=lat_ref)
    var sp = PtrF(unsafe_from_address=lon_ref)
    var dp = PtrF(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        var t = surface_position_with_ref(
            fp.unsafe_load(i),
            ap.unsafe_load(i),
            bp.unsafe_load(i),
            rp.unsafe_load(i),
            sp.unsafe_load(i),
        )
        dp.unsafe_store(2 * i, t[0])
        dp.unsafe_store(2 * i + 1, t[1])

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)


@export("mp_surface_position_pair")
def mp_surface_position_pair(
    lat_even: Int,
    lon_even: Int,
    lat_odd: Int,
    lon_odd: Int,
    lat_ref: Float64,
    lon_ref: Float64,
    even_is_newer: Int,
    dst: Int,
) abi("C") -> Int:
    var p = PtrF(unsafe_from_address=dst)
    var t = surface_position_pair(
        lat_even, lon_even, lat_odd, lon_odd, lat_ref, lon_ref, even_is_newer != 0
    )
    p.unsafe_store(0, t[0])
    p.unsafe_store(1, t[1])
    return 0 if _is_nan(t[0]) else 1


@export("mp_surface_position_pair_batch")
def mp_surface_position_pair_batch(
    lat_even: Int,
    lon_even: Int,
    lat_odd: Int,
    lon_odd: Int,
    lat_ref: Int,
    lon_ref: Int,
    newer: Int,
    n: Int,
    dst: Int,
) abi("C"):
    var ep = PtrI(unsafe_from_address=lat_even)
    var gp = PtrI(unsafe_from_address=lon_even)
    var op = PtrI(unsafe_from_address=lat_odd)
    var qp = PtrI(unsafe_from_address=lon_odd)
    var rp = PtrF(unsafe_from_address=lat_ref)
    var sp = PtrF(unsafe_from_address=lon_ref)
    var np = PtrI(unsafe_from_address=newer)
    var dp = PtrF(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        var t = surface_position_pair(
            ep.unsafe_load(i),
            gp.unsafe_load(i),
            op.unsafe_load(i),
            qp.unsafe_load(i),
            rp.unsafe_load(i),
            sp.unsafe_load(i),
            np.unsafe_load(i) != 0,
        )
        dp.unsafe_store(2 * i, t[0])
        dp.unsafe_store(2 * i + 1, t[1])

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)


# ------------------------------------------ pyModeS/decoder/bds/_helpers.py


@export("mp_signed")
def mp_signed(value: Int, width: Int, sign: Int) abi("C") -> Int:
    return signed(value, width, sign)


@export("mp_normalise_angle")
def mp_normalise_angle(deg: Float64) abi("C") -> Float64:
    return normalise_angle(deg)


@export("mp_wrong_status")
def mp_wrong_status(
    payload: Int, status_bit: Int, value_start: Int, value_width: Int
) abi("C") -> Int:
    return 1 if wrong_status(payload, status_bit, value_start, value_width) else 0


@export("mp_is_bds30")
def mp_is_bds30(payload: Int) abi("C") -> Int:
    return 1 if is_bds30(payload) else 0


@export("mp_wake_vortex_category")
def mp_wake_vortex_category(tc: Int, category: Int) abi("C") -> Int:
    return wake_vortex_category(tc, category)


# ============================================================ header decode
#
# Mirrors pyModeS.util: df, icao, typecode, crc, altcode, idcode.
#
# slots: 0 present  1 absent  2 df  3 icao  4 typecode  5 crc  6 crc_valid
#        7 altitude  8 idcode
comptime HDR_W = 9


@export("mp_decode_header")
def mp_decode_header(hi: Int, lo: Int, length: Int, n: Int, dst: Int) abi("C"):
    var hp = PtrI(unsafe_from_address=hi)
    var lp = PtrI(unsafe_from_address=lo)
    var gp = PtrI(unsafe_from_address=length)
    var dp = PtrF(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        var h = hp.unsafe_load(i)
        var l = lp.unsafe_load(i)
        var total = gp.unsafe_load(i)
        var row = dp.unsafe_offset(HDR_W * i)
        var present = 0
        var absent = 0

        # util.df: int(msg[:2], 16) >> 3, clamped at 24
        var top = _field(h, l, total, 0, 5)
        var dfv = 24 if top >= 24 else top
        row.unsafe_store(2, Float64(dfv))
        present |= 1

        # util.typecode: ME bits 32-36, DF17/18 only
        present |= 4
        if dfv == 17 or dfv == 18:
            row.unsafe_store(4, Float64(_field(h, l, total, 32, 5)))
        else:
            absent |= 4

        # util.crc
        var rem = crc_remainder_wide(h, l, total)
        row.unsafe_store(5, Float64(rem))
        row.unsafe_store(6, 1.0 if rem == 0 else 0.0)
        present |= 8 | 16

        # util.icao
        present |= 2
        if dfv == 11 or dfv == 17 or dfv == 18:
            row.unsafe_store(3, Float64(_field(h, l, total, 8, 24)))
        elif (
            dfv == 0 or dfv == 4 or dfv == 5 or dfv == 16 or dfv == 20 or dfv == 21
        ):
            row.unsafe_store(3, Float64(rem))
        else:
            absent |= 2

        # util.altcode / util.idcode: the 13-bit field at bits 19-31
        var ac = _field(h, l, total, 19, 13)
        present |= 32 | 64
        if dfv == 0 or dfv == 4 or dfv == 16 or dfv == 20:
            var alt = altcode_to_altitude(ac)
            if alt is None:
                absent |= 32
            else:
                row.unsafe_store(7, Float64(alt.value()))
        else:
            absent |= 32
        if dfv == 5 or dfv == 21:
            row.unsafe_store(8, Float64(ac))
        else:
            absent |= 64

        row.unsafe_store(0, Float64(present))
        row.unsafe_store(1, Float64(absent))

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)


# =================================================================== BDS 0,5
#
# slots: 0 present  1 absent  2 altitude  3 surveillance_status  4 nic_b
#        5 cpr_format  6 cpr_lat  7 cpr_lon  8 nuc_p
comptime B05_W = 9


@export("mp_decode_bds05")
def mp_decode_bds05(payloads: Int, tcs: Int, n: Int, dst: Int) abi("C"):
    var pp = PtrI(unsafe_from_address=payloads)
    var tp = PtrI(unsafe_from_address=tcs)
    var dp = PtrF(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        var row = dp.unsafe_offset(B05_W * i)
        var r = decode_bds05(pp.unsafe_load(i), tp.unsafe_load(i))
        var present = 0x7F
        var absent = 0
        if r.altitude is None:
            absent |= 1
        else:
            row.unsafe_store(2, Float64(r.altitude.value()))
        row.unsafe_store(3, Float64(r.surveillance_status))
        row.unsafe_store(4, Float64(r.nic_b))
        row.unsafe_store(5, Float64(r.cpr_format))
        row.unsafe_store(6, Float64(r.cpr_lat))
        row.unsafe_store(7, Float64(r.cpr_lon))
        row.unsafe_store(8, Float64(r.nuc_p))
        row.unsafe_store(0, Float64(present))
        row.unsafe_store(1, Float64(absent))

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)


# =================================================================== BDS 0,6
#
# slots: 0 present  1 absent  2 movement  3 groundspeed  4 track
#        5 track_status  6 cpr_format  7 cpr_lat  8 cpr_lon
comptime B06_W = 9


@export("mp_decode_bds06")
def mp_decode_bds06(payloads: Int, n: Int, dst: Int) abi("C"):
    var pp = PtrI(unsafe_from_address=payloads)
    var dp = PtrF(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        var row = dp.unsafe_offset(B06_W * i)
        var r = decode_bds06(pp.unsafe_load(i))
        var present = 0x7F
        var absent = 0
        if r.groundspeed is None:
            absent |= 2
        else:
            row.unsafe_store(3, r.groundspeed.value())
        if r.track is None:
            absent |= 4
        else:
            row.unsafe_store(4, r.track.value())
        row.unsafe_store(2, Float64(r.movement))
        row.unsafe_store(5, Float64(r.track_status))
        row.unsafe_store(6, Float64(r.cpr_format))
        row.unsafe_store(7, Float64(r.cpr_lat))
        row.unsafe_store(8, Float64(r.cpr_lon))
        row.unsafe_store(0, Float64(present))
        row.unsafe_store(1, Float64(absent))

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)


# =================================================================== BDS 0,8
#
# slots: 0 present  1 absent  2 category  3 wake_vortex
# 8 bytes of chars per message hold the stripped callsign.
comptime B08_W = 4


@export("mp_decode_bds08")
def mp_decode_bds08(payloads: Int, n: Int, dst: Int, chars: Int) abi("C") -> Int:
    var pp = PtrI(unsafe_from_address=payloads)
    var dp = PtrF(unsafe_from_address=dst)
    var cp = PtrU8(unsafe_from_address=chars)

    def body(i: Int) {imm}:
        var row = dp.unsafe_offset(B08_W * i)
        var r = decode_bds08(pp.unsafe_load(i))
        row.unsafe_store(2, Float64(r.category))
        row.unsafe_store(3, Float64(r.wake_vortex))
        row.unsafe_store(0, 7.0)
        row.unsafe_store(1, 0.0)
        _ = _str(cp, 8 * i, r.callsign)

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)
    return 0


# =================================================================== BDS 0,9
#
# slots: 0 present  1 absent  2 subtype  3 nac_v  4 groundspeed  5 track
#        6 heading  7 airspeed  8 airspeed_type  9 vr_source
#        10 vertical_rate  11 geo_minus_baro
comptime B09_W = 12


@export("mp_decode_bds09")
def mp_decode_bds09(payloads: Int, n: Int, dst: Int) abi("C"):
    var pp = PtrI(unsafe_from_address=payloads)
    var dp = PtrF(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        var row = dp.unsafe_offset(B09_W * i)
        var r = decode_bds09(pp.unsafe_load(i))
        var present = 0
        var absent = 0

        # bits 0 subtype, 1 nac_v, 2 groundspeed, 3 track, 4 heading,
        # 5 airspeed, 6 airspeed_type, 7 vr_source, 8 vertical_rate,
        # 9 geo_minus_baro
        row.unsafe_store(2, Float64(r.subtype))
        row.unsafe_store(3, Float64(r.nac_v))
        present |= 3 | 896

        if r.subtype == 1 or r.subtype == 2:
            present |= 12
            if r.groundspeed is None:
                absent |= 4
            else:
                row.unsafe_store(4, Float64(r.groundspeed.value()))
            if r.track is None:
                absent |= 8
            else:
                row.unsafe_store(5, r.track.value())
        elif r.subtype == 3 or r.subtype == 4:
            present |= 112
            if r.heading is None:
                absent |= 16
            else:
                row.unsafe_store(6, r.heading.value())
            if r.airspeed is None:
                absent |= 32
            else:
                row.unsafe_store(7, Float64(r.airspeed.value()))
            row.unsafe_store(8, Float64(r.airspeed_type))
        row.unsafe_store(9, Float64(r.vr_source))
        if r.vertical_rate is None:
            absent |= 256
        else:
            row.unsafe_store(10, Float64(r.vertical_rate.value()))
        if r.geo_minus_baro is None:
            absent |= 512
        else:
            row.unsafe_store(11, Float64(r.geo_minus_baro.value()))

        row.unsafe_store(0, Float64(present))
        row.unsafe_store(1, Float64(absent))

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)


# =================================================================== BDS 3,0
#
# slots: 0 present  1 absent  2 threat_type_indicator  3 issued_ra
#        4 corrective  5 downward_sense  6 increased_rate  7 sense_reversal
#        8 altitude_crossing  9 positive  10 no_below  11 no_above
#        12 no_left  13 no_right  14 ra_terminated  15 multiple_threat
#        16 threat_icao  17 threat_altitude  18 threat_range  19 threat_bearing
# 6 bytes of chars per message hold the 6 hex digits of threat_icao.
comptime B30_W = 20


@export("mp_decode_bds30")
def mp_decode_bds30(payloads: Int, n: Int, dst: Int, chars: Int) abi("C") -> Int:
    var pp = PtrI(unsafe_from_address=payloads)
    var dp = PtrF(unsafe_from_address=dst)
    var cp = PtrU8(unsafe_from_address=chars)

    def body(i: Int) {imm}:
        var row = dp.unsafe_offset(B30_W * i)
        var r = decode_bds30(pp.unsafe_load(i))
        var present = 0
        var absent = 0
        for k in range(14):
            present |= 1 << k
        row.unsafe_store(2, Float64(r.threat_type_indicator))
        row.unsafe_store(3, 1.0 if r.issued_ra else 0.0)
        row.unsafe_store(4, 1.0 if r.corrective else 0.0)
        row.unsafe_store(5, 1.0 if r.downward_sense else 0.0)
        row.unsafe_store(6, 1.0 if r.increased_rate else 0.0)
        row.unsafe_store(7, 1.0 if r.sense_reversal else 0.0)
        row.unsafe_store(8, 1.0 if r.altitude_crossing else 0.0)
        row.unsafe_store(9, 1.0 if r.positive else 0.0)
        row.unsafe_store(10, 1.0 if r.no_below else 0.0)
        row.unsafe_store(11, 1.0 if r.no_above else 0.0)
        row.unsafe_store(12, 1.0 if r.no_left else 0.0)
        row.unsafe_store(13, 1.0 if r.no_right else 0.0)
        row.unsafe_store(14, 1.0 if r.ra_terminated else 0.0)
        row.unsafe_store(15, 1.0 if r.multiple_threat else 0.0)
        if r.threat_type_indicator == 1:
            present |= 1 << 14
            row.unsafe_store(16, Float64(r.threat_icao))
            _hex6(cp, 6 * i, r.threat_icao)
        elif r.threat_type_indicator == 2:
            present |= 7 << 15
            if r.threat_altitude is None:
                absent |= 1 << 15
            else:
                row.unsafe_store(17, Float64(r.threat_altitude.value()))
            if r.threat_range is None:
                absent |= 1 << 16
            else:
                row.unsafe_store(18, r.threat_range.value())
            if r.threat_bearing is None:
                absent |= 1 << 17
            else:
                row.unsafe_store(19, r.threat_bearing.value())
        row.unsafe_store(0, Float64(present))
        row.unsafe_store(1, Float64(absent))

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)
    return 0


# =================================================================== BDS 6,1
#
# slots: 0 present  1 absent  2 subtype  3 emergency_state
#        4..21 the BDS 3,0 fields in the same order as B30_W, shifted by 2
#        (threat_type_indicator 4, issued_ra 5, ... threat_bearing 21)
# 8 bytes of chars per message hold the 4-digit squawk (subtype 1) or the
# 6 hex digits of threat_icao (subtype 2); only one is ever present.
comptime B61_W = 22


@export("mp_decode_bds61")
def mp_decode_bds61(payloads: Int, n: Int, dst: Int, chars: Int) abi("C") -> Int:
    var pp = PtrI(unsafe_from_address=payloads)
    var dp = PtrF(unsafe_from_address=dst)
    var cp = PtrU8(unsafe_from_address=chars)

    def body(i: Int) {imm}:
        var row = dp.unsafe_offset(B61_W * i)
        var r = decode_bds61(pp.unsafe_load(i))
        var present = 1
        var absent = 0
        row.unsafe_store(2, Float64(r.subtype))
        if r.subtype == 1:
            present |= 6
            row.unsafe_store(3, Float64(r.emergency_state))
            _ = _str(cp, 8 * i, r.squawk)
        elif r.subtype == 2:
            var ra = r.ra.copy()
            for k in range(14):
                present |= 1 << (3 + k)
            row.unsafe_store(4, Float64(ra.threat_type_indicator))
            row.unsafe_store(5, 1.0 if ra.issued_ra else 0.0)
            row.unsafe_store(6, 1.0 if ra.corrective else 0.0)
            row.unsafe_store(7, 1.0 if ra.downward_sense else 0.0)
            row.unsafe_store(8, 1.0 if ra.increased_rate else 0.0)
            row.unsafe_store(9, 1.0 if ra.sense_reversal else 0.0)
            row.unsafe_store(10, 1.0 if ra.altitude_crossing else 0.0)
            row.unsafe_store(11, 1.0 if ra.positive else 0.0)
            row.unsafe_store(12, 1.0 if ra.no_below else 0.0)
            row.unsafe_store(13, 1.0 if ra.no_above else 0.0)
            row.unsafe_store(14, 1.0 if ra.no_left else 0.0)
            row.unsafe_store(15, 1.0 if ra.no_right else 0.0)
            row.unsafe_store(16, 1.0 if ra.ra_terminated else 0.0)
            row.unsafe_store(17, 1.0 if ra.multiple_threat else 0.0)
            if ra.threat_type_indicator == 1:
                present |= 1 << 17
                row.unsafe_store(18, Float64(ra.threat_icao))
                _hex6(cp, 8 * i, ra.threat_icao)
            elif ra.threat_type_indicator == 2:
                present |= 7 << 18
                if ra.threat_altitude is None:
                    absent |= 1 << 18
                else:
                    row.unsafe_store(19, Float64(ra.threat_altitude.value()))
                if ra.threat_range is None:
                    absent |= 1 << 19
                else:
                    row.unsafe_store(20, ra.threat_range.value())
                if ra.threat_bearing is None:
                    absent |= 1 << 20
                else:
                    row.unsafe_store(21, ra.threat_bearing.value())
        row.unsafe_store(0, Float64(present))
        row.unsafe_store(1, Float64(absent))

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)
    return 0


# =================================================================== BDS 6,2
#
# slots: 0 present  1 absent  2 subtype  3 selected_altitude
#        4 selected_altitude_source  5 baro_pressure_setting
#        6 selected_heading  7 nac_p  8 nic_baro  9 sil  10 autopilot
#        11 vnav_mode  12 altitude_hold_mode  13 approach_mode
#        14 lnav_mode  15 tcas_operational
comptime B62_W = 16


@export("mp_decode_bds62")
def mp_decode_bds62(payloads: Int, n: Int, dst: Int) abi("C"):
    var pp = PtrI(unsafe_from_address=payloads)
    var dp = PtrF(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        var row = dp.unsafe_offset(B62_W * i)
        var r = decode_bds62(pp.unsafe_load(i))
        var present = 0x3FFF
        var absent = 0
        row.unsafe_store(2, Float64(r.subtype))
        # bits 0 subtype, 1 selected_altitude, 2 selected_altitude_source,
        # 3 baro_pressure_setting, 4 selected_heading, 5 nac_p, 6 nic_baro,
        # 7 sil, 8 autopilot, 9 vnav_mode, 10 altitude_hold_mode,
        # 11 approach_mode, 12 lnav_mode, 13 tcas_operational
        if r.selected_altitude is None:
            absent |= 2
        else:
            row.unsafe_store(3, Float64(r.selected_altitude.value()))
        row.unsafe_store(4, Float64(r.selected_altitude_source))
        if r.baro_pressure_setting is None:
            absent |= 8
        else:
            row.unsafe_store(5, r.baro_pressure_setting.value())
        if r.selected_heading is None:
            absent |= 16
        else:
            row.unsafe_store(6, r.selected_heading.value())
        row.unsafe_store(7, Float64(r.nac_p))
        row.unsafe_store(8, Float64(r.nic_baro))
        row.unsafe_store(9, Float64(r.sil))
        if r.autopilot is None:
            absent |= 256
        else:
            row.unsafe_store(10, 1.0 if r.autopilot.value() else 0.0)
        if r.vnav_mode is None:
            absent |= 512
        else:
            row.unsafe_store(11, 1.0 if r.vnav_mode.value() else 0.0)
        if r.altitude_hold_mode is None:
            absent |= 1024
        else:
            row.unsafe_store(12, 1.0 if r.altitude_hold_mode.value() else 0.0)
        if r.approach_mode is None:
            absent |= 2048
        else:
            row.unsafe_store(13, 1.0 if r.approach_mode.value() else 0.0)
        if r.lnav_mode is None:
            absent |= 4096
        else:
            row.unsafe_store(14, 1.0 if r.lnav_mode.value() else 0.0)
        row.unsafe_store(15, 1.0 if r.tcas_operational else 0.0)
        row.unsafe_store(0, Float64(present))
        row.unsafe_store(1, Float64(absent))

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)


# =================================================================== BDS 6,5
#
# slots: 0 present  1 absent  2 subtype  3 version  4 nic_supplement_a
#        5 nac_p  6 sil  7 hrd  8 capability_class  9 operational_mode
#        10 nic_baro  11 sil_supplement
comptime B65_W = 12


@export("mp_decode_bds65")
def mp_decode_bds65(payloads: Int, n: Int, dst: Int) abi("C"):
    var pp = PtrI(unsafe_from_address=payloads)
    var dp = PtrF(unsafe_from_address=dst)

    def body(i: Int) {imm}:
        var row = dp.unsafe_offset(B65_W * i)
        var r = decode_bds65(pp.unsafe_load(i))
        var present = 0xFF
        var absent = 0
        row.unsafe_store(2, Float64(r.subtype))
        row.unsafe_store(3, Float64(r.version))
        row.unsafe_store(4, Float64(r.nic_supplement_a))
        row.unsafe_store(5, Float64(r.nac_p))
        row.unsafe_store(6, Float64(r.sil))
        row.unsafe_store(7, Float64(r.hrd))
        row.unsafe_store(8, Float64(r.capability_class))
        row.unsafe_store(9, Float64(r.operational_mode))
        if r.has_nic_baro:
            present |= 1 << 8
            row.unsafe_store(10, Float64(r.nic_baro))
        if r.has_sil_supplement:
            present |= 1 << 9
            row.unsafe_store(11, Float64(r.sil_supplement))
        row.unsafe_store(0, Float64(present))
        row.unsafe_store(1, Float64(absent))

    if n < PAR_THRESHOLD:
        for i in range(n):
            body(i)
    else:

        def work(i: Int) {imm}:
            body(i)

        parallelize(work, n, PAR_WORKERS)
