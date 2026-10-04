"""pyModeS.position._cpr — CPR position decoding primitives."""

from __future__ import annotations

import math

from .._lib import addr, f64, i64, lib, out_f64, scratch_2f


def cprNL(lat: float) -> int:
    """Number of CPR longitude zones at `lat`, per DO-260B A.1.7.2."""
    return lib().mp_cprNL(float(lat))


def cprNL_batch(lats) -> list[int]:
    """`cprNL` over many latitudes, one kernel call."""
    a = f64(lats)
    n = a.size
    if n == 0:
        return []
    out = i64([0] * n)
    lib().mp_cprNL_batch(addr(a), n, addr(out))
    return out.tolist()


def airborne_position_with_ref(
    cpr_format: int,
    cpr_lat_raw: int,
    cpr_lon_raw: int,
    lat_ref: float,
    lon_ref: float,
) -> tuple[float, float]:
    """Resolve an airborne CPR frame against a nearby reference."""
    buf = scratch_2f()
    lib().mp_airborne_position_with_ref(
        cpr_format, cpr_lat_raw, cpr_lon_raw, lat_ref, lon_ref, addr(buf)
    )
    return float(buf[0]), float(buf[1])


def airborne_position_pair(
    cpr_lat_even_raw: int,
    cpr_lon_even_raw: int,
    cpr_lat_odd_raw: int,
    cpr_lon_odd_raw: int,
    *,
    even_is_newer: bool,
) -> tuple[float, float] | None:
    """Resolve absolute lat/lon from an even/odd CPR pair."""
    buf = scratch_2f()
    ok = lib().mp_airborne_position_pair(
        cpr_lat_even_raw,
        cpr_lon_even_raw,
        cpr_lat_odd_raw,
        cpr_lon_odd_raw,
        1 if even_is_newer else 0,
        addr(buf),
    )
    return (float(buf[0]), float(buf[1])) if ok else None


def surface_position_with_ref(
    cpr_format: int,
    cpr_lat_raw: int,
    cpr_lon_raw: int,
    lat_ref: float,
    lon_ref: float,
) -> tuple[float, float]:
    """Resolve a surface CPR frame against a nearby reference."""
    buf = scratch_2f()
    lib().mp_surface_position_with_ref(
        cpr_format, cpr_lat_raw, cpr_lon_raw, lat_ref, lon_ref, addr(buf)
    )
    return float(buf[0]), float(buf[1])


def surface_position_pair(
    cpr_lat_even_raw: int,
    cpr_lon_even_raw: int,
    cpr_lat_odd_raw: int,
    cpr_lon_odd_raw: int,
    *,
    lat_ref: float,
    lon_ref: float,
    even_is_newer: bool,
) -> tuple[float, float] | None:
    """Resolve an even/odd surface CPR pair against a receiver location."""
    buf = scratch_2f()
    ok = lib().mp_surface_position_pair(
        cpr_lat_even_raw,
        cpr_lon_even_raw,
        cpr_lat_odd_raw,
        cpr_lon_odd_raw,
        lat_ref,
        lon_ref,
        1 if even_is_newer else 0,
        addr(buf),
    )
    return (float(buf[0]), float(buf[1])) if ok else None


# ----------------------------------------------------------- batch helpers


def airborne_position_pair_batch(
    cpr_lat_even_raw,
    cpr_lon_even_raw,
    cpr_lat_odd_raw,
    cpr_lon_odd_raw,
    even_is_newer,
) -> list[tuple[float, float] | None]:
    """`airborne_position_pair` over many pairs, one kernel call."""
    lat_e = i64(cpr_lat_even_raw)
    lon_e = i64(cpr_lon_even_raw)
    lat_o = i64(cpr_lat_odd_raw)
    lon_o = i64(cpr_lon_odd_raw)
    newer = i64(even_is_newer)
    n = lat_e.size
    if n == 0:
        return []
    out = out_f64(n, 2)
    lib().mp_airborne_position_pair_batch(
        addr(lat_e), addr(lon_e), addr(lat_o), addr(lon_o), addr(newer), n, addr(out)
    )
    return [
        None if math.isnan(row[0]) else (float(row[0]), float(row[1]))
        for row in out.reshape(-1, 2)
    ]


def surface_position_pair_batch(
    cpr_lat_even_raw,
    cpr_lon_even_raw,
    cpr_lat_odd_raw,
    cpr_lon_odd_raw,
    lat_ref,
    lon_ref,
    even_is_newer,
) -> list[tuple[float, float] | None]:
    """`surface_position_pair` over many pairs, one kernel call."""
    lat_e = i64(cpr_lat_even_raw)
    lon_e = i64(cpr_lon_even_raw)
    lat_o = i64(cpr_lat_odd_raw)
    lon_o = i64(cpr_lon_odd_raw)
    ref_lat = f64(lat_ref)
    ref_lon = f64(lon_ref)
    newer = i64(even_is_newer)
    n = lat_e.size
    if n == 0:
        return []
    out = out_f64(n, 2)
    lib().mp_surface_position_pair_batch(
        addr(lat_e),
        addr(lon_e),
        addr(lat_o),
        addr(lon_o),
        addr(ref_lat),
        addr(ref_lon),
        addr(newer),
        n,
        addr(out),
    )
    return [
        None if math.isnan(row[0]) else (float(row[0]), float(row[1]))
        for row in out.reshape(-1, 2)
    ]


def airborne_position_with_ref_batch(
    cpr_format,
    cpr_lat_raw,
    cpr_lon_raw,
    lat_ref,
    lon_ref,
) -> list[tuple[float, float]]:
    """`airborne_position_with_ref` over many frames, one kernel call."""
    fmt = i64(cpr_format)
    lat = i64(cpr_lat_raw)
    lon = i64(cpr_lon_raw)
    ref_lat = f64(lat_ref)
    ref_lon = f64(lon_ref)
    n = fmt.size
    if n == 0:
        return []
    out = out_f64(n, 2)
    lib().mp_airborne_position_with_ref_batch(
        addr(fmt), addr(lat), addr(lon), addr(ref_lat), addr(ref_lon), n, addr(out)
    )
    return [(float(r[0]), float(r[1])) for r in out.reshape(-1, 2)]


def surface_position_with_ref_batch(
    cpr_format,
    cpr_lat_raw,
    cpr_lon_raw,
    lat_ref,
    lon_ref,
) -> list[tuple[float, float]]:
    """`surface_position_with_ref` over many frames, one kernel call."""
    fmt = i64(cpr_format)
    lat = i64(cpr_lat_raw)
    lon = i64(cpr_lon_raw)
    ref_lat = f64(lat_ref)
    ref_lon = f64(lon_ref)
    n = fmt.size
    if n == 0:
        return []
    out = out_f64(n, 2)
    lib().mp_surface_position_with_ref_batch(
        addr(fmt), addr(lat), addr(lon), addr(ref_lat), addr(ref_lon), n, addr(out)
    )
    return [(float(r[0]), float(r[1])) for r in out.reshape(-1, 2)]
