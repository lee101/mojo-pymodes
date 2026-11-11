"""The Python surface mirrors upstream's names and signatures, function by function.

A port that renames an argument or reorders two of them is not a drop-in, and
that is invisible to a numerical parity test: the test would be comparing this
repo against upstream through the port's own wrapper, which is exactly the
thing under suspicion. So compare `inspect.signature` directly, against the
real installed pyModeS 3.6.0.

Parameter names, order, kind (positional / keyword-only) and defaults must all
match. Annotation text is normalised first: `from __future__ import
annotations` turns every annotation into a string, so upstream's `dict[str,
Any]` and this repo's `dict` would otherwise read as a mismatch when the
signature is in fact identical.
"""

from __future__ import annotations

import inspect

import pytest

import mojopymodes as pm
from pyModeS import _altcode, _bits, _callsign, _idcode, util
from pyModeS.decoder.bds import _helpers
from pyModeS.decoder.bds import (
    bds05,
    bds06,
    bds08,
    bds09,
    bds30,
    bds61,
    bds62,
    bds65,
)
from pyModeS.position import _airports
from pyModeS.position._cpr import (
    airborne_position_pair,
    airborne_position_with_ref,
    cprNL,
    surface_position_pair,
    surface_position_with_ref,
)

# name -> (upstream function, the ported function with the same signature)
# Every entry here is a public pyModeS function this repo claims to cover.
SURFACE = {
    "extract_unsigned": (_bits.extract_unsigned, pm.extract_unsigned),
    "extract_signed": (_bits.extract_signed, pm.extract_signed),
    "crc_remainder": (_bits.crc_remainder, pm.crc_remainder),
    "altcode_to_altitude": (_altcode.altcode_to_altitude, pm.altcode_to_altitude),
    "decode_callsign": (_callsign.decode_callsign, pm.decode_callsign),
    "is_valid_callsign_char": (
        _callsign.is_valid_callsign_char,
        pm.is_valid_callsign_char,
    ),
    "idcode_to_squawk": (_idcode.idcode_to_squawk, pm.idcode_to_squawk),
    "cprNL": (cprNL, pm.cprNL),
    "airborne_position_with_ref": (
        airborne_position_with_ref,
        pm.airborne_position_with_ref,
    ),
    "airborne_position_pair": (airborne_position_pair, pm.airborne_position_pair),
    "surface_position_with_ref": (
        surface_position_with_ref,
        pm.surface_position_with_ref,
    ),
    "surface_position_pair": (surface_position_pair, pm.surface_position_pair),
    "signed": (_helpers.signed, pm.signed),
    "normalise_angle": (_helpers.normalise_angle, pm.normalise_angle),
    "wrong_status": (_helpers.wrong_status, pm.wrong_status),
    "decode_bds05": (bds05.decode_bds05, pm.decoder.bds.decode_bds05),
    "decode_bds06": (bds06.decode_bds06, pm.decoder.bds.decode_bds06),
    "decode_bds08": (bds08.decode_bds08, pm.decoder.bds.decode_bds08),
    "decode_bds09": (bds09.decode_bds09, pm.decoder.bds.decode_bds09),
    "is_bds30": (bds30.is_bds30, pm.decoder.bds.is_bds30),
    "decode_bds30": (bds30.decode_bds30, pm.decoder.bds.decode_bds30),
    "decode_bds61": (bds61.decode_bds61, pm.decoder.bds.decode_bds61),
    "decode_bds62": (bds62.decode_bds62, pm.decoder.bds.decode_bds62),
    "decode_bds65": (bds65.decode_bds65, pm.decoder.bds.decode_bds65),
    "df": (util.df, pm.util.df),
    "icao": (util.icao, pm.util.icao),
    "typecode": (util.typecode, pm.util.typecode),
    "crc": (util.crc, pm.util.crc),
    "altcode": (util.altcode, pm.util.altcode),
    "idcode": (util.idcode, pm.util.idcode),
    "hex2bin": (util.hex2bin, pm.util.hex2bin),
    "bin2int": (util.bin2int, pm.util.bin2int),
    "hex2int": (util.hex2int, pm.util.hex2int),
    "bin2hex": (util.bin2hex, pm.util.bin2hex),
    "resolve_surface_ref": (
        _airports.resolve_surface_ref,
        pm.position.resolve_surface_ref,
    ),
}


def _shape(fn) -> str:
    """Parameter names, kinds and defaults, with no annotations and no return.

    Annotations differ cosmetically -- this repo runs
    `from __future__ import annotations`, so every annotation is a string --
    and the return annotation carries no information about the call contract.
    """
    sig = inspect.signature(fn)
    parts = []
    for name, p in sig.parameters.items():
        if p.kind is p.VAR_POSITIONAL:
            parts.append(f"*{name}")
        elif p.kind is p.VAR_KEYWORD:
            parts.append(f"**{name}")
        else:
            if p.kind is p.KEYWORD_ONLY:
                parts.append("*")
            default = "" if p.default is p.empty else f"={p.default!r}"
            parts.append(f"{name}{default}")
    return ",".join(parts)


@pytest.mark.parametrize("name", sorted(SURFACE))
def test_signature_matches_upstream(name):
    ref, ours = SURFACE[name]
    assert _shape(ours) == _shape(ref), name


@pytest.mark.parametrize("name", sorted(SURFACE))
def test_keyword_arguments_are_accepted(name):
    """Every upstream parameter must still be callable by name.

    A dropped or renamed parameter binds positionally and passes the numeric
    parity tests, because this repo's own wrapper is what calls the kernel. It
    breaks the moment a caller writes `decode_bds05(payload=p, tc=tc)`.
    """
    ref, ours = SURFACE[name]
    sig = inspect.signature(ref)
    names = [
        pname
        for pname, p in sig.parameters.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert names, name
    for pname in names:
        # bind_partial, not bind: the point is whether the keyword *exists*, and
        # binding without the other arguments would fail for a valid signature.
        try:
            inspect.signature(ours).bind_partial(**{pname: None})
        except TypeError:
            pytest.fail(f"{name} does not accept keyword {pname!r}")


def test_keyword_only_arguments_are_keyword_only():
    """The two upstream keyword-only markers must survive the port."""
    import pyModeS.decoder.bds.bds05 as m05
    import pyModeS.position._cpr as cpr

    for ref, ours in (
        (m05.decode_bds05, pm.decoder.bds.decode_bds05),
        (cpr.airborne_position_pair, pm.airborne_position_pair),
        (cpr.surface_position_pair, pm.surface_position_pair),
    ):
        ref_kw = {
            n for n, p in inspect.signature(ref).parameters.items()
            if p.kind is p.KEYWORD_ONLY
        }
        our_kw = {
            n for n, p in inspect.signature(ours).parameters.items()
            if p.kind is p.KEYWORD_ONLY
        }
        assert our_kw == ref_kw, (ref.__name__, our_kw, ref_kw)


def test_every_covered_name_is_reachable_from_the_package_root():
    for name in SURFACE:
        assert hasattr(pm, name) or hasattr(pm.util, name) or hasattr(
            pm.position, name
        ) or hasattr(pm.decoder.bds, name), name


def test_no_extra_upstream_names_are_promised():
    """The package root only exposes names this repo actually implements.

    Every upstream *function* this repo covers is listed above; everything
    else public on the root here is an addition or one of the supporting
    modules, not a claim of a pyModeS name that does not exist.
    """
    added = {
        "altcodes_to_altitudes",
        "decode_callsigns",
        "decode_fields",
        "headers",
        "idcodes_to_squawks",
        "payload",
        "payloads",
        "airborne_position_pair_batch",
        "airborne_position_with_ref_batch",
        "cprNL_batch",
        "surface_position_pair_batch",
        "surface_position_with_ref_batch",
    }
    modules = {"decoder", "position", "util", "batch"}
    public = {
        n for n in dir(pm)
        if not n.startswith("_") and n != "annotations"
    }
    known = set(SURFACE) | added | modules | {"__version__"}
    assert public - known == set(), sorted(public - known)
