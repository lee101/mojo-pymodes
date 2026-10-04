"""mojopymodes -- pyModeS's ADS-B decoding primitives, in Mojo.

The public surface mirrors pyModeS 3.6.0: `util` keeps the low-level
hex/bit/CRC helpers under the same names, `position` keeps the CPR decoders,
`decoder.bds` keeps the per-register decoders, and every one of them returns
the same value the upstream function returns.

    >>> import mojopymodes as pm
    >>> pm.util.df("8D406B902015A678D4D220AA4BDA")
    17
    >>> pm.decoder.bds.bds08.decode_bds08(pm.payload(msg))["callsign"]
    'EZY85MH'

Everything numeric runs in a compiled Mojo shared library; this package is
ctypes, numpy and the string/mapping glue the C ABI cannot carry.
"""

from __future__ import annotations

from . import _bits, _altcode, _callsign, _idcode, _helpers, decoder, position, util
from ._bits import crc_remainder, extract_signed, extract_unsigned
from ._altcode import altcode_to_altitude
from ._callsign import decode_callsign, is_valid_callsign_char
from ._idcode import idcode_to_squawk
from ._helpers import normalise_angle, signed, wrong_status
from .batch import (
    altcodes_to_altitudes,
    decode_callsigns,
    decode_fields,
    headers,
    idcodes_to_squawks,
    payload,
    payloads,
)
from .position import (
    airborne_position_pair,
    airborne_position_pair_batch,
    airborne_position_with_ref,
    airborne_position_with_ref_batch,
    cprNL,
    cprNL_batch,
    surface_position_pair,
    surface_position_pair_batch,
    surface_position_with_ref,
    surface_position_with_ref_batch,
)
from .util import (
    altcode,
    bin2hex,
    bin2int,
    crc,
    df,
    hex2bin,
    hex2int,
    icao,
    idcode,
    typecode,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "airborne_position_pair",
    "airborne_position_pair_batch",
    "airborne_position_with_ref",
    "airborne_position_with_ref_batch",
    "altcode",
    "altcode_to_altitude",
    "altcodes_to_altitudes",
    "bin2hex",
    "bin2int",
    "cprNL",
    "cprNL_batch",
    "crc",
    "crc_remainder",
    "decode_callsign",
    "decode_callsigns",
    "decode_fields",
    "df",
    "extract_signed",
    "extract_unsigned",
    "headers",
    "hex2bin",
    "hex2int",
    "icao",
    "idcode",
    "idcode_to_squawk",
    "idcodes_to_squawks",
    "is_valid_callsign_char",
    "normalise_angle",
    "payload",
    "payloads",
    "signed",
    "surface_position_pair",
    "surface_position_pair_batch",
    "surface_position_with_ref",
    "surface_position_with_ref_batch",
    "typecode",
    "wrong_status",
]
