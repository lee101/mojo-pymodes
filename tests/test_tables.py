"""The static tables in src/tables.mojo are copies of upstream's, byte for byte.

`build/gen_tables.py` derives them mechanically; this test re-derives them
from the installed pyModeS and compares against what is actually in the
file, so a hand edit or a toolchain bump cannot quietly change a CRC table.
"""

from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "src", "tables.mojo")

def _literals(name: str, dtype: str) -> list:
    text = open(TABLES).read()
    body = text.split(f"comptime {name} = StaticTuple[{dtype},", 1)[1]
    body = body.split("(", 1)[1].split("\n)", 1)[0]
    if dtype == "Float64":
        return [float(x) for x in re.findall(r"-?\d+\.\d+", body)]
    return [int(x) for x in re.findall(r"-?\d+", body)]


def test_crc_table():
    from pyModeS._bits import _CRC_TABLE

    assert _literals("_CRC_TABLE", "Int") == list(_CRC_TABLE)


def test_callsign_table():
    from pyModeS._callsign import _CALLSIGN_TABLE

    assert _literals("_CALLSIGN_TABLE", "UInt8") == [ord(c) for c in _CALLSIGN_TABLE]


def test_nl_boundaries():
    from pyModeS.position._cpr import _NL_BOUNDARIES

    assert _literals("_NL_BOUNDARIES", "Float64") == list(_NL_BOUNDARIES)


def test_nucp_lookup():
    from pyModeS._uncertainty import TC_NUCp_lookup

    assert _literals("TC_NUCp_lookup", "Int") == [
        TC_NUCp_lookup.get(tc, 0) for tc in range(32)
    ]


def test_wake_vortex_indices():
    """Every upstream (tc, category) resolves to the same string."""
    from mojopymodes._layout import WAKE_VORTEX_NAMES
    from mojopymodes._lib import lib
    from pyModeS.decoder.bds.bds08 import _WAKE_VORTEX

    for (tc, cat), name in _WAKE_VORTEX.items():
        idx = lib().mp_wake_vortex_category(tc, cat)
        assert WAKE_VORTEX_NAMES[idx] == name, (tc, cat)


def test_generator_reproduces_every_table():
    """src/tables.mojo is exactly what build/gen_tables.py produces."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_gen_tables", os.path.join(ROOT, "build", "gen_tables.py")
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    assert _literals("_CRC_TABLE", "Int") == gen.crc_table()
    assert _literals("_CALLSIGN_TABLE", "UInt8") == gen.callsign_table()
    assert _literals("_NL_BOUNDARIES", "Float64") == gen.nl_boundaries()
    assert _literals("TC_NUCp_lookup", "Int") == gen.nucp_table()
    assert _literals("_WAKE_VORTEX", "Int") == gen.wake_vortex_table()
