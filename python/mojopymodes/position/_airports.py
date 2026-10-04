"""Surface-position reference lookup: ICAO airport code or (lat, lon) tuple.

The airport table itself is data, not computation, so it stays in Python and
is imported from the installed upstream package rather than duplicated here.
"""

from __future__ import annotations

from pyModeS.data.airports import AIRPORTS


def resolve_surface_ref(surface_ref) -> tuple[float, float]:
    """Resolve a surface-position reference to a (lat, lon) tuple."""
    if isinstance(surface_ref, tuple):
        return surface_ref
    try:
        return AIRPORTS[surface_ref]
    except KeyError:
        raise ValueError(f"unknown airport code: {surface_ref}") from None
