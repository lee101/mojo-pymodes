"""BDS register decoders, ported from pyModeS.decoder.bds.

Each module exposes the same function name and argument list as its upstream
counterpart and returns the same dict. The `*_batch` sibling is not an
upstream function: it runs the same kernel over a whole array in one call.
"""

from .bds05 import decode_bds05, decode_bds05_batch
from .bds06 import decode_bds06, decode_bds06_batch
from .bds08 import decode_bds08, decode_bds08_batch
from .bds09 import decode_bds09, decode_bds09_batch
from .bds30 import decode_bds30, decode_bds30_batch, is_bds30
from .bds61 import decode_bds61, decode_bds61_batch
from .bds62 import decode_bds62, decode_bds62_batch
from .bds65 import decode_bds65, decode_bds65_batch

__all__ = [
    "decode_bds05",
    "decode_bds05_batch",
    "decode_bds06",
    "decode_bds06_batch",
    "decode_bds08",
    "decode_bds08_batch",
    "decode_bds09",
    "decode_bds09_batch",
    "decode_bds30",
    "decode_bds30_batch",
    "decode_bds61",
    "decode_bds61_batch",
    "decode_bds62",
    "decode_bds62_batch",
    "decode_bds65",
    "decode_bds65_batch",
    "is_bds30",
]
