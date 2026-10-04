"""Faster hex -> (hi, lo, len) conversion for a list of Mode-S hex messages.

The current `to_words` pads every message with `rjust` (a Python string per
message) and then folds 16 nibbles with 16 full-array shift/or passes. This
compares cheaper shapes.

    flock /tmp/mojo-bench.lock pixi run python bench/micro_words.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

import corpus  # noqa: E402

N = 1_000_000
REPEAT = 3
REPEAT_S = 20


def best_of(fn, repeat=REPEAT):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def row(label, seconds):
    print(f"{label:44s}: {seconds * 1e3:9.1f} ms  ({seconds / N * 1e9:6.0f} ns/msg)", flush=True)


def current(msgs):
    lengths = np.array([len(m) for m in msgs], dtype=np.int64)
    width = int(lengths.max())
    padded = (width + 15) // 16 * 16
    blob = "".join(m.rjust(padded, "0") for m in msgs).encode("ascii")
    nibbles = np.frombuffer(blob, dtype=np.uint8).reshape(len(msgs), padded)
    values = np.where(nibbles <= 57, nibbles - 48, nibbles - 55).astype(np.uint64)
    words = []
    for c in range(padded // 16):
        chunk = values[:, c * 16 : (c + 1) * 16]
        word = np.zeros(len(msgs), dtype=np.uint64)
        for j in range(16):
            word = (word << np.uint64(4)) | chunk[:, j]
        words.append(word)
    hi = words[0] if len(words) > 1 else np.zeros(len(msgs), dtype=np.uint64)
    return hi.astype(np.int64), words[-1].astype(np.int64), lengths * 4


def uniform_length(msgs):
    """All messages the same length: skip rjust entirely."""
    lengths = np.array([len(m) for m in msgs], dtype=np.int64)
    width = int(lengths.max())
    padded = (width + 15) // 16 * 16
    if int(lengths.min()) == padded:
        blob = "".join(msgs).encode("ascii")
    else:
        blob = "".join(m.rjust(padded, "0") for m in msgs).encode("ascii")
    nibbles = np.frombuffer(blob, dtype=np.uint8).reshape(len(msgs), padded)
    values = np.where(nibbles <= 57, nibbles - 48, nibbles - 55).astype(np.uint64)
    words = []
    for c in range(padded // 16):
        chunk = values[:, c * 16 : (c + 1) * 16]
        word = np.zeros(len(msgs), dtype=np.uint64)
        for j in range(16):
            word = (word << np.uint64(4)) | chunk[:, j]
        words.append(word)
    hi = words[0] if len(words) > 1 else np.zeros(len(msgs), dtype=np.uint64)
    return hi.astype(np.int64), words[-1].astype(np.int64), lengths * 4


def frombuffer_bits(msgs):
    """Skip the per-message length array when every message is the same size."""
    lengths = np.full(len(msgs), 28, dtype=np.int64)
    blob = "".join(msgs).encode("ascii")
    n = len(msgs)
    # 28 hex digits -> 112 bits -> hi = top 48 bits, lo = bottom 64 bits.
    raw = np.frombuffer(blob, dtype=np.uint8).reshape(n, 28)
    vals = np.where(raw <= 57, raw - 48, raw - 55).astype(np.uint64)
    lo = np.zeros(n, dtype=np.uint64)
    for j in range(16, 28):
        lo = (lo << np.uint64(4)) | vals[:, j]
    hi = np.zeros(n, dtype=np.uint64)
    for j in range(0, 12):
        hi = (hi << np.uint64(4)) | vals[:, j]
    return hi.astype(np.int64), lo.astype(np.int64), lengths * 4


def uniform_length_noastype(msgs):
    lengths = np.array([len(m) for m in msgs], dtype=np.int64)
    width = int(lengths.max())
    padded = (width + 15) // 16 * 16
    if int(lengths.min()) == padded:
        blob = "".join(msgs).encode("ascii")
    else:
        blob = "".join(m.rjust(padded, "0") for m in msgs).encode("ascii")
    nibbles = np.frombuffer(blob, dtype=np.uint8).reshape(len(msgs), padded)
    values = np.where(nibbles <= 57, nibbles - 48, nibbles - 55).astype(np.uint64)
    words = []
    for c in range(padded // 16):
        chunk = values[:, c * 16 : (c + 1) * 16]
        word = np.zeros(len(msgs), dtype=np.uint64)
        for j in range(16):
            word = (word << np.uint64(4)) | chunk[:, j]
        words.append(word)
    hi = words[0] if len(words) > 1 else np.zeros(len(msgs), dtype=np.uint64)
    # view as int64 instead of astype: bit-identical for values below 2**63,
    # and the top word is non-negative by construction.
    return hi.view(np.int64), words[-1].view(np.int64), lengths * 4


def main() -> None:
    base = corpus.long_corpus()
    msgs = [base[i % len(base)] for i in range(N)]
    print(f"uniform length: {set(len(m) for m in msgs[:1000])}")
    ref = current(msgs)
    for name, fn in (
        ("current", current),
        ("skip rjust when uniform", uniform_length),
        ("fixed 28-digit, no rjust", frombuffer_bits),
        ("skip rjust + view not astype", uniform_length_noastype),
    ):
        got = fn(msgs)
        assert all(np.array_equal(a, b) for a, b in zip(ref, got)), name
        row(name, best_of(lambda fn=fn: fn(msgs)))

    print()
    print("parts")
    lengths = np.array([len(m) for m in msgs], dtype=np.int64)
    row("  [len(m) for m in msgs]", best_of(lambda: [len(m) for m in msgs], REPEAT_S))
    row("  np.array of those lengths", best_of(lambda: np.array([len(m) for m in msgs], dtype=np.int64), REPEAT_S))
    row("  ''.join(msgs)", best_of(lambda: "".join(msgs), REPEAT_S))
    row("  ''.join(rjust each)", best_of(lambda: "".join(m.rjust(28, "0") for m in msgs), REPEAT_S))
    blob = "".join(msgs).encode("ascii")
    row("  frombuffer+reshape", best_of(lambda: np.frombuffer(blob, dtype=np.uint8).reshape(N, 28), REPEAT_S))
    raw = np.frombuffer(blob, dtype=np.uint8).reshape(N, 28)
    row("  nibble map (where+astype)", best_of(lambda: np.where(raw <= 57, raw - 48, raw - 55).astype(np.uint64), REPEAT_S))


if __name__ == "__main__":
    main()
