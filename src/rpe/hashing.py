"""Deterministic pseudo-random numbers from integer keys: experiment assignment and test cells must
be reproducible from the id alone, on any machine."""

from __future__ import annotations

import numpy as np


def hash_uniform(*keys: np.ndarray | int) -> np.ndarray:
    """A uniform(0, 1) per element from integer keys (splitmix64 mixing)."""
    x = np.zeros(np.broadcast(*[np.asarray(k) for k in keys]).shape, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in keys:
            x = x ^ (np.asarray(k).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15) + (x << np.uint64(6)) + (x >> np.uint64(2)))
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)
