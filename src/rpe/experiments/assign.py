"""Experiment assignment: by title on the sell side (copies of a title share buyers, so they must
share an arm), by seller on the buy side (a seller must see consistent offers)."""

from __future__ import annotations

import numpy as np

from rpe.hashing import hash_uniform


def arm(unit_ids: np.ndarray, salt: int, treatment_share: float = 0.5) -> np.ndarray:
    """True = treatment. Stable for a unit id and salt."""
    return hash_uniform(np.asarray(unit_ids), salt) < treatment_share


def seller_arm(salt: int, treatment_share: float = 0.5):
    return lambda sellers: arm(sellers, salt, treatment_share)
