"""Surrogate index for long-term outcomes: an offer test cannot wait four months to see whether
sellers come back, so it predicts the long-term outcome from what it sees in days (satisfaction,
basket sent, value) with the seller-value model fitted on history, and compares arms on that."""

from __future__ import annotations

import pandas as pd

from rpe.experiments.analysis import Estimate, welch


def surrogate_effect(model, first_baskets: pd.DataFrame) -> Estimate:
    """first_baskets: one row per seller in the test with columns sat, sent, kept_eur, returning, treat."""
    p = model.p_return(
        first_baskets.sat.to_numpy(),
        first_baskets.sent.to_numpy().astype(float),
        first_baskets.kept_eur.to_numpy(),
        first_baskets.returning.to_numpy().astype(float),
    )
    return welch(p, first_baskets.treat.to_numpy())


def realised_effect(first_baskets: pd.DataFrame) -> Estimate:
    return welch(first_baskets.returned.to_numpy().astype(float), first_baskets.treat.to_numpy())
