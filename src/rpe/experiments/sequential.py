"""Always-valid inference for experiments that are looked at every day.

Pricing experiments get checked daily. A fixed-horizon test checked daily and stopped at the first
p < 0.05 has a false positive rate far above 5%. The mixture sequential probability ratio test
(mSPRT, normal mixture) gives a p-value that stays valid however often it is looked at, at the cost
of some power."""

from __future__ import annotations

import numpy as np


def msprt_lr(diff: np.ndarray, var: np.ndarray, tau2: float) -> np.ndarray:
    """Mixture likelihood ratio for H0: difference = 0, with N(0, tau2) mixing over the effect.
    `var` is the variance of the difference estimate at each look."""
    return np.sqrt(var / (var + tau2)) * np.exp(diff**2 * tau2 / (2 * var * (var + tau2)))


def always_valid_p(diff: np.ndarray, var: np.ndarray, tau2: float) -> np.ndarray:
    """Running always-valid p-values over a sequence of looks."""
    lr = msprt_lr(np.asarray(diff, float), np.asarray(var, float), tau2)
    return np.minimum(1.0, 1.0 / np.maximum.accumulate(np.maximum(lr, 1e-300)))


def peeking_study(
    n_experiments: int = 2000,
    looks: int = 28,
    per_look: int = 200,
    effect: float = 0.0,
    sd: float = 1.0,
    tau2: float | None = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Simulated experiments checked at every look: share that ever reject with a naive t-test vs
    with the always-valid p-value. With effect = 0 these are false positive rates."""
    rng = np.random.default_rng(seed)
    t = rng.normal(effect, sd, (n_experiments, looks, per_look)).sum(axis=2).cumsum(axis=1)
    c = rng.normal(0.0, sd, (n_experiments, looks, per_look)).sum(axis=2).cumsum(axis=1)
    n = per_look * np.arange(1, looks + 1)
    diff = t / n - c / n
    var = 2 * sd**2 / n
    z = np.abs(diff) / np.sqrt(var)
    naive_any = (z > 1.959964).any(axis=1).mean()
    naive_end = (z[:, -1] > 1.959964).mean()
    tau2 = tau2 if tau2 is not None else (0.1 * sd) ** 2
    p = np.array([always_valid_p(diff[i], var, tau2) for i in range(n_experiments)])
    seq_any = (p.min(axis=1) < alpha).mean()
    stop = np.where((p < alpha).any(axis=1), (p < alpha).argmax(axis=1) + 1, looks)
    return {
        "naive_reject_any_look": float(naive_any),
        "naive_reject_at_end": float(naive_end),
        "always_valid_reject": float(seq_any),
        "mean_looks_to_stop": float(stop.mean()),
    }
