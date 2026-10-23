"""Fixed-horizon analysis: difference in means with Welch intervals, CUPED variance reduction,
ratio metrics by the delta method, and the sample-ratio check that should precede any readout."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class Estimate:
    diff: float
    se: float
    lo: float
    hi: float
    p: float
    n_t: int
    n_c: int
    mean_c: float

    @property
    def rel(self) -> float:
        return self.diff / abs(self.mean_c) if self.mean_c else float("nan")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["rel"] = self.rel
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}


def welch(y: np.ndarray, treat: np.ndarray, alpha: float = 0.05) -> Estimate:
    y, treat = np.asarray(y, float), np.asarray(treat, bool)
    a, b = y[treat], y[~treat]
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    d = float(a.mean() - b.mean())
    dof = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)) if se > 0 else len(y) - 2
    q = stats.t.ppf(1 - alpha / 2, dof)
    p = float(2 * stats.t.sf(abs(d) / se, dof)) if se > 0 else 1.0
    return Estimate(d, se, float(d - q * se), float(d + q * se), p, len(a), len(b), float(b.mean()))


def cuped(y: np.ndarray, x: np.ndarray, treat: np.ndarray, alpha: float = 0.05) -> tuple[Estimate, float]:
    """Welch on y - theta (x - mean x), theta = cov(y, x) / var(x) on the pooled data. x must be
    measured before assignment. Returns the estimate and the share of variance removed."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(y, x)[0, 1] / x.var(ddof=1) if x.var() > 0 else 0.0
    adj = y - theta * (x - x.mean())
    return welch(adj, treat, alpha), float(1 - adj.var() / y.var()) if y.var() > 0 else 0.0


def delta_ratio(num: np.ndarray, den: np.ndarray, treat: np.ndarray, alpha: float = 0.05) -> Estimate:
    """Difference of ratio metrics sum(num)/sum(den) per arm, units = randomisation clusters."""
    num, den, treat = np.asarray(num, float), np.asarray(den, float), np.asarray(treat, bool)

    def part(m):
        n, d = num[m], den[m]
        r = n.sum() / d.sum()
        k = len(n)
        md = d.mean()
        var = (n.var(ddof=1) - 2 * r * np.cov(n, d)[0, 1] + r * r * d.var(ddof=1)) / (k * md * md)
        return r, var, k

    rt, vt, kt = part(treat)
    rc, vc, kc = part(~treat)
    se = float(np.sqrt(vt + vc))
    d = float(rt - rc)
    z = stats.norm.ppf(1 - alpha / 2)
    p = float(2 * stats.norm.sf(abs(d) / se)) if se > 0 else 1.0
    return Estimate(d, se, float(d - z * se), float(d + z * se), p, kt, kc, float(rc))


def srm(n_t: int, n_c: int, expected_share: float = 0.5) -> float:
    """p-value of the sample-ratio-mismatch chi-square test; below 0.001 the assignment is broken."""
    n = n_t + n_c
    e_t, e_c = n * expected_share, n * (1 - expected_share)
    chi = (n_t - e_t) ** 2 / e_t + (n_c - e_c) ** 2 / e_c
    return float(stats.chi2.sf(chi, 1))


def sample_size(sd: float, mde: float, alpha: float = 0.05, power: float = 0.8, share: float = 0.5) -> int:
    """Units in total for a two-sided test of a difference in means of `mde`."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(z**2 * sd**2 * (1 / share + 1 / (1 - share)) / mde**2))
