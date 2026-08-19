"""
Raw SVI and the two no-arbitrage conditions.

Raw SVI, per expiry, in total implied variance against log-moneyness:

    w(k) = a + b * ( rho * (k - m) + sqrt( (k - m)^2 + sigma^2 ) )

Analytic derivatives, which matter because finite differencing them and then
feeding the result into the butterfly test produces spurious violations at the
grid edges:

    w'(k)  = b * ( rho + (k - m) / sqrt((k - m)^2 + sigma^2) )
    w''(k) = b * sigma^2 / ((k - m)^2 + sigma^2)^(3/2)

Note w'' >= 0 whenever b >= 0, so a raw SVI slice is always convex in k. That
is *not* the same as being free of butterfly arbitrage, which is the point of
the Durrleman check below and the reason people get this wrong.

Parameter constraints
---------------------
    b >= 0                            slice opens upward
    |rho| < 1                         otherwise a wing has negative slope
    sigma > 0                         sigma is the curvature scale, not a vol
    a + b*sigma*sqrt(1 - rho^2) >= 0  minimum of w is non-negative

The last one is the binding one and the easiest to drop by accident. Its left
side is exactly min_k w(k), attained at k = m - rho*sigma/sqrt(1-rho^2).

Butterfly arbitrage
-------------------
Durrleman's function:

    g(k) = (1 - k*w'/(2w))^2 - (w'^2 / 4) * (1/w + 1/4) + w''/2

The slice admits no butterfly arbitrage iff g(k) >= 0 everywhere. g < 0 means
the implied risk-neutral density goes negative there, so a butterfly spread
struck around that point has negative cost and non-negative payoff.

Calendar arbitrage
------------------
Total variance must be non-decreasing in T at fixed k. If w(k, T2) < w(k, T1)
for T2 > T1 then a calendar spread is free money. Checked across fitted slices
on a shared k grid rather than on quoted strikes, since strikes differ by
expiry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

PARAM_NAMES = ("a", "b", "rho", "m", "sigma")


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma], dtype=float)

    def min_total_variance(self) -> float:
        """min_k w(k), the left side of the binding constraint."""
        return self.a + self.b * self.sigma * np.sqrt(max(1.0 - self.rho**2, 0.0))

    def violations(self) -> list[str]:
        """Parameter-domain problems, empty when the slice is admissible."""
        out = []
        if self.b < 0:
            out.append(f"b = {self.b:.6g} < 0")
        if abs(self.rho) >= 1:
            out.append(f"|rho| = {abs(self.rho):.6g} >= 1")
        if self.sigma <= 0:
            out.append(f"sigma = {self.sigma:.6g} <= 0")
        if self.min_total_variance() < 0:
            out.append(f"min w = {self.min_total_variance():.6g} < 0")
        return out


def svi_w(k, p: SVIParams):
    """Total implied variance."""
    k = np.asarray(k, dtype=float)
    return p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma**2))


def svi_dw(k, p: SVIParams):
    k = np.asarray(k, dtype=float)
    return p.b * (p.rho + (k - p.m) / np.sqrt((k - p.m) ** 2 + p.sigma**2))


def svi_d2w(k, p: SVIParams):
    k = np.asarray(k, dtype=float)
    return p.b * p.sigma**2 / ((k - p.m) ** 2 + p.sigma**2) ** 1.5


def durrleman_g(k, p: SVIParams):
    """
    g(k) >= 0 everywhere iff the slice is free of butterfly arbitrage.
    Uses the analytic derivatives above, not finite differences.
    """
    k = np.asarray(k, dtype=float)
    w = svi_w(k, p)
    dw = svi_dw(k, p)
    d2w = svi_d2w(k, p)

    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = (1.0 - k * dw / (2.0 * w)) ** 2
        term2 = (dw**2 / 4.0) * (1.0 / w + 0.25)
    return term1 - term2 + d2w / 2.0


def butterfly_report(p: SVIParams, k_min=-1.5, k_max=1.5, n=601) -> dict:
    """Scan g on a grid and report where and how badly it goes negative."""
    k = np.linspace(k_min, k_max, n)
    g = durrleman_g(k, p)
    bad = g < 0
    return {
        "n_grid": int(n),
        "n_negative": int(bad.sum()),
        "min_g": float(np.nanmin(g)),
        "argmin_k": float(k[int(np.nanargmin(g))]),
        "k_negative_range": (float(k[bad].min()), float(k[bad].max())) if bad.any() else None,
        "free_of_butterfly": bool(not bad.any()),
    }


def calendar_report(slices: dict[float, SVIParams], k_min=-1.5, k_max=1.5, n=601) -> dict:
    """
    Total variance must not decrease in T at fixed k. Compares consecutive
    fitted slices on a shared grid.
    """
    k = np.linspace(k_min, k_max, n)
    maturities = sorted(slices)
    pairs = []
    total_bad = 0

    for t1, t2 in zip(maturities[:-1], maturities[1:]):
        diff = svi_w(k, slices[t2]) - svi_w(k, slices[t1])
        bad = diff < 0
        total_bad += int(bad.sum())
        pairs.append(
            {
                "T_short": t1,
                "T_long": t2,
                "n_negative": int(bad.sum()),
                "min_dw": float(diff.min()),
                "argmin_k": float(k[int(np.argmin(diff))]),
            }
        )

    return {
        "pairs": pairs,
        "n_negative_total": total_bad,
        "free_of_calendar": total_bad == 0,
    }


def _initial_guess(k, w) -> np.ndarray:
    """
    Crude but stable start. a near the minimum observed variance, b from the
    overall slope scale, m at the observed minimum, sigma at the strike spread.
    A bad start is the usual reason SVI fits land in a local minimum, so this
    is deliberately data-driven rather than a fixed constant.
    """
    k, w = np.asarray(k, float), np.asarray(w, float)
    spread = max(k.max() - k.min(), 1e-3)
    return np.array(
        [
            max(w.min() * 0.5, 1e-6),
            max((w.max() - w.min()) / spread, 1e-3),
            -0.3,
            k[int(np.argmin(w))],
            max(spread / 4.0, 1e-3),
        ]
    )


def fit_slice(k, w, weights=None, enforce_butterfly=False) -> tuple[SVIParams, dict]:
    """
    Weighted least squares on total variance subject to the parameter
    constraints, via SLSQP.

    Fitting in total variance rather than in vol is deliberate: SVI is linear-ish
    in w, and least squares in vol space silently overweights short maturities
    because w = sigma^2 * T.

    enforce_butterfly adds min_k g(k) >= 0 as a hard constraint. Off by default,
    because the interesting result is *whether* an unconstrained raw SVI fit
    violates it. Turn it on to see what the fit costs in RMSE once you forbid it.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(k) < 5:
        raise ValueError(f"need at least 5 quotes to fit 5 parameters, got {len(k)}")

    wt = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)
    wt = wt / wt.sum()

    def unpack(x) -> SVIParams:
        return SVIParams(*x)

    def objective(x):
        resid = svi_w(k, unpack(x)) - w
        return float(np.sum(wt * resid**2))

    cons = [
        # a + b*sigma*sqrt(1-rho^2) >= 0
        {"type": "ineq", "fun": lambda x: unpack(x).min_total_variance()},
    ]
    if enforce_butterfly:
        g_grid = np.linspace(k.min() - 0.2, k.max() + 0.2, 121)
        cons.append({"type": "ineq", "fun": lambda x: float(np.min(durrleman_g(g_grid, unpack(x))))})

    bounds = [
        (-2.0, 5.0),      # a
        (1e-8, 10.0),     # b
        (-0.999, 0.999),  # rho
        (-2.0, 2.0),      # m
        (1e-6, 5.0),      # sigma
    ]

    res = minimize(
        objective,
        _initial_guess(k, w),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    p = unpack(res.x)

    resid = svi_w(k, p) - w
    return p, {
        "success": bool(res.success),
        "message": str(res.message),
        "n_quotes": int(len(k)),
        "rmse_w": float(np.sqrt(np.mean(resid**2))),
        "max_abs_resid_w": float(np.max(np.abs(resid))),
        "param_violations": p.violations(),
    }
