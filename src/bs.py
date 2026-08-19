"""
Black-76 pricing and implied volatility inversion.

Forward form rather than spot form. Deribit options are on futures, so the
forward is quoted directly and using it avoids having to assume a dividend or
carry. For equity chains you would build F from spot, rate and dividends first
and the rest of the code is unchanged.

    C = exp(-rT) * [F * N(d1) - K * N(d2)]
    P = exp(-rT) * [K * N(-d2) - F * N(-d1)]
    d1 = (ln(F/K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

The inversion is where the first arbitrage check falls out for free. A call
price has to sit inside

    exp(-rT) * max(F - K, 0)  <  C  <  exp(-rT) * F

Outside those bounds no positive volatility reproduces the price, so the quote
is either stale, mismarked, or a genuine arbitrage. Rather than returning NaN
quietly, `implied_vol` says which bound was violated, because the count of
violated quotes in a chain is a result worth reporting rather than an error to
suppress.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

VOL_LOWER, VOL_UPPER = 1e-6, 5.0


class NoImpliedVol(ValueError):
    """Raised when no positive volatility reproduces the quoted price."""


def black76(forward, strike, maturity, vol, rate=0.0, is_call=True):
    """Undiscounted-forward Black-76 price. Vectorised over all arguments."""
    F = np.asarray(forward, dtype=float)
    K = np.asarray(strike, dtype=float)
    T = np.asarray(maturity, dtype=float)
    s = np.asarray(vol, dtype=float)

    disc = np.exp(-np.asarray(rate, dtype=float) * T)
    sqrt_t = s * np.sqrt(T)

    # at zero variance the option is worth its discounted intrinsic
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * s**2 * T) / sqrt_t
        d2 = d1 - sqrt_t

    call = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    price = np.where(is_call, call, put)

    intrinsic = disc * np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    return np.where(sqrt_t <= 0, intrinsic, price)


def price_bounds(forward, strike, maturity, rate=0.0, is_call=True):
    """Lower and upper no-arbitrage bounds on the option price."""
    disc = np.exp(-rate * maturity)
    if is_call:
        return disc * max(forward - strike, 0.0), disc * forward
    return disc * max(strike - forward, 0.0), disc * strike


def implied_vol(price, forward, strike, maturity, rate=0.0, is_call=True):
    """
    Invert Black-76 by Brent. Raises NoImpliedVol naming the violated bound
    rather than returning NaN, so the caller can count violations instead of
    silently dropping quotes.
    """
    lo, hi = price_bounds(forward, strike, maturity, rate, is_call)
    if price <= lo:
        raise NoImpliedVol(
            f"price {price:.6g} at or below intrinsic {lo:.6g} "
            f"(K={strike:g}, T={maturity:.4f})"
        )
    if price >= hi:
        raise NoImpliedVol(
            f"price {price:.6g} at or above upper bound {hi:.6g} "
            f"(K={strike:g}, T={maturity:.4f})"
        )

    def objective(s):
        return float(black76(forward, strike, maturity, s, rate, is_call)) - price

    try:
        return float(brentq(objective, VOL_LOWER, VOL_UPPER, xtol=1e-10, maxiter=200))
    except ValueError as exc:  # no sign change inside the bracket
        raise NoImpliedVol(
            f"no root in [{VOL_LOWER}, {VOL_UPPER}] for K={strike:g}, T={maturity:.4f}"
        ) from exc


def log_moneyness(strike, forward):
    """k = ln(K / F). Zero at the forward, not at spot."""
    return np.log(np.asarray(strike, dtype=float) / np.asarray(forward, dtype=float))


def total_variance(vol, maturity):
    """w = sigma^2 * T. SVI is written in total variance, not vol."""
    return np.asarray(vol, dtype=float) ** 2 * np.asarray(maturity, dtype=float)


def vol_from_total_variance(w, maturity):
    return np.sqrt(np.maximum(np.asarray(w, dtype=float), 0.0) / maturity)
