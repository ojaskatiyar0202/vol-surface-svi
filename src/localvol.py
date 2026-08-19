"""
Dupire local volatility from the fitted surface.

In terms of total implied variance w(k, T) with k = ln(K/F):

    sigma_loc^2(k, T) = (dw/dT) / g(k)

where g is *exactly* the Durrleman function from svi.py. That is the whole point
of this module and the one line worth being able to say out loud:

    local volatility is well defined if and only if the surface is free of
    butterfly arbitrage

because g is the denominator. Where g goes negative, local variance goes
negative, and there is no local volatility model consistent with those quotes.
The no-arbitrage condition is not a hygiene check bolted on afterwards; it is
the existence condition.

The numerator carries the calendar condition in the same way. dw/dT < 0 is
calendar arbitrage, and it also drives local variance negative. So the two
checks in svi.py are precisely the numerator and denominator of this formula
being non-negative.

dw/dT is taken by finite difference across fitted slices, since raw SVI is
fitted per expiry and says nothing about the term direction. Central differences
in the interior, one-sided at the ends.
"""

from __future__ import annotations

import numpy as np

from .svi import SVIParams, durrleman_g, svi_w


def dw_dT(k, slices: dict[float, SVIParams]) -> tuple[np.ndarray, np.ndarray]:
    """
    Finite-difference the term derivative of total variance on a k grid.
    Returns (maturities, dwdT) with dwdT shaped (n_maturities, n_k).
    """
    if len(slices) < 2:
        raise ValueError("need at least two expiries to differentiate in T")

    k = np.asarray(k, dtype=float)
    T = np.array(sorted(slices), dtype=float)
    W = np.vstack([svi_w(k, slices[t]) for t in T])

    out = np.empty_like(W)
    out[0] = (W[1] - W[0]) / (T[1] - T[0])
    out[-1] = (W[-1] - W[-2]) / (T[-1] - T[-2])
    for i in range(1, len(T) - 1):
        out[i] = (W[i + 1] - W[i - 1]) / (T[i + 1] - T[i - 1])
    return T, out


def local_variance(k, slices: dict[float, SVIParams]) -> dict:
    """
    Local variance surface plus a full accounting of where it fails to exist.

    Failures are reported, not clipped. A NaN in this surface is a statement
    about the quotes, and quietly filling it with a neighbour is how people end
    up calibrating to an arbitrage they never noticed.
    """
    k = np.asarray(k, dtype=float)
    T, dwdT = dw_dT(k, slices)

    G = np.vstack([durrleman_g(k, slices[t]) for t in T])

    with np.errstate(divide="ignore", invalid="ignore"):
        lv = dwdT / G

    bad_g = G <= 0
    bad_num = dwdT < 0
    undefined = bad_g | bad_num

    lv_clean = np.where(undefined, np.nan, lv)

    return {
        "maturities": T,
        "log_moneyness": k,
        "local_variance": lv_clean,
        "local_vol": np.sqrt(np.maximum(lv_clean, 0.0)),
        "g": G,
        "dw_dT": dwdT,
        "n_butterfly_failures": int(bad_g.sum()),
        "n_calendar_failures": int(bad_num.sum()),
        "n_undefined": int(undefined.sum()),
        "n_points": int(undefined.size),
        "fraction_defined": float(1.0 - undefined.sum() / undefined.size),
    }
