"""
Chains in, canonical frame out.

Every downstream function expects:
    maturity  float, year fraction
    forward   float
    strike    float
    is_call   bool
    price     float, in the same units as forward
    bid, ask  float, optional, used for fit weights and for the inside-spread test

Deribit is the default source: public, no auth, many expiries, wide strikes,
which is what you need to see wing behaviour. Note Deribit quotes option prices
in units of the underlying, so a BTC option priced at 0.05 costs 0.05 BTC. The
loader converts to absolute units using the index price, since mixing the two is
the single easiest way to get a nonsense surface.

Network calls are not made here. `parse_deribit_book_summary` takes the parsed
JSON so the loader is testable offline and so you can cache a chain to disk and
re-run against it, which you want anyway for reproducibility.

    import requests
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    js = requests.get(url, params={"currency": "BTC", "kind": "option"}).json()
    df = parse_deribit_book_summary(js["result"], index_price=...)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .svi import SVIParams, svi_w

REQUIRED = ["maturity", "forward", "strike", "is_call", "price"]

_NAME = re.compile(r"^(?P<ccy>[A-Z]+)-(?P<expiry>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>\d+)-(?P<kind>[CP])$")


def parse_instrument_name(name: str) -> dict:
    """BTC-27JUN26-80000-C -> currency, expiry date, strike, call flag."""
    m = _NAME.match(name.strip().upper())
    if not m:
        raise ValueError(f"unrecognised instrument name: {name!r}")
    return {
        "currency": m["ccy"],
        "expiry": datetime.strptime(m["expiry"], "%d%b%y").replace(tzinfo=timezone.utc),
        "strike": float(m["strike"]),
        "is_call": m["kind"] == "C",
    }


def parse_deribit_book_summary(rows, index_price: float, now=None) -> pd.DataFrame:
    """
    Convert a get_book_summary_by_currency result into the canonical frame.
    Prices are converted from underlying units to absolute units.
    """
    now = now or datetime.now(timezone.utc)
    out = []
    for r in rows:
        try:
            meta = parse_instrument_name(r["instrument_name"])
        except ValueError:
            continue

        mark = r.get("mark_price")
        bid, ask = r.get("bid_price"), r.get("ask_price")
        if mark is None:
            continue

        years = (meta["expiry"] - now).total_seconds() / (365.25 * 24 * 3600)
        if years <= 0:
            continue

        out.append(
            {
                "maturity": years,
                "forward": float(r.get("underlying_price") or index_price),
                "strike": meta["strike"],
                "is_call": meta["is_call"],
                "price": float(mark) * index_price,
                "bid": float(bid) * index_price if bid else np.nan,
                "ask": float(ask) * index_price if ask else np.nan,
            }
        )

    df = pd.DataFrame(out)
    if df.empty:
        raise ValueError("no usable option rows parsed")
    return df.sort_values(["maturity", "strike"]).reset_index(drop=True)


def synthetic_chain(
    true_slices: dict[float, SVIParams] | None = None,
    forward: float = 100.0,
    n_strikes: int = 21,
    k_range: float = 0.9,
    spread_vol: float = 0.004,
    noise_vol: float = 0.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[float, SVIParams]]:
    """
    Build a chain from known SVI slices so the fitter can be checked against
    ground truth. With noise_vol = 0 a correct fitter recovers the parameters
    almost exactly, which is the strongest test available without market data.

    Returns (frame, true_slices).
    """
    from .bs import black76, vol_from_total_variance

    rng = np.random.default_rng(seed)
    if true_slices is None:
        true_slices = {
            0.25: SVIParams(a=0.010, b=0.090, rho=-0.35, m=0.010, sigma=0.180),
            0.50: SVIParams(a=0.022, b=0.130, rho=-0.32, m=0.015, sigma=0.220),
            1.00: SVIParams(a=0.045, b=0.190, rho=-0.28, m=0.020, sigma=0.280),
        }

    rows = []
    for T, p in true_slices.items():
        k = np.linspace(-k_range, k_range, n_strikes)
        strikes = forward * np.exp(k)
        vols = vol_from_total_variance(svi_w(k, p), T)
        if noise_vol > 0:
            vols = np.maximum(vols + rng.normal(0.0, noise_vol, size=vols.shape), 1e-4)

        for K, s, kk in zip(strikes, vols, k):
            is_call = kk >= 0  # OTM side, as a real chain would quote
            mid = float(black76(forward, K, T, s, 0.0, is_call))
            half = 0.5 * abs(
                float(black76(forward, K, T, s + spread_vol, 0.0, is_call)) - mid
            )
            rows.append(
                {
                    "maturity": T,
                    "forward": forward,
                    "strike": float(K),
                    "is_call": bool(is_call),
                    "price": mid,
                    "bid": max(mid - half, 1e-12),
                    "ask": mid + half,
                }
            )

    return pd.DataFrame(rows).sort_values(["maturity", "strike"]).reset_index(drop=True), true_slices
