"""
Chain -> implied vols -> fitted slices -> arbitrage report -> local vol.

Kept separate from svi.py so that module stays about the parameterisation and
this one is about the workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bs import NoImpliedVol, implied_vol, log_moneyness, total_variance
from .localvol import local_variance
from .svi import (SVIParams, butterfly_report, calendar_report, fit_slice,
                  svi_w)


def add_implied_vols(df: pd.DataFrame, rate: float = 0.0) -> tuple[pd.DataFrame, list[str]]:
    """
    Invert every quote. Quotes with no admissible implied vol are dropped and
    the reasons returned, because the count of unpriceable quotes is a finding
    about the chain rather than noise to hide.
    """
    vols, failures = [], []
    for r in df.itertuples():
        try:
            vols.append(
                implied_vol(r.price, r.forward, r.strike, r.maturity, rate, r.is_call)
            )
        except NoImpliedVol as exc:
            vols.append(np.nan)
            failures.append(str(exc))

    out = df.copy()
    out["implied_vol"] = vols
    out["k"] = log_moneyness(out["strike"], out["forward"])
    out["w"] = total_variance(out["implied_vol"], out["maturity"])
    return out.dropna(subset=["implied_vol"]).reset_index(drop=True), failures


def _weights(g: pd.DataFrame) -> np.ndarray | None:
    """
    Inverse bid-ask width, so tight quotes pull the fit harder than wide wing
    quotes. Uniform if the chain carries no bid or ask.
    """
    if not {"bid", "ask"}.issubset(g.columns) or g[["bid", "ask"]].isna().all().all():
        return None
    width = (g["ask"] - g["bid"]).to_numpy(float)
    width = np.where(np.isfinite(width) & (width > 0), width, np.nanmedian(width))
    return 1.0 / np.maximum(width, 1e-12)


def fit_surface(df: pd.DataFrame, enforce_butterfly: bool = False) -> tuple[dict, pd.DataFrame]:
    """Fit one raw SVI slice per expiry. Returns (slices, per-slice diagnostics)."""
    slices, diags = {}, []
    for T, g in df.groupby("maturity"):
        if len(g) < 5:
            diags.append({"maturity": T, "n_quotes": len(g), "success": False,
                          "message": "too few quotes", "rmse_w": np.nan})
            continue

        p, info = fit_slice(g["k"].to_numpy(), g["w"].to_numpy(),
                            weights=_weights(g), enforce_butterfly=enforce_butterfly)
        slices[float(T)] = p

        # residual in vol points is what a trader reads, not variance
        model_vol = np.sqrt(np.maximum(svi_w(g["k"].to_numpy(), p), 0.0) / T)
        vol_resid = model_vol - g["implied_vol"].to_numpy()

        inside = np.nan
        if {"bid", "ask"}.issubset(g.columns) and g[["bid", "ask"]].notna().all().all():
            from .bs import black76
            mp = black76(g["forward"], g["strike"], T, model_vol, 0.0, g["is_call"])
            inside = float(np.mean((mp >= g["bid"].to_numpy()) & (mp <= g["ask"].to_numpy())))

        diags.append({
            "maturity": float(T),
            "n_quotes": info["n_quotes"],
            "success": info["success"],
            "rmse_vol_bp": float(np.sqrt(np.mean(vol_resid**2)) * 10_000),
            "max_abs_vol_bp": float(np.max(np.abs(vol_resid)) * 10_000),
            "frac_inside_spread": inside,
            "param_violations": "; ".join(info["param_violations"]) or "none",
            **{k: getattr(p, k) for k in ("a", "b", "rho", "m", "sigma")},
        })

    return slices, pd.DataFrame(diags)


def arbitrage_report(slices: dict[float, SVIParams], k_min=-1.0, k_max=1.0, n=601) -> dict:
    """Butterfly per slice, calendar across slices."""
    bf = {T: butterfly_report(p, k_min, k_max, n) for T, p in slices.items()}
    return {
        "butterfly": bf,
        "calendar": calendar_report(slices, k_min, k_max, n),
        "n_slices_with_butterfly_arb": sum(1 for r in bf.values() if not r["free_of_butterfly"]),
    }


def build(df: pd.DataFrame, rate: float = 0.0, enforce_butterfly: bool = False,
          k_min: float = -1.0, k_max: float = 1.0, n_k: int = 601) -> dict:
    """Full workflow, returning everything needed for the results tables."""
    with_vols, failures = add_implied_vols(df, rate)
    slices, diags = fit_surface(with_vols, enforce_butterfly)
    if len(slices) < 2:
        raise ValueError(f"only {len(slices)} slice(s) fitted, need 2+ for local vol")

    k = np.linspace(k_min, k_max, n_k)
    return {
        "quotes": with_vols,
        "n_quotes_in": len(df),
        "n_quotes_used": len(with_vols),
        "inversion_failures": failures,
        "slices": slices,
        "diagnostics": diags,
        "arbitrage": arbitrage_report(slices, k_min, k_max, n_k),
        "local_vol": local_variance(k, slices),
    }
