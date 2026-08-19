"""
Entry point.

    python -m src.run                          # synthetic chain, end to end
    python -m src.run --noise-vol 0.006        # add quote noise
    python -m src.run --enforce-butterfly      # forbid butterfly arb in the fit
    python -m src.run --chain data/btc.json --index-price 65000
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import parse_deribit_book_summary, synthetic_chain
from .surface import build


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default=None, help="cached Deribit book-summary JSON")
    ap.add_argument("--index-price", type=float, default=None)
    ap.add_argument("--noise-vol", type=float, default=0.0)
    ap.add_argument("--enforce-butterfly", action="store_true")
    ap.add_argument("--rate", type=float, default=0.0)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    if args.chain:
        js = json.loads(Path(args.chain).read_text())
        rows = js["result"] if "result" in js else js
        df = parse_deribit_book_summary(rows, index_price=args.index_price)
        source = f"deribit:{Path(args.chain).name}"
    else:
        df, truth = synthetic_chain(noise_vol=args.noise_vol)
        source = f"synthetic(noise_vol={args.noise_vol})"

    res = build(df, rate=args.rate, enforce_butterfly=args.enforce_butterfly)

    print(f"source: {source}")
    print(f"quotes: {res['n_quotes_used']} used of {res['n_quotes_in']}, "
          f"{len(res['inversion_failures'])} not invertible")
    if res["inversion_failures"]:
        for f in res["inversion_failures"][:5]:
            print(f"   {f}")

    pd.set_option("display.width", 200, "display.max_columns", 30,
                  "display.float_format", lambda v: f"{v:,.4f}")
    print("\nper-expiry fit")
    cols = ["maturity","n_quotes","rmse_vol_bp","max_abs_vol_bp","frac_inside_spread",
            "a","b","rho","m","sigma","param_violations"]
    print(res["diagnostics"][cols].to_string(index=False))

    print("\nbutterfly, Durrleman g on [-1, 1]")
    for T, r in sorted(res["arbitrage"]["butterfly"].items()):
        status = "clean" if r["free_of_butterfly"] else f"ARB at k in {r['k_negative_range']}"
        print(f"  T={T:.4f}  min g={r['min_g']:+.6f} at k={r['argmin_k']:+.3f}  "
              f"{r['n_negative']}/{r['n_grid']} negative  {status}")

    print("\ncalendar, total variance across expiries")
    for pr in res["arbitrage"]["calendar"]["pairs"]:
        status = "clean" if pr["n_negative"] == 0 else "ARB"
        print(f"  T {pr['T_short']:.4f} -> {pr['T_long']:.4f}  "
              f"min dw={pr['min_dw']:+.6f} at k={pr['argmin_k']:+.3f}  "
              f"{pr['n_negative']} negative  {status}")

    lv = res["local_vol"]
    print(f"\nDupire local vol, sigma_loc^2 = (dw/dT) / g(k)")
    print(f"  grid {lv['local_variance'].shape[0]} expiries x {lv['local_variance'].shape[1]} strikes")
    print(f"  defined at {100*lv['fraction_defined']:.2f}% of points")
    print(f"  butterfly failures (g <= 0):   {lv['n_butterfly_failures']}")
    print(f"  calendar failures (dw/dT < 0): {lv['n_calendar_failures']}")
    with np.errstate(all="ignore"):
        v = lv["local_vol"][np.isfinite(lv["local_vol"])]
        if v.size:
            print(f"  local vol range: {v.min():.4f} to {v.max():.4f}")

    out = Path(args.outdir); out.mkdir(exist_ok=True)
    res["diagnostics"].to_csv(out / "fit_diagnostics.csv", index=False)
    with open(out / "summary.json", "w") as fh:
        json.dump({
            "source": source,
            "n_quotes_in": res["n_quotes_in"],
            "n_quotes_used": res["n_quotes_used"],
            "slices": {str(T): p.__dict__ for T, p in res["slices"].items()},
            "butterfly": {str(T): {k: v for k, v in r.items() if k != "k_negative_range"}
                          for T, r in res["arbitrage"]["butterfly"].items()},
            "calendar": res["arbitrage"]["calendar"],
            "local_vol": {k: res["local_vol"][k] for k in
                          ("n_butterfly_failures","n_calendar_failures","n_undefined",
                           "n_points","fraction_defined")},
        }, fh, indent=2, default=float)
    print(f"\nwrote {out}/")

    if not args.chain:
        print("\nSYNTHETIC CHAIN. These numbers describe the generator, not a market.")


if __name__ == "__main__":
    main()
