# Arbitrage-free volatility surface construction

Raw SVI fitted per expiry, screened for butterfly and calendar arbitrage, then
converted to a Dupire local volatility surface. Following Gatheral and Jacquier
(2014).

## The one idea this repo is built around

Dupire local variance, written in total implied variance `w(k, T)` with
`k = ln(K/F)`:

    sigma_loc^2(k, T) = (dw/dT) / g(k)

where `g` is **exactly** Durrleman's butterfly function

    g(k) = (1 - k*w'/(2w))^2 - (w'^2 / 4) * (1/w + 1/4) + w''/2

They are the same function. Expand the standard Dupire denominator term by term
and you get `g` back. Which means:

> Local volatility exists if and only if the surface is free of butterfly
> arbitrage, because the no-arbitrage condition **is** the denominator.

And the numerator carries the calendar condition the same way: `dw/dT < 0` is a
calendar spread arbitrage, and it also drives local variance negative. So the two
arbitrage screens are not hygiene checks bolted on afterwards. They are precisely
the numerator and denominator of the local vol formula being non-negative.

`localvol.py` reports where the surface fails rather than clipping it. A NaN in
that output is a statement about the quotes, and filling it from a neighbour is
how you end up calibrated to an arbitrage you never noticed.

## Method

1. **Invert.** Black-76 by Brent, forward form since Deribit options are on
   futures. Quotes outside the no-arbitrage price bounds raise an error naming
   the violated bound instead of returning NaN, so unpriceable quotes get counted
   as a finding.
2. **Fit.** Raw SVI, `w(k) = a + b(rho(k-m) + sqrt((k-m)^2 + sigma^2))`, by
   weighted least squares under SLSQP. Weights are inverse bid-ask width so tight
   quotes pull harder than wide wing quotes. Constraints: `b >= 0`, `|rho| < 1`,
   `sigma > 0`, and `a + b*sigma*sqrt(1-rho^2) >= 0`, the last being
   `min_k w(k) >= 0` and the one that is easiest to drop by accident.
3. **Screen butterfly.** `g(k) >= 0` on a grid, using analytic first and second
   derivatives. Finite-differencing them and feeding the result into the test
   produces spurious violations at grid edges.
4. **Screen calendar.** Total variance non-decreasing in `T` at fixed `k`, on a
   shared grid rather than on quoted strikes, since strikes differ by expiry.
5. **Local vol.** The formula above, with `dw/dT` by central differences across
   fitted slices.

Two deliberate choices. Fitting in total variance rather than in vol, because
`w = sigma^2 T` means least squares in vol space silently overweights short
maturities. And `--enforce-butterfly` is **off** by default, because the
interesting question is whether an unconstrained fit violates the condition; turn
it on to measure what forbidding it costs in RMSE.

## Running it

```bash
pip install -r requirements.txt
pytest tests/ -q                        # 25 tests
python -m src.run                       # synthetic chain, end to end
python -m src.run --noise-vol 0.006     # add quote noise
python -m src.run --enforce-butterfly
python -m src.run --chain data/btc.json --index-price 65000
```

## Data

Deribit is the intended source: public, no auth, many expiries, wide strikes,
which is what you need to see wing behaviour. Equity chains work if you build the
forward from spot, rate and dividends first.

```python
import requests
url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
js = requests.get(url, params={"currency": "BTC", "kind": "option"}).json()
df = parse_deribit_book_summary(js["result"], index_price=...)
```

Deribit quotes option prices in units of the underlying, so a BTC option at 0.05
costs 0.05 BTC. The loader converts to absolute units. Mixing the two is the
fastest way to a nonsense surface. Cache the JSON to disk and re-run against it,
since a surface that cannot be reproduced cannot be debugged.

**No network calls happen in this repo.** `parse_deribit_book_summary` takes
parsed JSON so it stays testable offline.

## Results, synthetic

Chain generated from known SVI slices, so the fitter can be checked against
ground truth. **These numbers describe the generator, not a market.**

Clean chain, 63 quotes across three expiries:

| T | Quotes | RMSE (bp vol) | Max abs (bp) | Inside spread | a | b | rho | m | sigma |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 21 | 0.03 | 0.07 | 100% | 0.0100 | 0.0900 | −0.3500 | 0.0100 | 0.1800 |
| 0.50 | 21 | 0.004 | 0.01 | 100% | 0.0220 | 0.1300 | −0.3200 | 0.0150 | 0.2200 |
| 1.00 | 21 | 0.001 | 0.002 | 100% | 0.0450 | 0.1900 | −0.2800 | 0.0200 | 0.2800 |

Recovered parameters match the generating values to four decimal places. Local
vol defined at 100% of grid points, ranging 0.265 to 0.967.

With 60bp of vol noise, RMSE rises to 45–61bp and the fraction of quotes priced
inside the spread falls to 24–38%, since the spread is only ±40bp wide.

## What did not work, and what is still missing

**The butterfly screen never fired on synthetic data.** Sweeping quote noise from
0 to 400bp across six seeds, zero slices violated `g >= 0`:

| Noise (vol) | Mean RMSE | Slices with butterfly arb (6 seeds) |
|---|---|---|
| 0.000 | 0.01bp | 0 |
| 0.010 | 100bp | 0 |
| 0.020 | 196bp | 0 |
| 0.040 | 402bp | 0 |

The parameter constraints, especially `b >= 0` forcing convexity, are apparently
strong enough that iid quote noise cannot push a fitted slice into butterfly
arbitrage. This is **not** evidence that raw SVI is safe. It says the generator
produces internally consistent quotes and noise around them stays consistent.
Real violations come from a different mechanism: stale wing quotes, crossed
markets, and strikes that disagree with each other rather than with a smooth
curve. `test_durrleman_detects_a_known_bad_slice` confirms the checker fires on a
hand-built violating parameter set, so the screen works and simply has nothing to
catch here. Testing it on a real chain is the open question.

**Time value underflow is asymmetric.** A deep ITM call at short maturity prices
to exactly intrinsic, since its time value is below the last representable bit,
and no vol is recoverable. The mirror OTM strike prices to about 1e-23 and still
inverts to 1e-14 accuracy, because its lower bound is zero so Brent brackets
fine. Invert the OTM side and get the ITM side by parity. Assuming both tails
break is wrong, and both cases are pinned by tests.

**Not attempted.** SSVI or eSSVI, which is where Gatheral and Jacquier go next
precisely to guarantee no calendar arbitrage across slices by construction rather
than checking after the fact. No term interpolation beyond finite differences.
`dw/dT` from three expiries is crude. No smoothing of quotes before fitting, no
outlier rejection, and no treatment of the discrete-dividend or funding-rate
issues that arise on equity chains.

## Reference

Gatheral, J. and Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
*Quantitative Finance*, 14(1), 59–71.
