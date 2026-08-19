import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.bs import (NoImpliedVol, black76, implied_vol, price_bounds,
                    total_variance, vol_from_total_variance)
from src.data import parse_instrument_name, synthetic_chain
from src.localvol import local_variance
from src.surface import add_implied_vols, build, fit_surface
from src.svi import SVIParams, durrleman_g, fit_slice, svi_d2w, svi_dw, svi_w

P = SVIParams(a=0.02, b=0.13, rho=-0.32, m=0.015, sigma=0.22)


def test_put_call_parity():
    F, K, T, s = 100.0, 110.0, 0.5, 0.3
    c = float(black76(F, K, T, s, 0.0, True))
    p = float(black76(F, K, T, s, 0.0, False))
    assert c - p == pytest.approx(F - K, abs=1e-10)


def test_zero_vol_gives_intrinsic():
    assert float(black76(100.0, 90.0, 1.0, 0.0, 0.0, True)) == pytest.approx(10.0)
    assert float(black76(100.0, 110.0, 1.0, 0.0, 0.0, True)) == pytest.approx(0.0)


def test_price_monotone_in_vol():
    p = [float(black76(100.0, 100.0, 1.0, s)) for s in (0.1, 0.2, 0.4, 0.8)]
    assert p == sorted(p)


def test_implied_vol_roundtrip():
    """
    Round-trip only where the option carries resolvable time value. Deep ITM or
    deep OTM at short maturity has time value below double precision, and those
    cases are covered separately below.
    """
    for K in (60.0, 100.0, 160.0):
        for T in (0.1, 1.0, 3.0):
            for s in (0.15, 0.45, 1.2):
                px = float(black76(100.0, K, T, s, 0.0, True))
                lo, _ = price_bounds(100.0, K, T, 0.0, True)
                if px - lo < 1e-10:
                    continue
                assert implied_vol(px, 100.0, K, T, 0.0, True) == pytest.approx(s, rel=1e-6)


def test_deep_itm_time_value_underflow_raises():
    """
    A deep ITM call at short maturity prices to exactly intrinsic, because its
    time value is smaller than the last representable bit of 40.0. There is no
    recoverable vol and the code must say so rather than invent one.

    Practical consequence: invert the OTM side of the chain and get the ITM side
    by put-call parity. synthetic_chain picks the OTM side deliberately.
    """
    px = float(black76(100.0, 60.0, 0.1, 0.15, 0.0, True))
    assert px == pytest.approx(40.0, abs=1e-12)
    with pytest.raises(NoImpliedVol, match="intrinsic"):
        implied_vol(px, 100.0, 60.0, 0.1, 0.0, True)


def test_deep_otm_still_inverts_despite_tiny_price():
    """
    The mirror case does NOT fail. A 160 call at T=0.1 prices to about 1e-23,
    but its lower bound is zero so Brent still brackets a sign change and
    recovers the vol to ~1e-14. Worth pinning: the failure mode is asymmetric,
    and assuming both tails break is wrong.
    """
    px = float(black76(100.0, 160.0, 0.1, 0.15, 0.0, True))
    assert 0 < px < 1e-20
    assert implied_vol(px, 100.0, 160.0, 0.1, 0.0, True) == pytest.approx(0.15, rel=1e-9)


def test_otm_side_inverts_where_itm_side_underflows():
    """The same 60-strike is invertible as a put, which is why you use OTM."""
    px = float(black76(100.0, 60.0, 0.1, 0.15, 0.0, False))
    assert implied_vol(px, 100.0, 60.0, 0.1, 0.0, False) == pytest.approx(0.15, rel=1e-6)


def test_below_intrinsic_is_rejected_by_name():
    lo, _ = price_bounds(100.0, 80.0, 1.0, 0.0, True)
    with pytest.raises(NoImpliedVol, match="intrinsic"):
        implied_vol(lo * 0.99, 100.0, 80.0, 1.0, 0.0, True)


def test_above_upper_bound_is_rejected_by_name():
    with pytest.raises(NoImpliedVol, match="upper bound"):
        implied_vol(101.0, 100.0, 80.0, 1.0, 0.0, True)


def test_total_variance_roundtrip():
    assert vol_from_total_variance(total_variance(0.3, 2.0), 2.0) == pytest.approx(0.3)


def test_derivatives_match_finite_differences():
    k = np.linspace(-0.8, 0.8, 41)
    # separate steps: the second difference divides by h^2, so 1e-6 loses four
    # digits to cancellation. 1e-4 is the sweet spot here (~3e-7 relative).
    h1, h2 = 1e-6, 1e-4
    fd1 = (svi_w(k + h1, P) - svi_w(k - h1, P)) / (2 * h1)
    fd2 = (svi_w(k + h2, P) - 2 * svi_w(k, P) + svi_w(k - h2, P)) / h2**2
    np.testing.assert_allclose(svi_dw(k, P), fd1, rtol=1e-6)
    np.testing.assert_allclose(svi_d2w(k, P), fd2, rtol=1e-5)


def test_slice_is_convex_when_b_positive():
    assert (svi_d2w(np.linspace(-2, 2, 101), P) >= 0).all()


def test_minimum_total_variance_is_attained_where_predicted():
    k_star = P.m - P.rho * P.sigma / np.sqrt(1 - P.rho**2)
    assert svi_w(k_star, P) == pytest.approx(P.min_total_variance(), rel=1e-10)
    k = np.linspace(k_star - 2, k_star + 2, 4001)
    assert svi_w(k, P).min() >= P.min_total_variance() - 1e-12


def test_parameter_violations_detected():
    assert SVIParams(0.02, -1.0, -0.3, 0.0, 0.2).violations()
    assert SVIParams(0.02, 0.1, 1.5, 0.0, 0.2).violations()
    assert SVIParams(0.02, 0.1, -0.3, 0.0, -0.2).violations()
    assert SVIParams(-5.0, 0.1, -0.3, 0.0, 0.2).violations()
    assert P.violations() == []


def test_durrleman_g_positive_for_benign_slice():
    assert (durrleman_g(np.linspace(-1.0, 1.0, 401), P) > 0).all()


def test_durrleman_detects_a_known_bad_slice():
    bad = SVIParams(a=0.001, b=1.2, rho=-0.95, m=0.0, sigma=0.02)
    assert (durrleman_g(np.linspace(-1.0, 1.0, 801), bad) < 0).any()


def test_fit_recovers_known_parameters_without_noise():
    k = np.linspace(-0.9, 0.9, 25)
    p, info = fit_slice(k, svi_w(k, P))
    assert info["success"]
    assert info["rmse_w"] < 1e-7
    np.testing.assert_allclose(svi_w(k, p), svi_w(k, P), atol=1e-7)


def test_fit_rejects_underdetermined_slice():
    with pytest.raises(ValueError, match="at least 5"):
        fit_slice(np.array([0.0, 0.1, 0.2]), np.array([0.02, 0.02, 0.03]))


def test_fitted_slices_respect_parameter_constraints():
    df, _ = synthetic_chain(noise_vol=0.0)
    with_vols, _ = add_implied_vols(df)
    slices, diags = fit_surface(with_vols)
    assert len(slices) == 3
    for p in slices.values():
        assert p.violations() == []
    assert (diags["rmse_vol_bp"] < 5.0).all()


def test_build_end_to_end_on_clean_chain():
    df, truth = synthetic_chain(noise_vol=0.0)
    res = build(df)
    assert res["n_quotes_used"] == res["n_quotes_in"]
    assert res["inversion_failures"] == []
    assert res["arbitrage"]["n_slices_with_butterfly_arb"] == 0
    assert res["arbitrage"]["calendar"]["free_of_calendar"]
    assert res["local_vol"]["fraction_defined"] == 1.0


def test_local_vol_matches_manual_formula():
    df, _ = synthetic_chain(noise_vol=0.0)
    res = build(df)
    lv = res["local_vol"]
    manual = lv["dw_dT"] / lv["g"]
    ok = np.isfinite(lv["local_variance"])
    np.testing.assert_allclose(lv["local_variance"][ok], manual[ok], rtol=1e-10)


def test_local_vol_undefined_where_calendar_fails():
    slices = {
        0.5: SVIParams(a=0.10, b=0.10, rho=-0.3, m=0.0, sigma=0.2),
        1.0: SVIParams(a=0.02, b=0.10, rho=-0.3, m=0.0, sigma=0.2),
    }
    out = local_variance(np.linspace(-0.5, 0.5, 51), slices)
    assert out["n_calendar_failures"] > 0
    assert out["fraction_defined"] < 1.0


def test_local_vol_needs_two_expiries():
    with pytest.raises(ValueError, match="at least two"):
        local_variance(np.linspace(-0.5, 0.5, 11), {0.5: P})


def test_instrument_name_parsing():
    m = parse_instrument_name("BTC-27JUN26-80000-C")
    assert m["currency"] == "BTC" and m["strike"] == 80000.0 and m["is_call"]
    assert parse_instrument_name("ETH-26SEP25-4000-P")["is_call"] is False
    with pytest.raises(ValueError):
        parse_instrument_name("not-an-instrument")


def test_noisy_chain_still_fits_reasonably():
    df, _ = synthetic_chain(noise_vol=0.006, seed=7)
    res = build(df)
    assert (res["diagnostics"]["rmse_vol_bp"] < 150).all()
