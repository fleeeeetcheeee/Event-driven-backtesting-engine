"""
Metrics, checked against closed-form answers on constructed series.

Where a metric has a known analytic value for a given input, that value is the
assertion. Where it does not, the test pins a documented convention (Sortino's
denominator, two-way turnover) that would otherwise drift silently.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from evbt.analytics.attribution import attribute, newey_west_lags
from evbt.analytics.capacity import (
    capacity_curve,
    estimate_capacity,
    realised_cost_rate,
)
from evbt.analytics.metrics import (
    annualised_return,
    annualised_turnover,
    annualised_volatility,
    calmar_ratio,
    conditional_value_at_risk,
    evaluate,
    hit_rate,
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    sharpe_standard_error,
    sortino_ratio,
    total_return,
    value_at_risk,
)


def nav_from_returns(returns, start=100.0) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=len(returns) + 1)
    return pd.Series(start * np.cumprod(np.r_[1.0, 1.0 + np.asarray(returns)]), index=idx)


class TestReturnAndRisk:
    def test_total_return(self):
        assert total_return(pd.Series([100.0, 110.0])) == pytest.approx(0.10)

    def test_annualised_return_is_geometric(self):
        """252 periods of exactly 1% compounds to 1.01^252 - 1 over one year."""
        nav = nav_from_returns([0.01] * 252)
        assert annualised_return(nav) == pytest.approx(1.01**252 - 1.0)

    def test_annualised_vol_scales_by_root_time(self):
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.0, 0.01, 2520))
        assert annualised_volatility(returns) == pytest.approx(
            returns.std(ddof=1) * math.sqrt(252)
        )

    def test_constant_returns_have_zero_vol_and_zero_sharpe(self):
        """Division by a zero standard deviation must not produce infinity."""
        returns = pd.Series([0.001] * 100)
        assert annualised_volatility(returns) == pytest.approx(0.0)
        assert sharpe_ratio(returns) == 0.0


class TestSharpe:
    def test_matches_the_definition(self):
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0.0005, 0.01, 1000))
        expected = returns.mean() / returns.std(ddof=1) * math.sqrt(252)
        assert sharpe_ratio(returns) == pytest.approx(expected)

    def test_risk_free_rate_is_subtracted(self):
        returns = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 1000))
        assert sharpe_ratio(returns, risk_free_rate=0.05) < sharpe_ratio(returns)

    def test_standard_error_shrinks_with_sample_size(self):
        assert sharpe_standard_error(1.0, 3000) < sharpe_standard_error(1.0, 250)

    def test_standard_error_matches_lo_2002(self):
        # sqrt((1 + 1.0^2 / 2) / 756)
        assert sharpe_standard_error(1.0, 756) == pytest.approx(
            math.sqrt(1.5 / 756)
        )

    def test_three_years_of_daily_data_gives_a_wide_interval(self):
        """The number behind the module docstring's warning."""
        assert sharpe_standard_error(1.0, 756) == pytest.approx(0.0445, abs=0.002)


class TestSortino:
    def test_penalises_only_downside(self):
        upside = pd.Series([0.01, 0.02, 0.03, -0.001] * 50)
        symmetric = pd.Series([0.01, -0.02, 0.03, -0.001] * 50)
        assert sortino_ratio(upside) > sortino_ratio(symmetric)

    def test_denominator_divides_by_all_observations(self):
        """
        Not by the count of negatives. The wrong convention inflates the ratio
        most for strategies that rarely lose.
        """
        returns = pd.Series([0.01] * 9 + [-0.02])
        downside = math.sqrt((0.02**2) / 10)  # 10 observations, not 1
        expected = returns.mean() / downside * math.sqrt(252)
        assert sortino_ratio(returns) == pytest.approx(expected)

    def test_no_downside_is_infinite(self):
        assert sortino_ratio(pd.Series([0.01] * 50)) == float("inf")


class TestDrawdown:
    def test_simple_peak_to_trough(self):
        nav = pd.Series([100.0, 120.0, 90.0, 110.0])
        assert max_drawdown(nav) == pytest.approx(-0.25)  # 120 -> 90

    def test_monotonic_series_has_no_drawdown(self):
        assert max_drawdown(pd.Series([100.0, 110.0, 120.0])) == pytest.approx(0.0)

    def test_duration_counts_periods_below_the_peak(self):
        nav = pd.Series([100.0, 120.0, 90.0, 95.0, 99.0, 130.0])
        assert max_drawdown_duration(nav) == 3

    def test_calmar_is_return_over_drawdown(self):
        nav = nav_from_returns([0.01, -0.02, 0.015, -0.01] * 63)
        assert calmar_ratio(nav) == pytest.approx(
            annualised_return(nav) / abs(max_drawdown(nav))
        )


class TestTails:
    def test_var_is_the_empirical_quantile(self):
        returns = pd.Series(np.linspace(-0.10, 0.10, 101))
        assert value_at_risk(returns, 0.95) == pytest.approx(np.quantile(returns, 0.05))

    def test_cvar_is_at_least_as_bad_as_var(self):
        returns = pd.Series(np.random.default_rng(3).normal(0.0, 0.01, 2000))
        assert conditional_value_at_risk(returns) <= value_at_risk(returns)

    def test_hit_rate(self):
        assert hit_rate(pd.Series([0.01, -0.01, 0.01, 0.01])) == pytest.approx(0.75)


class TestTurnover:
    def test_two_way_annualised_turnover(self):
        """
        252 bars, mean NAV 100, 100 of notional traded per bar -> 252x a year
        two-way. Quoted two-way because that is what costs are charged on.
        """
        curve = pd.DataFrame(
            {"nav": [100.0] * 252, "turnover_notional": [100.0] * 252},
            index=pd.bdate_range("2020-01-01", periods=252),
        )
        assert annualised_turnover(curve) == pytest.approx(252.0)

    def test_no_trading_is_zero_turnover(self):
        curve = pd.DataFrame({"nav": [100.0] * 10, "turnover_notional": [0.0] * 10})
        assert annualised_turnover(curve) == pytest.approx(0.0)


class TestEvaluate:
    def _curve(self, n=800, seed=4):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0004, 0.008, n)
        nav = nav_from_returns(returns)
        return pd.DataFrame(
            {
                "nav": nav,
                "turnover_notional": [0.0] * len(nav),
                "gross_exposure": nav.to_numpy(),
            }
        )

    def test_produces_a_full_report(self):
        report = evaluate(self._curve())
        assert report.n_periods == 801
        assert report.sharpe_stderr > 0
        assert report.sharpe_tstat == pytest.approx(report.sharpe / report.sharpe_stderr)

    def test_short_sample_is_flagged(self):
        report = evaluate(self._curve(n=100))
        assert any("too wide" in note for note in report.notes)

    def test_implausible_sharpe_is_flagged(self):
        """The spec is explicit that a Sharpe of 4 is a red flag to report."""
        curve = pd.DataFrame({"nav": nav_from_returns([0.001] * 600 + [-0.0001] * 200)})
        report = evaluate(curve)
        assert report.sharpe > 3.0
        assert any("implausibly high" in note for note in report.notes)

    def test_empty_curve_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            evaluate(pd.DataFrame())

    def test_cost_drag_is_annualised_against_mean_nav(self):
        curve = pd.DataFrame({"nav": [100.0] * 252, "turnover_notional": [0.0] * 252})
        fills = pd.DataFrame({"total_cost": [1.0] * 10})
        report = evaluate(curve, fills)
        assert report.total_costs == pytest.approx(10.0)
        assert report.cost_drag_annualised == pytest.approx(10.0 / 100.0)


class TestAttribution:
    def test_recovers_a_planted_beta_and_alpha(self):
        rng = np.random.default_rng(5)
        n = 2000
        idx = pd.bdate_range("2015-01-01", periods=n)
        market = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
        alpha = 0.0002
        strategy = alpha + 0.6 * market + pd.Series(rng.normal(0, 0.002, n), index=idx)

        result = attribute(strategy, pd.DataFrame({"MKT": market}))

        assert result.exposures[0].beta == pytest.approx(0.6, abs=0.02)
        # Tolerance is three standard errors of the alpha estimate:
        # residual sd 0.002 over n=2000 gives SE ~ 4.5e-5. A tighter bound
        # would be asserting that noise is absent, not that the fit is right.
        assert result.alpha == pytest.approx(alpha, abs=1.5e-4)
        # Signal variance (0.6 * 0.01)^2 against noise variance 0.002^2 puts
        # the population R-squared at 0.90; the sample lands just under it.
        assert result.r_squared > 0.85

    def test_pure_beta_shows_no_significant_alpha(self):
        """
        The question an interviewer asks first: is this alpha or beta?

        The strategy is a levered index plus idiosyncratic noise. The noise is
        not decoration — with an exact multiple of the market the regression
        fits perfectly, residuals collapse to ~1e-19, and every t-statistic
        becomes a 0/0 artifact of order 1e16. Real return series always carry
        residual variance, so the noise is what makes this test meaningful
        rather than degenerate.
        """
        rng = np.random.default_rng(6)
        n = 1500
        idx = pd.bdate_range("2015-01-01", periods=n)
        market = pd.Series(rng.normal(0.0004, 0.01, n), index=idx)
        strategy = 1.3 * market + pd.Series(rng.normal(0, 0.003, n), index=idx)

        result = attribute(strategy, pd.DataFrame({"MKT": market}))

        assert result.exposures[0].beta == pytest.approx(1.3, abs=0.02)
        assert not result.alpha_is_significant

    def test_multiple_factors(self):
        rng = np.random.default_rng(7)
        n = 1500
        idx = pd.bdate_range("2015-01-01", periods=n)
        factors = pd.DataFrame(
            {
                "MKT": rng.normal(0.0003, 0.010, n),
                "SMB": rng.normal(0.0001, 0.005, n),
                "HML": rng.normal(0.0001, 0.006, n),
            },
            index=idx,
        )
        strategy = (
            0.8 * factors["MKT"] - 0.3 * factors["SMB"] + 0.5 * factors["HML"]
        ) + pd.Series(rng.normal(0, 0.001, n), index=idx)

        result = attribute(strategy, factors)
        betas = {e.name: e.beta for e in result.exposures}

        assert betas["MKT"] == pytest.approx(0.8, abs=0.02)
        assert betas["SMB"] == pytest.approx(-0.3, abs=0.05)
        assert betas["HML"] == pytest.approx(0.5, abs=0.05)

    def test_newey_west_widens_errors_on_autocorrelated_residuals(self):
        """
        The reason HAC is the default. OLS understates standard errors when
        residuals persist, inflating t-stats on exactly the strategies that hold
        positions longest.
        """
        rng = np.random.default_rng(8)
        n = 1200
        idx = pd.bdate_range("2015-01-01", periods=n)
        market = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)

        noise = np.zeros(n)
        for t in range(1, n):
            noise[t] = 0.9 * noise[t - 1] + rng.normal(0, 0.002)
        strategy = 0.5 * market + pd.Series(noise, index=idx)

        no_lags = attribute(strategy, pd.DataFrame({"MKT": market}), lags=0)
        hac = attribute(strategy, pd.DataFrame({"MKT": market}))

        assert hac.newey_west_lags > 0
        assert hac.alpha_stderr > no_lags.alpha_stderr

    def test_lag_rule_of_thumb(self):
        # floor(4 * (1000/100)^(2/9)) = floor(4 * 10^0.2222) = floor(6.69) = 6
        assert newey_west_lags(1000) == 6

    def test_misaligned_calendars_shrink_the_sample_visibly(self):
        idx_a = pd.bdate_range("2020-01-01", periods=100)
        idx_b = pd.bdate_range("2020-03-01", periods=100)
        strategy = pd.Series(np.random.default_rng(9).normal(0, 0.01, 100), index=idx_a)
        factor = pd.DataFrame(
            {"MKT": np.random.default_rng(10).normal(0, 0.01, 100)}, index=idx_b
        )

        result = attribute(strategy, factor)
        assert result.n_observations < 100

    def test_too_few_observations_rejected(self):
        idx = pd.bdate_range("2020-01-01", periods=2)
        with pytest.raises(ValueError, match="at least 3"):
            attribute(
                pd.Series([0.01, 0.02], index=idx),
                pd.DataFrame({"MKT": [0.01, 0.02]}, index=idx),
            )


class TestCapacity:
    def test_capacity_scales_as_the_square_of_the_alpha_to_cost_ratio(self):
        # gross 4%, cost 1% at 100M -> (0.04/0.01)^2 = 16x -> 1.6B
        est = estimate_capacity(100e6, 0.04, 0.01)
        assert est.capacity_aum == pytest.approx(1.6e9)
        assert est.net_alpha == pytest.approx(0.03)

    def test_halving_the_cost_rate_quadruples_capacity(self):
        cheap = estimate_capacity(100e6, 0.04, 0.005)
        dear = estimate_capacity(100e6, 0.04, 0.01)
        assert cheap.capacity_aum == pytest.approx(4.0 * dear.capacity_aum)

    def test_linear_impact_is_far_more_pessimistic(self):
        root = estimate_capacity(100e6, 0.04, 0.01, scaling_exponent=0.5)
        linear = estimate_capacity(100e6, 0.04, 0.01, scaling_exponent=1.0)
        assert linear.capacity_aum < root.capacity_aum

    def test_costless_strategy_has_unbounded_capacity(self):
        """Which is exactly what a fixed-bps cost model implicitly asserts."""
        assert estimate_capacity(100e6, 0.04, 0.0).capacity_aum == float("inf")

    def test_curve_shows_alpha_eroding_with_size(self):
        curve = capacity_curve(100e6, 0.04, 0.01)
        assert curve["net_alpha"].is_monotonic_decreasing
        assert curve["alpha_retained"].iloc[0] > curve["alpha_retained"].iloc[-1]

    def test_realised_cost_rate_from_a_run(self):
        curve = pd.DataFrame({"nav": [1_000_000.0] * 252})
        fills = pd.DataFrame({"total_cost": [1000.0] * 10})
        assert realised_cost_rate(fills, curve) == pytest.approx(0.01)

    def test_non_positive_aum_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            estimate_capacity(0.0, 0.04, 0.01)
