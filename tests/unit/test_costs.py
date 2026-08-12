"""
Cost models, and the Almgren-Chriss optimal schedule.

The schedule tests check the two analytic limits rather than a table of
numbers, because the limits are what the formula *means*: risk-neutral is
TWAP, risk-averse is front-loaded. A test that reproduces sinh() from sinh()
proves nothing.
"""

from __future__ import annotations

import math

import pytest

from evbt.execution.costs import (
    AlmgrenChrissImpact,
    BpsCommission,
    CompositeImpact,
    FixedBpsSpread,
    PerShareCommission,
    SquareRootImpact,
    VolatilityScaledSpread,
    ZeroCommission,
    ZeroImpact,
    almgren_chriss_schedule,
)
from tests.conftest import make_bar


class TestCommission:
    def test_zero(self):
        assert ZeroCommission().calculate(100, 50.0) == 0.0

    def test_per_share_rate(self):
        # 1,000 shares at 0.005 = 5.00, above the 1.00 minimum
        assert PerShareCommission().calculate(1_000, 50.0) == pytest.approx(5.0)

    def test_per_share_minimum_binds_on_small_orders(self):
        # 10 shares at 0.005 = 0.05, floored at 1.00
        assert PerShareCommission().calculate(10, 50.0) == pytest.approx(1.0)

    def test_per_share_cap_binds_on_penny_stocks(self):
        # 1,000 shares at $0.10 = $100 notional; 1% cap = $1.00, below the
        # $5.00 per-share charge that would otherwise apply.
        model = PerShareCommission(rate_per_share=0.005, minimum=1.0, max_pct_of_notional=0.01)
        assert model.calculate(1_000, 0.10) == pytest.approx(1.0)

    def test_bps_scales_with_notional(self):
        # 1 bp of 100 * 50 = 5,000 -> 0.50
        assert BpsCommission(bps=1.0).calculate(100, 50.0) == pytest.approx(0.5)

    def test_zero_quantity_costs_nothing(self):
        assert BpsCommission(bps=10.0, minimum=5.0).calculate(0, 50.0) == 0.0


class TestSpread:
    def test_fixed_bps_charges_half_the_quoted_spread(self):
        # 10 bps quoted on a 100 price -> 5 bps half-spread -> 0.05
        bar = make_bar(close=100.0)
        assert FixedBpsSpread(bps=10.0).half_spread(bar, "X") == pytest.approx(0.05)

    def test_per_symbol_override(self):
        bar = make_bar(symbol="ILLIQ", close=100.0)
        model = FixedBpsSpread(bps=5.0, overrides={"ILLIQ": 50.0})
        assert model.half_spread(bar, "ILLIQ") == pytest.approx(0.25)

    def test_volatility_scaled_widens_with_vol(self):
        calm = make_bar(close=100.0, volatility=0.01)
        wild = make_bar(close=100.0, volatility=0.08)
        model = VolatilityScaledSpread(k=0.05, min_bps=1.0)

        assert model.half_spread(wild) > model.half_spread(calm)
        assert model.half_spread(wild) == pytest.approx(100.0 * 0.05 * 0.08)

    def test_volatility_scaled_falls_back_to_the_floor_without_vol(self):
        bar = make_bar(close=100.0, volatility=None)
        model = VolatilityScaledSpread(k=0.05, min_bps=2.0)
        assert model.half_spread(bar) == pytest.approx(100.0 * 2.0 * 1e-4 / 2.0)


class TestImpact:
    def test_participation_is_capped_at_one(self):
        bar = make_bar(volume=1_000.0)
        assert ZeroImpact.participation(5_000, bar) == 1.0

    def test_zero_volume_bar_is_treated_as_full_participation(self):
        """Rather than dividing by zero and poisoning the curve with NaN."""
        bar = make_bar(volume=0.0)
        assert ZeroImpact.participation(1.0, bar) == 1.0

    def test_almgren_chriss_is_linear_in_participation(self):
        bar = make_bar(close=100.0, volume=1_000_000.0)
        model = AlmgrenChrissImpact(eta=0.01, gamma=0.0)

        at_10pct = model.temporary(100_000, bar)
        at_20pct = model.temporary(200_000, bar)

        assert at_10pct == pytest.approx(0.01 * 0.10 * 100.0)  # 10 bps
        assert at_20pct == pytest.approx(2 * at_10pct)

    def test_permanent_impact_is_charged_at_half(self):
        """The `gamma * X^2 / 2` term: the price walks away as you trade."""
        bar = make_bar(close=100.0, volume=1_000_000.0)
        model = AlmgrenChrissImpact(eta=0.0, gamma=0.01)

        assert model.permanent(100_000, bar) == pytest.approx(0.01 * 0.10 * 100.0)
        assert model.cost_per_share(100_000, bar) == pytest.approx(
            0.5 * 0.01 * 0.10 * 100.0
        )

    def test_square_root_law_matches_its_formula(self):
        # Y * sigma * sqrt(Q/V) * price = 0.5 * 0.02 * sqrt(0.25) * 100
        bar = make_bar(close=100.0, volume=1_000_000.0, volatility=0.02)
        model = SquareRootImpact(Y=0.5, permanent_fraction=0.0)

        expected = 0.5 * 0.02 * math.sqrt(0.25) * 100.0
        assert model.temporary(250_000, bar) == pytest.approx(expected)

    def test_square_root_is_concave_in_size(self):
        """
        Doubling size raises cost per share by sqrt(2), not 2. This concavity
        is why capacity scales as the square of an acceptable cost.
        """
        bar = make_bar(close=100.0, volume=1_000_000.0, volatility=0.02)
        model = SquareRootImpact(Y=0.5, permanent_fraction=0.0)

        small = model.temporary(100_000, bar)
        big = model.temporary(200_000, bar)
        assert big / small == pytest.approx(math.sqrt(2.0))

    def test_square_root_falls_back_to_a_default_vol(self):
        bar = make_bar(close=100.0, volume=1_000_000.0, volatility=None)
        model = SquareRootImpact(Y=0.5, default_volatility=0.03, permanent_fraction=0.0)
        assert model.temporary(1_000_000, bar) == pytest.approx(0.5 * 0.03 * 100.0)

    def test_composite_sums_its_parts(self):
        bar = make_bar(close=100.0, volume=1_000_000.0, volatility=0.02)
        linear = AlmgrenChrissImpact(eta=0.01, gamma=0.0)
        root = SquareRootImpact(Y=0.5, permanent_fraction=0.0)
        both = CompositeImpact([linear, root])

        assert both.temporary(100_000, bar) == pytest.approx(
            linear.temporary(100_000, bar) + root.temporary(100_000, bar)
        )


class TestAlmgrenChrissSchedule:
    def test_risk_neutral_limit_is_twap(self):
        """
        lambda -> 0 means kappa -> 0 and the trajectory becomes linear in time.
        A trader indifferent to risk trades evenly, because impact is convex in
        rate. This is the whole argument for TWAP.
        """
        schedule = almgren_chriss_schedule(
            1_000, 10, volatility=0.02, eta=0.01, gamma=0.001, risk_aversion=0.0
        )
        expected = [1_000 * (1 - j / 10) for j in range(11)]
        assert schedule == pytest.approx(expected)

    def test_small_risk_aversion_approaches_twap(self):
        schedule = almgren_chriss_schedule(
            1_000, 10, volatility=0.02, eta=0.01, gamma=0.0, risk_aversion=1e-9
        )
        twap = [1_000 * (1 - j / 10) for j in range(11)]
        assert schedule == pytest.approx(twap, rel=1e-3)

    def test_risk_aversion_front_loads_the_trade(self):
        """Urgency buys certainty at the price of impact."""
        patient = almgren_chriss_schedule(
            1_000, 10, volatility=0.02, eta=0.01, gamma=0.0, risk_aversion=1e-4
        )
        urgent = almgren_chriss_schedule(
            1_000, 10, volatility=0.02, eta=0.01, gamma=0.0, risk_aversion=1e-1
        )
        # Less inventory remaining at the halfway point means faster trading.
        assert urgent[5] < patient[5]

    def test_boundary_conditions_hold(self):
        schedule = almgren_chriss_schedule(
            5_000, 20, volatility=0.03, eta=0.02, gamma=0.001, risk_aversion=1e-3
        )
        assert schedule[0] == pytest.approx(5_000.0)
        assert schedule[-1] == pytest.approx(0.0, abs=1e-9)
        assert len(schedule) == 21

    def test_trajectory_is_monotonically_decreasing(self):
        schedule = almgren_chriss_schedule(
            5_000, 20, volatility=0.03, eta=0.02, gamma=0.001, risk_aversion=1e-2
        )
        assert all(a >= b for a, b in zip(schedule, schedule[1:]))

    def test_dominant_permanent_impact_is_rejected(self):
        """
        eta_tilde <= 0 makes the problem ill-posed — trading faster would appear
        free — and is a calibration error worth stopping on, not clamping.
        """
        with pytest.raises(ValueError, match="eta_tilde"):
            almgren_chriss_schedule(
                1_000, 10, volatility=0.02, eta=0.001, gamma=0.01, risk_aversion=1e-3
            )

    def test_zero_shares_is_a_flat_schedule(self):
        assert almgren_chriss_schedule(
            0, 5, volatility=0.02, eta=0.01, gamma=0.0, risk_aversion=1e-3
        ) == [0.0] * 6

    def test_at_least_one_interval_required(self):
        with pytest.raises(ValueError, match="n_intervals"):
            almgren_chriss_schedule(
                100, 0, volatility=0.02, eta=0.01, gamma=0.0, risk_aversion=1e-3
            )
