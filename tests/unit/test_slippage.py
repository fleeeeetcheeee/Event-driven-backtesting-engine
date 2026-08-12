"""Slippage models. The invariant under test everywhere: slippage is adverse."""

from __future__ import annotations

import math

import pytest

from evbt.execution.slippage import (
    FixedBpsSlippage,
    ParticipationSlippage,
    SpreadCrossingSlippage,
    ZeroSlippage,
)
from tests.conftest import make_bar


def test_zero_slippage():
    assert ZeroSlippage().cost_per_share(100, make_bar(), 100.0) == 0.0


def test_fixed_bps_is_independent_of_size():
    model = FixedBpsSlippage(bps=5.0)
    bar = make_bar(volume=1_000_000.0)

    assert model.cost_per_share(100, bar, 100.0) == pytest.approx(0.05)
    assert model.cost_per_share(500_000, bar, 100.0) == pytest.approx(0.05)


class TestParticipationSlippage:
    def test_matches_its_formula(self):
        # k * sigma * sqrt(Q/V) * price = 0.1 * 0.02 * sqrt(0.25) * 100
        bar = make_bar(volume=1_000_000.0, volatility=0.02)
        model = ParticipationSlippage(k=0.1)

        assert model.cost_per_share(250_000, bar, 100.0) == pytest.approx(
            0.1 * 0.02 * math.sqrt(0.25) * 100.0
        )

    def test_grows_with_participation(self):
        bar = make_bar(volume=1_000_000.0, volatility=0.02)
        model = ParticipationSlippage()

        small = model.cost_per_share(10_000, bar, 100.0)
        large = model.cost_per_share(500_000, bar, 100.0)
        assert large > small

    def test_grows_with_volatility(self):
        model = ParticipationSlippage()
        calm = make_bar(volume=1_000_000.0, volatility=0.01)
        wild = make_bar(volume=1_000_000.0, volatility=0.05)

        assert model.cost_per_share(100_000, wild, 100.0) > model.cost_per_share(
            100_000, calm, 100.0
        )

    def test_concave_in_size(self):
        """Four times the size costs twice as much per share, not four times."""
        bar = make_bar(volume=1_000_000.0, volatility=0.02)
        model = ParticipationSlippage()

        one = model.cost_per_share(50_000, bar, 100.0)
        four = model.cost_per_share(200_000, bar, 100.0)
        assert four / one == pytest.approx(2.0)

    def test_falls_back_to_a_default_vol(self):
        bar = make_bar(volume=1_000_000.0, volatility=None)
        model = ParticipationSlippage(k=0.1, default_volatility=0.03)
        assert model.cost_per_share(1_000_000, bar, 100.0) == pytest.approx(
            0.1 * 0.03 * 100.0
        )

    def test_zero_volume_is_full_participation(self):
        bar = make_bar(volume=0.0, volatility=0.02)
        model = ParticipationSlippage(k=0.1)
        assert model.cost_per_share(1, bar, 100.0) == pytest.approx(0.1 * 0.02 * 100.0)

    def test_noise_is_reproducible_from_the_seed(self):
        bar = make_bar(volume=1_000_000.0, volatility=0.02)

        def draws():
            model = ParticipationSlippage(noise_std=0.5, seed=42)
            return [model.cost_per_share(100_000, bar, 100.0) for _ in range(10)]

        assert draws() == draws()

    def test_noise_never_makes_slippage_favourable(self):
        bar = make_bar(volume=1_000_000.0, volatility=0.02)
        model = ParticipationSlippage(noise_std=5.0, seed=7)
        assert all(
            model.cost_per_share(100_000, bar, 100.0) >= 0.0 for _ in range(500)
        )

    def test_reset_replays_the_same_sequence(self):
        bar = make_bar(volume=1_000_000.0, volatility=0.02)
        model = ParticipationSlippage(noise_std=0.5, seed=1)

        first = [model.cost_per_share(100_000, bar, 100.0) for _ in range(5)]
        model.reset()
        assert [model.cost_per_share(100_000, bar, 100.0) for _ in range(5)] == first


def test_spread_crossing_uses_the_bar_range():
    bar = make_bar(high=102.0, low=98.0)
    assert SpreadCrossingSlippage(fraction=0.1).cost_per_share(
        100, bar, 100.0
    ) == pytest.approx(0.4)
