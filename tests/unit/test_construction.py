"""Sizers, and the translation from target weights to orders."""

from __future__ import annotations

from datetime import datetime
from itertools import count

import pytest

from evbt.core.events import (
    MarkToMarketEvent,
    OrderSide,
    SignalDirection,
    SignalEvent,
)
from evbt.portfolio.construction import (
    EqualWeightSizer,
    ExplicitWeightSizer,
    PortfolioConstructor,
    ProportionalSizer,
)
from evbt.portfolio.portfolio import Portfolio
from evbt.portfolio.risk import RiskLimits, RiskManager
from tests.conftest import make_bar, make_fill

T0 = datetime(2024, 1, 2)


def signal(symbol, direction=SignalDirection.LONG, strength=1.0):
    return SignalEvent(timestamp=T0, symbol=symbol, direction=direction, strength=strength)


def marked_portfolio(prices: dict[str, float], cash: float = 100_000.0) -> Portfolio:
    pf = Portfolio(cash)
    bars = {s: make_bar(symbol=s, timestamp=T0, close=p) for s, p in prices.items()}
    pf.on_mark_to_market(MarkToMarketEvent(timestamp=T0, bars=bars))
    return pf


def ids():
    counter = count(1)
    return lambda: f"O{next(counter)}"


class TestExplicitWeightSizer:
    def test_passes_strength_through_untouched(self):
        """Required for the done criterion: published weights must not be reshaped."""
        out = ExplicitWeightSizer().target_weights(
            [signal("A", strength=0.5), signal("B", SignalDirection.SHORT, strength=-0.5)]
        )
        assert out == pytest.approx({"A": 0.5, "B": -0.5})

    def test_exit_targets_zero(self):
        out = ExplicitWeightSizer().target_weights([signal("A", SignalDirection.EXIT, 9.0)])
        assert out == {"A": 0.0}


class TestEqualWeightSizer:
    def test_splits_gross_across_both_sides(self):
        out = EqualWeightSizer(gross_leverage=1.0).target_weights(
            [signal("A"), signal("B"), signal("C", SignalDirection.SHORT)]
        )
        assert out["A"] == pytest.approx(0.25)
        assert out["B"] == pytest.approx(0.25)
        assert out["C"] == pytest.approx(-0.5)

    def test_book_is_dollar_neutral_when_both_sides_are_present(self):
        out = EqualWeightSizer().target_weights(
            [signal("A"), signal("B", SignalDirection.SHORT)]
        )
        assert sum(out.values()) == pytest.approx(0.0)

    def test_long_only_gets_the_full_gross(self):
        out = EqualWeightSizer(gross_leverage=1.0).target_weights(
            [signal("A"), signal("B")]
        )
        assert sum(out.values()) == pytest.approx(1.0)

    def test_no_signals_is_an_empty_book(self):
        assert EqualWeightSizer().target_weights([]) == {}


class TestProportionalSizer:
    def test_weights_track_strength_and_normalise_to_gross(self):
        out = ProportionalSizer(gross_leverage=1.0).target_weights(
            [signal("A", strength=3.0), signal("B", SignalDirection.SHORT, strength=1.0)]
        )
        assert out["A"] == pytest.approx(0.75)
        assert out["B"] == pytest.approx(-0.25)
        assert sum(abs(w) for w in out.values()) == pytest.approx(1.0)

    def test_all_zero_strength_gives_a_flat_book(self):
        out = ProportionalSizer().target_weights([signal("A", strength=0.0)])
        assert out == {"A": 0.0}


class TestOrderGeneration:
    def test_sizes_from_nav_and_the_last_known_mark(self):
        pf = marked_portfolio({"A": 50.0})
        ctor = PortfolioConstructor(sizer=ExplicitWeightSizer())

        orders = ctor.build_orders([signal("A", strength=0.5)], pf, T0, ids())

        assert len(orders) == 1
        # 0.5 * 100,000 / 50 = 1,000 shares
        assert orders[0].quantity == pytest.approx(1_000.0)
        assert orders[0].side is OrderSide.BUY

    def test_orders_the_difference_from_the_current_position(self):
        pf = marked_portfolio({"A": 50.0})
        pf.on_fill(make_fill(symbol="A", side=OrderSide.BUY, quantity=600, price=50.0))
        ctor = PortfolioConstructor(sizer=ExplicitWeightSizer())

        orders = ctor.build_orders([signal("A", strength=0.5)], pf, T0, ids())
        # Target 1,000 shares against 600 held -> buy 400.
        assert orders[0].quantity == pytest.approx(400.0)

    def test_negative_weight_produces_a_sell(self):
        pf = marked_portfolio({"A": 50.0})
        ctor = PortfolioConstructor(sizer=ExplicitWeightSizer())
        orders = ctor.build_orders(
            [signal("A", SignalDirection.SHORT, strength=-0.5)], pf, T0, ids()
        )
        assert orders[0].side is OrderSide.SELL

    def test_symbol_without_a_mark_is_skipped(self):
        """Guessing a price would be inventing data."""
        pf = marked_portfolio({"A": 50.0})
        ctor = PortfolioConstructor(sizer=ExplicitWeightSizer())
        assert ctor.build_orders([signal("NEVERTRADED", strength=0.5)], pf, T0, ids()) == []

    def test_min_trade_notional_suppresses_dust(self):
        pf = marked_portfolio({"A": 50.0})
        pf.on_fill(make_fill(symbol="A", side=OrderSide.BUY, quantity=999, price=50.0))
        ctor = PortfolioConstructor(
            sizer=ExplicitWeightSizer(), min_trade_notional=1_000.0
        )
        # The residual trade is ~1 share = $50, well under the floor.
        assert ctor.build_orders([signal("A", strength=0.5)], pf, T0, ids()) == []

    def test_round_lots_truncates_toward_zero(self):
        pf = marked_portfolio({"A": 30.0})
        ctor = PortfolioConstructor(
            sizer=ExplicitWeightSizer(), round_lots=True, lot_size=100
        )
        orders = ctor.build_orders([signal("A", strength=0.5)], pf, T0, ids())
        # 0.5 * 100,000 / 30 = 1,666.67 -> 1,600
        assert orders[0].quantity == pytest.approx(1_600.0)

    def test_liquidate_unsignalled_closes_dropped_names(self):
        pf = marked_portfolio({"A": 50.0, "B": 50.0})
        pf.on_fill(make_fill(symbol="B", side=OrderSide.BUY, quantity=100, price=50.0))
        ctor = PortfolioConstructor(sizer=ExplicitWeightSizer())

        orders = ctor.build_orders(
            [signal("A", strength=0.5)], pf, T0, ids(), liquidate_unsignalled=True
        )
        sells = [o for o in orders if o.symbol == "B"]
        assert sells and sells[0].side is OrderSide.SELL
        assert sells[0].quantity == pytest.approx(100.0)

    def test_unsignalled_positions_are_held_by_default(self):
        pf = marked_portfolio({"A": 50.0, "B": 50.0})
        pf.on_fill(make_fill(symbol="B", side=OrderSide.BUY, quantity=100, price=50.0))
        ctor = PortfolioConstructor(sizer=ExplicitWeightSizer())

        orders = ctor.build_orders([signal("A", strength=0.5)], pf, T0, ids())
        assert [o.symbol for o in orders] == ["A"]

    def test_risk_limits_are_applied_before_orders_exist(self):
        pf = marked_portfolio({"A": 50.0})
        risk = RiskManager(RiskLimits(max_position_weight=0.10))
        ctor = PortfolioConstructor(sizer=ExplicitWeightSizer(), risk=risk)

        orders = ctor.build_orders([signal("A", strength=0.5)], pf, T0, ids())
        # Clipped from 50% to 10% of NAV: 10,000 / 50 = 200 shares.
        assert orders[0].quantity == pytest.approx(200.0)
        assert risk.summary()["max_position_weight"] == 1

    def test_target_weight_is_recorded_on_the_order(self):
        pf = marked_portfolio({"A": 50.0})
        ctor = PortfolioConstructor(sizer=ExplicitWeightSizer())
        orders = ctor.build_orders([signal("A", strength=0.5)], pf, T0, ids())
        assert orders[0].metadata["target_weight"] == pytest.approx(0.5)

    def test_wiped_out_portfolio_emits_nothing(self):
        pf = marked_portfolio({"A": 50.0}, cash=100_000.0)
        pf.cash = -50_000.0
        ctor = PortfolioConstructor(sizer=ExplicitWeightSizer())
        assert ctor.build_orders([signal("A", strength=0.5)], pf, T0, ids()) == []

    def test_orders_are_emitted_in_deterministic_symbol_order(self):
        pf = marked_portfolio({"C": 10.0, "A": 10.0, "B": 10.0})
        ctor = PortfolioConstructor(sizer=ExplicitWeightSizer())
        orders = ctor.build_orders(
            [signal("C", strength=0.1), signal("A", strength=0.1), signal("B", strength=0.1)],
            pf,
            T0,
            ids(),
        )
        assert [o.symbol for o in orders] == ["A", "B", "C"]
