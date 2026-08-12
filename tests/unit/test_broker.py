"""
Broker behaviour, and above all the no-lookahead barrier.

`TestNoLookaheadBarrier` is the most important class in this file. If it ever
goes green for the wrong reason, every number this engine produces is fiction.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from evbt.core.events import (
    EventType,
    MarketEvent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from evbt.execution.broker import LookaheadError, SimulatedBroker
from evbt.execution.costs import BpsCommission, FixedBpsSpread, AlmgrenChrissImpact
from evbt.execution.slippage import FixedBpsSlippage
from tests.conftest import make_bar, make_order

T0 = datetime(2024, 1, 2)
T1 = datetime(2024, 1, 3)
T2 = datetime(2024, 1, 4)


def slice_at(timestamp, **bar_kwargs) -> MarketEvent:
    bar = make_bar(symbol="X", timestamp=timestamp, **bar_kwargs)
    return MarketEvent(timestamp=timestamp, bars={"X": bar})


class TestNoLookaheadBarrier:
    def test_order_fills_on_the_following_bar(self):
        broker = SimulatedBroker()
        broker.submit(make_order(timestamp=T0, quantity=100))

        fills = broker.on_market(slice_at(T1, open_=105.0))

        assert len(fills) == 1
        assert fills[0].timestamp == T1
        assert fills[0].fill_price == pytest.approx(105.0)

    def test_order_stamped_at_the_bar_it_would_fill_on_raises(self):
        """
        There is no legitimate path that produces this. Returning False instead
        would hide the upstream bug while making the strategy look inactive.
        """
        broker = SimulatedBroker()
        broker.submit(make_order(timestamp=T1, quantity=100))

        with pytest.raises(LookaheadError, match="strictly after"):
            broker.on_market(slice_at(T1))

    def test_order_does_not_fill_on_an_earlier_bar(self):
        broker = SimulatedBroker()
        broker.submit(make_order(timestamp=T1, quantity=100, time_in_force=TimeInForce.GTC))

        assert broker.on_market(slice_at(T0)) == []
        assert len(broker.open_orders) == 1

    def test_fill_price_is_the_next_open_not_the_decision_close(self):
        """
        The gap between the two is real execution risk, and the engine leaves
        it in. A same-bar-close fill would erase it.
        """
        broker = SimulatedBroker()
        broker.submit(make_order(timestamp=T0, quantity=100))

        fills = broker.on_market(slice_at(T1, open_=110.0, close=100.0))
        assert fills[0].fill_price == pytest.approx(110.0)


class TestOrderTypes:
    def test_market_and_moo_reference_the_open(self):
        for order_type in (OrderType.MARKET, OrderType.MARKET_ON_OPEN):
            broker = SimulatedBroker()
            broker.submit(make_order(timestamp=T0, order_type=order_type))
            fills = broker.on_market(slice_at(T1, open_=103.0, close=107.0))
            assert fills[0].fill_price == pytest.approx(103.0)

    def test_moc_references_the_close(self):
        broker = SimulatedBroker()
        broker.submit(make_order(timestamp=T0, order_type=OrderType.MARKET_ON_CLOSE))
        fills = broker.on_market(slice_at(T1, open_=103.0, close=107.0))
        assert fills[0].fill_price == pytest.approx(107.0)


class TestLimitOrders:
    def _buy_limit(self, broker, limit=99.0):
        broker.submit(
            make_order(
                timestamp=T0,
                order_type=OrderType.LIMIT,
                limit_price=limit,
                time_in_force=TimeInForce.GTC,
            )
        )

    def test_no_fill_when_the_price_never_reaches_the_limit(self):
        broker = SimulatedBroker()
        self._buy_limit(broker, limit=95.0)
        assert broker.on_market(slice_at(T1, open_=100.0, low=99.0, high=101.0)) == []

    def test_merely_touching_the_limit_does_not_fill_by_default(self):
        """
        Assuming a fill when the low exactly equals your limit assumes you were
        at the front of a queue you were somewhere in the middle of. This is the
        most common way mean-reversion backtests fabricate returns.
        """
        broker = SimulatedBroker(require_trade_through=True)
        self._buy_limit(broker, limit=99.0)
        assert broker.on_market(slice_at(T1, open_=100.0, low=99.0, high=101.0)) == []

    def test_touching_fills_when_trade_through_is_not_required(self):
        broker = SimulatedBroker(require_trade_through=False)
        self._buy_limit(broker, limit=99.0)
        fills = broker.on_market(slice_at(T1, open_=100.0, low=99.0, high=101.0))
        assert len(fills) == 1 and fills[0].fill_price == pytest.approx(99.0)

    def test_trading_through_fills_at_the_limit(self):
        broker = SimulatedBroker()
        self._buy_limit(broker, limit=99.0)
        fills = broker.on_market(slice_at(T1, open_=100.0, low=98.0, high=101.0))
        assert fills[0].fill_price == pytest.approx(99.0)

    def test_gapping_through_gives_price_improvement(self):
        """Filled at the open, which is better than the limit. This is real."""
        broker = SimulatedBroker()
        self._buy_limit(broker, limit=99.0)
        fills = broker.on_market(slice_at(T1, open_=95.0, low=94.0, high=96.0))
        assert fills[0].fill_price == pytest.approx(95.0)

    def test_sell_limit_needs_the_high_to_trade_through(self):
        broker = SimulatedBroker()
        broker.submit(
            make_order(
                timestamp=T0,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                limit_price=101.0,
                time_in_force=TimeInForce.GTC,
            )
        )
        assert broker.on_market(slice_at(T1, open_=100.0, high=101.0, low=99.0)) == []
        fills = broker.on_market(slice_at(T2, open_=100.0, high=102.0, low=99.0))
        assert fills[0].fill_price == pytest.approx(101.0)

    def test_resting_limit_pays_no_spread_or_slippage(self):
        """It supplies liquidity rather than demanding it; impact still applies."""
        broker = SimulatedBroker(
            spread=FixedBpsSpread(bps=50.0), slippage=FixedBpsSlippage(bps=50.0)
        )
        self._buy_limit(broker, limit=99.0)
        fills = broker.on_market(slice_at(T1, open_=100.0, low=98.0, high=101.0))

        assert fills[0].spread_cost == 0.0
        assert fills[0].slippage_cost == 0.0
        assert fills[0].fill_price == pytest.approx(99.0)


class TestPartialFills:
    def test_fill_is_capped_at_max_participation(self):
        # 10% of 1,000,000 shares = 100,000
        broker = SimulatedBroker(max_participation=0.10)
        broker.submit(
            make_order(timestamp=T0, quantity=500_000, time_in_force=TimeInForce.GTC)
        )

        fills = broker.on_market(slice_at(T1, volume=1_000_000.0))
        assert fills[0].quantity == pytest.approx(100_000.0)
        assert fills[0].is_partial

    def test_large_order_is_worked_across_bars(self):
        broker = SimulatedBroker(max_participation=0.10)
        broker.submit(
            make_order(timestamp=T0, quantity=250_000, time_in_force=TimeInForce.GTC)
        )

        filled = []
        for day in (T1, T2, datetime(2024, 1, 5)):
            filled += [f.quantity for f in broker.on_market(slice_at(day, volume=1_000_000.0))]

        assert filled == pytest.approx([100_000.0, 100_000.0, 50_000.0])
        assert broker.open_orders == []

    def test_final_slice_is_not_flagged_partial(self):
        broker = SimulatedBroker(max_participation=0.10)
        broker.submit(
            make_order(timestamp=T0, quantity=150_000, time_in_force=TimeInForce.GTC)
        )
        broker.on_market(slice_at(T1, volume=1_000_000.0))
        last = broker.on_market(slice_at(T2, volume=1_000_000.0))[0]
        assert not last.is_partial

    def test_disabling_partial_fills_ignores_the_cap(self):
        broker = SimulatedBroker(max_participation=0.10, allow_partial_fills=False)
        broker.submit(make_order(timestamp=T0, quantity=500_000))
        fills = broker.on_market(slice_at(T1, volume=1_000_000.0))
        assert fills[0].quantity == pytest.approx(500_000.0)

    def test_zero_volume_bar_produces_no_fill(self):
        """Filling there invents a counterparty that was not present."""
        broker = SimulatedBroker()
        broker.submit(make_order(timestamp=T0, time_in_force=TimeInForce.GTC))
        assert broker.on_market(slice_at(T1, volume=0.0)) == []


class TestTimeInForce:
    def test_day_order_expires_after_one_session(self):
        broker = SimulatedBroker(max_participation=0.10)
        broker.submit(
            make_order(timestamp=T0, quantity=500_000, time_in_force=TimeInForce.DAY)
        )

        produced = broker.on_market(slice_at(T1, volume=1_000_000.0))
        kinds = [e.type for e in produced]

        assert EventType.FILL in kinds and EventType.ORDER_EXPIRED in kinds
        assert broker.open_orders == []

    def test_gtc_order_survives(self):
        broker = SimulatedBroker(max_participation=0.10)
        broker.submit(
            make_order(timestamp=T0, quantity=500_000, time_in_force=TimeInForce.GTC)
        )
        broker.on_market(slice_at(T1, volume=1_000_000.0))
        assert len(broker.open_orders) == 1

    def test_order_without_a_bar_does_not_age(self):
        """A halted symbol never exposed the order to a session."""
        broker = SimulatedBroker()
        broker.submit(
            make_order(symbol="HALTED", timestamp=T0, time_in_force=TimeInForce.DAY)
        )
        assert broker.on_market(slice_at(T1)) == []
        assert len(broker.open_orders) == 1


class TestCostApplication:
    def test_buys_fill_above_and_sells_below_the_reference(self):
        broker = SimulatedBroker(spread=FixedBpsSpread(bps=10.0))

        broker.submit(make_order(timestamp=T0, side=OrderSide.BUY))
        buy = broker.on_market(slice_at(T1, open_=100.0, close=100.0))[0]

        broker2 = SimulatedBroker(spread=FixedBpsSpread(bps=10.0))
        broker2.submit(make_order(timestamp=T0, side=OrderSide.SELL))
        sell = broker2.on_market(slice_at(T1, open_=100.0, close=100.0))[0]

        assert buy.fill_price > 100.0 > sell.fill_price

    def test_cost_components_are_reported_separately(self):
        broker = SimulatedBroker(
            commission=BpsCommission(bps=1.0),
            spread=FixedBpsSpread(bps=10.0),
            impact=AlmgrenChrissImpact(eta=0.01, gamma=0.0),
            slippage=FixedBpsSlippage(bps=2.0),
        )
        broker.submit(make_order(timestamp=T0, quantity=100_000))
        fill = broker.on_market(slice_at(T1, open_=100.0, volume=1_000_000.0))[0]

        # half-spread 5 bps, slippage 2 bps, impact 0.01 * 0.1 = 10 bps
        assert fill.spread_cost == pytest.approx(100_000 * 100.0 * 5e-4)
        assert fill.slippage_cost == pytest.approx(100_000 * 100.0 * 2e-4)
        assert fill.impact_cost == pytest.approx(100_000 * 0.01 * 0.10 * 100.0)
        assert fill.commission > 0
        assert fill.reference_price == pytest.approx(100.0)

    def test_reference_price_is_preserved_for_shortfall_analysis(self):
        broker = SimulatedBroker(spread=FixedBpsSpread(bps=100.0))
        broker.submit(make_order(timestamp=T0))
        fill = broker.on_market(slice_at(T1, open_=100.0))[0]

        assert fill.reference_price == pytest.approx(100.0)
        assert fill.fill_price > fill.reference_price


class TestOrderManagement:
    def test_duplicate_order_id_rejected(self):
        broker = SimulatedBroker()
        broker.submit(make_order(order_id="DUP", timestamp=T0))
        with pytest.raises(ValueError, match="duplicate"):
            broker.submit(make_order(order_id="DUP", timestamp=T0))

    def test_cancel_removes_a_resting_order(self):
        broker = SimulatedBroker()
        broker.submit(make_order(order_id="O1", timestamp=T0))
        assert broker.cancel("O1")
        assert broker.open_orders == []
        assert not broker.cancel("O1")

    def test_cancel_all_can_be_scoped_to_a_symbol(self):
        broker = SimulatedBroker()
        broker.submit(make_order(symbol="A", order_id="O1", timestamp=T0))
        broker.submit(make_order(symbol="B", order_id="O2", timestamp=T0))

        assert broker.cancel_all(symbol="A") == 1
        assert [o.symbol for o in broker.open_orders] == ["B"]

    def test_order_ids_are_unique_and_ordered(self):
        broker = SimulatedBroker()
        ids = [broker.next_order_id() for _ in range(3)]
        assert ids == sorted(ids) and len(set(ids)) == 3

    def test_rejection_is_recorded(self):
        broker = SimulatedBroker()
        event = broker.reject(make_order(timestamp=T0), "no cash")
        assert event.reason == "no cash"
        assert broker.stats["rejected"] == 1

    def test_matching_is_deterministic_across_runs(self):
        def run():
            broker = SimulatedBroker(max_participation=1.0)
            for i in range(20):
                broker.submit(make_order(order_id=f"O{i:03d}", timestamp=T0))
            return [f.order_id for f in broker.on_market(slice_at(T1))]

        assert run() == run() == run()

    def test_invalid_participation_rejected(self):
        with pytest.raises(ValueError, match="max_participation"):
            SimulatedBroker(max_participation=0.0)
        with pytest.raises(ValueError, match="max_participation"):
            SimulatedBroker(max_participation=1.5)
