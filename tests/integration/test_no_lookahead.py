"""
The engine's own acceptance test: no information from the future can reach a
decision, and no decision can transact at a price it caused.

This file is to this project what `test_no_lookahead.py` is to Project 01. Every
other number the engine produces is conditional on these passing. They are
written so that the *only* way to make them pass is for the barrier to actually
hold — each one has a specific, named way a broken engine would fail it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
import pytest

from evbt.core.events import MarketEvent, SignalEvent
from evbt.data.frame import DataFrameDataHandler
from evbt.engine import Backtest
from evbt.execution.broker import SimulatedBroker
from evbt.portfolio.construction import ExplicitWeightSizer, PortfolioConstructor
from evbt.portfolio.portfolio import Portfolio
from evbt.strategy.base import BuyAndHold, Strategy, StrategyContext
from tests.conftest import flat_price_bars, trending_bars


def run(bars, strategy, actions=None, cash=100_000.0, **kwargs) -> tuple:
    data = DataFrameDataHandler(bars, actions)
    portfolio = Portfolio(cash)
    broker = SimulatedBroker()
    constructor = PortfolioConstructor(sizer=ExplicitWeightSizer())
    engine = Backtest(data, strategy, portfolio, broker, constructor, **kwargs)
    return engine.run(), portfolio


class SpikeChaser(Strategy):
    """
    Goes fully long the bar *after* it observes a return above `threshold`.

    If the engine leaked, this strategy would buy at the pre-spike price and
    capture the jump. It cannot: it learns of the spike from that bar's close,
    and the earliest it can transact is the following open, by which time the
    price has already moved.
    """

    name = "spike_chaser"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.entered = False

    def on_bar(self, event: MarketEvent, ctx: StrategyContext) -> Iterable[SignalEvent]:
        if self.entered:
            return []
        returns = ctx.data.returns("A", 1)
        if returns.size and returns[-1] > self.threshold:
            self.entered = True
            return [ctx.long("A", strength=1.0)]
        return []


class GreedyPeeker(Strategy):
    """
    Actively tries to see the future through every accessor it is given.

    Records what the widest possible request returns. The assertion is that the
    handler hands back only what has already streamed — the interface offers no
    argument that could reach further.
    """

    name = "greedy_peeker"

    def __init__(self):
        self.observations: list[tuple[datetime, int, float]] = []

    def on_bar(self, event: MarketEvent, ctx: StrategyContext) -> Iterable[SignalEvent]:
        history = ctx.data.history("A", 10_000)
        self.observations.append(
            (event.timestamp, len(history), max(b.close for b in history))
        )
        return []


class TestFillTiming:
    def test_no_fill_can_precede_the_signal_that_caused_it(self):
        bars = trending_bars({"A": 100.0}, n_days=20, daily_return=0.01)
        result, _ = run(bars, BuyAndHold())

        signal_time = result.equity_curve.index[0]
        assert (result.fills["timestamp"] > signal_time).all()

    def test_decision_price_and_fill_price_differ_by_the_real_gap(self):
        """
        Sized off day 0's close of 100, filled at day 1's open of 110. The 10%
        gap is genuine execution risk; an engine that fills at the decision bar
        deletes it and books the difference as alpha.
        """
        bars = trending_bars({"A": 100.0}, n_days=5, daily_return=0.10)
        result, _ = run(bars, BuyAndHold())

        fill = result.fills.iloc[0]
        assert fill["fill_price"] == pytest.approx(110.0)
        assert fill["quantity"] == pytest.approx(1_000.0)  # 100,000 / 100, not / 110

    def test_a_spike_cannot_be_captured_by_the_bar_that_reveals_it(self):
        """
        Prices: 100 flat, then a jump to 200, then flat at 200. A strategy that
        reacts to the jump buys at 200 and earns nothing from it. A leaking
        engine would fill at 100 and double its money.
        """
        dates = pd.bdate_range("2024-01-01", periods=10)
        prices = [100.0] * 4 + [200.0] * 6
        bars = pd.DataFrame(
            [
                {
                    "timestamp": d,
                    "symbol": "A",
                    "open": p,
                    "high": p,
                    "low": p,
                    "close": p,
                    "volume": 1e7,
                }
                for d, p in zip(dates, prices)
            ]
        )

        result, _ = run(bars, SpikeChaser(threshold=0.5))

        assert len(result.fills) == 1
        assert result.fills.iloc[0]["fill_price"] == pytest.approx(200.0)
        assert result.total_return == pytest.approx(0.0, abs=1e-9)


class TestDataVisibility:
    def test_history_never_extends_past_the_current_bar(self):
        bars = trending_bars({"A": 100.0}, n_days=15, daily_return=0.05)
        peeker = GreedyPeeker()
        run(bars, peeker)

        for i, (_, length, highest) in enumerate(peeker.observations):
            expected_bars = i + 1
            expected_max = 100.0 * 1.05**i
            assert length == expected_bars
            assert highest == pytest.approx(expected_max)

    def test_the_last_observation_sees_the_whole_series_and_no_more(self):
        bars = trending_bars({"A": 100.0}, n_days=15, daily_return=0.05)
        peeker = GreedyPeeker()
        run(bars, peeker)

        assert peeker.observations[-1][1] == 15
        assert peeker.observations[-1][2] == pytest.approx(100.0 * 1.05**14)


class TestDeterminism:
    def test_identical_inputs_produce_identical_output(self):
        bars = trending_bars({"A": 100.0, "B": 50.0}, n_days=30, daily_return=0.003)

        curves = []
        for _ in range(3):
            result, _ = run(bars, BuyAndHold())
            curves.append(result.equity_curve["nav"].tolist())

        assert curves[0] == curves[1] == curves[2]

    def test_fill_sequence_is_stable(self):
        bars = trending_bars(
            {c: 100.0 for c in "ABCDEFGH"}, n_days=10, daily_return=0.01
        )

        def fill_keys():
            result, _ = run(bars, BuyAndHold())
            return list(zip(result.fills["symbol"], result.fills["quantity"]))

        assert fill_keys() == fill_keys()


class TestOrderLifecycle:
    def test_orders_never_fill_on_the_bar_they_were_created_on(self):
        """
        The engine-level version of the broker's `LookaheadError` guard. If the
        signal-to-order path ever stamps an order with the wrong time, the
        broker raises and this test fails loudly rather than silently profiting.
        """
        bars = trending_bars({"A": 100.0, "B": 50.0}, n_days=10, daily_return=0.01)
        result, _ = run(bars, BuyAndHold())

        # Reaching here without a LookaheadError is the assertion; the counts
        # confirm the run actually exercised the path.
        assert result.n_orders == 2
        assert result.n_fills == 2

    def test_a_flat_strategy_trades_nothing_and_holds_nav(self):
        bars = flat_price_bars({"A": 100.0}, n_days=10)

        class DoNothing(Strategy):
            name = "do_nothing"

            def on_bar(self, event, ctx):
                return []

        result, _ = run(bars, DoNothing())
        assert result.n_orders == 0
        assert result.final_nav == pytest.approx(100_000.0)
