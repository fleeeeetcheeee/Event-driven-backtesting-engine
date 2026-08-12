"""
End-to-end accounting against answers computed by hand in the docstrings.

These are the tests that would catch a plausible-looking but wrong equity
curve. Each scenario is small enough to work out on paper, which is the point:
an assertion whose expected value came out of the code proves only that the
code is self-consistent.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import pytest

from evbt.core.events import MarketEvent, SignalEvent
from evbt.data.frame import DataFrameDataHandler
from evbt.engine import Backtest
from evbt.execution.broker import SimulatedBroker
from evbt.execution.costs import BpsCommission, FixedBpsSpread
from evbt.portfolio.construction import ExplicitWeightSizer, PortfolioConstructor
from evbt.portfolio.financing import FinancingModel
from evbt.portfolio.portfolio import Portfolio
from evbt.strategy.base import BuyAndHold, Strategy, StrategyContext
from tests.conftest import flat_price_bars


class EnterOnce(Strategy):
    """Signals a fixed book on the first bar, then never again."""

    name = "enter_once"

    def __init__(self, weights: dict[str, float]):
        self.weights = weights
        self.done = False

    def on_bar(self, event: MarketEvent, ctx: StrategyContext) -> Iterable[SignalEvent]:
        if self.done:
            return []
        self.done = True
        return [
            ctx.long(s, strength=w) if w > 0 else ctx.short(s, strength=w)
            for s, w in self.weights.items()
        ]


def build(bars, strategy, actions=None, cash=100_000.0, broker=None, financing=None):
    data = DataFrameDataHandler(bars, actions)
    portfolio = Portfolio(cash, financing=financing)
    return Backtest(
        data,
        strategy,
        portfolio,
        broker or SimulatedBroker(),
        PortfolioConstructor(sizer=ExplicitWeightSizer()),
    )


class TestFrictionlessBaseline:
    def test_flat_prices_and_no_costs_preserve_nav_exactly(self):
        """The null result. Any P&L here is a bug, not a return."""
        result = build(flat_price_bars({"A": 100.0}, n_days=10), BuyAndHold()).run()
        assert result.final_nav == pytest.approx(100_000.0)
        assert result.total_return == pytest.approx(0.0, abs=1e-12)

    def test_equity_curve_has_one_row_per_bar(self):
        result = build(flat_price_bars({"A": 100.0}, n_days=10), BuyAndHold()).run()
        assert len(result.equity_curve) == 10


class TestDividendsAndSplits:
    """
    One symbol, 10 business days from 2024-01-01, $100,000 of capital, no costs.

        prices    100 on days 0-5, 50 on days 6-9
        dividend  $2.00 per share, ex-date day 3 (2024-01-04)
        split     2-for-1, ex-date day 6 (2024-01-09)

    Walking it through:

      day 0   NAV 100,000, mark 100 -> target 1,000 shares, order placed
      day 1   fill 1,000 @ open 100 -> cash 0, position 100,000, NAV 100,000
      day 3   dividend 1,000 x 2.00 = 2,000 -> cash 2,000, NAV 102,000
      day 6   split: 1,000 -> 2,000 shares, basis 100 -> 50, price series
              halves to 50 -> position value 2,000 x 50 = 100,000, NAV 102,000
      day 9   unchanged, NAV 102,000

    The split must be economically neutral; the dividend must be the only
    source of return.
    """

    @staticmethod
    def _scenario():
        dates = pd.bdate_range("2024-01-01", periods=10)
        prices = [100.0] * 6 + [50.0] * 4
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
        actions = pd.DataFrame(
            [
                {
                    "timestamp": dates[3],
                    "symbol": "A",
                    "action": "DIVIDEND",
                    "cash_amount": 2.0,
                    "split_ratio": 1.0,
                },
                {
                    "timestamp": dates[6],
                    "symbol": "A",
                    "action": "SPLIT",
                    "cash_amount": 0.0,
                    "split_ratio": 2.0,
                },
            ]
        )
        return bars, actions

    def test_final_nav_is_capital_plus_the_dividend(self):
        bars, actions = self._scenario()
        result = build(bars, BuyAndHold(), actions).run()
        assert result.final_nav == pytest.approx(102_000.0)

    def test_share_count_doubles_and_basis_halves(self):
        bars, actions = self._scenario()
        engine = build(bars, BuyAndHold(), actions)
        engine.run()

        position = engine.portfolio.positions["A"]
        assert position.quantity == pytest.approx(2_000.0)
        assert position.average_cost == pytest.approx(50.0)

    def test_the_split_bar_does_not_move_nav(self):
        bars, actions = self._scenario()
        result = build(bars, BuyAndHold(), actions).run()

        navs = result.equity_curve["nav"].tolist()
        assert navs[5] == pytest.approx(navs[6])

    def test_dividend_is_the_only_source_of_return(self):
        bars, actions = self._scenario()
        engine = build(bars, BuyAndHold(), actions)
        result = engine.run()

        assert engine.portfolio.cumulative_dividends == pytest.approx(2_000.0)
        assert engine.portfolio.positions["A"].unrealized_pnl(50.0) == pytest.approx(0.0)
        assert result.total_return == pytest.approx(0.02)


class TestShortAccounting:
    """
    Short half the book at a flat price of 100, borrow at 5% annual, ACT/360.

      day 0   NAV 100,000, mark 100 -> target -500 shares
      day 1   sell 500 @ 100 -> cash 150,000, position -50,000, NAV 100,000
      carry   50,000 notional x 5% x (days / 360)

    Marks run 2024-01-01 to 2024-01-12, which is 11 calendar days of accrual
    (the Jan 5 -> Jan 8 mark spans the weekend and is charged three days, as it
    must be: borrow accrues on calendar days, not sessions).

      total borrow = 50,000 x 0.05 x 11 / 360 = 76.3889
      final NAV    = 100,000 - 76.3889 = 99,923.61
    """

    @staticmethod
    def _run():
        engine = build(
            flat_price_bars({"A": 100.0}, n_days=10),
            EnterOnce({"A": -0.5}),
            financing=FinancingModel(borrow_rate=0.05),
        )
        return engine.run(), engine.portfolio

    def test_short_sale_credits_cash(self):
        _, portfolio = self._run()
        assert portfolio.quantity("A") == pytest.approx(-500.0)
        assert portfolio.cash > 100_000.0

    def test_borrow_cost_matches_the_hand_calculation(self):
        _, portfolio = self._run()
        expected = 50_000.0 * 0.05 * 11.0 / 360.0
        assert portfolio.cumulative_financing == pytest.approx(expected)

    def test_final_nav_is_capital_less_borrow(self):
        result, _ = self._run()
        assert result.final_nav == pytest.approx(100_000.0 - 76.38888888, abs=1e-6)

    def test_weekend_is_charged_three_days(self):
        """
        Borrow accrues on calendar days. An engine that charges per *bar* would
        undercount by 40% over a year, and the error is invisible because the
        equity curve still looks smooth.
        """
        _, portfolio = self._run()
        curve = portfolio.equity_curve()
        daily = curve["cumulative_financing"].diff()

        # Index 5 is the 2024-01-08 mark, which spans Fri -> Mon.
        assert daily.iloc[5] == pytest.approx(3.0 * daily.iloc[4])


class TestCostAccounting:
    """
    Costs must be fully accounted: the NAV shortfall against a frictionless run
    equals the sum of the reported cost components, to the cent.
    """

    @staticmethod
    def _scenario():
        return flat_price_bars({"A": 100.0, "B": 50.0}, n_days=10)

    def test_nav_shortfall_equals_reported_costs(self):
        bars = self._scenario()

        free = build(bars, BuyAndHold()).run()
        costly_engine = build(
            bars,
            BuyAndHold(),
            broker=SimulatedBroker(
                commission=BpsCommission(bps=2.0), spread=FixedBpsSpread(bps=10.0)
            ),
        )
        costly = costly_engine.run()

        shortfall = free.final_nav - costly.final_nav
        reported = costly.fills["total_cost"].sum()
        assert shortfall == pytest.approx(reported, rel=1e-9)

    def test_cost_components_are_individually_correct(self):
        """
        Buying 500 shares of A at 100 and 1,000 of B at 50 is 50,000 of notional
        each. A 10 bp spread means a 5 bp half-spread: 25.00 per name. A 2 bp
        commission on the (slightly higher) fill price is about 10.00 per name.
        """
        engine = build(
            self._scenario(),
            BuyAndHold(),
            broker=SimulatedBroker(
                commission=BpsCommission(bps=2.0), spread=FixedBpsSpread(bps=10.0)
            ),
        )
        result = engine.run()

        assert result.fills["spread_cost"].sum() == pytest.approx(50.0)
        assert result.fills["commission"].sum() == pytest.approx(20.02, abs=0.05)
        assert result.fills["impact_cost"].sum() == pytest.approx(0.0)

    def test_buys_fill_above_the_reference_price(self):
        engine = build(
            self._scenario(),
            BuyAndHold(),
            broker=SimulatedBroker(spread=FixedBpsSpread(bps=10.0)),
        )
        result = engine.run()
        assert (result.fills["fill_price"] > result.fills["reference_price"]).all()


class TestRebalancing:
    def test_a_restating_strategy_closes_names_it_drops(self):
        class Restater(Strategy):
            name = "restater"

            def __init__(self):
                self.day = 0

            def on_bar(self, event, ctx):
                self.day += 1
                if self.day == 1:
                    return [ctx.long("A", strength=0.5), ctx.long("B", strength=0.5)]
                if self.day == 5:
                    return [ctx.long("A", strength=1.0)]
                return []

        data = DataFrameDataHandler(flat_price_bars({"A": 100.0, "B": 50.0}, n_days=10))
        portfolio = Portfolio(100_000.0)
        engine = Backtest(
            data,
            Restater(),
            portfolio,
            SimulatedBroker(),
            PortfolioConstructor(sizer=ExplicitWeightSizer()),
            liquidate_unsignalled=True,
        )
        engine.run()

        assert portfolio.quantity("B") == pytest.approx(0.0)
        assert portfolio.quantity("A") == pytest.approx(1_000.0)

    def test_nav_is_conserved_across_a_frictionless_rebalance(self):
        class Restater(Strategy):
            name = "restater"

            def __init__(self):
                self.day = 0

            def on_bar(self, event, ctx):
                self.day += 1
                if self.day in (1, 5):
                    w = 0.5 if self.day == 1 else 0.3
                    return [ctx.long("A", strength=w), ctx.long("B", strength=1.0 - w)]
                return []

        data = DataFrameDataHandler(flat_price_bars({"A": 100.0, "B": 50.0}, n_days=10))
        portfolio = Portfolio(100_000.0)
        engine = Backtest(
            data,
            Restater(),
            portfolio,
            SimulatedBroker(),
            PortfolioConstructor(sizer=ExplicitWeightSizer()),
            liquidate_unsignalled=True,
        )
        result = engine.run()

        assert result.final_nav == pytest.approx(100_000.0)
        assert result.n_fills > 2  # the rebalance really happened
