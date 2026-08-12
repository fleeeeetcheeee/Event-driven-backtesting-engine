"""
Portfolio accounting against hand-computed answers.

The NAV identity is the thing under test throughout:

    NAV = cash + sum(signed quantity * mark)

and in particular that it behaves correctly for shorts, where the cash from the
sale and the negative market value of the borrowed shares have to net out.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from evbt.core.events import (
    CorporateActionEvent,
    CorporateActionType,
    MarkToMarketEvent,
    OrderSide,
)
from evbt.portfolio.financing import FinancingModel
from evbt.portfolio.portfolio import InsufficientCashError, Portfolio
from tests.conftest import make_bar, make_fill, make_order

T0 = datetime(2024, 1, 2)
T1 = datetime(2024, 1, 3)


def mark(portfolio: Portfolio, timestamp, prices: dict[str, float]) -> None:
    bars = {s: make_bar(symbol=s, timestamp=timestamp, close=p) for s, p in prices.items()}
    portfolio.on_mark_to_market(MarkToMarketEvent(timestamp=timestamp, bars=bars))


class TestNavIdentity:
    def test_starts_at_initial_cash(self):
        assert Portfolio(1_000_000).nav == 1_000_000

    def test_buying_moves_value_from_cash_to_positions_without_changing_nav(self):
        pf = Portfolio(100_000)
        pf.on_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))

        assert pf.cash == pytest.approx(95_000.0)
        assert pf.positions_value == pytest.approx(5_000.0)
        assert pf.nav == pytest.approx(100_000.0)

    def test_short_sale_credits_cash_and_books_a_negative_market_value(self):
        # Sell 100 @ 50: +5,000 cash, -5,000 market value, NAV unchanged.
        pf = Portfolio(100_000)
        pf.on_fill(make_fill(side=OrderSide.SELL, quantity=100, price=50.0))

        assert pf.cash == pytest.approx(105_000.0)
        assert pf.positions_value == pytest.approx(-5_000.0)
        assert pf.nav == pytest.approx(100_000.0)

    def test_short_gains_when_the_mark_falls(self):
        pf = Portfolio(100_000)
        pf.on_fill(make_fill(side=OrderSide.SELL, quantity=100, price=50.0))
        mark(pf, T0, {"X": 40.0})

        # cash 105,000, position -100 * 40 = -4,000 -> NAV 101,000
        assert pf.nav == pytest.approx(101_000.0)

    def test_commission_reduces_nav_immediately(self):
        pf = Portfolio(100_000)
        pf.on_fill(
            make_fill(side=OrderSide.BUY, quantity=100, price=50.0, commission=7.0)
        )
        assert pf.nav == pytest.approx(99_993.0)


class TestExposures:
    def _long_short_book(self) -> Portfolio:
        pf = Portfolio(100_000)
        pf.on_fill(make_fill(symbol="L", side=OrderSide.BUY, quantity=100, price=100.0))
        pf.on_fill(make_fill(symbol="S", side=OrderSide.SELL, quantity=200, price=25.0))
        mark(pf, T0, {"L": 100.0, "S": 25.0})
        return pf

    def test_gross_is_the_sum_of_absolute_values(self):
        # long 10,000 + short 5,000
        assert self._long_short_book().gross_exposure == pytest.approx(15_000.0)

    def test_net_is_the_signed_sum(self):
        assert self._long_short_book().net_exposure == pytest.approx(5_000.0)

    def test_side_exposures(self):
        pf = self._long_short_book()
        assert pf.long_exposure == pytest.approx(10_000.0)
        assert pf.short_exposure == pytest.approx(5_000.0)  # reported positive

    def test_leverage_is_relative_to_nav(self):
        pf = self._long_short_book()
        assert pf.gross_leverage == pytest.approx(15_000.0 / pf.nav)
        assert pf.net_leverage == pytest.approx(5_000.0 / pf.nav)


class TestCorporateActions:
    def test_dividend_credits_a_long(self):
        pf = Portfolio(100_000)
        pf.on_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        pf.on_corporate_action(
            CorporateActionEvent(
                timestamp=T0,
                symbol="X",
                action=CorporateActionType.DIVIDEND,
                cash_amount=0.25,
            )
        )
        assert pf.cash == pytest.approx(95_000.0 + 25.0)
        assert pf.cumulative_dividends == pytest.approx(25.0)

    def test_dividend_debits_a_short(self):
        pf = Portfolio(100_000)
        pf.on_fill(make_fill(side=OrderSide.SELL, quantity=100, price=50.0))
        pf.on_corporate_action(
            CorporateActionEvent(
                timestamp=T0,
                symbol="X",
                action=CorporateActionType.DIVIDEND,
                cash_amount=0.25,
            )
        )
        assert pf.cash == pytest.approx(105_000.0 - 25.0)

    def test_dividend_on_an_unheld_symbol_is_a_no_op(self):
        pf = Portfolio(100_000)
        pf.on_corporate_action(
            CorporateActionEvent(
                timestamp=T0,
                symbol="NOTHELD",
                action=CorporateActionType.DIVIDEND,
                cash_amount=5.0,
            )
        )
        assert pf.cash == pytest.approx(100_000.0)

    def test_split_leaves_nav_unchanged(self):
        """
        The invariant that makes split handling testable: a split is a
        relabelling, not an economic event.
        """
        pf = Portfolio(100_000)
        pf.on_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        mark(pf, T0, {"X": 50.0})
        before = pf.nav

        pf.on_corporate_action(
            CorporateActionEvent(
                timestamp=T1,
                symbol="X",
                action=CorporateActionType.SPLIT,
                split_ratio=2.0,
            )
        )

        assert pf.quantity("X") == pytest.approx(200.0)
        assert pf.mark_price("X") == pytest.approx(25.0)
        assert pf.nav == pytest.approx(before)

    def test_split_restates_a_stale_mark_even_when_flat(self):
        """
        A flat position still needs its mark restated, or the next NAV computed
        before the symbol's following bar is wrong by the split ratio.
        """
        pf = Portfolio(100_000)
        mark(pf, T0, {"X": 50.0})
        pf.on_corporate_action(
            CorporateActionEvent(
                timestamp=T1,
                symbol="X",
                action=CorporateActionType.SPLIT,
                split_ratio=2.0,
            )
        )
        assert pf.mark_price("X") == pytest.approx(25.0)


class TestFinancing:
    def test_short_accrues_borrow_over_the_elapsed_days(self):
        # 100 shares short at 50 = 5,000 notional. 5% annual, ACT/360, one day.
        pf = Portfolio(100_000, financing=FinancingModel(borrow_rate=0.05))
        pf.on_fill(make_fill(side=OrderSide.SELL, quantity=100, price=50.0))
        mark(pf, T0, {"X": 50.0})
        cash_before = pf.cash

        mark(pf, T1, {"X": 50.0})

        expected = 5_000.0 * 0.05 * 1.0 / 360.0
        assert cash_before - pf.cash == pytest.approx(expected)
        assert pf.cumulative_financing == pytest.approx(expected)

    def test_longs_are_never_charged_borrow(self):
        pf = Portfolio(100_000, financing=FinancingModel(borrow_rate=0.05))
        pf.on_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        mark(pf, T0, {"X": 50.0})
        cash_before = pf.cash

        mark(pf, T1, {"X": 50.0})
        assert pf.cash == pytest.approx(cash_before)

    def test_hard_to_borrow_override_applies(self):
        pf = Portfolio(
            100_000,
            financing=FinancingModel(borrow_rate=0.005, hard_to_borrow={"X": 0.50}),
        )
        pf.on_fill(make_fill(side=OrderSide.SELL, quantity=100, price=50.0))
        mark(pf, T0, {"X": 50.0})
        cash_before = pf.cash

        mark(pf, T1, {"X": 50.0})
        assert cash_before - pf.cash == pytest.approx(5_000.0 * 0.50 / 360.0)

    def test_cash_earns_credit_interest_when_enabled(self):
        pf = Portfolio(100_000, financing=FinancingModel(cash_credit_rate=0.05))
        mark(pf, T0, {})
        mark(pf, T1, {})
        assert pf.cash == pytest.approx(100_000.0 * (1 + 0.05 / 360.0))

    def test_defaults_charge_nothing(self):
        """Financing must be switched on deliberately, never inherited."""
        pf = Portfolio(100_000)
        pf.on_fill(make_fill(side=OrderSide.SELL, quantity=100, price=50.0))
        mark(pf, T0, {"X": 50.0})
        mark(pf, T1, {"X": 50.0})
        assert pf.cumulative_financing == pytest.approx(0.0)


class TestCashConstraint:
    def test_overdraft_raises_when_margin_is_disabled(self):
        pf = Portfolio(1_000, allow_margin=False)
        with pytest.raises(InsufficientCashError, match="margin disabled"):
            pf.on_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))

    def test_overdraft_is_allowed_when_margin_is_enabled(self):
        pf = Portfolio(1_000, allow_margin=True)
        pf.on_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        assert pf.cash == pytest.approx(-4_000.0)

    def test_can_afford_screens_buys_without_margin(self):
        pf = Portfolio(1_000, allow_margin=False)
        ok, reason = pf.can_afford(make_order(quantity=100), 50.0)
        assert not ok and "insufficient cash" in reason

    def test_can_afford_permits_sells_without_margin(self):
        pf = Portfolio(1_000, allow_margin=False)
        ok, _ = pf.can_afford(make_order(side=OrderSide.SELL, quantity=100), 50.0)
        assert ok


class TestRecording:
    def test_equity_curve_gains_a_row_per_mark(self):
        pf = Portfolio(100_000)
        mark(pf, T0, {"X": 50.0})
        mark(pf, T1, {"X": 51.0})

        curve = pf.equity_curve()
        assert len(curve) == 2
        assert list(curve.index) == [T0, T1]

    def test_turnover_is_reported_per_period_then_reset(self):
        pf = Portfolio(100_000)
        pf.on_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        mark(pf, T0, {"X": 50.0})
        mark(pf, T1, {"X": 50.0})

        curve = pf.equity_curve()
        assert curve["turnover_notional"].iloc[0] == pytest.approx(5_000.0)
        assert curve["turnover_notional"].iloc[1] == pytest.approx(0.0)

    def test_target_quantity_sizes_off_the_last_known_mark(self):
        pf = Portfolio(100_000)
        mark(pf, T0, {"X": 50.0})
        assert pf.target_quantity("X", 0.5) == pytest.approx(1_000.0)

    def test_target_quantity_is_zero_without_a_mark(self):
        assert Portfolio(100_000).target_quantity("UNKNOWN", 0.5) == 0.0
