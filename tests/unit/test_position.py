"""
Position arithmetic, with the three cases the module docstring calls out:
sign flips, average cost on partial closes, and shorts owing dividends.

Every expected value here is computed by hand in the test's comment. If a
number needs the implementation to explain it, the test is not testing anything.
"""

from __future__ import annotations

import pytest

from evbt.core.events import OrderSide
from evbt.portfolio.position import Position
from tests.conftest import make_fill


class TestOpeningAndAdding:
    def test_opening_long_sets_basis_to_fill_price(self):
        pos = Position("X")
        realized = pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))

        assert pos.quantity == 100
        assert pos.average_cost == 50.0
        assert realized == 0.0

    def test_opening_short_holds_positive_basis(self):
        """`average_cost` is the price shares were sold at, and stays positive."""
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.SELL, quantity=100, price=50.0))

        assert pos.quantity == -100
        assert pos.average_cost == 50.0
        assert pos.is_short

    def test_adding_weights_the_average(self):
        # 100 @ 50 then 100 @ 60 -> 200 @ 55
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=100, price=60.0))

        assert pos.quantity == 200
        assert pos.average_cost == pytest.approx(55.0)

    def test_adding_to_a_short_weights_the_average(self):
        # -100 @ 50 then -100 @ 60 -> -200 @ 55
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.SELL, quantity=100, price=50.0))
        pos.apply_fill(make_fill(side=OrderSide.SELL, quantity=100, price=60.0))

        assert pos.quantity == -200
        assert pos.average_cost == pytest.approx(55.0)


class TestPartialClose:
    def test_realises_pnl_on_the_closed_portion_only(self):
        # Long 100 @ 50, sell 40 @ 60 -> realised 40 * 10 = 400
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        realized = pos.apply_fill(make_fill(side=OrderSide.SELL, quantity=40, price=60.0))

        assert realized == pytest.approx(400.0)
        assert pos.realized_pnl == pytest.approx(400.0)
        assert pos.quantity == 60

    def test_basis_of_the_residual_is_unchanged(self):
        """Closing part of a position must not rewrite the basis of the rest."""
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        pos.apply_fill(make_fill(side=OrderSide.SELL, quantity=40, price=60.0))

        assert pos.average_cost == pytest.approx(50.0)

    def test_short_profits_when_covering_lower(self):
        # Short 100 @ 50, buy back 40 @ 40 -> realised 40 * 10 = 400
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.SELL, quantity=100, price=50.0))
        realized = pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=40, price=40.0))

        assert realized == pytest.approx(400.0)
        assert pos.quantity == -60

    def test_flattening_zeroes_the_basis(self):
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        pos.apply_fill(make_fill(side=OrderSide.SELL, quantity=100, price=60.0))

        assert pos.quantity == 0
        assert pos.average_cost == 0.0
        assert not pos.is_open
        assert pos.unrealized_pnl(70.0) == 0.0


class TestSignFlip:
    """
    The case a naive `quantity += delta` gets wrong.

    Long 100 @ 50, sell 150 @ 60. Correct: realise 100 * (60 - 50) = 1000, then
    open a short of 50 with basis 60. Wrong: quantity becomes -50 while the
    basis stays 50, which then reports a further 500 of unrealised profit that
    never existed.
    """

    def test_long_to_short_realises_only_the_closed_shares(self):
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        realized = pos.apply_fill(make_fill(side=OrderSide.SELL, quantity=150, price=60.0))

        assert realized == pytest.approx(1000.0)
        assert pos.quantity == -50

    def test_new_side_takes_the_fill_price_as_basis(self):
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        pos.apply_fill(make_fill(side=OrderSide.SELL, quantity=150, price=60.0))

        assert pos.average_cost == pytest.approx(60.0)
        # Marked back at the flip price, the new short shows no P&L at all.
        assert pos.unrealized_pnl(60.0) == pytest.approx(0.0)

    def test_short_to_long_flip(self):
        # Short 100 @ 50, buy 150 @ 40 -> realise 100 * 10 = 1000, long 50 @ 40
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.SELL, quantity=100, price=50.0))
        realized = pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=150, price=40.0))

        assert realized == pytest.approx(1000.0)
        assert pos.quantity == 50
        assert pos.average_cost == pytest.approx(40.0)


class TestUnrealisedPnl:
    def test_long_gains_when_price_rises(self):
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        assert pos.unrealized_pnl(60.0) == pytest.approx(1000.0)

    def test_short_gains_when_price_falls(self):
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.SELL, quantity=100, price=50.0))
        assert pos.unrealized_pnl(40.0) == pytest.approx(1000.0)
        assert pos.unrealized_pnl(60.0) == pytest.approx(-1000.0)

    def test_market_value_of_a_short_is_negative(self):
        """A short is a liability; summing signed values is what makes NAV work."""
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.SELL, quantity=100, price=50.0))
        assert pos.market_value(50.0) == pytest.approx(-5000.0)


class TestSplits:
    def test_two_for_one_doubles_shares_and_halves_basis(self):
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        pos.apply_split(2.0)

        assert pos.quantity == 200
        assert pos.average_cost == pytest.approx(25.0)

    def test_split_preserves_position_value(self):
        """The invariant worth asserting: quantity * basis is unchanged."""
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        before = pos.quantity * pos.average_cost
        pos.apply_split(3.0)

        assert pos.quantity * pos.average_cost == pytest.approx(before)

    def test_reverse_split(self):
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=100, price=10.0))
        pos.apply_split(0.1)  # 1-for-10

        assert pos.quantity == pytest.approx(10.0)
        assert pos.average_cost == pytest.approx(100.0)

    def test_split_applies_to_a_short_too(self):
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.SELL, quantity=100, price=50.0))
        pos.apply_split(2.0)

        assert pos.quantity == -200
        assert pos.average_cost == pytest.approx(25.0)

    def test_non_positive_ratio_rejected(self):
        with pytest.raises(ValueError):
            Position("X").apply_split(-1.0)


class TestDividends:
    def test_long_receives(self):
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        assert pos.apply_dividend(0.25) == pytest.approx(25.0)

    def test_short_pays(self):
        """
        The omission that overstates every short book by its dividend yield.
        """
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.SELL, quantity=100, price=50.0))
        assert pos.apply_dividend(0.25) == pytest.approx(-25.0)
        assert pos.dividends_received == pytest.approx(-25.0)

    def test_dividend_does_not_touch_the_cost_basis(self):
        pos = Position("X")
        pos.apply_fill(make_fill(side=OrderSide.BUY, quantity=100, price=50.0))
        pos.apply_dividend(0.25)
        assert pos.average_cost == pytest.approx(50.0)


class TestTotalPnl:
    def test_aggregates_trading_carry_and_costs(self):
        # Buy 100 @ 50 with 1.00 commission, collect 0.25/share, pay 3.00 borrow,
        # mark at 55. Unrealised 500, dividends 25, less 3 borrow and 1 commission.
        pos = Position("X")
        pos.apply_fill(
            make_fill(side=OrderSide.BUY, quantity=100, price=50.0, commission=1.0)
        )
        pos.apply_dividend(0.25)
        pos.accrue_borrow(3.0)

        assert pos.total_pnl(55.0) == pytest.approx(500.0 + 25.0 - 3.0 - 1.0)


def test_fill_for_the_wrong_symbol_is_rejected():
    pos = Position("X")
    with pytest.raises(ValueError, match="applied to position"):
        pos.apply_fill(make_fill(symbol="Y"))
