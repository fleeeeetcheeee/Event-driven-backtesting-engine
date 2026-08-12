"""Sign conventions and the validation that keeps them honest."""

from __future__ import annotations

from datetime import datetime

import pytest

from evbt.core.events import (
    Bar,
    CorporateActionEvent,
    CorporateActionType,
    OrderSide,
    OrderType,
)
from tests.conftest import make_fill, make_order

T0 = datetime(2024, 1, 2)


class TestOrderValidation:
    def test_negative_quantity_rejected(self):
        """Direction lives in `side`; a signed quantity is always a bug."""
        with pytest.raises(ValueError, match="positive"):
            make_order(quantity=-100)

    def test_zero_quantity_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            make_order(quantity=0)

    def test_limit_order_without_price_rejected(self):
        with pytest.raises(ValueError, match="limit_price"):
            make_order(order_type=OrderType.LIMIT)

    def test_signed_quantity_applies_side(self):
        assert make_order(side=OrderSide.BUY, quantity=100).signed_quantity == 100
        assert make_order(side=OrderSide.SELL, quantity=100).signed_quantity == -100


class TestFillCashEffects:
    def test_buy_consumes_cash_and_pays_commission(self):
        fill = make_fill(side=OrderSide.BUY, quantity=100, price=50.0, commission=1.0)
        assert fill.gross_value == 5000.0
        assert fill.cash_delta == -5001.0

    def test_sell_releases_cash_and_still_pays_commission(self):
        fill = make_fill(side=OrderSide.SELL, quantity=100, price=50.0, commission=1.0)
        assert fill.cash_delta == 4999.0

    def test_total_cost_sums_every_component(self):
        fill = make_fill(quantity=100, price=50.0, commission=1.0)
        fill.spread_cost = 2.0
        fill.impact_cost = 3.0
        fill.slippage_cost = 4.0
        assert fill.total_cost == 10.0


class TestCorporateActionValidation:
    def test_non_positive_split_ratio_rejected(self):
        with pytest.raises(ValueError, match="split_ratio"):
            CorporateActionEvent(
                timestamp=T0,
                symbol="X",
                action=CorporateActionType.SPLIT,
                split_ratio=0.0,
            )

    def test_negative_dividend_rejected(self):
        with pytest.raises(ValueError, match="dividend"):
            CorporateActionEvent(
                timestamp=T0,
                symbol="X",
                action=CorporateActionType.DIVIDEND,
                cash_amount=-1.0,
            )


class TestBar:
    def test_typical_price(self):
        bar = Bar("X", T0, open=10.0, high=12.0, low=8.0, close=11.0, volume=1.0)
        assert bar.typical_price == pytest.approx((12 + 8 + 11) / 3)

    def test_contains_price_is_inclusive_of_the_range(self):
        bar = Bar("X", T0, open=10.0, high=12.0, low=8.0, close=11.0, volume=1.0)
        assert bar.contains_price(8.0)
        assert bar.contains_price(12.0)
        assert not bar.contains_price(12.01)
