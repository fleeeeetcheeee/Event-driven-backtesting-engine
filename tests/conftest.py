"""Shared fixtures. Synthetic data with hand-computable answers throughout."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from evbt.core.events import Bar, FillEvent, OrderEvent, OrderSide, OrderType


def make_bar(
    symbol: str = "X",
    timestamp: datetime | None = None,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
    volume: float = 1_000_000.0,
    volatility: float | None = 0.02,
) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=timestamp or datetime(2024, 1, 2),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        volatility=volatility,
    )


def make_fill(
    symbol: str = "X",
    side: OrderSide = OrderSide.BUY,
    quantity: float = 100.0,
    price: float = 100.0,
    commission: float = 0.0,
    timestamp: datetime | None = None,
    order_id: str = "O1",
) -> FillEvent:
    return FillEvent(
        timestamp=timestamp or datetime(2024, 1, 2),
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        fill_price=price,
        commission=commission,
    )


def make_order(
    symbol: str = "X",
    side: OrderSide = OrderSide.BUY,
    quantity: float = 100.0,
    timestamp: datetime | None = None,
    order_id: str = "O1",
    order_type: OrderType = OrderType.MARKET,
    limit_price: float | None = None,
    **kwargs,
) -> OrderEvent:
    return OrderEvent(
        timestamp=timestamp or datetime(2024, 1, 1),
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        **kwargs,
    )


def flat_price_bars(
    symbols: dict[str, float],
    n_days: int = 10,
    start: str = "2024-01-01",
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """Constant prices — any P&L in a test using these is a bug, not a return."""
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for date in dates:
        for symbol, price in symbols.items():
            rows.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


def trending_bars(
    symbols: dict[str, float],
    n_days: int = 10,
    daily_return: float = 0.01,
    start: str = "2024-01-01",
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Geometric price paths with open == high == low == close on each bar.

    Collapsing the bar to a single price is deliberate: it removes intrabar
    ambiguity so a fill price can be predicted exactly by hand, which is what
    makes the accounting tests assertions rather than approximations.
    """
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for i, date in enumerate(dates):
        for symbol, base in symbols.items():
            price = base * (1.0 + daily_return) ** i
            rows.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def bar():
    return make_bar()


@pytest.fixture
def two_symbol_flat():
    return flat_price_bars({"AAA": 100.0, "BBB": 50.0}, n_days=10)


@pytest.fixture
def two_symbol_trending():
    return trending_bars({"AAA": 100.0, "BBB": 50.0}, n_days=10, daily_return=0.01)
