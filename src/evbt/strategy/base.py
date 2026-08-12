"""
The strategy interface.

A strategy sees three things: the bars that have already printed, its own
current book, and the clock. It returns opinions. It does not size, does not
place orders, and — critically — has no way to reach data it has not been given.

The interface is deliberately narrow. Every widening of it is a new way to leak
the future: a strategy handed the raw DataFrame will eventually index into it,
and a strategy allowed to place orders directly will eventually place one that
fills on the bar it was computed from.

`StrategyContext` is the only handle a strategy gets. It exposes the data
handler (whose accessors can only reach already-streamed bars — see
`data.base`) and a read-only view of the portfolio.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Iterable, Optional

from evbt.core.events import FillEvent, MarketEvent, SignalDirection, SignalEvent

if TYPE_CHECKING:  # pragma: no cover
    from evbt.data.base import DataHandler
    from evbt.portfolio.portfolio import Portfolio


@dataclass
class StrategyContext:
    """Everything a strategy is permitted to see."""

    data: "DataHandler"
    portfolio: "Portfolio"
    now: datetime

    # --- convenience passthroughs, so strategies read cleanly --------------

    def price(self, symbol: str) -> Optional[float]:
        """Latest close for `symbol`, or None if it has not traded yet."""
        return self.data.latest_price(symbol)

    def quantity(self, symbol: str) -> float:
        return self.portfolio.quantity(symbol)

    def weight(self, symbol: str) -> float:
        """Current weight of `symbol` as a fraction of NAV."""
        nav = self.portfolio.nav
        if nav <= 0:
            return 0.0
        position = self.portfolio.positions.get(symbol)
        if position is None or not position.is_open:
            return 0.0
        return position.market_value(self.portfolio.mark_price(symbol)) / nav

    @property
    def nav(self) -> float:
        return self.portfolio.nav

    def long(self, symbol: str, strength: float = 1.0, **metadata) -> SignalEvent:
        return SignalEvent(
            timestamp=self.now,
            symbol=symbol,
            direction=SignalDirection.LONG,
            strength=strength,
            metadata=metadata,
        )

    def short(self, symbol: str, strength: float = 1.0, **metadata) -> SignalEvent:
        return SignalEvent(
            timestamp=self.now,
            symbol=symbol,
            direction=SignalDirection.SHORT,
            strength=strength,
            metadata=metadata,
        )

    def exit(self, symbol: str, **metadata) -> SignalEvent:
        return SignalEvent(
            timestamp=self.now,
            symbol=symbol,
            direction=SignalDirection.EXIT,
            strength=0.0,
            metadata=metadata,
        )


class Strategy(ABC):
    """
    Base class for trading strategies.

    Only `on_bar` is required. The other hooks exist for strategies that need
    to react to their own executions or to set up state once.
    """

    name: str = "strategy"

    def initialize(self, ctx: StrategyContext) -> None:
        """Called once, before the first bar. Optional."""

    @abstractmethod
    def on_bar(self, event: MarketEvent, ctx: StrategyContext) -> Iterable[SignalEvent]:
        """
        React to a bar slice. Return the signals for this timestamp.

        Return an empty sequence to do nothing — which is what a strategy that
        rebalances monthly should return on the other twenty days.
        """

    def on_fill(self, fill: FillEvent, ctx: StrategyContext) -> None:
        """Called after each of this strategy's fills is applied. Optional."""

    def finalize(self, ctx: StrategyContext) -> None:
        """Called once after the last bar. Optional."""


class BuyAndHold(Strategy):
    """
    Buy an equal-weight basket on the first bar and hold it.

    The benchmark every strategy has to beat, and the simplest end-to-end test
    of the engine: with zero costs, its return must equal the equal-weight
    return of the basket plus dividends, which can be checked by hand.
    """

    name = "buy_and_hold"

    def __init__(self, symbols: Optional[list[str]] = None, gross_leverage: float = 1.0):
        self.symbols = symbols
        self.gross_leverage = gross_leverage
        self._invested = False

    def on_bar(self, event: MarketEvent, ctx: StrategyContext) -> Iterable[SignalEvent]:
        if self._invested:
            return []
        universe = self.symbols or event.symbols
        tradeable = [s for s in universe if ctx.price(s) is not None]
        if not tradeable:
            return []
        self._invested = True
        weight = self.gross_leverage / len(tradeable)
        return [ctx.long(s, strength=weight) for s in tradeable]
