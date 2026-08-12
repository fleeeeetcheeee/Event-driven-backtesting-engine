"""
Portfolio construction: signals in, orders out.

The stage between "the strategy has an opinion" and "the broker has an order",
and the reason `SignalEvent` deliberately carries no share count. Separating
them buys three things:

  - the same signal can be run at $10M and at $10B without editing the strategy,
    which is what makes a capacity analysis possible at all;
  - risk limits apply to the book the strategy wants, not to whichever orders
    happen to arrive first;
  - alpha and sizing can be attributed separately, so "the signal was right but
    the sizing was wrong" is a statement the backtest can actually support.

The pipeline:

    signals -> Sizer -> target weights -> RiskManager -> compliant weights
            -> diff against current book -> orders

Sizing uses the *last known* price and the *current* NAV. Both are known at
signal time; the fill happens on a later bar at a price nobody knows yet. That
gap is genuine execution risk and the engine leaves it in — sizing against the
fill price would require knowing it, which is precisely the lookahead this
project exists to avoid.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Sequence

from evbt.core.events import (
    OrderEvent,
    OrderSide,
    OrderType,
    SignalDirection,
    SignalEvent,
    TimeInForce,
)
from evbt.portfolio.risk import RiskManager

if TYPE_CHECKING:  # pragma: no cover
    from evbt.portfolio.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Sizers: signals -> target weights
# ---------------------------------------------------------------------------


class Sizer(ABC):
    """Converts a set of signals at one timestamp into target weights of NAV."""

    @abstractmethod
    def target_weights(self, signals: Sequence[SignalEvent]) -> dict[str, float]:
        """Signed target weights. Positive is long, negative is short."""


@dataclass
class ExplicitWeightSizer(Sizer):
    """
    Takes `signal.strength` as the target weight verbatim.

    For strategies that already know the book they want — a factor replication
    with published weights, a risk-parity allocation solved elsewhere. It is
    also what the done-criterion replication uses, because reproducing a
    published series to basis points requires the weights to pass through
    completely untouched.
    """

    def target_weights(self, signals: Sequence[SignalEvent]) -> dict[str, float]:
        out: dict[str, float] = {}
        for signal in signals:
            if signal.direction is SignalDirection.EXIT:
                out[signal.symbol] = 0.0
            else:
                out[signal.symbol] = float(signal.strength)
        return out


@dataclass
class EqualWeightSizer(Sizer):
    """
    Splits `gross_leverage` equally across longs and shorts.

    The default for a cross-sectional signal that ranks but does not size. Each
    side gets half the gross when both are present, so the book is
    dollar-neutral by construction; a signal with only longs gets the full gross
    on that side.

    Equal weighting is a stronger baseline than it looks. It has no estimation
    error, which is exactly the failure mode that makes mean-variance optimal
    weights underperform out of sample (DeMiguel, Garlappi & Uppal, 2009). Any
    fancier sizing scheme should be shown to beat this, not assumed to.
    """

    gross_leverage: float = 1.0

    def target_weights(self, signals: Sequence[SignalEvent]) -> dict[str, float]:
        longs = [s for s in signals if s.direction is SignalDirection.LONG]
        shorts = [s for s in signals if s.direction is SignalDirection.SHORT]
        exits = [s for s in signals if s.direction is SignalDirection.EXIT]

        out: dict[str, float] = {s.symbol: 0.0 for s in exits}
        n_sides = (1 if longs else 0) + (1 if shorts else 0)
        if n_sides == 0:
            return out

        per_side = self.gross_leverage / n_sides
        for group, sign in ((longs, 1.0), (shorts, -1.0)):
            if not group:
                continue
            weight = sign * per_side / len(group)
            for signal in group:
                out[signal.symbol] = weight
        return out


@dataclass
class ProportionalSizer(Sizer):
    """
    Weights proportional to signal strength, normalised to a gross target.

    The natural sizing when `strength` is a real conviction — an expected
    return, a z-score, a factor exposure — rather than a label. Note the
    implicit assumption: weight proportional to expected return is optimal only
    if risk is equal across names. When it is not, this overweights the volatile
    names, and the fix is a risk model (Project 10), not a bigger constant.
    """

    gross_leverage: float = 1.0

    def target_weights(self, signals: Sequence[SignalEvent]) -> dict[str, float]:
        raw: dict[str, float] = {}
        for signal in signals:
            if signal.direction is SignalDirection.EXIT:
                raw[signal.symbol] = 0.0
                continue
            sign = 1.0 if signal.direction is SignalDirection.LONG else -1.0
            raw[signal.symbol] = sign * abs(float(signal.strength))

        total = sum(abs(w) for w in raw.values())
        if total <= 1e-12:
            return {s: 0.0 for s in raw}
        scale = self.gross_leverage / total
        return {s: w * scale for s, w in raw.items()}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@dataclass
class PortfolioConstructor:
    """
    Turns signals into orders against a live portfolio.

    Parameters
    ----------
    sizer
        Signals to target weights.
    risk
        Applied to the target weights before any order is generated.
    min_trade_notional
        Trades below this are skipped. Not a cosmetic filter: a per-order
        commission minimum turns a stream of dust trades into a real drag, and
        a rebalance that chases every 0.01% drift is a backtest of a strategy
        no one would run. Setting it to zero reproduces the naive behaviour.
    round_lots
        Round order quantities to whole shares (or to `lot_size`). Fractional
        shares are convenient and, at institutional size, immaterial — but they
        also quietly remove the rounding drag that a small account really pays.
    order_type / time_in_force
        Applied to every generated order. MARKET/DAY is the honest default: it
        fills at the next open at a price that includes costs, and it does not
        assume a limit order that may never have been hit.
    """

    sizer: Sizer
    risk: Optional[RiskManager] = None
    min_trade_notional: float = 0.0
    round_lots: bool = False
    lot_size: int = 1
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY

    def build_orders(
        self,
        signals: Sequence[SignalEvent],
        portfolio: "Portfolio",
        timestamp,
        order_id_factory,
        *,
        liquidate_unsignalled: bool = False,
    ) -> list[OrderEvent]:
        """
        Produce the orders that move `portfolio` toward the sized, risk-checked
        target book.

        `liquidate_unsignalled` controls what happens to a held position the
        strategy said nothing about this bar. False — the default — holds it,
        which is right for a strategy that signals only on changes. True closes
        it, which is right for a strategy that re-states its entire book every
        rebalance. Getting this backwards is a quiet way to run a completely
        different strategy: a full-restatement strategy with the flag off
        accumulates every position it ever opened.
        """
        targets = self.sizer.target_weights(signals)

        nav = portfolio.nav
        if nav <= 0:
            # Wiped out. Emitting orders against a non-positive NAV produces
            # nonsense sizes; stopping is the honest outcome and the equity
            # curve already records the ruin.
            return []

        current_weights = self._current_weights(portfolio, nav)

        if liquidate_unsignalled:
            for symbol in current_weights:
                targets.setdefault(symbol, 0.0)

        if self.risk is not None:
            targets = self.risk.apply(targets, current_weights, timestamp)

        orders: list[OrderEvent] = []
        for symbol in sorted(targets):
            order = self._order_for(
                symbol, targets[symbol], portfolio, nav, timestamp, order_id_factory
            )
            if order is not None:
                orders.append(order)
        return orders

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _current_weights(portfolio: "Portfolio", nav: float) -> dict[str, float]:
        return {
            symbol: position.market_value(portfolio.mark_price(symbol)) / nav
            for symbol, position in portfolio.open_positions.items()
        }

    def _order_for(
        self,
        symbol: str,
        target_weight: float,
        portfolio: "Portfolio",
        nav: float,
        timestamp,
        order_id_factory,
    ) -> Optional[OrderEvent]:
        price = portfolio.mark_price(symbol)
        if price is None or price <= 0:
            # Never traded, or a corrupt mark. Cannot size against it, and
            # guessing a price would be inventing data.
            return None

        target_quantity = target_weight * nav / price
        delta = target_quantity - portfolio.quantity(symbol)

        if self.round_lots:
            delta = math.floor(abs(delta) / self.lot_size) * self.lot_size * (
                1.0 if delta > 0 else -1.0
            )

        if abs(delta) < 1e-9 or abs(delta) * price < self.min_trade_notional:
            return None

        return OrderEvent(
            timestamp=timestamp,
            order_id=order_id_factory(),
            symbol=symbol,
            side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
            quantity=abs(delta),
            order_type=self.order_type,
            time_in_force=self.time_in_force,
            metadata={"target_weight": target_weight},
        )
