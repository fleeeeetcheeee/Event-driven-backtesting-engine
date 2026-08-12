"""
The simulated broker: order lifecycle, fill rules, and the no-lookahead barrier.

This is the module the whole project exists for. Everything else — events,
portfolio arithmetic, analytics — is bookkeeping that a vectorised backtester
also has to do somehow. What a vectorised backtester cannot do is answer "when,
exactly, could this order have filled, and for how much?", and that question is
where backtest P&L is manufactured out of nothing.

The barrier
-----------
    An order submitted at time t may only fill against a bar strictly after t.

Enforced in `_can_match`, and it is an assertion rather than a convention: an
order whose timestamp equals the bar's is a bug somewhere upstream, and the
broker raises instead of quietly filling it. In a pandas backtest the same bug
is `signal * returns` with no shift, it is one character wide, and it produces
a beautiful equity curve.

The three optimistic assumptions this module refuses to make
------------------------------------------------------------
**1. That you get the whole bar.** Fills are capped at `max_participation` of
the bar's volume. An order for 500k shares of a name that trades 1M a day does
not fill in one print at the open; it is worked over days, at prices that move
while you work it. `allow_partial_fills=False` turns this off and is available
mostly to demonstrate how much it flatters a result.

**2. That a limit order fills whenever the price is touched.** By default a fill
requires the bar to trade *through* the limit, not merely to it. Touching your
price means the market reached the front of a queue you were somewhere in the
middle of; assuming a fill there is the single most common way to fabricate
returns in a mean-reversion backtest, because your limits get hit precisely on
the bars where you would in fact have been left behind.

**3. That your order is free of the market.** Every fill pays spread, impact and
slippage, computed off the bar it actually fills on — so an order worked across
five bars pays five different, individually-computed costs, because the bars it
consumes have different volumes and volatilities.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Optional

from evbt.core.events import (
    Bar,
    Event,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderExpiredEvent,
    OrderRejectedEvent,
    OrderSide,
    OrderType,
    TimeInForce,
)
from evbt.execution.costs import (
    CommissionModel,
    ImpactModel,
    SpreadModel,
    ZeroCommission,
    ZeroImpact,
    ZeroSpread,
)
from evbt.execution.slippage import SlippageModel, ZeroSlippage

log = logging.getLogger(__name__)


class LookaheadError(RuntimeError):
    """An order was matched against a bar it could not have seen the future of."""


@dataclass
class RestingOrder:
    """An accepted order awaiting execution, with its partial-fill progress."""

    order: OrderEvent
    filled_quantity: float = 0.0
    bars_exposed: int = 0

    @property
    def remaining(self) -> float:
        return self.order.quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        # Tolerance in shares. Float accumulation across a dozen partial fills
        # leaves residues around 1e-10 that would otherwise keep an order alive
        # forever, re-charging commission minimums on each near-zero slice.
        return self.remaining <= 1e-9


class SimulatedBroker:
    """
    Matches resting orders against incoming bars and prices the resulting fills.

    Parameters
    ----------
    max_participation
        Cap on the fraction of a bar's volume a single order may take. 0.1 is a
        common institutional working limit. Also the knob that makes capacity
        bind: at 10% of volume, a $1B position in a name trading $50M a day
        takes twenty sessions to build, and the strategy's signal has decayed
        by then.
    require_trade_through
        Whether a limit order needs the bar to trade strictly past its price.
        True is the conservative and correct default; see the module docstring.
    allow_partial_fills
        When False, an order either fills entirely on a bar or not at all,
        ignoring the participation cap.
    day_order_lifetime_bars
        How many bars a DAY order is exposed for before expiring. 1 = the next
        session only, which is what DAY means.
    """

    def __init__(
        self,
        *,
        commission: Optional[CommissionModel] = None,
        spread: Optional[SpreadModel] = None,
        impact: Optional[ImpactModel] = None,
        slippage: Optional[SlippageModel] = None,
        max_participation: float = 0.1,
        require_trade_through: bool = True,
        allow_partial_fills: bool = True,
        day_order_lifetime_bars: int = 1,
    ) -> None:
        self.commission = commission or ZeroCommission()
        self.spread = spread or ZeroSpread()
        self.impact = impact or ZeroImpact()
        self.slippage = slippage or ZeroSlippage()

        if not 0.0 < max_participation <= 1.0:
            raise ValueError(
                f"max_participation must be in (0, 1], got {max_participation}"
            )
        self.max_participation = max_participation
        self.require_trade_through = require_trade_through
        self.allow_partial_fills = allow_partial_fills
        self.day_order_lifetime_bars = day_order_lifetime_bars

        self._resting: dict[str, RestingOrder] = {}
        self._order_ids = itertools.count(1)
        self.rejections: list[OrderRejectedEvent] = []
        self.expirations: list[OrderExpiredEvent] = []

    # --- order entry -------------------------------------------------------

    def next_order_id(self) -> str:
        return f"O{next(self._order_ids):08d}"

    def submit(self, order: OrderEvent) -> None:
        """Accept an order onto the book. It becomes eligible from the next bar."""
        if order.order_id in self._resting:
            raise ValueError(f"duplicate order_id {order.order_id}")
        self._resting[order.order_id] = RestingOrder(order=order)

    def reject(self, order: OrderEvent, reason: str, timestamp=None) -> OrderRejectedEvent:
        """
        Record a rejection. Returned so the engine can queue it as an event —
        rejections are data, not a log line. A strategy that wanted to trade and
        could not is a strategy that was not backtested.
        """
        event = OrderRejectedEvent(
            timestamp=timestamp or order.timestamp, order=order, reason=reason
        )
        self.rejections.append(event)
        return event

    def cancel(self, order_id: str) -> bool:
        return self._resting.pop(order_id, None) is not None

    def cancel_all(self, symbol: Optional[str] = None) -> int:
        """Cancel every resting order, or every one in `symbol`. Returns the count."""
        doomed = [
            oid
            for oid, resting in self._resting.items()
            if symbol is None or resting.order.symbol == symbol
        ]
        for oid in doomed:
            del self._resting[oid]
        return len(doomed)

    @property
    def open_orders(self) -> list[OrderEvent]:
        return [r.order for r in self._resting.values()]

    # --- matching ----------------------------------------------------------

    def on_market(self, event: MarketEvent) -> list[Event]:
        """
        Match every resting order against this bar slice.

        Returns fills and expirations in submission order, so a run is
        reproducible regardless of dict iteration details.
        """
        produced: list[Event] = []
        completed: list[str] = []

        for order_id, resting in sorted(self._resting.items()):
            bar = event.bars.get(resting.order.symbol)
            if bar is None:
                # No bar for this symbol today — halted, delisted, or simply not
                # in this slice. The order rests; it does not expire, because it
                # was never exposed to a session.
                continue

            if not self._can_match(resting.order, bar):
                continue

            resting.bars_exposed += 1
            fill = self._try_fill(resting, bar)
            if fill is not None:
                produced.append(fill)

            if resting.is_complete:
                completed.append(order_id)
            elif self._has_expired(resting):
                produced.append(
                    OrderExpiredEvent(
                        timestamp=bar.timestamp,
                        order=resting.order,
                        filled_quantity=resting.filled_quantity,
                    )
                )
                self.expirations.append(produced[-1])
                completed.append(order_id)

        for order_id in completed:
            del self._resting[order_id]

        return produced

    def _can_match(self, order: OrderEvent, bar: Bar) -> bool:
        """
        The no-lookahead barrier.

        An order stamped at or after the bar's timestamp cannot legitimately
        trade on it: the strategy that produced the order saw that bar's close.
        This raises rather than returning False, because there is no legitimate
        path that produces such an order and silently skipping it would hide the
        upstream bug while making the strategy look merely inactive.
        """
        if order.timestamp > bar.timestamp:
            return False
        if order.timestamp == bar.timestamp:
            raise LookaheadError(
                f"order {order.order_id} ({order.symbol}) is stamped at "
                f"{order.timestamp}, the same time as the bar it would fill "
                f"against. An order may only fill on a bar strictly after the "
                f"information that produced it."
            )
        return True

    def _has_expired(self, resting: RestingOrder) -> bool:
        if resting.order.time_in_force is TimeInForce.GTC:
            return False
        return resting.bars_exposed >= self.day_order_lifetime_bars

    def _try_fill(self, resting: RestingOrder, bar: Bar) -> Optional[FillEvent]:
        order = resting.order

        reference = self._reference_price(order, bar)
        if reference is None:
            return None

        quantity = self._fillable_quantity(resting, bar)
        if quantity <= 0:
            return None

        # A resting limit order supplies liquidity rather than demanding it, so
        # it neither crosses the spread nor slips: by definition it transacted
        # at its own price. It still moves the market, so impact is still
        # charged. Market orders pay all three.
        is_liquidity_taking = order.order_type is not OrderType.LIMIT

        spread_per_share = (
            self.spread.cost_per_share(bar, order.symbol) if is_liquidity_taking else 0.0
        )
        slippage_per_share = (
            self.slippage.cost_per_share(quantity, bar, reference)
            if is_liquidity_taking
            else 0.0
        )
        impact_per_share = self.impact.cost_per_share(quantity, bar)

        adverse = spread_per_share + slippage_per_share + impact_per_share
        fill_price = reference + order.side.sign * adverse

        if fill_price <= 0:
            # Only reachable on corrupt data (a near-zero price with a huge
            # adverse term). Refusing the fill is right: a negative transaction
            # price would make the portfolio arithmetic meaningless downstream.
            log.warning(
                "refusing fill for %s at non-positive price %.6f (reference %.6f)",
                order.symbol,
                fill_price,
                reference,
            )
            return None

        commission = self.commission.calculate(quantity, fill_price, order.symbol)
        resting.filled_quantity += quantity

        return FillEvent(
            timestamp=bar.timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            fill_price=fill_price,
            commission=commission,
            spread_cost=spread_per_share * quantity,
            impact_cost=impact_per_share * quantity,
            slippage_cost=slippage_per_share * quantity,
            reference_price=reference,
            is_partial=not resting.is_complete,
            metadata=dict(order.metadata),
        )

    def _reference_price(self, order: OrderEvent, bar: Bar) -> Optional[float]:
        """
        The price this order transacts around, before execution costs.

        MARKET and MOO reference the open — the first price available after the
        decision. MOC references the close. LIMIT references its own limit,
        unless the bar opened already through it, in which case the order fills
        at the open: price improvement is real and refusing it would be its own
        distortion.

        Returns None when a limit order's condition is not met on this bar.
        """
        if order.order_type in (OrderType.MARKET, OrderType.MARKET_ON_OPEN):
            return bar.open
        if order.order_type is OrderType.MARKET_ON_CLOSE:
            return bar.close

        limit = order.limit_price
        assert limit is not None  # guaranteed by OrderEvent.__post_init__

        if order.side is OrderSide.BUY:
            if bar.open <= limit:
                return bar.open  # gapped through: filled at the better price
            reached = bar.low < limit if self.require_trade_through else bar.low <= limit
            return limit if reached else None
        else:
            if bar.open >= limit:
                return bar.open
            reached = bar.high > limit if self.require_trade_through else bar.high >= limit
            return limit if reached else None

    def _fillable_quantity(self, resting: RestingOrder, bar: Bar) -> float:
        """How much of the remaining order this bar can absorb."""
        remaining = resting.remaining
        if not self.allow_partial_fills:
            return remaining
        if bar.volume <= 0:
            # No volume means no fill. Filling against a zero-volume bar invents
            # a counterparty that demonstrably was not there.
            return 0.0
        return min(remaining, self.max_participation * bar.volume)

    # --- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Clear all state. Used between walk-forward folds."""
        self._resting.clear()
        self.rejections.clear()
        self.expirations.clear()
        # Re-seed any stochastic slippage model so each fold replays identically.
        reset = getattr(self.slippage, "reset", None)
        if callable(reset):
            reset()

    @property
    def stats(self) -> dict[str, int]:
        return {
            "resting": len(self._resting),
            "rejected": len(self.rejections),
            "expired": len(self.expirations),
        }
