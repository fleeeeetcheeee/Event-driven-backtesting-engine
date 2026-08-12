"""
The event vocabulary of the simulation.

Everything that happens in a backtest is one of these objects moving through
the queue. Nothing in the engine reads market data or mutates the portfolio
outside of an event — that is what makes the simulation auditable, and it is
the whole reason for not writing this vectorised.

The causal chain, and the direction it is allowed to flow:

    MarketEvent  ->  SignalEvent  ->  OrderEvent  ->  FillEvent  ->  portfolio
    (data)           (strategy)      (portfolio)     (broker)

Plus CorporateActionEvent, which is injected by the data layer and consumed by
the portfolio directly. It has no strategy or order stage: a dividend is
something that happens *to* you.

Sign conventions
----------------
Orders carry a positive `quantity` and an explicit `side`. Positions carry a
*signed* quantity (negative = short). Mixing these two conventions is one of
the most common accounting bugs in a backtester, so the split is enforced by
construction: `OrderEvent.__post_init__` rejects a non-positive quantity.

Timestamp semantics
-------------------
Every event's timestamp is the moment the event becomes *knowable*. A
MarketEvent for a daily bar is stamped at that bar's close. This matters: a
signal computed from bar `t` is itself stamped at `t`'s close, so the broker
can tell that it must not fill against bar `t`. See `execution.broker`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Mapping, Optional


class EventType(str, Enum):
    CORPORATE_ACTION = "CORPORATE_ACTION"
    MARKET = "MARKET"
    FILL = "FILL"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_EXPIRED = "ORDER_EXPIRED"
    MARK_TO_MARKET = "MARK_TO_MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"


class EventPriority(IntEnum):
    """
    Tie-break order for events sharing a timestamp. Lower runs first.

    This ordering *is* the trading day. Each phase depends on the one above it
    having already happened, and the whole sequence exists in the queue rather
    than being buried in an engine method so that a trace of the run reads as
    a literal narrative of the day:

      1. CORPORATE_ACTION  splits restate the share count, and dividends pay on
                           the position held *going into* the ex-date. Both must
                           land before anything trades today.
      2. MARKET            today's bar becomes visible.
      3. FILL              resting orders — placed on an *earlier* bar, never
                           this one — match against today's prices.
      4. ORDER_EXPIRED /   unfilled DAY orders die; rejections are recorded.
         ORDER_REJECTED
      5. MARK_TO_MARKET    the book is valued at today's closes, carry is
                           accrued, and the equity curve gains a row.
      6. SIGNAL            the strategy sees today's bar and the freshly marked
                           portfolio.
      7. ORDER             orders born from those signals rest until tomorrow.

    Two placements carry the design's weight. FILL before MARK_TO_MARKET means
    the equity curve includes today's trades. SIGNAL after MARK_TO_MARKET means
    a strategy sizing against NAV sees today's NAV, not yesterday's — the
    alternative silently sizes every position off stale capital, which compounds
    into a real distortion over a long run.
    """

    CORPORATE_ACTION = 0
    MARKET = 1
    FILL = 2
    ORDER_REJECTED = 3
    ORDER_EXPIRED = 3
    MARK_TO_MARKET = 4
    SIGNAL = 5
    ORDER = 6


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        """+1 for BUY, -1 for SELL. Converts an order into a position delta."""
        return 1 if self is OrderSide.BUY else -1


class OrderType(str, Enum):
    MARKET = "MARKET"        # fills at the next bar's open, plus costs
    LIMIT = "LIMIT"          # fills only if a later bar's range reaches the limit
    MARKET_ON_OPEN = "MOO"   # fills at the next bar's open, no intrabar drift
    MARKET_ON_CLOSE = "MOC"  # fills at the next bar's close


class TimeInForce(str, Enum):
    DAY = "DAY"    # cancelled if unfilled at the end of the next bar
    GTC = "GTC"    # rests until filled or explicitly cancelled


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"   # close whatever is held, direction-agnostic


class CorporateActionType(str, Enum):
    DIVIDEND = "DIVIDEND"
    SPLIT = "SPLIT"


@dataclass(frozen=True)
class Bar:
    """
    One OHLCV bar. **Unadjusted** prices — see the README on why.

    Storing raw prices and replaying splits/dividends as events is more work
    than using an adjusted series, but an adjusted series makes exact cash
    accounting impossible: you cannot credit a $0.24 dividend on a share count
    that has been retroactively rescaled by every split since.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Bar-level volatility proxy, used by the slippage model. Supplied by the
    # data layer (typically a trailing realised vol), not derived here, so the
    # engine never has to look at more than one bar at a time.
    volatility: Optional[float] = None

    @property
    def typical_price(self) -> float:
        """(H+L+C)/3 — the standard stand-in for VWAP when VWAP is unavailable."""
        return (self.high + self.low + self.close) / 3.0

    def contains_price(self, price: float) -> bool:
        """Whether `price` traded during this bar, used for limit-order matching."""
        return self.low <= price <= self.high


@dataclass
class Event:
    """Base class. `timestamp` is when the event becomes knowable."""

    timestamp: datetime

    @property
    def type(self) -> EventType:  # pragma: no cover - overridden by every subclass
        raise NotImplementedError

    @property
    def priority(self) -> EventPriority:
        return EventPriority[self.type.name]


@dataclass
class MarketEvent(Event):
    """
    All market data for one timestamp, delivered as a single barrier.

    Cross-sectional strategies need the whole slice at once — you cannot rank a
    universe on a signal if symbols arrive one at a time — so the unit of market
    data is the slice, not the individual bar. For a tick-level run the slice
    simply has one entry.
    """

    bars: Mapping[str, Bar]

    @property
    def type(self) -> EventType:
        return EventType.MARKET

    @property
    def symbols(self) -> list[str]:
        return list(self.bars.keys())


@dataclass
class MarkToMarketEvent(Event):
    """
    Value the book at today's closes and write a row of the equity curve.

    An explicit event rather than a side effect of handling `MarketEvent`,
    because *when* valuation happens relative to fills and signals is a design
    decision with real consequences, and design decisions belong somewhere a
    reader can see them. Its position in `EventPriority` is that decision.
    """

    bars: Mapping[str, Bar]

    @property
    def type(self) -> EventType:
        return EventType.MARK_TO_MARKET


@dataclass
class CorporateActionEvent(Event):
    """
    A split or a cash dividend, stamped at its **ex-date**.

    Ex-date is the right anchor for both. A holder of record is entitled to the
    dividend if they held the shares going into the ex-date open, which is why
    this event is prioritised ahead of MARKET and FILL: the position it pays on
    is the position carried over from the previous bar.

    `split_ratio` follows the market convention: 2.0 means a 2-for-1, so share
    count multiplies by 2.0 and cost basis divides by 2.0.
    """

    symbol: str
    action: CorporateActionType
    cash_amount: float = 0.0   # dividend per share, in currency
    split_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.action is CorporateActionType.SPLIT and self.split_ratio <= 0:
            raise ValueError(f"split_ratio must be positive, got {self.split_ratio}")
        if self.action is CorporateActionType.DIVIDEND and self.cash_amount < 0:
            raise ValueError(f"dividend must be non-negative, got {self.cash_amount}")

    @property
    def type(self) -> EventType:
        return EventType.CORPORATE_ACTION


@dataclass
class SignalEvent(Event):
    """
    A strategy's *intent*, deliberately stripped of any notion of size.

    The separation matters. A strategy says "I want to be long AAPL with
    conviction 0.8"; the portfolio decides what that is worth in shares given
    NAV, risk limits, and existing positions. Collapsing the two — having the
    strategy emit share counts directly — is how backtests end up with position
    sizing silently baked into alpha, and makes the same signal impossible to
    re-run at a different capital base.

    `strength` is a signed conviction in [-1, 1] for cross-sectional strategies
    (it is the thing that gets ranked), or simply 1.0 for a binary signal.
    """

    symbol: str
    direction: SignalDirection
    strength: float = 1.0
    # Free-form provenance so a fill can be traced back to the rule that caused
    # it. Costs nothing and makes post-mortems on a 10-year run tractable.
    metadata: dict = field(default_factory=dict)

    @property
    def type(self) -> EventType:
        return EventType.SIGNAL


@dataclass
class OrderEvent(Event):
    """
    An instruction to the broker. `timestamp` is when the order was *submitted*,
    which is what the no-lookahead rule is enforced against.
    """

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(
                f"order quantity must be positive (direction lives in `side`), "
                f"got {self.quantity}"
            )
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT order requires a limit_price")

    @property
    def type(self) -> EventType:
        return EventType.ORDER

    @property
    def signed_quantity(self) -> float:
        return self.side.sign * self.quantity


@dataclass
class FillEvent(Event):
    """
    An execution. Carries its costs decomposed rather than pre-netted into the
    price, because "what did this strategy pay, and to whom" is a question the
    analytics layer has to answer separately from "what did it earn".

    `fill_price` is the price actually transacted at, i.e. it already includes
    spread and slippage. Commission is charged on top. `impact_cost` and
    `spread_cost` are reported in currency for attribution only — folding them
    into the price twice is an easy and expensive mistake.
    """

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float          # shares actually filled; may be < order quantity
    fill_price: float
    commission: float = 0.0
    spread_cost: float = 0.0
    impact_cost: float = 0.0
    slippage_cost: float = 0.0
    # Price before any execution cost was applied — the arrival/decision price.
    # Implementation shortfall is measured against this.
    reference_price: Optional[float] = None
    is_partial: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def type(self) -> EventType:
        return EventType.FILL

    @property
    def signed_quantity(self) -> float:
        return self.side.sign * self.quantity

    @property
    def gross_value(self) -> float:
        """Notional transacted, always positive."""
        return self.quantity * self.fill_price

    @property
    def cash_delta(self) -> float:
        """
        Effect on cash. A buy consumes cash (negative), a sell releases it.
        Commission always consumes cash regardless of side.
        """
        return -self.side.sign * self.gross_value - self.commission

    @property
    def total_cost(self) -> float:
        """All-in execution cost in currency, for attribution."""
        return self.commission + self.spread_cost + self.impact_cost + self.slippage_cost


@dataclass
class OrderRejectedEvent(Event):
    """
    A rejection is a first-class event, not a log line.

    Rejections are how a backtest tells you it could not do what the strategy
    asked — insufficient cash, a breached risk limit, a halted symbol. Swallowing
    them silently produces a backtest of a strategy that was never actually run,
    so the engine records every one and the report surfaces the count.
    """

    order: OrderEvent
    reason: str

    @property
    def type(self) -> EventType:
        return EventType.ORDER_REJECTED


@dataclass
class OrderExpiredEvent(Event):
    """A DAY order that reached the end of its life unfilled or partly filled."""

    order: OrderEvent
    filled_quantity: float = 0.0

    @property
    def type(self) -> EventType:
        return EventType.ORDER_EXPIRED
