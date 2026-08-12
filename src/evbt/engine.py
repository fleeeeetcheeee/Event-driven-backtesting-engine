"""
The event loop.

Structurally this is thirty lines: pull the next timestamp's events into the
queue, drain the queue, repeat. All the judgement lives in `EventPriority`,
which fixes what happens in what order within a timestamp, and in the broker,
which decides what can fill when. The loop itself deliberately makes no
decisions — if it did, they would be invisible.

    while data remains:
        push this timestamp's corporate actions and market event
        while the queue is non-empty:
            pop the highest-priority event and dispatch it

The queue empties at the end of every timestamp. Resting orders survive because
they live in the broker, not the queue — an order is not a pending event, it is
a standing instruction, and conflating the two is how backtests end up filling
yesterday's cancelled orders.

Phase completion
----------------
One subtlety. Portfolio construction needs the *whole* cross-section of signals
for a timestamp, not one at a time, so the engine has to know when the signal
phase is over. It peeks: after handling a signal, if the next queued event is
not another signal at the same timestamp, the phase has ended and the book is
built. This keeps signals as real events — inspectable in a trace — rather than
a list passed around behind the queue's back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from evbt.core.event_queue import EventQueue
from evbt.core.events import (
    CorporateActionEvent,
    Event,
    EventType,
    FillEvent,
    MarketEvent,
    MarkToMarketEvent,
    OrderEvent,
    OrderExpiredEvent,
    OrderRejectedEvent,
    SignalEvent,
)
from evbt.data.base import DataHandler
from evbt.execution.broker import SimulatedBroker
from evbt.portfolio.construction import PortfolioConstructor
from evbt.portfolio.portfolio import Portfolio
from evbt.strategy.base import Strategy, StrategyContext

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Everything a run produced. Analytics operate on this, not on the engine."""

    equity_curve: pd.DataFrame
    fills: pd.DataFrame
    final_positions: pd.DataFrame
    initial_capital: float
    final_nav: float
    start: Optional[datetime]
    end: Optional[datetime]
    n_events: int
    n_orders: int
    n_fills: int
    n_rejections: int
    n_expirations: int
    risk_violations: dict[str, int] = field(default_factory=dict)
    rejections: list[OrderRejectedEvent] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        return self.final_nav / self.initial_capital - 1.0

    def returns(self) -> pd.Series:
        """Period-over-period NAV returns, the input to every metric."""
        if self.equity_curve.empty:
            return pd.Series(dtype=float)
        return self.equity_curve["nav"].pct_change().dropna()

    def summary(self) -> str:  # pragma: no cover - presentation only
        lines = [
            f"period          {self.start} -> {self.end}",
            f"initial capital {self.initial_capital:,.2f}",
            f"final NAV       {self.final_nav:,.2f}",
            f"total return    {self.total_return:+.2%}",
            f"events          {self.n_events:,}",
            f"orders / fills  {self.n_orders:,} / {self.n_fills:,}",
            f"rejected        {self.n_rejections:,}",
            f"expired unfilled{self.n_expirations:,}",
        ]
        if self.risk_violations:
            bound = ", ".join(f"{k}={v}" for k, v in sorted(self.risk_violations.items()))
            lines.append(f"limits bound    {bound}")
        return "\n".join(lines)


class Backtest:
    """
    Wires the components together and runs the loop.

    Parameters
    ----------
    liquidate_unsignalled
        Passed through to the constructor on every rebalance. See
        `PortfolioConstructor.build_orders` — it is the difference between "the
        strategy states its whole book each time" and "the strategy signals only
        changes", and picking the wrong one silently runs a different strategy.
    cancel_open_orders_on_rebalance
        Cancel anything still resting before generating a new book. True is
        right for a periodic rebalancer: a stale order from the previous
        rebalance is chasing a target that no longer exists. False is right when
        large orders are meant to be worked across several bars.
    """

    def __init__(
        self,
        data: DataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        broker: SimulatedBroker,
        constructor: PortfolioConstructor,
        *,
        liquidate_unsignalled: bool = False,
        cancel_open_orders_on_rebalance: bool = True,
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.broker = broker
        self.constructor = constructor
        self.liquidate_unsignalled = liquidate_unsignalled
        self.cancel_open_orders_on_rebalance = cancel_open_orders_on_rebalance

        self.queue = EventQueue()
        self._pending_signals: list[SignalEvent] = []
        self._current_market: Optional[MarketEvent] = None
        self._n_events = 0
        self._n_orders = 0
        self._n_fills = 0
        self._start: Optional[datetime] = None
        self._end: Optional[datetime] = None

    # --- the loop ----------------------------------------------------------

    def run(self, progress: bool = False) -> BacktestResult:
        ctx = self._context(self.data.current_time or datetime.min)
        self.strategy.initialize(ctx)

        iterator = self._timestamps()
        if progress:
            iterator = self._with_progress(iterator)

        for _ in iterator:
            while self.queue:
                self._dispatch(self.queue.pop())

        self.strategy.finalize(self._context(self._end or datetime.min))
        return self._result()

    def _timestamps(self):
        """Feed one timestamp's events into the queue at a time."""
        while self.data.has_more():
            events = self.data.next_events()
            for event in events:
                self.queue.push(event)
                if self._start is None:
                    self._start = event.timestamp
                self._end = event.timestamp
            yield events

    @staticmethod
    def _with_progress(iterator):  # pragma: no cover - cosmetic
        try:
            from tqdm import tqdm
        except ImportError:
            return iterator
        return tqdm(iterator, desc="backtest", unit="bar")

    # --- dispatch ----------------------------------------------------------

    def _dispatch(self, event: Event) -> None:
        self._n_events += 1
        kind = event.type

        if kind is EventType.CORPORATE_ACTION:
            self.portfolio.on_corporate_action(event)  # type: ignore[arg-type]

        elif kind is EventType.MARKET:
            self._on_market(event)  # type: ignore[arg-type]

        elif kind is EventType.FILL:
            self._on_fill(event)  # type: ignore[arg-type]

        elif kind is EventType.MARK_TO_MARKET:
            self._on_mark_to_market(event)  # type: ignore[arg-type]

        elif kind is EventType.SIGNAL:
            self._on_signal(event)  # type: ignore[arg-type]

        elif kind is EventType.ORDER:
            self._on_order(event)  # type: ignore[arg-type]

        elif kind in (EventType.ORDER_REJECTED, EventType.ORDER_EXPIRED):
            self._on_order_terminal(event)  # type: ignore[arg-type]

        else:  # pragma: no cover - unreachable while EventType is exhaustive
            raise ValueError(f"unhandled event type {kind}")

    def _on_market(self, event: MarketEvent) -> None:
        """
        Today's prices arrive. Resting orders — all of them from earlier bars —
        get their chance, then the book is scheduled for valuation.
        """
        self._current_market = event
        for produced in self.broker.on_market(event):
            self.queue.push(produced)
        self.queue.push(MarkToMarketEvent(timestamp=event.timestamp, bars=event.bars))

    def _on_fill(self, fill: FillEvent) -> None:
        self.portfolio.on_fill(fill)
        self._n_fills += 1
        self.strategy.on_fill(fill, self._context(fill.timestamp))

    def _on_mark_to_market(self, event: MarkToMarketEvent) -> None:
        """
        Value the book, then let the strategy look at the world.

        The strategy runs here rather than on `MarketEvent` so that it sees a
        portfolio that already reflects today's fills and today's marks. A
        strategy sizing off a stale NAV is a subtle, permanent, compounding
        error, and it is invisible in the output.
        """
        self.portfolio.on_mark_to_market(event)

        if self._current_market is None:
            return

        ctx = self._context(event.timestamp)
        signals = list(self.strategy.on_bar(self._current_market, ctx))
        for signal in signals:
            if signal.timestamp != event.timestamp:
                # A strategy that stamps a signal with anything but "now" is
                # either confused or cheating; both are worth stopping for.
                raise ValueError(
                    f"strategy {self.strategy.name} emitted a signal for "
                    f"{signal.symbol} stamped {signal.timestamp}, but the "
                    f"current time is {event.timestamp}"
                )
            self.queue.push(signal)

    def _on_signal(self, signal: SignalEvent) -> None:
        """
        Accumulate signals, then build the book once the phase completes.

        Completion is detected by peeking: if the next queued event is not
        another signal at this timestamp, the cross-section is complete.
        """
        self._pending_signals.append(signal)

        nxt = self.queue.peek()
        phase_continues = (
            nxt is not None
            and nxt.type is EventType.SIGNAL
            and nxt.timestamp == signal.timestamp
        )
        if not phase_continues:
            self._rebalance(signal.timestamp)

    def _rebalance(self, timestamp: datetime) -> None:
        signals = self._pending_signals
        self._pending_signals = []

        if self.cancel_open_orders_on_rebalance:
            self.broker.cancel_all()

        orders = self.constructor.build_orders(
            signals,
            self.portfolio,
            timestamp,
            self.broker.next_order_id,
            liquidate_unsignalled=self.liquidate_unsignalled,
        )
        for order in orders:
            self.queue.push(order)

    def _on_order(self, order: OrderEvent) -> None:
        price = self.portfolio.mark_price(order.symbol)
        if price is not None:
            affordable, reason = self.portfolio.can_afford(order, price)
            if not affordable:
                self.queue.push(self.broker.reject(order, reason, order.timestamp))
                return
        self.broker.submit(order)
        self._n_orders += 1

    def _on_order_terminal(self, event: Event) -> None:
        """
        Rejections and expirations are recorded by the broker when created; this
        handler exists so they pass through the queue and appear in a trace in
        the right place, rather than being an out-of-band side effect.
        """
        if isinstance(event, OrderRejectedEvent):
            log.debug("order %s rejected: %s", event.order.order_id, event.reason)
        elif isinstance(event, OrderExpiredEvent):
            log.debug(
                "order %s expired with %g/%g filled",
                event.order.order_id,
                event.filled_quantity,
                event.order.quantity,
            )

    # --- helpers -----------------------------------------------------------

    def _context(self, now: datetime) -> StrategyContext:
        return StrategyContext(data=self.data, portfolio=self.portfolio, now=now)

    def _result(self) -> BacktestResult:
        risk = self.constructor.risk
        return BacktestResult(
            equity_curve=self.portfolio.equity_curve(),
            fills=self.portfolio.fills_frame(),
            final_positions=self.portfolio.positions_frame(),
            initial_capital=self.portfolio.initial_cash,
            final_nav=self.portfolio.nav,
            start=self._start,
            end=self._end,
            n_events=self._n_events,
            n_orders=self._n_orders,
            n_fills=self._n_fills,
            n_rejections=len(self.broker.rejections),
            n_expirations=len(self.broker.expirations),
            risk_violations=risk.summary() if risk is not None else {},
            rejections=list(self.broker.rejections),
        )
