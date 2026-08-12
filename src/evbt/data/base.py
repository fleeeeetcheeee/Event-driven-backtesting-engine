"""
Market data delivery, and the structural no-lookahead guarantee.

A backtester cannot prevent lookahead bias by asking strategies politely not to
peek. It has to make peeking impossible. The rule here:

    A strategy never receives a data container. It receives a `DataHandler`
    whose accessors can only return bars the handler has already streamed.

There is no method on `DataHandler` that takes a future timestamp, and no
attribute holding the full series. The history buffer is appended to as the
simulation advances, so `handler.history("AAPL", 20)` on 2010-03-01 returns the
20 bars up to 2010-03-01 and there is no argument you can pass to get the 21st.

This is the difference that matters versus a vectorised backtest. In pandas,
`df["close"].rolling(20).mean()` is one character away from
`df["close"].rolling(20).mean().shift(-1)`, and the second one is a strategy
that trades on tomorrow's average. Both produce a plausible equity curve.

Data is delivered in *slices*: everything knowable at one timestamp, handed
over at once. Cross-sectional strategies rank a universe, so they need the full
cross-section simultaneously; a per-symbol stream would force the strategy to
buffer, and buffering inside strategies is where lookahead creeps back in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Optional, Sequence

import numpy as np

from evbt.core.events import Bar, CorporateActionEvent, Event, MarketEvent


@dataclass
class MarketSlice:
    """Everything that becomes knowable at a single timestamp."""

    timestamp: datetime
    bars: dict[str, Bar]
    actions: list[CorporateActionEvent] = field(default_factory=list)


class DataHandler(ABC):
    """
    Base class for market data sources.

    Subclasses implement `_slices()`, a chronologically ordered generator of
    `MarketSlice`. Everything else — buffering, history, the accessors the
    strategy sees — is handled here so that no subclass can accidentally widen
    the visibility window.
    """

    def __init__(self, max_history: int = 512) -> None:
        self._max_history = max_history
        self._history: dict[str, deque[Bar]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self._current_time: Optional[datetime] = None
        self._iter: Optional[Iterator[MarketSlice]] = None
        self._buffered: Optional[MarketSlice] = None
        self._exhausted = False

    # --- subclass contract -------------------------------------------------

    @abstractmethod
    def _slices(self) -> Iterator[MarketSlice]:
        """Yield slices in strictly increasing timestamp order."""

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """Every symbol this handler can ever emit. Known up front by design —
        the universe is not a secret, only the future prices are."""

    # --- streaming ---------------------------------------------------------

    def _advance(self) -> None:
        """Pull the next slice into the buffer if the buffer is empty."""
        if self._buffered is not None or self._exhausted:
            return
        if self._iter is None:
            self._iter = self._slices()
        try:
            self._buffered = next(self._iter)
        except StopIteration:
            self._exhausted = True

    def has_more(self) -> bool:
        self._advance()
        return self._buffered is not None

    def peek_time(self) -> Optional[datetime]:
        """Timestamp of the next slice, without consuming it."""
        self._advance()
        return self._buffered.timestamp if self._buffered else None

    def next_events(self) -> list[Event]:
        """
        Consume the next slice and return its events, ready to be queued.

        Corporate actions come first in the returned list; the queue's priority
        ordering enforces this independently, but returning them in causal order
        keeps the sequence obvious when reading a trace.

        Side effect: the slice's bars enter the history buffer. This is the
        moment "now" advances, and the only moment those bars become visible.
        """
        self._advance()
        if self._buffered is None:
            raise StopIteration("no more market data")

        slice_ = self._buffered
        self._buffered = None

        if self._current_time is not None and slice_.timestamp <= self._current_time:
            raise ValueError(
                f"data slices must strictly increase in time: got "
                f"{slice_.timestamp} after {self._current_time}"
            )
        self._current_time = slice_.timestamp

        for symbol, bar in slice_.bars.items():
            self._history[symbol].append(bar)

        events: list[Event] = list(slice_.actions)
        if slice_.bars:
            events.append(MarketEvent(timestamp=slice_.timestamp, bars=dict(slice_.bars)))
        return events

    # --- accessors available to strategies ---------------------------------
    #
    # Everything below reads the history buffer, which by construction holds
    # only already-streamed bars. There is deliberately no `bar_at(timestamp)`.

    @property
    def current_time(self) -> Optional[datetime]:
        return self._current_time

    def latest_bar(self, symbol: str) -> Optional[Bar]:
        """Most recent bar for `symbol`, or None if it has never traded yet."""
        buf = self._history.get(symbol)
        return buf[-1] if buf else None

    def latest_price(self, symbol: str) -> Optional[float]:
        bar = self.latest_bar(symbol)
        return bar.close if bar else None

    def has_data(self, symbol: str) -> bool:
        """Whether `symbol` has traded at least once by now."""
        return bool(self._history.get(symbol))

    def history(self, symbol: str, n: int = 1) -> list[Bar]:
        """The most recent `n` bars, oldest first. Shorter than `n` early on."""
        buf = self._history.get(symbol)
        if not buf:
            return []
        return list(buf)[-n:]

    def series(self, symbol: str, field_name: str = "close", n: int = 1) -> np.ndarray:
        """
        One field of the most recent `n` bars as a float array, oldest first.

        Returns whatever is available rather than padding or raising: a symbol
        that IPO'd last week genuinely has five bars of history, and a strategy
        that needs sixty should check the length and abstain. Padding with NaN
        or with the first value invents data.
        """
        bars = self.history(symbol, n)
        return np.array([getattr(b, field_name) for b in bars], dtype=float)

    def returns(self, symbol: str, n: int = 1) -> np.ndarray:
        """Simple close-to-close returns over the most recent `n+1` bars."""
        closes = self.series(symbol, "close", n + 1)
        if closes.size < 2:
            return np.array([], dtype=float)
        return np.diff(closes) / closes[:-1]

    def cross_section(
        self, symbols: Optional[Sequence[str]] = None, field_name: str = "close"
    ) -> dict[str, float]:
        """
        Latest value of `field_name` for each symbol that currently has data.

        The workhorse for cross-sectional strategies. Symbols with no data yet
        are omitted rather than given a NaN, so a caller that ranks the result
        cannot accidentally rank a placeholder.
        """
        universe = symbols if symbols is not None else self._history.keys()
        out: dict[str, float] = {}
        for symbol in universe:
            bar = self.latest_bar(symbol)
            if bar is not None:
                out[symbol] = float(getattr(bar, field_name))
        return out

    def reset(self) -> None:
        """Rewind to the start. Used by the walk-forward runner between folds."""
        self._history.clear()
        self._current_time = None
        self._iter = None
        self._buffered = None
        self._exhausted = False
