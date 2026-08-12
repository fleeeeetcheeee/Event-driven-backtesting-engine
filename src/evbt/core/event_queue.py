"""
The simulation's event queue.

Two properties are non-negotiable, and both are easy to get wrong:

**1. Determinism.** The same inputs must produce a bit-identical run every
time. A plain FIFO queue is not enough once events can be scheduled for future
timestamps, and a bare heap on `(timestamp, priority)` is not enough either —
Python's `heapq` is not a stable sort, so two events with equal keys come out in
an order that depends on heap geometry. We therefore push
`(timestamp, priority, sequence)` where `sequence` is a monotonically
increasing counter. The counter is unique, so the tuple comparison always
resolves before it reaches the event object, and insertion order breaks the
final tie.

**2. No time travel.** Once the simulation clock has advanced past `t`, nothing
may schedule an event *at or before* `t`. That is not a stylistic rule: an
event inserted in the past is exactly what lookahead bias looks like from
inside the engine, and it is otherwise invisible. `EventQueue` raises on it.

Both of these exist because the failure modes are silent. A backtest with a
subtly non-deterministic fill order still produces a Sharpe ratio; it just
produces a different one each run, and you will not notice until you try to
reproduce a result months later.
"""

from __future__ import annotations

import heapq
import itertools
from datetime import datetime
from typing import Iterator, Optional

from evbt.core.events import Event


class TimeTravelError(RuntimeError):
    """Raised when an event is scheduled at or before the current sim time."""


class EventQueue:
    """A deterministic priority queue over `Event`, ordered by (time, priority, seq)."""

    def __init__(self) -> None:
        self._heap: list[tuple[datetime, int, int, Event]] = []
        self._counter = itertools.count()
        self._now: Optional[datetime] = None
        self._pushed = 0
        self._popped = 0

    # --- clock ------------------------------------------------------------

    @property
    def now(self) -> Optional[datetime]:
        """
        Timestamp of the most recently popped event — the simulation's notion
        of "now". `None` before the first pop.
        """
        return self._now

    # --- mutation ---------------------------------------------------------

    def push(self, event: Event, *, allow_now: bool = False) -> None:
        """
        Schedule `event`.

        Events strictly in the past are always a bug and raise. Events stamped
        *at* the current time are the normal case for a causal chain within one
        bar (a MarketEvent at `t` produces a SignalEvent at `t` produces an
        OrderEvent at `t`), so they are permitted — `EventPriority` is what
        keeps them correctly ordered relative to each other. `allow_now` exists
        only to make that intent explicit at call sites that need it; it is not
        a safety switch, since same-timestamp scheduling is legal either way.
        """
        if self._now is not None and event.timestamp < self._now:
            raise TimeTravelError(
                f"cannot schedule {event.type.value} at {event.timestamp} — "
                f"simulation clock is already at {self._now}. An event in the "
                f"past is lookahead bias."
            )
        heapq.heappush(
            self._heap,
            (event.timestamp, int(event.priority), next(self._counter), event),
        )
        self._pushed += 1

    def pop(self) -> Event:
        """Remove and return the next event, advancing the clock to its timestamp."""
        if not self._heap:
            raise IndexError("pop from an empty EventQueue")
        timestamp, _priority, _seq, event = heapq.heappop(self._heap)
        self._now = timestamp
        self._popped += 1
        return event

    # --- inspection -------------------------------------------------------

    def peek_time(self) -> Optional[datetime]:
        """Timestamp of the next event without consuming it. `None` if empty."""
        return self._heap[0][0] if self._heap else None

    def peek(self) -> Optional[Event]:
        """
        The next event without consuming it, and without advancing the clock.

        The engine uses this to detect the end of a phase — "is the next thing
        still a SIGNAL at this timestamp?" — which is how portfolio
        construction knows it has seen the whole cross-section.
        """
        return self._heap[0][3] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __iter__(self) -> Iterator[Event]:
        """Drain the queue in priority order. Consumes it."""
        while self._heap:
            yield self.pop()

    @property
    def stats(self) -> dict[str, int]:
        return {
            "pushed": self._pushed,
            "popped": self._popped,
            "pending": len(self._heap),
        }
