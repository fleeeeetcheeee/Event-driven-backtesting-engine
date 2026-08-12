"""
The queue's two guarantees: deterministic ordering, and no scheduling into the
past. Both failure modes are silent, so both are tested directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from evbt.core.event_queue import EventQueue, TimeTravelError
from evbt.core.events import (
    CorporateActionEvent,
    CorporateActionType,
    EventType,
    MarketEvent,
    OrderSide,
    SignalDirection,
    SignalEvent,
)
from tests.conftest import make_bar, make_fill, make_order

T0 = datetime(2024, 1, 2)
T1 = datetime(2024, 1, 3)


def _all_event_types_at(timestamp):
    """One event of each type, constructed in deliberately scrambled order."""
    return [
        SignalEvent(timestamp=timestamp, symbol="X", direction=SignalDirection.LONG),
        make_fill(timestamp=timestamp),
        MarketEvent(timestamp=timestamp, bars={"X": make_bar(timestamp=timestamp)}),
        make_order(timestamp=timestamp),
        CorporateActionEvent(
            timestamp=timestamp,
            symbol="X",
            action=CorporateActionType.DIVIDEND,
            cash_amount=0.5,
        ),
    ]


def test_events_at_one_timestamp_come_out_in_trading_day_order():
    queue = EventQueue()
    for event in _all_event_types_at(T0):
        queue.push(event)

    order = [event.type for event in queue]
    assert order == [
        EventType.CORPORATE_ACTION,
        EventType.MARKET,
        EventType.FILL,
        EventType.SIGNAL,
        EventType.ORDER,
    ]


def test_timestamp_dominates_priority():
    """An earlier low-priority event still precedes a later high-priority one."""
    queue = EventQueue()
    queue.push(make_order(timestamp=T0))  # lowest priority, earlier
    queue.push(
        CorporateActionEvent(
            timestamp=T1,
            symbol="X",
            action=CorporateActionType.DIVIDEND,
            cash_amount=1.0,
        )
    )  # highest priority, later

    assert [e.timestamp for e in queue] == [T0, T1]


def test_equal_key_events_preserve_insertion_order():
    """
    Ties break on insertion sequence, not on heap geometry.

    Without the sequence counter this passes intermittently — which is the
    worst possible outcome, because a run reproduces until it doesn't.
    """
    queue = EventQueue()
    for i in range(50):
        queue.push(make_order(order_id=f"O{i:03d}", timestamp=T0))

    assert [e.order_id for e in queue] == [f"O{i:03d}" for i in range(50)]


def test_repeated_runs_are_identical():
    def drain():
        queue = EventQueue()
        for event in _all_event_types_at(T0):
            queue.push(event)
        for event in _all_event_types_at(T1):
            queue.push(event)
        return [(e.timestamp, e.type) for e in queue]

    assert drain() == drain() == drain()


def test_scheduling_into_the_past_raises():
    queue = EventQueue()
    queue.push(make_order(timestamp=T1))
    queue.pop()  # clock is now T1

    with pytest.raises(TimeTravelError, match="lookahead"):
        queue.push(make_order(timestamp=T0, order_id="O2"))


def test_scheduling_at_the_current_time_is_allowed():
    """The normal causal chain within one bar: market -> signal -> order at t."""
    queue = EventQueue()
    queue.push(MarketEvent(timestamp=T0, bars={"X": make_bar(timestamp=T0)}))
    queue.pop()

    queue.push(SignalEvent(timestamp=T0, symbol="X", direction=SignalDirection.LONG))
    assert len(queue) == 1


def test_clock_advances_only_on_pop():
    queue = EventQueue()
    assert queue.now is None
    queue.push(make_order(timestamp=T0))
    assert queue.now is None
    queue.pop()
    assert queue.now == T0


def test_peek_does_not_consume_or_advance_clock():
    queue = EventQueue()
    queue.push(make_order(timestamp=T0))

    peeked = queue.peek()
    assert peeked is not None and peeked.timestamp == T0
    assert len(queue) == 1
    assert queue.now is None
    assert queue.peek_time() == T0


def test_empty_queue_behaviour():
    queue = EventQueue()
    assert not queue
    assert queue.peek() is None
    assert queue.peek_time() is None
    with pytest.raises(IndexError):
        queue.pop()


def test_stats_track_flow():
    queue = EventQueue()
    queue.push(make_order(timestamp=T0))
    queue.push(make_order(timestamp=T1, order_id="O2"))
    queue.pop()
    assert queue.stats == {"pushed": 2, "popped": 1, "pending": 1}
