"""
The data handler's structural guarantee: a strategy cannot reach a bar that has
not been streamed.

There is deliberately no API that takes a future timestamp, so the tests here
assert the *absence* of a capability as well as the presence of correct
behaviour — the visible window must contain exactly what has streamed and
nothing beyond it.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from evbt.core.events import CorporateActionType, EventType
from evbt.data.frame import DataFrameDataHandler
from tests.conftest import flat_price_bars, trending_bars


class TestStreaming:
    def test_slices_arrive_in_chronological_order(self):
        handler = DataFrameDataHandler(flat_price_bars({"A": 10.0}, n_days=5))

        seen = []
        while handler.has_more():
            for event in handler.next_events():
                if event.type is EventType.MARKET:
                    seen.append(event.timestamp)

        assert seen == sorted(seen) and len(seen) == 5

    def test_a_slice_carries_the_whole_cross_section(self):
        handler = DataFrameDataHandler(
            flat_price_bars({"A": 10.0, "B": 20.0, "C": 30.0}, n_days=2)
        )
        event = [e for e in handler.next_events() if e.type is EventType.MARKET][0]
        assert sorted(event.symbols) == ["A", "B", "C"]

    def test_universe_is_known_up_front(self):
        """The symbol list is not a secret; only the future prices are."""
        handler = DataFrameDataHandler(flat_price_bars({"A": 10.0, "B": 20.0}, n_days=3))
        assert handler.symbols == ["A", "B"]

    def test_exhaustion(self):
        handler = DataFrameDataHandler(flat_price_bars({"A": 10.0}, n_days=2))
        handler.next_events()
        handler.next_events()
        assert not handler.has_more()
        with pytest.raises(StopIteration):
            handler.next_events()


class TestNoLookahead:
    def test_history_contains_only_streamed_bars(self):
        handler = DataFrameDataHandler(
            trending_bars({"A": 100.0}, n_days=10, daily_return=0.01)
        )

        for expected_length in range(1, 11):
            handler.next_events()
            assert len(handler.history("A", 100)) == expected_length

    def test_latest_price_is_the_current_bar_not_a_later_one(self):
        handler = DataFrameDataHandler(
            trending_bars({"A": 100.0}, n_days=5, daily_return=0.10)
        )
        handler.next_events()  # day 0
        assert handler.latest_price("A") == pytest.approx(100.0)
        handler.next_events()  # day 1
        assert handler.latest_price("A") == pytest.approx(110.0)

    def test_series_returns_what_exists_rather_than_padding(self):
        """
        A symbol with five bars genuinely has five bars. Padding would invent
        data, and a strategy needing sixty should check the length and abstain.
        """
        handler = DataFrameDataHandler(flat_price_bars({"A": 10.0}, n_days=10))
        for _ in range(3):
            handler.next_events()
        assert len(handler.series("A", "close", 60)) == 3

    def test_unstreamed_symbol_reports_no_data(self):
        handler = DataFrameDataHandler(flat_price_bars({"A": 10.0}, n_days=2))
        assert not handler.has_data("A")
        assert handler.latest_bar("A") is None
        handler.next_events()
        assert handler.has_data("A")

    def test_cross_section_omits_symbols_with_no_data(self):
        """Omitted rather than NaN, so a caller cannot rank a placeholder."""
        bars = flat_price_bars({"A": 10.0}, n_days=3)
        late = flat_price_bars({"B": 20.0}, n_days=1, start="2024-01-03")
        handler = DataFrameDataHandler(pd.concat([bars, late], ignore_index=True))

        handler.next_events()
        assert handler.cross_section() == {"A": 10.0}

    def test_returns_needs_two_bars(self):
        handler = DataFrameDataHandler(
            trending_bars({"A": 100.0}, n_days=5, daily_return=0.02)
        )
        handler.next_events()
        assert handler.returns("A", 5).size == 0
        handler.next_events()
        assert handler.returns("A", 5) == pytest.approx([0.02])


class TestCorporateActions:
    def test_actions_are_emitted_on_their_ex_date(self):
        bars = flat_price_bars({"A": 10.0}, n_days=5)
        dates = sorted(bars["timestamp"].unique())
        actions = pd.DataFrame(
            [
                {
                    "timestamp": dates[2],
                    "symbol": "A",
                    "action": "DIVIDEND",
                    "cash_amount": 0.5,
                    "split_ratio": 1.0,
                }
            ]
        )
        handler = DataFrameDataHandler(bars, actions)

        emitted = []
        while handler.has_more():
            for event in handler.next_events():
                if event.type is EventType.CORPORATE_ACTION:
                    emitted.append(event)

        assert len(emitted) == 1
        assert emitted[0].timestamp == pd.Timestamp(dates[2]).to_pydatetime()
        assert emitted[0].action is CorporateActionType.DIVIDEND

    def test_actions_precede_the_market_event_in_the_slice(self):
        bars = flat_price_bars({"A": 10.0}, n_days=2)
        dates = sorted(bars["timestamp"].unique())
        actions = pd.DataFrame(
            [{"timestamp": dates[0], "symbol": "A", "action": "SPLIT",
              "cash_amount": 0.0, "split_ratio": 2.0}]
        )
        handler = DataFrameDataHandler(bars, actions)

        kinds = [e.type for e in handler.next_events()]
        assert kinds == [EventType.CORPORATE_ACTION, EventType.MARKET]


class TestValidation:
    def test_missing_columns_rejected(self):
        with pytest.raises(ValueError, match="missing required columns"):
            DataFrameDataHandler(pd.DataFrame({"timestamp": [], "symbol": []}))

    def test_duplicate_bar_rejected(self):
        bars = flat_price_bars({"A": 10.0}, n_days=1)
        with pytest.raises(ValueError, match="duplicate bar"):
            DataFrameDataHandler(pd.concat([bars, bars], ignore_index=True))


class TestVolatility:
    def test_trailing_vol_appears_once_min_periods_is_met(self):
        handler = DataFrameDataHandler(
            trending_bars({"A": 100.0}, n_days=10, daily_return=0.01),
            volatility_window=5,
            volatility_min_periods=3,
        )
        vols = []
        while handler.has_more():
            handler.next_events()
            vols.append(handler.latest_bar("A").volatility)

        assert vols[0] is None and vols[1] is None
        assert vols[-1] is not None

    def test_constant_returns_give_zero_vol(self):
        handler = DataFrameDataHandler(
            trending_bars({"A": 100.0}, n_days=10, daily_return=0.01),
            volatility_window=5,
            volatility_min_periods=3,
        )
        while handler.has_more():
            handler.next_events()
        assert handler.latest_bar("A").volatility == pytest.approx(0.0, abs=1e-12)


class TestRestriction:
    def test_restricted_handler_covers_only_its_window(self):
        handler = DataFrameDataHandler(flat_price_bars({"A": 10.0}, n_days=20))
        sub = handler.restricted_to(datetime(2024, 1, 8), datetime(2024, 1, 12))

        stamps = []
        while sub.has_more():
            for event in sub.next_events():
                if event.type is EventType.MARKET:
                    stamps.append(event.timestamp)

        assert min(stamps) >= datetime(2024, 1, 8)
        assert max(stamps) <= datetime(2024, 1, 12)

    def test_restriction_inherits_the_parent_volatility_estimate(self):
        """
        Otherwise every walk-forward fold restarts its rolling window and spends
        its first weeks with no vol estimate, hence no slippage.
        """
        parent = DataFrameDataHandler(
            trending_bars({"A": 100.0}, n_days=40, daily_return=0.01),
            volatility_window=21,
            volatility_min_periods=5,
        )
        sub = parent.restricted_to(datetime(2024, 2, 12), datetime(2024, 2, 23))
        sub.next_events()

        assert sub.latest_bar("A").volatility is not None


def test_reset_rewinds():
    handler = DataFrameDataHandler(flat_price_bars({"A": 10.0}, n_days=5))
    while handler.has_more():
        handler.next_events()

    handler.reset()
    assert handler.has_more()
    assert handler.latest_bar("A") is None
