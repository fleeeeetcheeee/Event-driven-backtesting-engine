"""
The workhorse data handler: an in-memory long-format DataFrame.

Every other source (Parquet store, CSV, synthetic generator) normalises into
this shape and delegates here, so there is exactly one implementation of the
bar-construction and slice-ordering logic to get right.

Expected `bars` schema (long format, one row per symbol per timestamp):

    timestamp  symbol  open  high  low  close  volume

Optional `actions` schema:

    timestamp  symbol  action    cash_amount  split_ratio
                       DIVIDEND  0.24         1.0
                       SPLIT     0.0          4.0

`timestamp` on an action is its **ex-date**.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from evbt.core.events import Bar, CorporateActionEvent, CorporateActionType
from evbt.data.base import DataHandler, MarketSlice

BAR_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]


class DataFrameDataHandler(DataHandler):
    """
    Streams slices from a pandas DataFrame held in memory.

    Fine up to a few million rows, which covers a 30-year daily backtest on a
    1,000-name universe (~7.5M rows) on a normal laptop. Beyond that — intraday,
    or tick — stream from Parquet in date chunks instead; the `DataHandler`
    contract is identical, which is the point of having the contract.
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        actions: Optional[pd.DataFrame] = None,
        *,
        max_history: int = 512,
        volatility_window: int = 21,
        volatility_min_periods: int = 5,
    ) -> None:
        super().__init__(max_history=max_history)

        missing = set(BAR_COLUMNS) - set(bars.columns)
        if missing:
            raise ValueError(f"bars is missing required columns: {sorted(missing)}")

        # A caller may supply `volatility` precomputed. `restricted_to` does
        # exactly that, so a walk-forward fold inherits the vol estimate from
        # the full history instead of restarting its rolling window at the fold
        # boundary — which would leave the first few weeks of every fold with
        # no vol estimate and therefore no slippage.
        carried = BAR_COLUMNS + (["volatility"] if "volatility" in bars.columns else [])
        frame = bars.loc[:, carried].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.sort_values(["timestamp", "symbol"], kind="mergesort")

        duplicated = frame.duplicated(subset=["timestamp", "symbol"])
        if duplicated.any():
            example = frame.loc[duplicated, ["timestamp", "symbol"]].iloc[0]
            raise ValueError(
                f"duplicate bar for {example['symbol']} at {example['timestamp']}; "
                "a symbol may appear at most once per timestamp"
            )

        if "volatility" not in frame.columns:
            frame["volatility"] = self._trailing_volatility(
                frame, volatility_window, volatility_min_periods
            )
        self._frame = frame.reset_index(drop=True)
        self._symbols = sorted(frame["symbol"].unique().tolist())
        self._actions_by_time = self._index_actions(actions)

    # --- construction helpers ---------------------------------------------

    @staticmethod
    def _trailing_volatility(
        frame: pd.DataFrame, window: int, min_periods: int
    ) -> pd.Series:
        """
        Per-symbol trailing standard deviation of close-to-close returns.

        Point-in-time safe without needing a shift: the return at bar `t` is
        computed from closes at `t-1` and `t`, both of which are known at `t`'s
        close, and the rolling window looks strictly backwards. The engine only
        ever consumes this to size slippage on a fill at `t+1` or later.

        Returned in the same row order as `frame`, aligned by index.
        """
        by_symbol = frame.groupby("symbol", sort=False)["close"]
        returns = by_symbol.pct_change(fill_method=None)
        return (
            returns.groupby(frame["symbol"], sort=False)
            .rolling(window=window, min_periods=min_periods)
            .std()
            .reset_index(level=0, drop=True)
        )

    @staticmethod
    def _index_actions(
        actions: Optional[pd.DataFrame],
    ) -> dict[datetime, list[CorporateActionEvent]]:
        if actions is None or actions.empty:
            return {}

        required = {"timestamp", "symbol", "action"}
        missing = required - set(actions.columns)
        if missing:
            raise ValueError(f"actions is missing required columns: {sorted(missing)}")

        frame = actions.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.sort_values(["timestamp", "symbol"], kind="mergesort")

        indexed: dict[datetime, list[CorporateActionEvent]] = {}
        for row in frame.itertuples(index=False):
            action = CorporateActionType(str(row.action).upper())
            event = CorporateActionEvent(
                timestamp=row.timestamp.to_pydatetime(),
                symbol=row.symbol,
                action=action,
                cash_amount=float(getattr(row, "cash_amount", 0.0) or 0.0),
                split_ratio=float(getattr(row, "split_ratio", 1.0) or 1.0),
            )
            indexed.setdefault(event.timestamp, []).append(event)
        return indexed

    # --- DataHandler contract ---------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def _slices(self) -> Iterator[MarketSlice]:
        for timestamp, group in self._frame.groupby("timestamp", sort=True):
            bars: dict[str, Bar] = {}
            for row in group.itertuples(index=False):
                volatility = (
                    None
                    if row.volatility is None or np.isnan(row.volatility)
                    else float(row.volatility)
                )
                bars[row.symbol] = Bar(
                    symbol=row.symbol,
                    timestamp=timestamp.to_pydatetime(),
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                    volatility=volatility,
                )
            yield MarketSlice(
                timestamp=timestamp.to_pydatetime(),
                bars=bars,
                actions=list(self._actions_by_time.get(timestamp.to_pydatetime(), [])),
            )

    # --- convenience -------------------------------------------------------

    @property
    def start(self) -> datetime:
        return self._frame["timestamp"].iloc[0].to_pydatetime()

    @property
    def end(self) -> datetime:
        return self._frame["timestamp"].iloc[-1].to_pydatetime()

    def restricted_to(
        self, start: datetime, end: datetime, *, end_inclusive: bool = True
    ) -> "DataFrameDataHandler":
        """
        A new handler covering `[start, end]`, or `[start, end)`.

        Used by the walk-forward runner to build per-fold handlers. Returns a
        new object rather than mutating: a fold that could reach back into the
        parent's rows would defeat the point of splitting.

        `end_inclusive=False` is what the walk-forward runner uses. With
        inclusive bounds on both ends and a zero embargo, the single bar at
        `train_end == test_start` lands in *both* windows — one bar of genuine
        leakage, invisible in any summary statistic, and caught only by
        comparing the windows directly.
        """
        after_start = self._frame["timestamp"] >= pd.Timestamp(start)
        before_end = (
            self._frame["timestamp"] <= pd.Timestamp(end)
            if end_inclusive
            else self._frame["timestamp"] < pd.Timestamp(end)
        )
        rows = self._frame.loc[after_start & before_end, BAR_COLUMNS + ["volatility"]]

        def in_window(timestamp: datetime) -> bool:
            return start <= timestamp <= end if end_inclusive else start <= timestamp < end

        actions = [
            event
            for timestamp, events in self._actions_by_time.items()
            if in_window(timestamp)
            for event in events
        ]
        actions_frame = (
            pd.DataFrame(
                [
                    {
                        "timestamp": e.timestamp,
                        "symbol": e.symbol,
                        "action": e.action.value,
                        "cash_amount": e.cash_amount,
                        "split_ratio": e.split_ratio,
                    }
                    for e in actions
                ]
            )
            if actions
            else None
        )
        return DataFrameDataHandler(
            rows, actions_frame, max_history=self._max_history
        )
