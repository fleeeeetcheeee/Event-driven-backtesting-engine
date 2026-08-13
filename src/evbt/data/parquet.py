"""
Out-of-core market data: stream bars from a Parquet store instead of RAM.

`DataFrameDataHandler` holds the whole history in memory, which is fine to a few
million rows — a 30-year daily backtest on a 1,000-name universe is about 7.5M —
and hopeless for intraday. This handler reads the same data in date chunks, so
peak memory is set by the chunk size rather than by the length of the backtest.

It reads the shape Project 01 writes: Hive-partitioned `date=YYYY-MM-DD/`
directories of Parquet, queried through DuckDB. Plain directories of Parquet and
single files work too — DuckDB's globbing handles all three, and the partition
column is projected out of the path automatically when present.

Why DuckDB rather than pyarrow directly
---------------------------------------
Predicate pushdown. A chunk query carries its date range into the scan, so only
the relevant partition directories are opened. Reading the whole store and
filtering in pandas would defeat the entire point of chunking.

The no-lookahead guarantee is unchanged and comes for free: `DataHandler`'s
history buffer is appended to as slices are yielded, and this class only
implements `_slices()`. There is no accessor here that could reach a future
bar, because there are no accessors here at all.

Column mapping
--------------
Project 01 stores unadjusted prices as `close_unadj` alongside a `close_adj`
that is *not* point-in-time correct — its own README calls that pair the
sharpest edge in the codebase. This handler therefore requires an explicit
column mapping rather than guessing, and `PIT_STORE_COLUMNS` names the correct
one. Silently defaulting to a column called `close` would pick the wrong series
on that store.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator, Mapping, Optional

import numpy as np
import pandas as pd

from evbt.core.events import Bar, CorporateActionEvent, CorporateActionType
from evbt.data.base import DataHandler, MarketSlice

log = logging.getLogger(__name__)

# The mapping for Project 01's processed price store, read from that project's
# `price_cleaner.py` rather than assumed.
#
# Note the asymmetry, which is a trap: only *close* carries the unadjusted /
# adjusted split. `open`, `high` and `low` are stored raw under plain names,
# while close exists twice — as `close_unadj` and `close_adj`. Only
# `close_unadj` is point-in-time correct; `close_adj` is computed at write time
# using every split known then, so it embeds future information by
# construction, and Project 01's own README calls the pair the sharpest edge in
# that codebase.
PIT_STORE_COLUMNS = {
    "timestamp": "date",
    "symbol": "ticker",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close_unadj",
    "volume": "volume",
}

REQUIRED_FIELDS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")


class ParquetDataHandler(DataHandler):
    """
    Streams `MarketSlice`s from a Parquet store, one date chunk at a time.

    Parameters
    ----------
    source
        A directory (globbed recursively for `*.parquet`) or a single file.
    columns
        Maps this engine's field names to the store's column names. Required —
        see the module docstring on why guessing is unsafe.
    chunk_days
        How many distinct dates to load per query. Larger is faster and uses
        more memory; the default keeps a wide universe comfortably under a few
        hundred MB.
    actions
        Optional corporate-action frame, in the same shape
        `DataFrameDataHandler` takes. Held in memory: even a full US equity
        history of splits and dividends is small.
    volatility_window / volatility_min_periods
        Trailing close-to-close volatility, computed per chunk with a warm-up
        tail carried across chunk boundaries so the estimate does not restart.
    """

    def __init__(
        self,
        source: Path | str,
        *,
        columns: Mapping[str, str],
        chunk_days: int = 256,
        actions: Optional[pd.DataFrame] = None,
        symbols: Optional[list[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        max_history: int = 512,
        volatility_window: int = 21,
        volatility_min_periods: int = 5,
    ) -> None:
        super().__init__(max_history=max_history)

        missing = [f for f in REQUIRED_FIELDS if f not in columns]
        if missing:
            raise ValueError(f"columns mapping is missing {missing}")
        if chunk_days < 1:
            raise ValueError(f"chunk_days must be at least 1, got {chunk_days}")

        self.source = Path(source)
        if not self.source.exists():
            raise FileNotFoundError(f"no Parquet store at {self.source}")

        self.columns = dict(columns)
        self.chunk_days = chunk_days
        self.symbol_filter = list(symbols) if symbols else None
        self.start = start
        self.end = end
        self.volatility_window = volatility_window
        self.volatility_min_periods = volatility_min_periods

        self._actions_by_time = self._index_actions(actions)
        self._connection = None
        self._dates: Optional[list[pd.Timestamp]] = None
        self._symbols: Optional[list[str]] = None
        # Tail of the previous chunk, so the rolling volatility window spans
        # chunk boundaries instead of resetting at each one.
        self._warmup: Optional[pd.DataFrame] = None

    # --- DuckDB plumbing ---------------------------------------------------

    @property
    def _scan(self) -> str:
        """The DuckDB relation for the store, with Hive partitions honoured."""
        if self.source.is_dir():
            pattern = str(self.source / "**" / "*.parquet")
            return f"read_parquet('{pattern}', hive_partitioning=true, union_by_name=true)"
        return f"read_parquet('{self.source}')"

    def _connect(self):
        if self._connection is None:
            import duckdb

            self._connection = duckdb.connect(database=":memory:")
        return self._connection

    def _where(self) -> str:
        date_column = self.columns["timestamp"]
        clauses = []
        if self.start is not None:
            clauses.append(f"{date_column} >= DATE '{self.start:%Y-%m-%d}'")
        if self.end is not None:
            clauses.append(f"{date_column} <= DATE '{self.end:%Y-%m-%d}'")
        if self.symbol_filter:
            quoted = ", ".join(f"'{s}'" for s in self.symbol_filter)
            clauses.append(f"{self.columns['symbol']} IN ({quoted})")
        return f"WHERE {' AND '.join(clauses)}" if clauses else ""

    def _load_index(self) -> None:
        """Fetch the distinct dates and symbols once, up front."""
        if self._dates is not None:
            return

        connection = self._connect()
        date_column = self.columns["timestamp"]
        symbol_column = self.columns["symbol"]

        dates = connection.execute(
            f"SELECT DISTINCT {date_column} AS d FROM {self._scan} "
            f"{self._where()} ORDER BY d"
        ).fetchall()
        self._dates = [pd.Timestamp(row[0]) for row in dates]

        symbols = connection.execute(
            f"SELECT DISTINCT {symbol_column} AS s FROM {self._scan} "
            f"{self._where()} ORDER BY s"
        ).fetchall()
        self._symbols = [row[0] for row in symbols]

        if not self._dates:
            raise ValueError(
                f"no rows in {self.source} matching the requested filters"
            )

    # --- DataHandler contract ---------------------------------------------

    @property
    def symbols(self) -> list[str]:
        self._load_index()
        return list(self._symbols or [])

    @property
    def dates(self) -> list[pd.Timestamp]:
        self._load_index()
        return list(self._dates or [])

    def _read_chunk(self, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
        """One chunk of bars, renamed into this engine's field names."""
        date_column = self.columns["timestamp"]
        selected = ", ".join(
            f"{self.columns[field]} AS {field}" for field in REQUIRED_FIELDS
        )
        where = self._where()
        # The chunk bounds go into the scan itself, which is the whole point:
        # DuckDB pushes them down and never opens the other partitions.
        bound = (
            f"{date_column} >= DATE '{first:%Y-%m-%d}' "
            f"AND {date_column} <= DATE '{last:%Y-%m-%d}'"
        )
        clause = f"{where} AND {bound}" if where else f"WHERE {bound}"

        frame = self._connect().execute(
            f"SELECT {selected} FROM {self._scan} {clause} "
            f"ORDER BY timestamp, symbol"
        ).df()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        return frame

    def _with_volatility(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Attach trailing volatility, carrying a warm-up tail across chunks.

        Without the carry, every chunk boundary would produce a stretch of bars
        with no vol estimate and therefore no slippage — an artefact of the
        chunk size, which is an implementation detail that must not be visible
        in results. `DataFrameDataHandler` has the same property by construction
        because it never chunks; this reproduces it.
        """
        combined = (
            pd.concat([self._warmup, chunk], ignore_index=True)
            if self._warmup is not None
            else chunk
        )
        combined = combined.sort_values(["symbol", "timestamp"], kind="mergesort")

        returns = combined.groupby("symbol", sort=False)["close"].pct_change(
            fill_method=None
        )
        combined["volatility"] = (
            returns.groupby(combined["symbol"], sort=False)
            .rolling(
                window=self.volatility_window, min_periods=self.volatility_min_periods
            )
            .std()
            .reset_index(level=0, drop=True)
        )

        # Keep the last `window` bars per symbol as the next chunk's warm-up.
        self._warmup = (
            combined.groupby("symbol", sort=False)
            .tail(self.volatility_window)
            .loc[:, list(REQUIRED_FIELDS)]
            .copy()
        )

        first_new = chunk["timestamp"].min()
        return combined[combined["timestamp"] >= first_new].sort_values(
            ["timestamp", "symbol"], kind="mergesort"
        )

    def _slices(self) -> Iterator[MarketSlice]:
        self._load_index()
        dates = self._dates or []

        for offset in range(0, len(dates), self.chunk_days):
            window = dates[offset : offset + self.chunk_days]
            chunk = self._with_volatility(self._read_chunk(window[0], window[-1]))

            for timestamp, group in chunk.groupby("timestamp", sort=True):
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
                moment = timestamp.to_pydatetime()
                yield MarketSlice(
                    timestamp=moment,
                    bars=bars,
                    actions=list(self._actions_by_time.get(moment, [])),
                )

    def reset(self) -> None:
        super().reset()
        self._warmup = None

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _index_actions(
        actions: Optional[pd.DataFrame],
    ) -> dict[datetime, list[CorporateActionEvent]]:
        if actions is None or actions.empty:
            return {}

        frame = actions.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        indexed: dict[datetime, list[CorporateActionEvent]] = {}
        for row in frame.sort_values(["timestamp", "symbol"]).itertuples(index=False):
            event = CorporateActionEvent(
                timestamp=row.timestamp.to_pydatetime(),
                symbol=row.symbol,
                action=CorporateActionType(str(row.action).upper()),
                cash_amount=float(getattr(row, "cash_amount", 0.0) or 0.0),
                split_ratio=float(getattr(row, "split_ratio", 1.0) or 1.0),
            )
            indexed.setdefault(event.timestamp, []).append(event)
        return indexed
