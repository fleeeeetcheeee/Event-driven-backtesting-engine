"""
The Parquet-backed data handler.

The assertion that matters most is equivalence: streaming from Parquet in
chunks must produce *exactly* the same slices as holding everything in memory.
If it does not, the chunk size — an implementation detail — is visible in
results, which would make every backtest depend on how much RAM the machine had.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from evbt.core.events import EventType
from evbt.data.frame import DataFrameDataHandler
from evbt.data.parquet import PIT_STORE_COLUMNS, ParquetDataHandler
from tests.conftest import trending_bars

COLUMNS = {
    "timestamp": "timestamp",
    "symbol": "symbol",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


@pytest.fixture
def bars() -> pd.DataFrame:
    return trending_bars(
        {"AAA": 100.0, "BBB": 50.0, "CCC": 25.0}, n_days=120, daily_return=0.004
    )


@pytest.fixture
def flat_store(tmp_path, bars) -> "object":
    """A single Parquet file."""
    path = tmp_path / "prices.parquet"
    bars.to_parquet(path, index=False)
    return path


@pytest.fixture
def hive_store(tmp_path, bars) -> "object":
    """Hive-partitioned by date, the shape Project 01 writes."""
    root = tmp_path / "hive"
    for timestamp, group in bars.groupby("timestamp"):
        directory = root / f"date={timestamp:%Y-%m-%d}"
        directory.mkdir(parents=True, exist_ok=True)
        group.to_parquet(directory / "part.parquet", index=False)
    return root


def drain(handler) -> list[tuple]:
    """Every market slice, as comparable plain tuples."""
    out = []
    while handler.has_more():
        for event in handler.next_events():
            if event.type is EventType.MARKET:
                out.append(
                    (
                        event.timestamp,
                        tuple(
                            (s, round(b.open, 9), round(b.close, 9), b.volume)
                            for s, b in sorted(event.bars.items())
                        ),
                    )
                )
    return out


class TestEquivalenceWithInMemory:
    def test_single_file_matches_the_in_memory_handler(self, flat_store, bars):
        assert drain(ParquetDataHandler(flat_store, columns=COLUMNS)) == drain(
            DataFrameDataHandler(bars)
        )

    def test_hive_partitioned_store_matches_too(self, hive_store, bars):
        assert drain(ParquetDataHandler(hive_store, columns=COLUMNS)) == drain(
            DataFrameDataHandler(bars)
        )

    @pytest.mark.parametrize("chunk_days", [1, 7, 50, 10_000])
    def test_chunk_size_is_invisible_in_the_output(self, flat_store, bars, chunk_days):
        """
        The property that makes chunking safe: an implementation detail must not
        change results. A backtest that depended on available RAM would not be
        reproducible.
        """
        assert drain(
            ParquetDataHandler(flat_store, columns=COLUMNS, chunk_days=chunk_days)
        ) == drain(DataFrameDataHandler(bars))

    @pytest.mark.parametrize("chunk_days", [1, 7, 50])
    def test_volatility_is_continuous_across_chunk_boundaries(
        self, flat_store, bars, chunk_days
    ):
        """
        Without a warm-up tail carried between chunks, every boundary would show
        a run of bars with no vol estimate — and therefore no slippage — purely
        as an artefact of the chunk size.
        """
        chunked = ParquetDataHandler(
            flat_store, columns=COLUMNS, chunk_days=chunk_days
        )
        memory = DataFrameDataHandler(bars)

        chunked_vols, memory_vols = [], []
        while chunked.has_more():
            chunked.next_events()
            memory.next_events()
            chunked_vols.append(chunked.latest_bar("AAA").volatility)
            memory_vols.append(memory.latest_bar("AAA").volatility)

        assert [v is None for v in chunked_vols] == [v is None for v in memory_vols]
        pairs = [(a, b) for a, b in zip(chunked_vols, memory_vols) if a is not None]
        for a, b in pairs:
            assert a == pytest.approx(b, rel=1e-9)


class TestFiltering:
    def test_symbol_filter(self, flat_store):
        handler = ParquetDataHandler(
            flat_store, columns=COLUMNS, symbols=["AAA", "CCC"]
        )
        assert handler.symbols == ["AAA", "CCC"]
        first = [e for e in handler.next_events() if e.type is EventType.MARKET][0]
        assert sorted(first.symbols) == ["AAA", "CCC"]

    def test_date_range_filter(self, flat_store):
        handler = ParquetDataHandler(
            flat_store,
            columns=COLUMNS,
            start=datetime(2024, 2, 1),
            end=datetime(2024, 2, 29),
        )
        stamps = [d for d in handler.dates]
        assert min(stamps) >= pd.Timestamp("2024-02-01")
        assert max(stamps) <= pd.Timestamp("2024-02-29")

    def test_filters_that_match_nothing_raise(self, flat_store):
        """Rather than silently backtesting an empty universe."""
        with pytest.raises(ValueError, match="no rows"):
            ParquetDataHandler(flat_store, columns=COLUMNS, symbols=["NOPE"]).symbols


class TestCorporateActions:
    def test_actions_are_emitted_on_their_ex_date(self, flat_store, bars):
        ex_date = sorted(bars["timestamp"].unique())[3]
        actions = pd.DataFrame(
            [
                {
                    "timestamp": ex_date,
                    "symbol": "AAA",
                    "action": "DIVIDEND",
                    "cash_amount": 0.5,
                    "split_ratio": 1.0,
                }
            ]
        )
        handler = ParquetDataHandler(flat_store, columns=COLUMNS, actions=actions)

        emitted = []
        while handler.has_more():
            emitted += [
                e for e in handler.next_events() if e.type is EventType.CORPORATE_ACTION
            ]

        assert len(emitted) == 1
        assert emitted[0].timestamp == pd.Timestamp(ex_date).to_pydatetime()


class TestValidation:
    def test_missing_column_mapping_rejected(self, flat_store):
        with pytest.raises(ValueError, match="missing"):
            ParquetDataHandler(flat_store, columns={"timestamp": "timestamp"})

    def test_missing_store_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ParquetDataHandler(tmp_path / "nope.parquet", columns=COLUMNS)

    def test_non_positive_chunk_rejected(self, flat_store):
        with pytest.raises(ValueError, match="chunk_days"):
            ParquetDataHandler(flat_store, columns=COLUMNS, chunk_days=0)

    def test_pit_store_mapping_selects_the_unadjusted_close(self):
        """
        Project 01 stores `close_adj` beside `close_unadj`, and only the latter
        is point-in-time correct — `close_adj` is computed at write time from
        every split known then, so it embeds the future by construction.

        Pinned because the two columns sit side by side in the same table and
        the wrong one produces a backtest that looks fine and is not.
        """
        assert PIT_STORE_COLUMNS["close"] == "close_unadj"
        assert not any(
            column.endswith("_adj") for column in PIT_STORE_COLUMNS.values()
        )

    def test_pit_store_mapping_matches_project_01s_actual_schema(self):
        """
        Only `close` carries the unadj/adj split in that store — open, high and
        low are written raw under plain names. Assuming a symmetric
        `open_unadj` / `high_unadj` naming (as I first did) yields a mapping
        that fails at query time against the real store.
        """
        assert PIT_STORE_COLUMNS["open"] == "open"
        assert PIT_STORE_COLUMNS["high"] == "high"
        assert PIT_STORE_COLUMNS["low"] == "low"
        assert PIT_STORE_COLUMNS["symbol"] == "ticker"
        assert PIT_STORE_COLUMNS["timestamp"] == "date"


def test_reset_rewinds_and_clears_the_warmup(flat_store):
    handler = ParquetDataHandler(flat_store, columns=COLUMNS, chunk_days=10)
    while handler.has_more():
        handler.next_events()

    handler.reset()
    assert handler.has_more()
    assert handler.latest_bar("AAA") is None
