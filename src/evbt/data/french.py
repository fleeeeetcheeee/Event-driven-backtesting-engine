"""
Parser for the Ken French data library's CSV format.

The format is genuinely awkward and worth describing, because getting it subtly
wrong is the most likely way this replication produces a plausible but incorrect
answer:

  - A prose preamble of arbitrary length, then several stacked tables in one
    file. `6_Portfolios_2x3.csv` holds ten of them.
  - Each table is introduced by a free-text title line, then a header row that
    begins with a bare comma (the date column has no name).
  - Monthly and annual blocks sit in the same file with the same columns,
    distinguished only by the width of the date field: `202606` vs `2026`.
  - Missing data is `-99.99` or `-999`, not blank.
  - Returns are in **percent**, not decimals.
  - A copyright line at the end.

Line numbers are not stable — French republishes monthly and every table moves.
So tables are located structurally (title, then a `,`-leading header, then rows
whose first field is a 4- or 6-digit integer) rather than by offset.

The trap worth naming
---------------------
The file contains both value-weighted and equal-weighted versions of the six
portfolios, with identical column names, in adjacent tables. **HML is built from
the value-weighted set.** Reading the equal-weighted table produces a series that
looks entirely plausible, correlates highly with the real HML, and is wrong. The
`weighting` argument is required rather than defaulted for that reason.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# French's sentinels for missing data.
MISSING_CODES = (-99.99, -999.0, -99.99e0)

_DATA_ROW = re.compile(r"^\s*(\d{4}|\d{6})\s*(?:,|$)")

# The six portfolios, in the column order French writes them. The renaming is
# not cosmetic: "SMALL LoBM" is small-cap *growth* (low book-to-market), and
# reading it as anything else inverts the sign of the factor.
SIX_PORTFOLIO_NAMES = {
    "SMALL LoBM": "SmallGrowth",
    "ME1 BM2": "SmallNeutral",
    "SMALL HiBM": "SmallValue",
    "BIG LoBM": "BigGrowth",
    "ME2 BM2": "BigNeutral",
    "BIG HiBM": "BigValue",
}


@dataclass
class FrenchTable:
    """One table extracted from a French CSV, with the title that introduced it."""

    title: str
    periodicity: str  # "monthly" or "annual"
    data: pd.DataFrame


def _read_text(source: Path | str) -> str:
    """Read a French CSV, transparently unwrapping a single-member zip."""
    path = Path(source)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError(
                    f"expected exactly one CSV in {path.name}, found {names}"
                )
            return archive.read(names[0]).decode("utf-8-sig", errors="replace")
    return path.read_text(encoding="utf-8-sig", errors="replace")


def parse_tables(source: Path | str) -> list[FrenchTable]:
    """
    Split a French CSV into its constituent tables.

    Structural, not positional: a table is a `,`-leading header row followed by
    consecutive rows whose first field is a 4- or 6-digit integer. The title is
    the nearest preceding non-empty line that is not itself data.
    """
    lines = _read_text(source).splitlines()
    tables: list[FrenchTable] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith(","):
            i += 1
            continue

        header = [c.strip() for c in line.split(",")]
        columns = header[1:]

        # Walk back to the most recent non-empty line for the title.
        title = ""
        for back in range(i - 1, -1, -1):
            if lines[back].strip():
                title = lines[back].strip()
                break

        rows: list[list[str]] = []
        j = i + 1
        while j < len(lines) and _DATA_ROW.match(lines[j]):
            rows.append([c.strip() for c in lines[j].split(",")])
            j += 1

        if rows:
            periodicity = "monthly" if len(rows[0][0]) == 6 else "annual"
            frame = pd.DataFrame(
                [r[1 : len(columns) + 1] for r in rows],
                columns=columns,
                index=[r[0] for r in rows],
            )
            frame = frame.apply(pd.to_numeric, errors="coerce")
            tables.append(FrenchTable(title=title, periodicity=periodicity, data=frame))

        i = max(j, i + 1)

    return tables


def _to_period_index(frame: pd.DataFrame, periodicity: str) -> pd.DataFrame:
    """
    Index by period end. `202606` becomes 2026-06-30.

    Month *end* rather than start because a monthly return is realised over the
    month and known at its close — the same convention the engine's bars use.
    """
    out = frame.copy()
    if periodicity == "monthly":
        out.index = pd.to_datetime(out.index, format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        out.index = pd.to_datetime(out.index, format="%Y") + pd.offsets.YearEnd(0)
    out.index.name = "date"
    return out


def _clean_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace French's missing sentinels with NaN and convert percent to decimal."""
    out = frame.copy()
    for code in MISSING_CODES:
        out = out.mask(np.isclose(out, code, atol=1e-9))
    return out / 100.0


def load_six_portfolios(
    source: Path | str, *, weighting: str, periodicity: str = "monthly"
) -> pd.DataFrame:
    """
    The six size x book-to-market portfolios, as decimal returns.

    `weighting` must be given explicitly — "value" or "equal". HML is built from
    the value-weighted portfolios; the equal-weighted table has identical column
    names and silently produces a wrong factor.
    """
    if weighting not in ("value", "equal"):
        raise ValueError(f"weighting must be 'value' or 'equal', got {weighting!r}")

    wanted = "Value Weighted" if weighting == "value" else "Equal Weighted"
    matches = [
        t
        for t in parse_tables(source)
        if t.periodicity == periodicity and wanted in t.title and "Returns" in t.title
    ]
    if len(matches) != 1:
        titles = [t.title for t in parse_tables(source)]
        raise ValueError(
            f"expected exactly one {periodicity} '{wanted} Returns' table, "
            f"found {len(matches)}. Tables present: {titles}"
        )

    frame = _to_period_index(matches[0].data, periodicity)
    missing = set(SIX_PORTFOLIO_NAMES) - set(frame.columns)
    if missing:
        raise ValueError(f"missing expected portfolio columns: {sorted(missing)}")

    frame = frame[list(SIX_PORTFOLIO_NAMES)].rename(columns=SIX_PORTFOLIO_NAMES)
    return _clean_returns(frame)


def load_factors(source: Path | str, periodicity: str = "monthly") -> pd.DataFrame:
    """The published Mkt-RF, SMB, HML and RF series, as decimal returns."""
    tables = [t for t in parse_tables(source) if t.periodicity == periodicity]
    matches = [t for t in tables if "HML" in t.data.columns]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {periodicity} factor table containing HML, "
            f"found {len(matches)}"
        )
    return _clean_returns(_to_period_index(matches[0].data, periodicity))


def construct_hml(portfolios: pd.DataFrame) -> pd.Series:
    """
    Rebuild HML from the six portfolios, exactly as French defines it:

        HML = 1/2 (SmallValue + BigValue) - 1/2 (SmallGrowth + BigGrowth)

    The two neutral portfolios are unused — HML is the high-minus-low spread and
    the middle third of the book-to-market sort contributes nothing to it. They
    are loaded anyway because the engine holds a book, and a book that omits
    them is a different portfolio from the one the weights describe.
    """
    required = ["SmallValue", "BigValue", "SmallGrowth", "BigGrowth"]
    missing = [c for c in required if c not in portfolios.columns]
    if missing:
        raise ValueError(f"portfolios frame is missing {missing}")

    value = 0.5 * (portfolios["SmallValue"] + portfolios["BigValue"])
    growth = 0.5 * (portfolios["SmallGrowth"] + portfolios["BigGrowth"])
    return (value - growth).rename("HML")


# --- bridging French's returns into engine bars -----------------------------


def returns_to_bars(
    returns: pd.DataFrame, *, initial_price: float = 100.0, volume: float = 1e12
) -> pd.DataFrame:
    """
    Convert a frame of periodic returns into OHLCV bars the engine can stream.

    The construction that makes the replication exact:

        close[t] = price at the end of period t
        open[t]  = price at the end of period t-1

    so the series is continuous across bars. The engine sizes orders at bar t's
    close and fills them at bar t+1's open — which is the *same price* — so
    there is no execution drift, and the portfolio's realised return over bar
    t+1 is exactly the input return for t+1. Any deviation the replication finds
    is therefore an accounting bug, not a timing artefact.

    `volume` is deliberately enormous so the participation cap never binds. This
    is a validation harness, not a simulation of trading these portfolios.
    """
    if returns.empty:
        raise ValueError("returns frame is empty")

    prices = initial_price * (1.0 + returns.fillna(0.0)).cumprod()

    rows = []
    for symbol in prices.columns:
        series = prices[symbol]
        previous = series.shift(1).fillna(initial_price)
        for timestamp in series.index:
            close = float(series.loc[timestamp])
            open_ = float(previous.loc[timestamp])
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": open_,
                    "high": max(open_, close),
                    "low": min(open_, close),
                    "close": close,
                    "volume": volume,
                }
            )

    return pd.DataFrame(rows).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
