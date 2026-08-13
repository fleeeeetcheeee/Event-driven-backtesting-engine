"""
The Ken French CSV parser, pinned against a synthetic file in the real format.

A fixture rather than the downloaded data, so these run with no network and no
`data/` directory. The fixture reproduces every structural feature that makes
the real format hazardous: a prose preamble, stacked tables, monthly and annual
blocks sharing column names, missing-data sentinels, percent units, and a
trailing copyright line.

The replication itself is pinned separately in
`tests/integration/test_hml_replication.py`, which skips when the real files
are absent.
"""

from __future__ import annotations

import textwrap
import zipfile

import numpy as np
import pandas as pd
import pytest

from evbt.data.french import (
    SIX_PORTFOLIO_NAMES,
    construct_hml,
    load_factors,
    load_six_portfolios,
    parse_tables,
    returns_to_bars,
)

# Value-weighted and equal-weighted deliberately carry *different* numbers, so
# a test that reads the wrong table fails loudly instead of coincidentally
# passing.
FIXTURE = textwrap.dedent(
    """\
    This file was created using the 202606 CRSP database.
    It contains value- and equal-weighted returns for portfolios formed on ME and BEME.

    Missing data are indicated by -99.99 or -999.


      Average Value Weighted Returns -- Monthly
    ,SMALL LoBM,ME1 BM2,SMALL HiBM,BIG LoBM,ME2 BM2,BIG HiBM
    192607,   1.0000,   0.8807,   3.0000,   2.0000,   1.9060,   4.0000
    192608,   0.5000,   1.4677,   1.5000,   1.0000,   2.7028,   2.5000
    192609, -99.9900, -99.9900, -99.9900, -99.9900, -99.9900, -99.9900

      Average Equal Weighted Returns -- Monthly
    ,SMALL LoBM,ME1 BM2,SMALL HiBM,BIG LoBM,ME2 BM2,BIG HiBM
    192607,   9.0000,   9.0000,   9.0000,   9.0000,   9.0000,   9.0000
    192608,   9.0000,   9.0000,   9.0000,   9.0000,   9.0000,   9.0000
    192609,   9.0000,   9.0000,   9.0000,   9.0000,   9.0000,   9.0000

      Average Value Weighted Returns -- Annual
    ,SMALL LoBM,ME1 BM2,SMALL HiBM,BIG LoBM,ME2 BM2,BIG HiBM
    1927,  10.0000,  11.0000,  12.0000,  13.0000,  14.0000,  15.0000

    Copyright 2026 Eugene F. Fama and Kenneth R. French
    """
)

FACTORS_FIXTURE = textwrap.dedent(
    """\
    This file was created using the 202606 CRSP database.

    ,Mkt-RF,SMB,HML,RF
    192607,   2.89,  -2.55,   2.00,   0.22
    192608,   2.64,  -1.14,   1.25,   0.25

     Annual Factors: January-December
    ,Mkt-RF,SMB,HML,RF
    1927,  29.47,  -2.34,   1.00,   3.12

    Copyright 2026 Eugene F. Fama and Kenneth R. French
    """
)


@pytest.fixture
def portfolios_csv(tmp_path):
    path = tmp_path / "6_Portfolios_2x3.csv"
    path.write_text(FIXTURE)
    return path


@pytest.fixture
def factors_csv(tmp_path):
    path = tmp_path / "F-F_Research_Data_Factors.csv"
    path.write_text(FACTORS_FIXTURE)
    return path


class TestTableExtraction:
    def test_finds_every_table(self, portfolios_csv):
        tables = parse_tables(portfolios_csv)
        assert len(tables) == 3

    def test_titles_are_captured(self, portfolios_csv):
        titles = [t.title for t in parse_tables(portfolios_csv)]
        assert "Average Value Weighted Returns -- Monthly" in titles
        assert "Average Equal Weighted Returns -- Monthly" in titles

    def test_monthly_and_annual_are_distinguished_by_date_width(self, portfolios_csv):
        kinds = [t.periodicity for t in parse_tables(portfolios_csv)]
        assert kinds == ["monthly", "monthly", "annual"]

    def test_preamble_and_copyright_are_not_mistaken_for_data(self, portfolios_csv):
        for table in parse_tables(portfolios_csv):
            assert not table.data.empty
            assert list(table.data.columns) == list(SIX_PORTFOLIO_NAMES)

    def test_reads_from_a_zip_transparently(self, tmp_path, portfolios_csv):
        archive = tmp_path / "6_Portfolios_2x3_CSV.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.write(portfolios_csv, arcname="6_Portfolios_2x3.csv")
        assert len(parse_tables(archive)) == 3


class TestLoadingPortfolios:
    def test_columns_are_renamed_to_readable_names(self, portfolios_csv):
        frame = load_six_portfolios(portfolios_csv, weighting="value")
        assert list(frame.columns) == [
            "SmallGrowth",
            "SmallNeutral",
            "SmallValue",
            "BigGrowth",
            "BigNeutral",
            "BigValue",
        ]

    def test_lobm_maps_to_growth_not_value(self, portfolios_csv):
        """
        The renaming that inverts the factor's sign if it is wrong. `SMALL LoBM`
        is low book-to-market, i.e. small-cap *growth*.
        """
        frame = load_six_portfolios(portfolios_csv, weighting="value")
        assert frame.loc["1926-07-31", "SmallGrowth"] == pytest.approx(0.01)
        assert frame.loc["1926-07-31", "SmallValue"] == pytest.approx(0.03)

    def test_percent_is_converted_to_decimal(self, portfolios_csv):
        frame = load_six_portfolios(portfolios_csv, weighting="value")
        assert frame.loc["1926-07-31", "BigValue"] == pytest.approx(0.04)

    def test_missing_sentinel_becomes_nan(self, portfolios_csv):
        frame = load_six_portfolios(portfolios_csv, weighting="value")
        assert frame.loc["1926-09-30"].isna().all()

    def test_index_is_month_end(self, portfolios_csv):
        frame = load_six_portfolios(portfolios_csv, weighting="value")
        assert list(frame.index[:2]) == [
            pd.Timestamp("1926-07-31"),
            pd.Timestamp("1926-08-31"),
        ]

    def test_value_and_equal_weighted_are_different_tables(self, portfolios_csv):
        """
        The trap: identical column names, adjacent tables, and reading the wrong
        one yields a plausible series that is simply not HML.
        """
        value = load_six_portfolios(portfolios_csv, weighting="value")
        equal = load_six_portfolios(portfolios_csv, weighting="equal")
        assert not value.equals(equal)
        assert equal.loc["1926-07-31"].tolist() == pytest.approx([0.09] * 6)

    def test_weighting_must_be_explicit_and_valid(self, portfolios_csv):
        with pytest.raises(ValueError, match="weighting must be"):
            load_six_portfolios(portfolios_csv, weighting="vw")

    def test_annual_table_is_selectable(self, portfolios_csv):
        frame = load_six_portfolios(
            portfolios_csv, weighting="value", periodicity="annual"
        )
        assert frame.index[0] == pd.Timestamp("1927-12-31")
        assert frame.loc["1927-12-31", "SmallGrowth"] == pytest.approx(0.10)


class TestLoadingFactors:
    def test_reads_the_monthly_factor_table(self, factors_csv):
        frame = load_factors(factors_csv)
        assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
        assert frame.loc["1926-07-31", "HML"] == pytest.approx(0.02)

    def test_annual_block_is_not_mixed_into_monthly(self, factors_csv):
        frame = load_factors(factors_csv)
        assert len(frame) == 2
        assert frame.index.max() == pd.Timestamp("1926-08-31")


class TestConstructHml:
    def test_matches_frenchs_definition(self, portfolios_csv):
        # 1/2(0.03 + 0.04) - 1/2(0.01 + 0.02) = 0.035 - 0.015 = 0.02
        frame = load_six_portfolios(portfolios_csv, weighting="value")
        assert construct_hml(frame).loc["1926-07-31"] == pytest.approx(0.02)

    def test_reproduces_the_fixtures_published_hml(self, portfolios_csv, factors_csv):
        """End-to-end on the fixture: reconstruction must equal the published column."""
        mine = construct_hml(load_six_portfolios(portfolios_csv, weighting="value"))
        theirs = load_factors(factors_csv)["HML"]
        both = mine.rename("mine").to_frame().join(theirs, how="inner").dropna()

        assert len(both) == 2
        assert (both["mine"] - both["HML"]).abs().max() == pytest.approx(0.0, abs=1e-12)

    def test_neutral_portfolios_do_not_affect_the_factor(self, portfolios_csv):
        frame = load_six_portfolios(portfolios_csv, weighting="value")
        before = construct_hml(frame).loc["1926-07-31"]
        frame.loc["1926-07-31", ["SmallNeutral", "BigNeutral"]] = 99.0
        assert construct_hml(frame).loc["1926-07-31"] == pytest.approx(before)

    def test_missing_columns_rejected(self):
        with pytest.raises(ValueError, match="missing"):
            construct_hml(pd.DataFrame({"SmallValue": [0.01]}))


class TestReturnsToBars:
    def _returns(self):
        idx = pd.date_range("2020-01-31", periods=4, freq="ME")
        return pd.DataFrame({"A": [0.10, 0.05, -0.02, 0.03]}, index=idx)

    def test_open_equals_the_previous_close(self):
        """
        The construction that makes the replication exact: a continuous price
        series means the engine's next-bar fill happens at the price it sized
        against, so no execution drift contaminates the comparison.
        """
        bars = returns_to_bars(self._returns())
        closes = bars["close"].tolist()
        opens = bars["open"].tolist()
        assert opens[1:] == pytest.approx(closes[:-1])

    def test_first_bar_opens_at_the_initial_price(self):
        bars = returns_to_bars(self._returns(), initial_price=100.0)
        assert bars["open"].iloc[0] == pytest.approx(100.0)
        assert bars["close"].iloc[0] == pytest.approx(110.0)

    def test_close_to_close_returns_round_trip(self):
        returns = self._returns()
        bars = returns_to_bars(returns)
        recovered = bars.set_index("timestamp")["close"].pct_change().dropna()
        assert recovered.to_numpy() == pytest.approx(returns["A"].to_numpy()[1:])

    def test_high_and_low_bracket_open_and_close(self):
        bars = returns_to_bars(self._returns())
        assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
        assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()

    def test_missing_returns_are_treated_as_flat(self):
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        returns = pd.DataFrame({"A": [0.10, np.nan, 0.10]}, index=idx)
        bars = returns_to_bars(returns)
        assert bars["close"].tolist() == pytest.approx([110.0, 110.0, 121.0])

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            returns_to_bars(pd.DataFrame())
