"""
The project's done criterion, as an executable test.

Reproduces Fama-French HML and asserts the engine matches French's published
monthly series. Skipped when the data files are absent — they are gitignored,
so a fresh clone runs the rest of the suite and reports these as skipped until
`scripts/fetch_french_data.py` has been run.

What makes this a test of the *engine* rather than of arithmetic: the engine
gets French's own portfolios as tradeable instruments and has to arrive at his
factor by actually trading them — sizing orders against NAV, filling them a bar
later, carrying a 100% short book, marking to market, and compounding. The
closed-form answer is known, so any accounting bug shows up as drift.

`test_engine_adds_no_error_beyond_french_rounding` is the sharpest of these. It
asserts the engine's error distribution is indistinguishable from the pure
arithmetic reconstruction's — that the engine contributes *nothing*, and the
entire residual is French's own 2-decimal rounding.
"""

from __future__ import annotations

import pandas as pd
import pytest

from evbt.config import config
from evbt.data.frame import DataFrameDataHandler
from evbt.data.french import (
    construct_hml,
    load_factors,
    load_six_portfolios,
    returns_to_bars,
)
from evbt.engine import Backtest
from evbt.execution.broker import SimulatedBroker
from evbt.portfolio.construction import ExplicitWeightSizer, PortfolioConstructor
from evbt.portfolio.portfolio import Portfolio
from evbt.strategy.base import FixedWeightPortfolio

PORTFOLIOS_ZIP = config.fama_french_dir / "6_Portfolios_2x3_CSV.zip"
FACTORS_ZIP = config.fama_french_dir / "F-F_Research_Data_Factors_CSV.zip"

pytestmark = pytest.mark.skipif(
    not (PORTFOLIOS_ZIP.exists() and FACTORS_ZIP.exists()),
    reason="French data not downloaded — run scripts/fetch_french_data.py",
)

HML_WEIGHTS = {
    "SmallValue": 0.5,
    "BigValue": 0.5,
    "SmallGrowth": -0.5,
    "BigGrowth": -0.5,
}

# French publishes factor returns rounded to two decimal places in percent, so
# the half-ulp of his own reporting is 0.005% = 0.5 bp. Nothing built from that
# series can agree more closely than this, and the spec asks for "a few basis
# points" — so 1.0 bp is a bound that is both achievable and strict enough to
# fail on any real accounting error.
TOLERANCE_BPS = 1.0


@pytest.fixture(scope="module")
def portfolios() -> pd.DataFrame:
    return load_six_portfolios(PORTFOLIOS_ZIP, weighting="value")


@pytest.fixture(scope="module")
def published() -> pd.Series:
    return load_factors(FACTORS_ZIP)["HML"]


@pytest.fixture(scope="module")
def engine_returns(portfolios: pd.DataFrame) -> pd.Series:
    """Run the replication once and share it across the module."""
    engine = Backtest(
        DataFrameDataHandler(returns_to_bars(portfolios), max_history=8),
        FixedWeightPortfolio(HML_WEIGHTS),
        Portfolio(1_000_000.0),
        SimulatedBroker(),  # zero costs: the published series is a gross return
        PortfolioConstructor(sizer=ExplicitWeightSizer()),
        liquidate_unsignalled=True,
    )
    return engine.run().returns()


def _diff_bps(left: pd.Series, right: pd.Series) -> pd.Series:
    joined = left.rename("l").to_frame().join(right.rename("r"), how="inner").dropna()
    return (joined["l"] - joined["r"]) * 1e4


class TestStageOneArithmetic:
    """Validate the reference data before the engine is involved."""

    def test_reconstruction_matches_published_hml(self, portfolios, published):
        diff = _diff_bps(construct_hml(portfolios), published)
        assert diff.abs().max() <= TOLERANCE_BPS

    def test_covers_the_full_published_history(self, portfolios, published):
        diff = _diff_bps(construct_hml(portfolios), published)
        assert len(diff) > 1_100  # ~100 years of monthly data
        assert diff.index.min() <= pd.Timestamp("1926-12-31")

    def test_equal_weighted_is_a_different_and_wrong_factor(self, published):
        """
        The control that proves the value-weighted choice is load-bearing
        rather than incidental. Reading the wrong table gives a series that
        still correlates ~0.93 with HML — plausible, and wrong by ~90 bps a
        month.
        """
        wrong = construct_hml(load_six_portfolios(PORTFOLIOS_ZIP, weighting="equal"))
        diff = _diff_bps(wrong, published)
        assert diff.abs().mean() > 10.0


class TestDoneCriterion:
    """The acceptance test: the engine must reproduce the published series."""

    def test_engine_matches_published_hml_within_tolerance(
        self, engine_returns, published
    ):
        diff = _diff_bps(engine_returns, published)
        assert diff.abs().max() <= TOLERANCE_BPS, (
            f"worst monthly deviation {diff.abs().max():.4f} bps exceeds "
            f"{TOLERANCE_BPS} bps"
        )

    def test_no_month_deviates_by_more_than_a_basis_point(
        self, engine_returns, published
    ):
        diff = _diff_bps(engine_returns, published)
        assert int((diff.abs() > 1.0).sum()) == 0

    def test_engine_adds_no_error_beyond_french_rounding(
        self, engine_returns, portfolios, published
    ):
        """
        The sharpest assertion available. The pure arithmetic reconstruction and
        the engine's traded result must have the same error against French — so
        the engine contributes nothing, and the whole residual is his rounding.

        If the engine had any accounting bug — a mis-signed short, a basis
        carried across a rebalance, a stale NAV used for sizing, an off-by-one
        in fill timing — the engine's error would exceed the arithmetic one.
        """
        arithmetic = _diff_bps(construct_hml(portfolios), published).abs()
        engine = _diff_bps(engine_returns, published).abs()

        assert engine.mean() == pytest.approx(arithmetic.mean(), abs=0.01)
        assert engine.max() == pytest.approx(arithmetic.max(), abs=0.01)

    def test_alignment_is_not_off_by_a_month(self, engine_returns, published):
        """
        A one-month shift would still produce a smooth, plausible equity curve.
        Shifting deliberately must make the fit dramatically worse — if it does
        not, the comparison is not testing what it claims to.
        """
        correct = _diff_bps(engine_returns, published).abs().mean()
        shifted = _diff_bps(engine_returns.shift(1).dropna(), published).abs().mean()
        assert shifted > 50 * correct


class TestEngineBehaviourDuringReplication:
    """The book the engine actually held, not just the returns it reported."""

    def _run(self, portfolios: pd.DataFrame):
        engine = Backtest(
            DataFrameDataHandler(returns_to_bars(portfolios), max_history=8),
            FixedWeightPortfolio(HML_WEIGHTS),
            Portfolio(1_000_000.0),
            SimulatedBroker(),
            PortfolioConstructor(sizer=ExplicitWeightSizer()),
            liquidate_unsignalled=True,
        )
        return engine.run()

    def test_book_is_dollar_neutral(self, portfolios):
        """
        HML is 100% long, 100% short, so the book is neutral *at each rebalance*.

        The assertion is on the median rather than the max, because the equity
        curve is snapshotted at month end — after a full month of drift and
        before the next rebalance fills. In a month where value beats growth by
        30 percentage points the two legs genuinely are no longer equal, and a
        tight bound on the max would be asserting that markets do not move.
        `test_net_drift_is_intramonth_not_a_leak` covers the tail.
        """
        curve = self._run(portfolios).equity_curve
        assert (curve["net_exposure"] / curve["nav"]).abs().median() < 0.03

    def test_net_drift_is_intramonth_not_a_leak(self, portfolios):
        """
        The tail of net exposure must be explained by within-month dispersion
        between the legs, not by a rebalance failing to happen.

        Checked by correlation: months with the largest net drift must be months
        with the largest absolute factor return. A book that simply stopped
        rebalancing would drift monotonically and show no such relationship.
        """
        curve = self._run(portfolios).equity_curve
        net = (curve["net_exposure"] / curve["nav"]).abs()
        net.index = pd.DatetimeIndex(net.index)

        hml = construct_hml(portfolios).abs()
        joined = net.rename("net").to_frame().join(hml.rename("hml"), how="inner")

        assert joined["net"].corr(joined["hml"]) > 0.5

    def test_gross_leverage_is_two_turns(self, portfolios):
        curve = self._run(portfolios).equity_curve
        assert (curve["gross_exposure"] / curve["nav"]).mean() == pytest.approx(
            2.0, abs=0.05
        )

    def test_nothing_was_rejected(self, portfolios):
        """A rejection would mean the engine silently ran a different strategy."""
        assert self._run(portfolios).n_rejections == 0

    def test_zero_cost_configuration_really_charged_nothing(self, portfolios):
        result = self._run(portfolios)
        assert result.fills["total_cost"].sum() == pytest.approx(0.0)

    def test_rebalanced_every_month(self, portfolios):
        """
        Four orders per bar. A fixed-weight factor is defined as rebalanced each
        period; letting weights drift would be a different portfolio.
        """
        result = self._run(portfolios)
        assert result.n_orders == 4 * len(portfolios)
