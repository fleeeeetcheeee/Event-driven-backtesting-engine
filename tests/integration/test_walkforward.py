"""
Walk-forward evaluation, and the leakage guarantees it exists to provide.

The framework's only real job is that nothing `fit` saw can overlap what
`score` measured. `TestNoLeakage` checks exactly that, by recording what the fit
callback was shown and comparing it against the test windows.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from evbt.data.frame import DataFrameDataHandler
from evbt.execution.broker import SimulatedBroker
from evbt.portfolio.construction import ExplicitWeightSizer, PortfolioConstructor
from evbt.portfolio.portfolio import Portfolio
from evbt.strategy.base import BuyAndHold
from evbt.walkforward.runner import Fold, WalkForward, generate_folds
from tests.conftest import trending_bars


class TestFoldGeneration:
    def test_consecutive_test_windows_do_not_share_a_bar(self):
        """
        Half-open windows tile the timeline exactly once. Inclusive windows
        would put the boundary bar in two test folds and double-count it in
        the stitched out-of-sample series.
        """
        data, runner = _harness()
        folds = generate_folds(
            datetime(2020, 1, 1),
            datetime(2023, 1, 1),
            train_period=timedelta(days=250),
            test_period=timedelta(days=120),
        )
        result = runner.run(folds)

        stamps = result.oos_returns.index.tolist()
        assert len(stamps) == len(set(stamps))

    def test_rolling_folds_do_not_overlap_train_and_test(self):
        folds = generate_folds(
            datetime(2020, 1, 1),
            datetime(2024, 1, 1),
            train_period=timedelta(days=365),
            test_period=timedelta(days=180),
        )
        assert len(folds) >= 3
        for fold in folds:
            assert fold.train_end <= fold.test_start
            assert fold.test_start < fold.test_end

    def test_test_windows_are_contiguous_and_ordered(self):
        folds = generate_folds(
            datetime(2020, 1, 1),
            datetime(2024, 1, 1),
            train_period=timedelta(days=365),
            test_period=timedelta(days=180),
        )
        for earlier, later in zip(folds, folds[1:]):
            assert earlier.test_end <= later.test_start

    def test_rolling_windows_have_constant_train_length(self):
        folds = generate_folds(
            datetime(2020, 1, 1),
            datetime(2024, 1, 1),
            train_period=timedelta(days=365),
            test_period=timedelta(days=180),
            anchored=False,
        )
        lengths = {(f.train_end - f.train_start).days for f in folds}
        assert lengths == {365}

    def test_anchored_windows_grow(self):
        folds = generate_folds(
            datetime(2020, 1, 1),
            datetime(2024, 1, 1),
            train_period=timedelta(days=365),
            test_period=timedelta(days=180),
            anchored=True,
        )
        lengths = [(f.train_end - f.train_start).days for f in folds]
        assert all(f.train_start == folds[0].train_start for f in folds)
        assert lengths == sorted(lengths) and lengths[-1] > lengths[0]

    def test_embargo_opens_a_gap(self):
        """
        The gap López de Prado's purging requires: a label at the end of the
        training window can depend on data inside the test window.
        """
        folds = generate_folds(
            datetime(2020, 1, 1),
            datetime(2024, 1, 1),
            train_period=timedelta(days=365),
            test_period=timedelta(days=180),
            embargo=timedelta(days=21),
        )
        for fold in folds:
            assert (fold.test_start - fold.train_end).days == 21

    def test_too_short_a_range_raises_rather_than_returning_nothing(self):
        """A silent zero-fold run reports a perfect record on no data."""
        with pytest.raises(ValueError, match="too short"):
            generate_folds(
                datetime(2020, 1, 1),
                datetime(2020, 6, 1),
                train_period=timedelta(days=365),
                test_period=timedelta(days=180),
            )

    def test_non_positive_periods_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            generate_folds(
                datetime(2020, 1, 1),
                datetime(2024, 1, 1),
                train_period=timedelta(0),
                test_period=timedelta(days=30),
            )


def _harness(n_days=800):
    bars = trending_bars(
        {"A": 100.0, "B": 50.0}, n_days=n_days, daily_return=0.0004, start="2020-01-01"
    )
    data = DataFrameDataHandler(bars)
    return data, WalkForward(
        data,
        strategy_factory=BuyAndHold,
        portfolio_factory=lambda: Portfolio(1_000_000.0),
        broker_factory=SimulatedBroker,
        constructor_factory=lambda: PortfolioConstructor(sizer=ExplicitWeightSizer()),
    )


class TestNoLeakage:
    def test_fit_never_sees_data_from_its_own_test_window(self):
        data, _ = _harness()
        seen: list[tuple[Fold, datetime]] = []

        def fit(strategy, train_data):
            stamps = []
            while train_data.has_more():
                for event in train_data.next_events():
                    stamps.append(event.timestamp)
            seen.append(max(stamps) if stamps else None)

        runner = WalkForward(
            data,
            strategy_factory=BuyAndHold,
            portfolio_factory=lambda: Portfolio(1_000_000.0),
            broker_factory=SimulatedBroker,
            constructor_factory=lambda: PortfolioConstructor(sizer=ExplicitWeightSizer()),
            fit=fit,
        )
        folds = generate_folds(
            datetime(2020, 1, 1),
            datetime(2023, 1, 1),
            train_period=timedelta(days=250),
            test_period=timedelta(days=120),
        )
        runner.run(folds)

        assert len(seen) == len(folds)
        for last_train_stamp, fold in zip(seen, folds):
            assert last_train_stamp is None or last_train_stamp < fold.test_start

    def test_each_fold_gets_a_fresh_strategy(self):
        """
        A reused instance carries state — a fitted parameter, an
        `already_invested` flag — from a later fold's past into its present.
        """
        data, runner = _harness()
        instances = []

        def factory():
            strategy = BuyAndHold()
            instances.append(strategy)
            return strategy

        runner.strategy_factory = factory
        folds = generate_folds(
            datetime(2020, 1, 1),
            datetime(2023, 1, 1),
            train_period=timedelta(days=250),
            test_period=timedelta(days=120),
        )
        runner.run(folds)

        assert len(instances) == len(folds)
        assert len({id(s) for s in instances}) == len(folds)

    def test_each_fold_starts_from_clean_capital(self):
        data, runner = _harness()
        folds = generate_folds(
            datetime(2020, 1, 1),
            datetime(2023, 1, 1),
            train_period=timedelta(days=250),
            test_period=timedelta(days=120),
        )
        result = runner.run(folds)

        assert all(f.result.initial_capital == 1_000_000.0 for f in result.folds)


class TestResults:
    def _run(self):
        data, runner = _harness()
        folds = generate_folds(
            datetime(2020, 1, 1),
            datetime(2023, 1, 1),
            train_period=timedelta(days=250),
            test_period=timedelta(days=120),
        )
        return runner.run(folds)

    def test_oos_returns_are_stitched_in_time_order(self):
        result = self._run()
        returns = result.oos_returns
        assert not returns.empty
        assert returns.index.is_monotonic_increasing

    def test_oos_curve_compounds_the_stitched_returns(self):
        result = self._run()
        curve = result.oos_equity_curve(initial_capital=100.0)
        assert curve["nav"].iloc[0] > 0
        assert len(curve) == len(result.oos_returns)

    def test_oos_report_is_computable(self):
        report = self._run().evaluate_oos()
        assert report.n_periods > 0

    def test_summary_frame_has_one_row_per_fold(self):
        result = self._run()
        frame = result.summary_frame()
        assert len(frame) == len(result.folds)
        assert set(["fold", "sharpe", "total_return", "test_start"]).issubset(frame.columns)

    def test_consistency_is_the_fraction_of_winning_folds(self):
        """
        Worth more than the average: a strategy that made all its money in one
        fold out of eight worked once, however good the pooled Sharpe looks.
        """
        result = self._run()
        assert 0.0 <= result.consistency <= 1.0
        winners = sum(1 for f in result.folds if f.report.total_return > 0)
        assert result.consistency == pytest.approx(winners / len(result.folds))

    def test_a_steadily_rising_market_wins_every_fold(self):
        """Sanity floor: buy-and-hold on a monotone uptrend cannot lose."""
        result = self._run()
        assert result.consistency == pytest.approx(1.0)

    def test_empty_result_reports_no_oos_returns(self):
        from evbt.walkforward.runner import WalkForwardResult

        empty = WalkForwardResult()
        assert empty.oos_returns.empty
        assert empty.consistency == 0.0
        with pytest.raises(ValueError, match="no out-of-sample"):
            empty.evaluate_oos()
