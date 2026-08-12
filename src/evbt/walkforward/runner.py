"""
Walk-forward evaluation.

A single backtest over one period answers "would this have worked?". Walk-
forward answers the question that matters: "would a decision made using only
past data have worked, repeatedly?" The difference is whether the parameters
were chosen with knowledge of the period they are evaluated on.

The structure is a sequence of folds, each with a training window used to fit or
select, and a test window that follows it in time and is scored:

    |---- train ----|--embargo--|-- test --|
                    |---- train ----|--embargo--|-- test --|
                                    |---- train ----|--embargo--|-- test --|

Two knobs are the whole point.

**Anchored vs rolling.** Anchored keeps every training window starting at the
beginning of history, so later folds train on more data. Rolling holds the
window length fixed and slides it. Anchored assumes the distant past is still
informative; rolling assumes it is not. Which is right is a claim about the
strategy, and the framework refuses to choose for you.

**The embargo.** A gap between the end of training and the start of testing.
It exists because a label at time t can depend on data after t — a five-day
forward return computed at the end of the training window overlaps the first
five days of the test window, and training on it is leakage even though every
timestamp looks correctly ordered. López de Prado (*Advances in Financial
Machine Learning*, ch. 7) calls this purging and embargoing; the embargo must be
at least as long as the forward-looking horizon of the labels. Default is zero
because the correct value is a property of the strategy, and a wrong non-zero
default would be worse than making the choice explicit.

What this deliberately does not do
----------------------------------
It does not select parameters. `fit` is a callback the caller supplies, and what
happens inside it — a grid search, a model fit, nothing at all — is the caller's
business. The framework's only job is to guarantee that whatever `fit` sees is
strictly in the past of what `score` is measured on, and that guarantee is what
`test_walkforward.py` checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Iterator, Optional, Sequence

import pandas as pd

from evbt.analytics.metrics import PerformanceReport, evaluate
from evbt.data.frame import DataFrameDataHandler
from evbt.engine import Backtest, BacktestResult
from evbt.execution.broker import SimulatedBroker
from evbt.portfolio.construction import PortfolioConstructor
from evbt.portfolio.portfolio import Portfolio
from evbt.strategy.base import Strategy

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """
    One train/test split. Both windows are **half-open**: `[start, end)`.

    Half-open rather than inclusive, and the reason is a one-bar leak that no
    summary statistic would reveal. With inclusive bounds and a zero embargo,
    `train_end == test_start`, so the bar at that instant is both trained on and
    scored. The same collision recurs between consecutive test windows, which
    would additionally double-count that bar in the stitched out-of-sample
    series. Half-open windows tile the timeline exactly once.
    """

    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"fold {self.index}: train {self.train_start:%Y-%m-%d}..{self.train_end:%Y-%m-%d}"
            f" -> test {self.test_start:%Y-%m-%d}..{self.test_end:%Y-%m-%d}"
        )


@dataclass
class FoldResult:
    fold: Fold
    result: BacktestResult
    report: PerformanceReport


@dataclass
class WalkForwardResult:
    """Per-fold results plus the stitched out-of-sample record."""

    folds: list[FoldResult] = field(default_factory=list)

    @property
    def oos_returns(self) -> pd.Series:
        """
        Every test window's returns concatenated in time order.

        This is the series to quote. It is the only one in the whole framework
        that was never seen during fitting, and stitching per-fold *returns*
        rather than NAVs is what makes the concatenation meaningful — each fold
        restarts from the same capital, so its NAV levels are not comparable
        across folds but its returns are.
        """
        pieces = [f.result.returns() for f in self.folds]
        pieces = [p for p in pieces if not p.empty]
        if not pieces:
            return pd.Series(dtype=float)
        return pd.concat(pieces).sort_index()

    def oos_equity_curve(self, initial_capital: float = 1.0) -> pd.DataFrame:
        """Compounded OOS returns as a NAV series, for metric computation."""
        returns = self.oos_returns
        if returns.empty:
            return pd.DataFrame()
        nav = initial_capital * (1.0 + returns).cumprod()
        return pd.DataFrame({"nav": nav})

    def evaluate_oos(self, periods_per_year: int = 252) -> PerformanceReport:
        curve = self.oos_equity_curve()
        if curve.empty:
            raise ValueError("no out-of-sample returns to evaluate")
        return evaluate(curve, periods_per_year=periods_per_year)

    def summary_frame(self) -> pd.DataFrame:
        """One row per fold — the table that shows whether results are stable."""
        return pd.DataFrame(
            [
                {
                    "fold": f.fold.index,
                    "train_start": f.fold.train_start,
                    "train_end": f.fold.train_end,
                    "test_start": f.fold.test_start,
                    "test_end": f.fold.test_end,
                    "total_return": f.report.total_return,
                    "sharpe": f.report.sharpe,
                    "max_drawdown": f.report.max_drawdown,
                    "turnover": f.report.annualised_turnover,
                    "n_fills": f.result.n_fills,
                }
                for f in self.folds
            ]
        )

    @property
    def consistency(self) -> float:
        """
        Fraction of folds with a positive return.

        Worth more than the average. A strategy that made all its money in one
        fold out of eight is a strategy that worked once, however good the
        pooled Sharpe looks.
        """
        if not self.folds:
            return 0.0
        return sum(1 for f in self.folds if f.report.total_return > 0) / len(self.folds)


def generate_folds(
    start: datetime,
    end: datetime,
    *,
    train_period: timedelta,
    test_period: timedelta,
    embargo: timedelta = timedelta(0),
    anchored: bool = False,
) -> list[Fold]:
    """
    Build the fold schedule. Windows never overlap between train and test.

    Raises if the range cannot fit even one fold, rather than returning an empty
    list — a silent zero-fold walk-forward reports a perfect record on no data.
    """
    if train_period <= timedelta(0) or test_period <= timedelta(0):
        raise ValueError("train_period and test_period must be positive")

    folds: list[Fold] = []
    train_start = start
    train_end = start + train_period

    while True:
        test_start = train_end + embargo
        test_end = test_start + test_period
        if test_end > end:
            break

        folds.append(
            Fold(
                index=len(folds),
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        train_end = test_end
        if not anchored:
            train_start = train_end - train_period

    if not folds:
        raise ValueError(
            f"range {start:%Y-%m-%d}..{end:%Y-%m-%d} is too short for "
            f"train={train_period.days}d + embargo={embargo.days}d + "
            f"test={test_period.days}d"
        )
    return folds


class WalkForward:
    """
    Runs a strategy fold by fold, scoring only the test windows.

    Parameters
    ----------
    strategy_factory
        Called once per fold to produce a *fresh* strategy. Reusing one instance
        across folds carries state — a fitted parameter, an `already_invested`
        flag — from a later fold's past into its own present, which is precisely
        the leak this framework exists to prevent.
    fit
        Optional `(strategy, train_data) -> None` hook, called with a data
        handler restricted to the training window before the test run. Whatever
        selection happens belongs here.
    portfolio_factory / broker_factory / constructor_factory
        Called per fold so each starts from clean state.
    """

    def __init__(
        self,
        data: DataFrameDataHandler,
        strategy_factory: Callable[[], Strategy],
        portfolio_factory: Callable[[], Portfolio],
        broker_factory: Callable[[], SimulatedBroker],
        constructor_factory: Callable[[], PortfolioConstructor],
        *,
        fit: Optional[Callable[[Strategy, DataFrameDataHandler], None]] = None,
        periods_per_year: int = 252,
        **backtest_kwargs,
    ) -> None:
        self.data = data
        self.strategy_factory = strategy_factory
        self.portfolio_factory = portfolio_factory
        self.broker_factory = broker_factory
        self.constructor_factory = constructor_factory
        self.fit = fit
        self.periods_per_year = periods_per_year
        self.backtest_kwargs = backtest_kwargs

    def run(self, folds: Sequence[Fold]) -> WalkForwardResult:
        out = WalkForwardResult()

        for fold in folds:
            strategy = self.strategy_factory()

            # Half-open on both windows — see `Fold`. This is the line that
            # keeps the boundary bar out of two places at once.
            if self.fit is not None:
                train_data = self.data.restricted_to(
                    fold.train_start, fold.train_end, end_inclusive=False
                )
                self.fit(strategy, train_data)

            test_data = self.data.restricted_to(
                fold.test_start, fold.test_end, end_inclusive=False
            )
            engine = Backtest(
                test_data,
                strategy,
                self.portfolio_factory(),
                self.broker_factory(),
                self.constructor_factory(),
                **self.backtest_kwargs,
            )
            result = engine.run()

            if result.equity_curve.empty:
                log.warning("fold %d produced no equity curve; skipping", fold.index)
                continue

            out.folds.append(
                FoldResult(
                    fold=fold,
                    result=result,
                    report=evaluate(
                        result.equity_curve,
                        result.fills,
                        periods_per_year=self.periods_per_year,
                    ),
                )
            )

        return out
