"""
Performance metrics.

Two conventions are stated explicitly because codebases silently disagree on
them, and the disagreement is large enough to change conclusions:

**Annualisation.** 252 trading days, configurable. A Sharpe computed on 252 and
compared against one computed on 260 differs by 1.6% for no reason at all.

**Sortino's denominator.** Downside deviation divides the sum of squared
shortfalls by the total number of observations, not by the number of negative
ones. Dividing by the count of negatives is a common implementation error that
inflates the ratio, and it inflates it *most* for strategies that rarely lose —
exactly the ones a reader is most inclined to believe.

On reporting Sharpe honestly
----------------------------
`sharpe_standard_error` implements Lo (2002):

    SE(SR) ~= sqrt((1 + SR^2 / 2) / n)

Over three years of daily data (n ~= 756) the standard error is roughly 0.037,
so a measured Sharpe of 1.0 carries a 95% interval of about [0.93, 1.07] — and
that is *before* accounting for having tried more than one strategy. `summary()`
reports the t-statistic alongside the ratio for this reason. The project spec is
explicit that a Sharpe of 4 on US equity long-short is a red flag to report, not
a result to celebrate; the machinery here is built to make that reportable.

Lo's formula assumes IID returns. Strategies with autocorrelated returns —
anything holding illiquid positions, anything smoothing marks — violate it and
their true standard errors are larger. Noted rather than corrected: the
correction needs an autocorrelation model this module does not have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Below this, a standard deviation is floating-point residue rather than risk.
#
# The guard is not hypothetical. `pd.Series([0.001] * 100).std(ddof=1)` is about
# 2e-19, not 0.0, so an exact `sigma == 0` check misses it and the Sharpe ratio
# comes back as 7.3e16. Any strategy that sits in cash, or holds a single
# position through a flat stretch, hits this — and a metric that reports 1e16
# instead of "undefined" is worse than one that crashes.
ZERO_VOL_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# Return and risk
# ---------------------------------------------------------------------------


def total_return(nav: pd.Series) -> float:
    if len(nav) < 2:
        return 0.0
    return float(nav.iloc[-1] / nav.iloc[0] - 1.0)


def annualised_return(nav: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """
    Geometric, not arithmetic. The arithmetic mean of returns overstates the
    compounded outcome by roughly half the variance, which for a 20%-vol
    strategy is 2% a year of return that was never earned.
    """
    if len(nav) < 2:
        return 0.0
    n_periods = len(nav) - 1
    growth = float(nav.iloc[-1] / nav.iloc[0])
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / n_periods) - 1.0


def annualised_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS
) -> float:
    """Sample standard deviation (ddof=1), scaled by sqrt(time)."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Annualised Sharpe on *excess* returns.

    `risk_free_rate` is annualised and converted to a per-period rate by simple
    division. Exact de-compounding would be `(1+rf)^(1/n) - 1`; the difference
    at any plausible rate is under a basis point a year, far inside the
    estimation error of the ratio itself.
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    sigma = float(excess.std(ddof=1))
    if not math.isfinite(sigma) or sigma < ZERO_VOL_TOLERANCE:
        return 0.0
    return float(excess.mean() / sigma * math.sqrt(periods_per_year))


def sharpe_standard_error(sharpe: float, n_observations: int) -> float:
    """
    Lo (2002) standard error of an annualised Sharpe ratio.

    Derived for IID returns. The formula is on the *per-period* Sharpe, so the
    annualisation factor has to be undone and reapplied; doing that inline is
    why this is a function rather than a one-liner at every call site.
    """
    if n_observations < 2:
        return float("inf")
    return math.sqrt((1.0 + 0.5 * sharpe**2) / n_observations)


def sortino_ratio(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Return per unit of *downside* deviation.

    The denominator divides by the total number of observations, not the number
    of negative ones — see the module docstring. Returns infinity when nothing
    ever fell below the target, which is a signal to look at the sample rather
    than a number to quote.
    """
    if len(returns) < 2:
        return 0.0
    per_period_target = target / periods_per_year
    shortfall = np.minimum(returns.to_numpy() - per_period_target, 0.0)
    downside = math.sqrt(float(np.mean(shortfall**2)))
    if downside < ZERO_VOL_TOLERANCE:
        return float("inf")
    mean_excess = float(returns.mean()) - per_period_target
    return mean_excess / downside * math.sqrt(periods_per_year)


def max_drawdown(nav: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction."""
    if len(nav) < 2:
        return 0.0
    return float((nav / nav.cummax() - 1.0).min())


def drawdown_series(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def max_drawdown_duration(nav: pd.Series) -> int:
    """
    Longest run of periods spent below a previous peak.

    Often the metric that actually decides whether a strategy is fundable: a
    12% drawdown recovered in a month is tolerable, the same drawdown lasting
    three years is not, and Sharpe cannot tell the two apart.
    """
    if len(nav) < 2:
        return 0
    underwater = nav < nav.cummax()
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def calmar_ratio(nav: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised return over the absolute max drawdown."""
    drawdown = max_drawdown(nav)
    if drawdown == 0:
        return float("inf")
    return annualised_return(nav, periods_per_year) / abs(drawdown)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a positive return. Not a measure of skill —
    a strategy can win 90% of days and still lose money."""
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR: the empirical quantile of the loss tail."""
    if returns.empty:
        return 0.0
    return float(np.quantile(returns.to_numpy(), 1.0 - confidence))


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected shortfall — the mean loss conditional on breaching VaR."""
    if returns.empty:
        return 0.0
    threshold = value_at_risk(returns, confidence)
    tail = returns[returns <= threshold]
    return float(tail.mean()) if not tail.empty else threshold


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


def annualised_turnover(
    equity_curve: pd.DataFrame, periods_per_year: int = TRADING_DAYS
) -> float:
    """
    Two-way notional traded per year as a multiple of average NAV.

    Reported two-way (buys plus sells) because that is what costs are charged
    on. Much of the literature quotes one-way turnover, which is half this — a
    factor of two worth stating rather than leaving the reader to guess.
    """
    if equity_curve.empty or "turnover_notional" not in equity_curve:
        return 0.0
    n_periods = len(equity_curve)
    if n_periods == 0:
        return 0.0
    mean_nav = float(equity_curve["nav"].mean())
    if mean_nav <= 0:
        return 0.0
    traded = float(equity_curve["turnover_notional"].sum())
    return traded / mean_nav * (periods_per_year / n_periods)


def average_gross_leverage(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty or "gross_exposure" not in equity_curve:
        return 0.0
    nav = equity_curve["nav"].replace(0.0, np.nan)
    return float((equity_curve["gross_exposure"] / nav).mean())


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class PerformanceReport:
    """Every headline number for one run, plus what they are conditional on."""

    n_periods: int
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe: float
    sharpe_stderr: float
    sharpe_tstat: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int
    hit_rate: float
    var_95: float
    cvar_95: float
    annualised_turnover: float
    average_gross_leverage: float
    total_costs: float
    cost_drag_annualised: float
    periods_per_year: int = TRADING_DAYS
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = {k: v for k, v in vars(self).items() if k != "notes"}
        return pd.DataFrame({"metric": list(rows), "value": list(rows.values())})

    def __str__(self) -> str:  # pragma: no cover - presentation only
        lines = [
            f"periods            {self.n_periods:,}",
            f"total return       {self.total_return:+.2%}",
            f"annualised return  {self.annualised_return:+.2%}",
            f"annualised vol     {self.annualised_volatility:.2%}",
            f"Sharpe             {self.sharpe:.2f}  (SE {self.sharpe_stderr:.2f},"
            f" t={self.sharpe_tstat:.2f})",
            f"Sortino            {self.sortino:.2f}",
            f"Calmar             {self.calmar:.2f}",
            f"max drawdown       {self.max_drawdown:.2%}"
            f"  ({self.max_drawdown_duration} periods underwater)",
            f"hit rate           {self.hit_rate:.1%}",
            f"VaR / CVaR (95%)   {self.var_95:.2%} / {self.cvar_95:.2%}",
            f"turnover (2-way)   {self.annualised_turnover:.2f}x / yr",
            f"avg gross leverage {self.average_gross_leverage:.2f}x",
            f"total costs        {self.total_costs:,.2f}"
            f"  ({self.cost_drag_annualised:.2%} / yr)",
        ]
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)


def evaluate(
    equity_curve: pd.DataFrame,
    fills: Optional[pd.DataFrame] = None,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> PerformanceReport:
    """
    Compute every metric for one run.

    Attaches a note whenever the sample is too short for the Sharpe to mean
    anything. Two years of daily data gives a Sharpe standard error near 0.05;
    six months gives 0.09, at which point a measured 1.0 and a measured 0.5 are
    not distinguishable, and reporting the point estimate alone is misleading.
    """
    if equity_curve.empty or "nav" not in equity_curve:
        raise ValueError("equity_curve must be non-empty and contain a 'nav' column")

    nav = equity_curve["nav"]
    returns = nav.pct_change().dropna()

    sharpe = sharpe_ratio(returns, risk_free_rate, periods_per_year)
    stderr = sharpe_standard_error(sharpe, len(returns))
    years = max(len(nav) / periods_per_year, 1e-9)

    costs = 0.0
    if fills is not None and not fills.empty and "total_cost" in fills:
        costs = float(fills["total_cost"].sum())
    mean_nav = float(nav.mean())
    cost_drag = costs / mean_nav / years if mean_nav > 0 else 0.0

    notes: list[str] = []
    if len(returns) < 2 * periods_per_year:
        notes.append(
            f"only {len(returns)} return observations "
            f"({years:.1f} years) — the Sharpe standard error of "
            f"{stderr:.2f} is too wide to support a point estimate"
        )
    if sharpe > 3.0:
        notes.append(
            f"Sharpe of {sharpe:.2f} on this sample is implausibly high for a "
            "realistic equity strategy — check costs, universe survivorship, "
            "and lookahead before believing it"
        )

    return PerformanceReport(
        n_periods=len(nav),
        total_return=total_return(nav),
        annualised_return=annualised_return(nav, periods_per_year),
        annualised_volatility=annualised_volatility(returns, periods_per_year),
        sharpe=sharpe,
        sharpe_stderr=stderr,
        sharpe_tstat=sharpe / stderr if stderr > 0 else 0.0,
        sortino=sortino_ratio(returns, risk_free_rate, periods_per_year),
        calmar=calmar_ratio(nav, periods_per_year),
        max_drawdown=max_drawdown(nav),
        max_drawdown_duration=max_drawdown_duration(nav),
        hit_rate=hit_rate(returns),
        var_95=value_at_risk(returns, 0.95),
        cvar_95=conditional_value_at_risk(returns, 0.95),
        annualised_turnover=annualised_turnover(equity_curve, periods_per_year),
        average_gross_leverage=average_gross_leverage(equity_curve),
        total_costs=costs,
        cost_drag_annualised=cost_drag,
        periods_per_year=periods_per_year,
        notes=notes,
    )
