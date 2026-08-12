"""
Capacity: how much money the strategy can run before its costs eat its edge.

The question that separates a backtest from an investable strategy, and the one
a naive engine cannot ask at all — if costs do not scale with size, capacity is
infinite and the model is silent.

The estimate
------------
Gross alpha per year is (roughly) invariant to AUM: doubling the book doubles
the dollars earned, so the *rate* stays put. Cost per dollar traded is not. Under
the square-root law,

    cost per share / price  =  Y * sigma * sqrt(Q / V)

and Q scales linearly with AUM, so cost as a *fraction of AUM* scales as
sqrt(AUM). Net alpha is therefore

    net(A)  =  gross  -  c * sqrt(A / A_0)

with `c` the annual cost rate measured at the backtest's own size `A_0`. Setting
net(A) = 0:

    A_max  =  A_0 * (gross / c)^2

The square is the entire economic content. Halving your cost rate quadruples
capacity; a strategy with twice the gross alpha of another, at the same cost
rate, holds four times the money. It is also why capacity is so sensitive to
the impact coefficient, and why a single point estimate is not a useful answer —
`capacity_curve` sweeps a range instead.

What this does not model
------------------------
Alpha decay with size. A large book takes longer to build, and a signal with a
five-day half-life has decayed materially before the position is on. That effect
is real, is strategy-specific, and needs the signal's decay profile (Project 6)
to quantify — so the numbers here are an *upper* bound on capacity, not a
central estimate. Treat them as "no more than this".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class CapacityEstimate:
    """Capacity implied by one backtest, with the assumptions that produced it."""

    backtest_aum: float
    gross_alpha: float          # annualised, before costs
    cost_rate: float            # annualised costs as a fraction of AUM
    net_alpha: float
    capacity_aum: float         # AUM at which net alpha reaches zero
    half_alpha_aum: float       # AUM at which half the gross alpha survives
    scaling_exponent: float

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return "\n".join(
            [
                f"backtest AUM      {self.backtest_aum:,.0f}",
                f"gross alpha       {self.gross_alpha:+.2%} / yr",
                f"cost rate         {self.cost_rate:.2%} / yr",
                f"net alpha         {self.net_alpha:+.2%} / yr",
                f"capacity (net=0)  {self.capacity_aum:,.0f}",
                f"half-alpha AUM    {self.half_alpha_aum:,.0f}",
                f"cost scaling      AUM^{self.scaling_exponent:.2f}",
            ]
        )


def estimate_capacity(
    backtest_aum: float,
    gross_annual_return: float,
    annual_cost_rate: float,
    *,
    scaling_exponent: float = 0.5,
) -> CapacityEstimate:
    """
    Capacity from a backtest's own gross return and measured cost rate.

    Parameters
    ----------
    scaling_exponent
        How cost per dollar grows with AUM. 0.5 is the square-root law and the
        default. 1.0 is linear (Almgren-Chriss), which is far more pessimistic
        and is worth running as the conservative bound. 0.0 would mean costs do
        not scale at all, i.e. infinite capacity — which is what a fixed-bps
        cost model implicitly assumes, and why such a model cannot answer this
        question.
    """
    if backtest_aum <= 0:
        raise ValueError("backtest_aum must be positive")
    if annual_cost_rate <= 0:
        return CapacityEstimate(
            backtest_aum=backtest_aum,
            gross_alpha=gross_annual_return,
            cost_rate=0.0,
            net_alpha=gross_annual_return,
            capacity_aum=float("inf"),
            half_alpha_aum=float("inf"),
            scaling_exponent=scaling_exponent,
        )

    net = gross_annual_return - annual_cost_rate

    def aum_for_surviving_alpha(target: float) -> float:
        """AUM at which gross - cost(A) equals `target`."""
        allowed = gross_annual_return - target
        if allowed <= 0:
            return 0.0
        return backtest_aum * (allowed / annual_cost_rate) ** (1.0 / scaling_exponent)

    return CapacityEstimate(
        backtest_aum=backtest_aum,
        gross_alpha=gross_annual_return,
        cost_rate=annual_cost_rate,
        net_alpha=net,
        capacity_aum=aum_for_surviving_alpha(0.0),
        half_alpha_aum=aum_for_surviving_alpha(gross_annual_return / 2.0),
        scaling_exponent=scaling_exponent,
    )


def capacity_curve(
    backtest_aum: float,
    gross_annual_return: float,
    annual_cost_rate: float,
    aum_levels: Optional[Sequence[float]] = None,
    *,
    scaling_exponent: float = 0.5,
) -> pd.DataFrame:
    """
    Net alpha across a range of AUM levels.

    Reported as a curve rather than a single number because the impact
    coefficient behind `annual_cost_rate` is calibrated to published estimates,
    not fitted to this data — so the honest output is a shape and a sensitivity,
    not a point.
    """
    if aum_levels is None:
        aum_levels = [backtest_aum * (10**k) for k in range(0, 5)]

    rows = []
    for aum in aum_levels:
        scaled_cost = annual_cost_rate * (aum / backtest_aum) ** scaling_exponent
        rows.append(
            {
                "aum": aum,
                "gross_alpha": gross_annual_return,
                "cost_rate": scaled_cost,
                "net_alpha": gross_annual_return - scaled_cost,
                "alpha_retained": (
                    (gross_annual_return - scaled_cost) / gross_annual_return
                    if gross_annual_return != 0
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def realised_cost_rate(
    fills: pd.DataFrame, equity_curve: pd.DataFrame, periods_per_year: int = 252
) -> float:
    """
    Annualised all-in cost as a fraction of average NAV, measured from a run.

    This is the `c` that `estimate_capacity` needs, and taking it from the
    engine's own fills rather than assuming a number is the point: it already
    reflects the participation rates the strategy actually ran at.
    """
    if fills.empty or equity_curve.empty or "total_cost" not in fills:
        return 0.0
    mean_nav = float(equity_curve["nav"].mean())
    if mean_nav <= 0:
        return 0.0
    years = max(len(equity_curve) / periods_per_year, 1e-9)
    return float(fills["total_cost"].sum()) / mean_nav / years
