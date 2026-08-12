"""
Transaction costs: commission, spread, and market impact.

The three are modelled separately because they behave differently and are
calibrated from different evidence, and because a backtest needs to be able to
say *which* of them killed a strategy:

  - **Commission** is contractual. Known exactly, scales with shares or notional.
  - **Spread** is the cost of demanding liquidity now. Scales with shares, not
    with size relative to the market.
  - **Impact** is the cost of *being large*. Scales super-linearly with the
    fraction of volume you take, and it is the term that determines capacity.

Only the third one has any theory behind it, and it is the one that decides
whether a strategy that looks good at $10M still works at $1B.

Impact: the two models here
---------------------------
**Almgren-Chriss linear impact.** From Almgren & Chriss (2000), "Optimal
execution of portfolio transactions". Trading X shares over T with N intervals
of length tau = T/N, holdings x_k and trades n_k = x_{k-1} - x_k:

    permanent:  price drifts by  g(v) = gamma * v      per unit time
    temporary:  you pay          h(v) = epsilon + eta * v   per share

with v = n_k / tau the trading rate. Expected implementation shortfall is

    E[X] = (gamma / 2) X^2  +  epsilon * sum |n_k|  +  (eta_tilde / tau) sum n_k^2
    eta_tilde = eta - gamma * tau / 2

and its variance is  V[X] = sigma^2 * sum tau * x_k^2.

The `gamma X^2 / 2` term is the whole reason permanent impact is charged at
*half* the final displacement: the price walks away linearly as you trade, so
the average price you pay reflects half the total move. Charging the full
displacement double-counts, and is a common error.

**Square-root law.** The empirically dominant form (Almgren et al. 2005; Torre's
BARRA market impact model; Grinold & Kahn):

    dP / P  =  Y * sigma * sqrt(Q / V)

with sigma the daily volatility, V the daily volume, Y an O(1) constant usually
fitted between 0.3 and 1.0. It fits observed institutional trade data far
better than a linear model, and it is why capacity degrades as the square root
of AUM rather than linearly. The spec asks for linear impact at minimum and a
nonlinear component ideally; both are here, and `CompositeImpact` runs them
together.

Calibration honesty
-------------------
The default coefficients below are round numbers from the published literature,
not values fitted to any dataset in this repo. They are a starting point for a
sensitivity analysis, not a measurement. Any result that depends on the exact
value of `eta` should be reported as a range across plausible coefficients —
`analytics.capacity` does exactly that.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from evbt.core.events import Bar, OrderSide

# One basis point as a fraction. Spelled out because the number of costing bugs
# caused by a factor of 10,000 is not small.
BPS = 1e-4


# ---------------------------------------------------------------------------
# Commission
# ---------------------------------------------------------------------------


class CommissionModel(ABC):
    """Contractual cost of a trade. Always non-negative, always paid in cash."""

    @abstractmethod
    def calculate(self, quantity: float, price: float, symbol: str = "") -> float:
        """Commission in currency for `quantity` shares at `price`."""


@dataclass
class ZeroCommission(CommissionModel):
    """
    No commission. The right choice for validation runs — the done-criterion
    replication has to match a published gross return series, and any cost at
    all makes that comparison meaningless. Never the right choice for a result.
    """

    def calculate(self, quantity: float, price: float, symbol: str = "") -> float:
        return 0.0


@dataclass
class PerShareCommission(CommissionModel):
    """
    Interactive-Brokers-style: a rate per share, floored at a minimum per order
    and capped at a percentage of notional.

    The cap matters for penny stocks, where a flat per-share rate can otherwise
    exceed the value of the trade — a good way to discover that a backtest is
    trading names it should have screened out.
    """

    rate_per_share: float = 0.005
    minimum: float = 1.0
    max_pct_of_notional: float = 0.01

    def calculate(self, quantity: float, price: float, symbol: str = "") -> float:
        if quantity <= 0:
            return 0.0
        raw = max(quantity * self.rate_per_share, self.minimum)
        return min(raw, quantity * price * self.max_pct_of_notional)


@dataclass
class BpsCommission(CommissionModel):
    """
    A flat number of basis points of notional, optionally floored.

    The usual institutional convention for all-in commission plus fees, and the
    easiest to reason about when quoting a strategy's cost sensitivity.
    """

    bps: float = 1.0
    minimum: float = 0.0

    def calculate(self, quantity: float, price: float, symbol: str = "") -> float:
        if quantity <= 0:
            return 0.0
        return max(quantity * price * self.bps * BPS, self.minimum)


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------


class SpreadModel(ABC):
    """
    Cost of crossing the bid-ask spread, expressed per share.

    A marketable order pays half the spread by construction: the mid is the
    fair price, and you transact at the far touch.
    """

    @abstractmethod
    def half_spread(self, bar: Bar, symbol: str = "") -> float:
        """Half-spread in currency per share."""

    def cost_per_share(self, bar: Bar, symbol: str = "") -> float:
        return self.half_spread(bar, symbol)


@dataclass
class ZeroSpread(SpreadModel):
    def half_spread(self, bar: Bar, symbol: str = "") -> float:
        return 0.0


@dataclass
class FixedBpsSpread(SpreadModel):
    """
    A constant spread in basis points of price, with per-symbol overrides.

    5 bps is a reasonable large-cap US equity assumption; small caps run 20-50
    and illiquid names far more. The override dict exists because using one
    number across a universe that spans both is the assumption most likely to
    flatter a small-cap strategy.
    """

    bps: float = 5.0
    overrides: Optional[dict[str, float]] = None

    def half_spread(self, bar: Bar, symbol: str = "") -> float:
        key = symbol or bar.symbol
        bps = (self.overrides or {}).get(key, self.bps)
        return bar.close * bps * BPS / 2.0


@dataclass
class VolatilityScaledSpread(SpreadModel):
    """
    Spread proportional to realised volatility, floored at a minimum.

    Spreads widen in stressed markets, and a strategy that trades *because*
    markets are stressed — most mean-reversion, most risk-off signals — meets
    its worst spreads exactly when it trades most. A constant spread hides that
    correlation entirely, and it is a first-order effect on such strategies.

    `k` is the ratio of half-spread to daily vol; 0.05 means a name with 2%
    daily vol quotes a 20 bps spread.
    """

    k: float = 0.05
    min_bps: float = 1.0

    def half_spread(self, bar: Bar, symbol: str = "") -> float:
        floor = bar.close * self.min_bps * BPS / 2.0
        if bar.volatility is None:
            return floor
        return max(floor, bar.close * self.k * bar.volatility)


# ---------------------------------------------------------------------------
# Market impact
# ---------------------------------------------------------------------------


class ImpactModel(ABC):
    """
    Price displacement caused by the trade itself.

    Returns cost **per share**, always positive; the broker applies the sign.
    Split into temporary and permanent because they are charged differently:
    temporary impact is paid entirely by this trade, permanent impact moves the
    market for everyone afterwards and this trade pays half of it on average.
    """

    @abstractmethod
    def temporary(self, quantity: float, bar: Bar) -> float:
        """Temporary impact per share, in currency."""

    @abstractmethod
    def permanent(self, quantity: float, bar: Bar) -> float:
        """Permanent displacement per share, in currency (full, not halved)."""

    def cost_per_share(self, quantity: float, bar: Bar) -> float:
        """
        All-in impact charged to this trade.

        Permanent impact enters at half its magnitude: the price walks away
        linearly over the course of the execution, so the average transacted
        price sits at the midpoint of the walk. This is the `gamma * X^2 / 2`
        term of Almgren-Chriss written per share.
        """
        return self.temporary(quantity, bar) + 0.5 * self.permanent(quantity, bar)

    @staticmethod
    def participation(quantity: float, bar: Bar) -> float:
        """
        Fraction of the bar's volume this trade represents.

        Guarded against zero-volume bars — halted or barely-traded names do
        occur in real data, and dividing by their volume produces an infinite
        cost that propagates as NaN through the entire equity curve. Returning
        1.0 says "you are the entire market", which is the correct reading of a
        trade on a bar with no volume, and it makes the trade appropriately
        expensive rather than nonsensical.
        """
        if bar.volume <= 0:
            return 1.0
        return min(abs(quantity) / bar.volume, 1.0)


@dataclass
class ZeroImpact(ImpactModel):
    def temporary(self, quantity: float, bar: Bar) -> float:
        return 0.0

    def permanent(self, quantity: float, bar: Bar) -> float:
        return 0.0


@dataclass
class AlmgrenChrissImpact(ImpactModel):
    """
    Linear impact in participation rate — the Almgren-Chriss form.

        temporary per share = eta * (Q / V) * price
        permanent per share = gamma * (Q / V) * price

    Coefficients are dimensionless fractions of price per unit participation.
    `eta = 0.01` means that executing 100% of a bar's volume costs 100 bps of
    temporary impact; taking a more typical 10% costs 10 bps.

    Linear impact is theoretically convenient — it is what makes the optimal
    execution problem solvable in closed form, see `optimal_schedule` — and
    empirically too weak at small sizes and too strong at large ones. Use it
    for the closed-form work and for comparability with the literature; use
    `SquareRootImpact` when the number is meant to be believed.
    """

    eta: float = 0.01     # temporary impact coefficient
    gamma: float = 0.005  # permanent impact coefficient

    def temporary(self, quantity: float, bar: Bar) -> float:
        return self.eta * self.participation(quantity, bar) * bar.close

    def permanent(self, quantity: float, bar: Bar) -> float:
        return self.gamma * self.participation(quantity, bar) * bar.close


@dataclass
class SquareRootImpact(ImpactModel):
    """
    The square-root law:  dP / P = Y * sigma * sqrt(Q / V).

    The best-supported empirical impact model, stable across markets, asset
    classes and decades. `Y` is fitted in the literature between roughly 0.3
    and 1.0; 0.5 is the usual round-number default.

    Concavity is the whole point. Doubling trade size raises cost per share by
    only sqrt(2), so cost *per share* falls with participation relative to a
    linear model at small sizes and rises more slowly at large ones. That
    concavity is what makes capacity scale as the square of the acceptable cost
    rather than linearly with it.

    `sigma` comes from the bar's trailing volatility. Bars without one — the
    first few of any symbol's history — fall back to `default_volatility`
    rather than silently costing nothing.
    """

    Y: float = 0.5
    default_volatility: float = 0.02
    permanent_fraction: float = 0.3  # share of total impact that persists

    def _total_per_share(self, quantity: float, bar: Bar) -> float:
        sigma = bar.volatility if bar.volatility is not None else self.default_volatility
        return self.Y * sigma * math.sqrt(self.participation(quantity, bar)) * bar.close

    def temporary(self, quantity: float, bar: Bar) -> float:
        return (1.0 - self.permanent_fraction) * self._total_per_share(quantity, bar)

    def permanent(self, quantity: float, bar: Bar) -> float:
        return self.permanent_fraction * self._total_per_share(quantity, bar)


@dataclass
class CompositeImpact(ImpactModel):
    """Sum of several impact models — e.g. a linear floor plus a square-root term."""

    models: list[ImpactModel]

    def temporary(self, quantity: float, bar: Bar) -> float:
        return sum(m.temporary(quantity, bar) for m in self.models)

    def permanent(self, quantity: float, bar: Bar) -> float:
        return sum(m.permanent(quantity, bar) for m in self.models)


# ---------------------------------------------------------------------------
# Optimal execution schedule
# ---------------------------------------------------------------------------


def almgren_chriss_schedule(
    total_shares: float,
    n_intervals: int,
    *,
    volatility: float,
    eta: float,
    gamma: float,
    risk_aversion: float,
    interval_length: float = 1.0,
) -> list[float]:
    """
    The Almgren-Chriss optimal execution trajectory.

    Minimising  E[cost] + lambda * Var[cost]  over trajectories x_0..x_N with
    x_0 = X and x_N = 0 gives

        x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T)

    where kappa solves

        2 * (cosh(kappa * tau) - 1) / tau^2  =  lambda * sigma^2 / eta_tilde
        eta_tilde = eta - gamma * tau / 2

    Read the two limits, because they are the entire economic content:

      - lambda -> 0 (risk-neutral): kappa -> 0 and the trajectory becomes
        linear in time. That is TWAP. A trader indifferent to risk should trade
        as slowly and evenly as possible, because impact cost is convex in rate.
      - lambda -> infinity (risk-averse): kappa -> infinity and the trajectory
        collapses toward the front. Urgency buys certainty at the price of
        impact.

    The trade-off between them is the efficient frontier of execution, and it
    is the answer to "why not just TWAP everything": TWAP leaves you holding
    inventory, and inventory has variance sigma^2 * tau * x^2 per interval.

    Returns the *holdings* trajectory `[x_0, ..., x_N]`, length `n_intervals+1`,
    starting at `total_shares` and ending at 0. Differencing it gives the trade
    list.
    """
    if n_intervals < 1:
        raise ValueError("n_intervals must be at least 1")
    if total_shares == 0:
        return [0.0] * (n_intervals + 1)

    tau = interval_length
    horizon = n_intervals * tau
    eta_tilde = eta - gamma * tau / 2.0
    if eta_tilde <= 0:
        raise ValueError(
            f"eta_tilde = eta - gamma*tau/2 = {eta_tilde:.6g} must be positive; "
            "permanent impact is overwhelming temporary impact, which makes the "
            "problem ill-posed (trading faster would appear free)"
        )

    if risk_aversion <= 0:
        # Risk-neutral limit: TWAP. Solved as a special case rather than by
        # taking kappa -> 0 numerically, where sinh(0)/sinh(0) is 0/0.
        return [total_shares * (1.0 - j / n_intervals) for j in range(n_intervals + 1)]

    # Solve the exact discrete relation rather than the small-tau approximation
    # kappa ~= sqrt(lambda * sigma^2 / eta_tilde). With daily bars tau is not
    # small relative to the horizon, and the approximation visibly drifts.
    rhs = risk_aversion * volatility**2 / eta_tilde
    kappa = math.acosh(1.0 + rhs * tau**2 / 2.0) / tau

    sinh_kt = math.sinh(kappa * horizon)
    return [
        total_shares * math.sinh(kappa * (horizon - j * tau)) / sinh_kt
        for j in range(n_intervals + 1)
    ]
