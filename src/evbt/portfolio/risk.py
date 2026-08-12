"""
Pre-trade risk constraints.

Constraints are applied to *target weights*, before any order exists. That
placement is the design decision worth defending: the alternative — letting the
strategy build whatever book it likes and rejecting individual orders at the
broker — produces a portfolio that depends on the order in which orders happen
to be submitted, which is arbitrary. Clipping the target book first gives a
deterministic, explainable result: this is the portfolio we wanted, this is the
nearest one we are allowed to hold, and here is every limit that bound.

Every adjustment is recorded as a `Violation`. A risk layer that silently
reshapes a book is worse than no risk layer, because the backtest then reports
the performance of a strategy nobody specified. The report surfaces which
limits bound and how often — if a limit binds on 90% of rebalances, the limit
is the strategy.

Order of application
--------------------
The sequence is fixed and matters, because later steps can undo earlier ones:

  1. **Position caps** — clip each |w_i|. Purely local.
  2. **Sector caps** — scale down any sector whose net exposure is too large.
     Can only reduce, so it cannot breach step 1.
  3. **Gross leverage** — scale the whole book. Uniform scaling preserves the
     relative bets, and cannot breach steps 1 or 2, both of which are also
     homogeneous of degree one in the weights.
  4. **Net leverage** — scale down whichever side dominates. This *can* breach
     nothing above it, again by being a pure reduction.
  5. **Turnover** — move only part of the way from the current book to the
     target. Applied last because it is a constraint on the *trade*, not on the
     portfolio, and because a partial move toward a compliant target is itself
     compliant: every constraint above is convex and satisfied at both ends of
     the segment, so it is satisfied everywhere on it.

That last argument is the reason this ordering works at all, and it is worth
stating explicitly: each constraint defines a convex set, the interpolation in
step 5 stays inside the intersection, so no step ever needs revisiting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Optional

EPS = 1e-12


@dataclass
class Violation:
    """One constraint that bound on one rebalance, with what it did."""

    timestamp: Optional[datetime]
    constraint: str
    detail: str
    requested: float
    allowed: float

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"{self.constraint}: {self.detail} "
            f"(requested {self.requested:.4f}, allowed {self.allowed:.4f})"
        )


@dataclass
class RiskLimits:
    """
    The constraint set. `None` disables a limit.

    Weights are fractions of NAV, signed. A market-neutral equity book at 1.0x
    gross would typically run something like
    `max_position_weight=0.02, max_gross_leverage=1.0, max_net_leverage=0.1,
    max_sector_weight=0.15, max_turnover=0.25`.
    """

    max_position_weight: Optional[float] = None
    max_gross_leverage: Optional[float] = None
    max_net_leverage: Optional[float] = None
    max_sector_weight: Optional[float] = None
    max_turnover: Optional[float] = None

    def __post_init__(self) -> None:
        for name in (
            "max_position_weight",
            "max_gross_leverage",
            "max_sector_weight",
            "max_turnover",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.max_net_leverage is not None and self.max_net_leverage < 0:
            raise ValueError("max_net_leverage must be non-negative")


class RiskManager:
    """
    Applies `RiskLimits` to a target book and records what bound.

    Parameters
    ----------
    limits
        The constraint set.
    sectors
        Symbol to sector mapping. Required only if `max_sector_weight` is set;
        symbols missing from it are pooled into an "UNKNOWN" sector, which is
        deliberately *not* exempt — an unmapped symbol should show up as a
        constraint problem, not vanish from the risk calculation.
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        sectors: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.sectors = dict(sectors or {})
        self.violations: list[Violation] = []

    # --- entry point -------------------------------------------------------

    def apply(
        self,
        targets: Mapping[str, float],
        current: Optional[Mapping[str, float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> dict[str, float]:
        """
        Return the nearest compliant book to `targets`.

        `current` is the present weight of each symbol, needed only for the
        turnover constraint. Symbols absent from `current` are assumed flat.
        """
        weights = {s: float(w) for s, w in targets.items()}
        current = dict(current or {})

        weights = self._apply_position_caps(weights, timestamp)
        weights = self._apply_sector_caps(weights, timestamp)
        weights = self._apply_gross_leverage(weights, timestamp)
        weights = self._apply_net_leverage(weights, timestamp)
        weights = self._apply_turnover(weights, current, timestamp)
        return weights

    # --- individual constraints --------------------------------------------

    def _apply_position_caps(
        self, weights: dict[str, float], timestamp: Optional[datetime]
    ) -> dict[str, float]:
        cap = self.limits.max_position_weight
        if cap is None:
            return weights

        out = {}
        for symbol, weight in weights.items():
            if abs(weight) > cap + EPS:
                self._record(
                    timestamp, "max_position_weight", symbol, abs(weight), cap
                )
                out[symbol] = cap if weight > 0 else -cap
            else:
                out[symbol] = weight
        return out

    def _apply_sector_caps(
        self, weights: dict[str, float], timestamp: Optional[datetime]
    ) -> dict[str, float]:
        cap = self.limits.max_sector_weight
        if cap is None:
            return weights

        by_sector: dict[str, list[str]] = {}
        for symbol in weights:
            by_sector.setdefault(self.sectors.get(symbol, "UNKNOWN"), []).append(symbol)

        out = dict(weights)
        for sector, symbols in by_sector.items():
            net = sum(out[s] for s in symbols)
            if abs(net) <= cap + EPS:
                continue
            self._record(timestamp, "max_sector_weight", sector, abs(net), cap)
            # Scale the sector's *net* exposure down while preserving the
            # relative sizing within it. Scaling every leg by the same factor
            # also shrinks the sector's gross, which is a side effect worth
            # knowing about: a sector that is internally hedged gets penalised
            # for its net even though its risk is small. A factor-model-based
            # constraint (Project 10) is the principled fix; this is the
            # standard exposure-based approximation.
            scale = cap / abs(net)
            for s in symbols:
                out[s] *= scale
        return out

    def _apply_gross_leverage(
        self, weights: dict[str, float], timestamp: Optional[datetime]
    ) -> dict[str, float]:
        cap = self.limits.max_gross_leverage
        if cap is None:
            return weights

        gross = sum(abs(w) for w in weights.values())
        if gross <= cap + EPS or gross <= EPS:
            return weights

        self._record(timestamp, "max_gross_leverage", "book", gross, cap)
        scale = cap / gross
        return {s: w * scale for s, w in weights.items()}

    def _apply_net_leverage(
        self, weights: dict[str, float], timestamp: Optional[datetime]
    ) -> dict[str, float]:
        cap = self.limits.max_net_leverage
        if cap is None:
            return weights

        net = sum(weights.values())
        if abs(net) <= cap + EPS:
            return weights

        self._record(timestamp, "max_net_leverage", "book", abs(net), cap)

        # Reduce the dominant side rather than adding to the other. Both fix the
        # net; only reduction is guaranteed not to breach the gross limit
        # already applied above, and only reduction avoids putting on positions
        # the strategy never asked for purely to satisfy an accounting identity.
        dominant_is_long = net > 0
        side = {
            s: w
            for s, w in weights.items()
            if (w > 0) == dominant_is_long and abs(w) > EPS
        }
        side_total = sum(abs(w) for w in side.values())
        excess = abs(net) - cap

        if side_total <= EPS:
            return weights
        # Cannot remove more than the side contains; if the excess exceeds it,
        # flatten that side entirely and accept the residual breach.
        reduction = min(excess, side_total)
        scale = 1.0 - reduction / side_total

        return {s: (w * scale if s in side else w) for s, w in weights.items()}

    def _apply_turnover(
        self,
        weights: dict[str, float],
        current: Mapping[str, float],
        timestamp: Optional[datetime],
    ) -> dict[str, float]:
        cap = self.limits.max_turnover
        if cap is None:
            return weights

        symbols = set(weights) | set(current)
        turnover = sum(
            abs(weights.get(s, 0.0) - current.get(s, 0.0)) for s in symbols
        )
        if turnover <= cap + EPS or turnover <= EPS:
            return weights

        self._record(timestamp, "max_turnover", "book", turnover, cap)
        # Move a fraction of the way from the current book to the target. This
        # is what a real turnover budget does — it slows the rebalance rather
        # than picking which names to skip, and it keeps the realised book a
        # convex combination of two compliant ones.
        alpha = cap / turnover
        return {
            s: current.get(s, 0.0)
            + alpha * (weights.get(s, 0.0) - current.get(s, 0.0))
            for s in symbols
        }

    # --- bookkeeping -------------------------------------------------------

    def _record(
        self,
        timestamp: Optional[datetime],
        constraint: str,
        detail: str,
        requested: float,
        allowed: float,
    ) -> None:
        self.violations.append(
            Violation(
                timestamp=timestamp,
                constraint=constraint,
                detail=detail,
                requested=requested,
                allowed=allowed,
            )
        )

    def summary(self) -> dict[str, int]:
        """How many times each constraint bound. Surfaced in the run report."""
        counts: dict[str, int] = {}
        for violation in self.violations:
            counts[violation.constraint] = counts.get(violation.constraint, 0) + 1
        return counts

    def reset(self) -> None:
        self.violations.clear()
