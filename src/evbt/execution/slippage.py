"""
Slippage: the gap between the price you aimed at and the price you got.

A note on double counting, because this is the part of a cost model that is
easiest to get quietly wrong
-----------------------------------------------------------------------------
Three effects are commonly all called "slippage", and this engine models them
in three different places:

1. **Delay between decision and execution.** A signal computed at Friday's
   close executes at Monday's open, and the market gapped over the weekend.
   *This is not modelled here at all* — it is already in the data. The broker
   fills against the real next bar, so the gap is whatever actually happened.
   Adding a term for it would charge twice for one effect. Engines that fill at
   the *same* bar's close have to model it, which is one reason they are wrong.

2. **Your own trade moving the price.** That is market impact, and it lives in
   `costs.ImpactModel`.

3. **Intrabar execution uncertainty** — what is modelled here. Even a small
   order is not one print at the open. It is worked over minutes against a book
   that keeps moving, and the achieved average differs from the reference
   price. The dispersion of that difference grows with intrabar volatility, and
   its *adverse mean* grows with participation, because taking a larger share of
   the bar means paying up through more of the book and giving other
   participants more time to read you.

Effects 2 and 3 genuinely overlap: both are increasing in participation, and a
model fitted to realised implementation shortfall data cannot separate them
without order-level attribution. Calibrating both independently from the same
shortfall figure will double count. The honest procedure is to fit the total
against shortfall data and then split it — or to set one of the two to zero and
let the other carry the whole cost. `ZeroSlippage` exists for that.

Sign convention: slippage is always adverse. Buys fill higher, sells lower.
Modelling it as mean-zero noise is a common and expensive mistake — the whole
point is that liquidity demanders lose to liquidity providers on average.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from evbt.core.events import Bar


class SlippageModel(ABC):
    """Adverse price movement per share, always returned as a positive number."""

    @abstractmethod
    def cost_per_share(self, quantity: float, bar: Bar, reference_price: float) -> float:
        """Slippage per share in currency. Positive means "worse for me"."""


@dataclass
class ZeroSlippage(SlippageModel):
    """
    No slippage. Correct for validation runs against published return series,
    and correct whenever the impact model has been calibrated to carry the whole
    execution cost (see the double-counting note above).
    """

    def cost_per_share(self, quantity: float, bar: Bar, reference_price: float) -> float:
        return 0.0


@dataclass
class FixedBpsSlippage(SlippageModel):
    """
    A flat number of basis points, independent of size.

    Crude, and its crudeness is instructive: because it does not scale with
    participation, it is exactly the assumption that makes a strategy look
    infinitely scalable. If a backtest's conclusions change materially between
    this model and `ParticipationSlippage`, the strategy's edge is a capacity
    question, not an alpha question.
    """

    bps: float = 5.0

    def cost_per_share(self, quantity: float, bar: Bar, reference_price: float) -> float:
        return reference_price * self.bps * 1e-4


@dataclass
class ParticipationSlippage(SlippageModel):
    """
    Slippage scaled by participation rate and bar volatility.

        per share = k * sigma * sqrt(Q / V) * price   [+ optional noise]

    The square root is not decoration. Working an order takes time roughly
    proportional to Q/V, and a price diffusing over that time drifts by
    sigma * sqrt(time). Concavity therefore falls out of the diffusion, not
    from a curve fit — which is the same argument that produces the square-root
    impact law, and the reason the two are hard to disentangle empirically.

    Parameters
    ----------
    k
        Scaling constant. 0.1 is a modest assumption for liquid US equities:
        a 2% daily-vol name traded at 10% of volume slips 0.1 * 0.02 * 0.316
        = 6.3 bps.
    default_volatility
        Used when a bar has no trailing vol yet — the first bars of a symbol's
        history. Falling back to zero would make the earliest, thinnest part of
        every backtest the cheapest to trade, which is precisely backwards.
    noise_std
        If positive, adds a seeded Gaussian draw scaled by the adverse term, so
        realised slippage varies trade to trade. The mean stays adverse. Off by
        default: a deterministic engine is far easier to debug, and randomness
        should be an explicit choice made when running Monte Carlo robustness
        checks rather than something inherited silently.
    seed
        Seeds the noise generator. Fixed by default so that even the stochastic
        path is reproducible.
    """

    k: float = 0.1
    default_volatility: float = 0.02
    noise_std: float = 0.0
    seed: int = 0
    _rng: Optional[np.random.Generator] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def cost_per_share(self, quantity: float, bar: Bar, reference_price: float) -> float:
        sigma = bar.volatility if bar.volatility is not None else self.default_volatility
        participation = 1.0 if bar.volume <= 0 else min(abs(quantity) / bar.volume, 1.0)
        adverse = self.k * sigma * math.sqrt(participation) * reference_price

        if self.noise_std <= 0.0:
            return adverse
        # Clipped at zero: a favourable draw large enough to flip the sign would
        # mean being paid to demand liquidity, which does not happen.
        draw = self._rng.normal(0.0, self.noise_std * adverse)
        return max(0.0, adverse + draw)

    def reset(self) -> None:
        """Re-seed. Called between walk-forward folds so each fold is repeatable."""
        self._rng = np.random.default_rng(self.seed)


@dataclass
class SpreadCrossingSlippage(SlippageModel):
    """
    Slippage as a multiple of the bar's own high-low range.

    A useful cross-check on `ParticipationSlippage` that needs no volume data at
    all — handy for instruments where reported volume is unreliable or absent
    (many FX and OTC series). `fraction` is the share of the bar's range given
    up; 0.1 on a bar that ranged 2% costs 20 bps.
    """

    fraction: float = 0.1

    def cost_per_share(self, quantity: float, bar: Bar, reference_price: float) -> float:
        return self.fraction * max(0.0, bar.high - bar.low)
