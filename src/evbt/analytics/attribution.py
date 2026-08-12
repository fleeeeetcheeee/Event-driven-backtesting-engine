"""
Factor decomposition of strategy returns.

The question this answers is the one an interviewer asks first: *is this alpha,
or is it beta you have not named yet?* Regressing strategy returns on a set of
known factors splits them into a part the factors explain and a part they do
not, and only the second part is a claim worth making.

    r_t - rf_t  =  alpha  +  sum_k beta_k * f_kt  +  e_t

Newey-West standard errors
--------------------------
Plain OLS standard errors assume the residuals are independent. Strategy
returns are not: they are autocorrelated by construction whenever positions
persist across periods, and the autocorrelation biases OLS standard errors
*downwards*, inflating t-statistics on exactly the strategies that hold longest.

The HAC correction (Newey & West, 1987) replaces the residual covariance with

    S = Omega_0 + sum_{l=1}^{L} w_l (Omega_l + Omega_l')
    Omega_l = sum_t u_t u_{t-l} x_t x_{t-l}'
    w_l = 1 - l / (L + 1)                 [the Bartlett kernel]

and the Bartlett weights are what guarantee S stays positive semi-definite. The
default lag length follows Newey-West's own rule of thumb, L = floor(4 (n/100)^(2/9)).

Implemented directly rather than pulled from statsmodels: this is a formula the
project spec expects to be derivable on a whiteboard, and thirty lines of numpy
is cheaper than a dependency.

One degenerate case is left uncaught deliberately. If the strategy is an exact
linear combination of the factors, the residuals collapse to floating-point
noise and every t-statistic becomes a 0/0 artifact of order 1e16. That input
cannot arise from a real return series — it means the "strategy" is literally
the factor — and special-casing it would add a branch that no real call ever
takes. Read an R-squared of 1.000 as a data problem, not a result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FactorExposure:
    """One coefficient with its inference."""

    name: str
    beta: float
    stderr: float
    tstat: float

    @property
    def significant(self) -> bool:
        """|t| > 2, the conventional bar. Not a substitute for a prior."""
        return abs(self.tstat) > 2.0


@dataclass
class AttributionResult:
    """The decomposition of a return series into factor and residual parts."""

    alpha: float                 # per period
    alpha_annualised: float
    alpha_stderr: float
    alpha_tstat: float
    exposures: list[FactorExposure]
    r_squared: float
    n_observations: int
    newey_west_lags: int

    @property
    def alpha_is_significant(self) -> bool:
        return abs(self.alpha_tstat) > 2.0

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "term": "alpha (annualised)",
                "estimate": self.alpha_annualised,
                "stderr": self.alpha_stderr,
                "tstat": self.alpha_tstat,
            }
        ]
        rows += [
            {"term": e.name, "estimate": e.beta, "stderr": e.stderr, "tstat": e.tstat}
            for e in self.exposures
        ]
        return pd.DataFrame(rows)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        lines = [
            f"alpha (ann.)  {self.alpha_annualised:+.2%}  "
            f"(t={self.alpha_tstat:.2f}, NW lags={self.newey_west_lags})",
        ]
        for exposure in self.exposures:
            flag = "*" if exposure.significant else " "
            lines.append(
                f"  {exposure.name:<12} {exposure.beta:+.3f} "
                f"(t={exposure.tstat:+.2f}){flag}"
            )
        lines.append(f"R^2           {self.r_squared:.3f}   n={self.n_observations}")
        return "\n".join(lines)


def newey_west_lags(n_observations: int) -> int:
    """Newey & West's rule of thumb: L = floor(4 * (n/100)^(2/9))."""
    if n_observations < 2:
        return 0
    return int(math.floor(4.0 * (n_observations / 100.0) ** (2.0 / 9.0)))


def _hac_covariance(x: np.ndarray, residuals: np.ndarray, lags: int) -> np.ndarray:
    """
    Newey-West HAC covariance of the OLS coefficient vector.

    `x` includes the intercept column. Returns the full covariance matrix, so
    the caller reads standard errors off its diagonal.
    """
    n, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)

    weighted = x * residuals[:, None]
    s = weighted.T @ weighted  # lag 0

    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = weighted[lag:].T @ weighted[:-lag]
        s = s + weight * (gamma + gamma.T)

    # Small-sample degrees-of-freedom correction, matching the usual convention.
    scale = n / max(n - k, 1)
    return xtx_inv @ (s * scale) @ xtx_inv


def attribute(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    risk_free_rate: Optional[pd.Series] = None,
    periods_per_year: int = 252,
    lags: Optional[int] = None,
) -> AttributionResult:
    """
    Regress strategy returns on factor returns with HAC inference.

    Both inputs are aligned on their index and the intersection is used, so a
    factor series on a different calendar silently shrinks the sample rather
    than misaligning it — the row count in the result is the check on that.

    `factor_returns` columns are the factors; name them (MKT, SMB, HML, ...) and
    the names carry through to the output.
    """
    frame = pd.concat([strategy_returns.rename("_strategy"), factor_returns], axis=1)
    frame = frame.dropna()
    if len(frame) < 3:
        raise ValueError(
            f"need at least 3 overlapping observations, got {len(frame)}; "
            "check that the strategy and factor series share a calendar"
        )

    y = frame["_strategy"].to_numpy(dtype=float)
    if risk_free_rate is not None:
        aligned_rf = risk_free_rate.reindex(frame.index).fillna(0.0).to_numpy(dtype=float)
        y = y - aligned_rf

    names = [c for c in frame.columns if c != "_strategy"]
    factors = frame[names].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(frame)), factors])

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta

    n_lags = lags if lags is not None else newey_west_lags(len(frame))
    covariance = _hac_covariance(x, residuals, n_lags)
    stderrs = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    total_ss = float(((y - y.mean()) ** 2).sum())
    resid_ss = float((residuals**2).sum())
    r_squared = 1.0 - resid_ss / total_ss if total_ss > 0 else 0.0

    def tstat(index: int) -> float:
        return float(beta[index] / stderrs[index]) if stderrs[index] > 0 else 0.0

    return AttributionResult(
        alpha=float(beta[0]),
        # Compounded, not multiplied: a per-period alpha of 2 bps is 5.2% a
        # year, not 5.04%, and the gap widens with the size of the alpha.
        alpha_annualised=float((1.0 + beta[0]) ** periods_per_year - 1.0),
        alpha_stderr=float(stderrs[0]),
        alpha_tstat=tstat(0),
        exposures=[
            FactorExposure(
                name=name,
                beta=float(beta[i + 1]),
                stderr=float(stderrs[i + 1]),
                tstat=tstat(i + 1),
            )
            for i, name in enumerate(names)
        ],
        r_squared=r_squared,
        n_observations=len(frame),
        newey_west_lags=n_lags,
    )
