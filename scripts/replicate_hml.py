"""
The project's done criterion: reproduce a documented strategy's published
monthly returns to within a few basis points.

The strategy is Fama-French HML, and the replication runs in two stages that
answer two different questions.

**Stage 1 — is the reference data what I think it is?**  Rebuild HML from the
six size x book-to-market portfolios using French's own definition,

    HML = 1/2 (SmallValue + BigValue) - 1/2 (SmallGrowth + BigGrowth)

and compare against his published HML. No engine involved: this is arithmetic
on two files. It has to pass first, because if it does not, a stage 2 mismatch
has two possible causes and no way to tell them apart.

**Stage 2 — is the engine's accounting correct?**  Feed those same six
portfolios to the engine as tradeable instruments, rebalance monthly to
`(+1/2, +1/2, -1/2, -1/2)` with every cost set to zero, and compare the
engine's realised NAV returns against the published HML.

Why stage 2 is a real test of the engine and not a tautology
------------------------------------------------------------
It exercises the whole machine — event loop, priority ordering, signal
generation, order sizing, next-bar fill rule, position accounting, short-side
mechanics, mark-to-market, NAV computation — and compares the output to a series
produced independently by someone else, decades ago, with none of this code.

The closed-form answer is known: a book holding weights `w_i` in assets
returning `r_i`, funded from cash, has NAV return exactly `sum(w_i * r_i)`. For
the HML weights that is exactly the factor return. So *any* deviation beyond
French's own 2-decimal rounding is an accounting bug — a mis-signed short, a
basis carried across a rebalance, a stale NAV used for sizing, an off-by-one in
fill timing. Each of those would produce a plausible-looking series that drifts.

Deliberately factored out: data quality. Feeding French's own portfolios in
means a mismatch cannot be blamed on the price source, which is what makes this
a clean test of the engine specifically. The separate question of whether HML
can be rebuilt bottom-up from free data is a different and much harder one — see
README and LOG.

Usage
-----
    python scripts/fetch_french_data.py       # once, to get the data
    python scripts/replicate_hml.py
    python scripts/replicate_hml.py --start 1963-07 --tolerance-bps 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evbt.config import config  # noqa: E402
from evbt.data.french import (  # noqa: E402
    construct_hml,
    load_factors,
    load_six_portfolios,
    returns_to_bars,
)
from evbt.data.frame import DataFrameDataHandler  # noqa: E402
from evbt.engine import Backtest  # noqa: E402
from evbt.execution.broker import SimulatedBroker  # noqa: E402
from evbt.portfolio.construction import (  # noqa: E402
    ExplicitWeightSizer,
    PortfolioConstructor,
)
from evbt.portfolio.portfolio import Portfolio  # noqa: E402
from evbt.strategy.base import FixedWeightPortfolio  # noqa: E402

# HML's definition, as weights on the six portfolios. The two neutral
# portfolios carry zero weight — HML is the high-minus-low spread and the middle
# third of the book-to-market sort contributes nothing.
HML_WEIGHTS = {
    "SmallValue": 0.5,
    "BigValue": 0.5,
    "SmallGrowth": -0.5,
    "BigGrowth": -0.5,
}

INITIAL_CAPITAL = 1_000_000.0


def describe(diff_bps: pd.Series, label: str, tolerance: float) -> bool:
    """Print the comparison and return whether it passed."""
    worst = float(diff_bps.abs().max())
    passed = worst <= tolerance

    print(f"\n  {label}")
    print(f"    months compared : {len(diff_bps):,}")
    print(f"    mean |diff|     : {diff_bps.abs().mean():.4f} bps")
    print(f"    median |diff|   : {diff_bps.abs().median():.4f} bps")
    print(f"    max  |diff|     : {worst:.4f} bps")
    print(f"    months > 1 bp   : {int((diff_bps.abs() > 1.0).sum())}")
    print(f"    verdict         : {'PASS' if passed else 'FAIL'} "
          f"(tolerance {tolerance:.1f} bps)")
    if not passed:
        print("\n    worst offenders:")
        for date, value in diff_bps.abs().nlargest(5).items():
            print(f"      {date:%Y-%m}  {value:.4f} bps")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None, help="first month, e.g. 1963-07")
    parser.add_argument("--end", default=None, help="last month, e.g. 2024-12")
    parser.add_argument(
        "--tolerance-bps",
        type=float,
        default=1.0,
        help="maximum acceptable absolute monthly deviation, in basis points",
    )
    args = parser.parse_args()

    portfolios_zip = config.fama_french_dir / "6_Portfolios_2x3_CSV.zip"
    factors_zip = config.fama_french_dir / "F-F_Research_Data_Factors_CSV.zip"
    for path in (portfolios_zip, factors_zip):
        if not path.exists():
            print(f"missing {path}\nRun: python scripts/fetch_french_data.py")
            return 1

    # Value-weighted, explicitly. The equal-weighted table has identical column
    # names and produces a wrong factor that correlates 0.93 with the real one.
    portfolios = load_six_portfolios(portfolios_zip, weighting="value")
    published = load_factors(factors_zip)["HML"]

    if args.start:
        portfolios = portfolios[portfolios.index >= pd.Timestamp(args.start)]
    if args.end:
        portfolios = portfolios[portfolios.index <= pd.Timestamp(args.end)]

    print("=" * 72)
    print("FAMA-FRENCH HML REPLICATION")
    print("=" * 72)
    print(f"  six portfolios  : {portfolios.index.min():%Y-%m} .. "
          f"{portfolios.index.max():%Y-%m}  ({len(portfolios):,} months)")
    print(f"  weighting       : value-weighted")
    if portfolios.isna().any().any():
        print(f"  WARNING: {int(portfolios.isna().sum().sum())} missing values present")

    # --- Stage 1 ----------------------------------------------------------
    print("\n" + "-" * 72)
    print("STAGE 1 — reconstruct HML from the six portfolios (no engine)")
    print("-" * 72)

    reconstructed = construct_hml(portfolios)
    # An inner join rather than `concat(axis=1)`: it states the alignment
    # intent explicitly, and silently mismatched calendars are exactly the kind
    # of error that would make this comparison meaningless while still printing
    # a number.
    stage1 = reconstructed.rename("mine").to_frame().join(
        published.rename("published"), how="inner"
    ).dropna()
    stage1_passed = describe(
        (stage1["mine"] - stage1["published"]) * 1e4,
        "reconstructed vs published HML",
        args.tolerance_bps,
    )

    # --- Stage 2 ----------------------------------------------------------
    print("\n" + "-" * 72)
    print("STAGE 2 — drive the engine with those portfolios (the done criterion)")
    print("-" * 72)

    bars = returns_to_bars(portfolios)
    engine = Backtest(
        DataFrameDataHandler(bars, max_history=8),
        FixedWeightPortfolio(HML_WEIGHTS),
        # Zero costs and zero financing throughout. The published series is a
        # gross academic factor return; charging it anything at all would make
        # the comparison meaningless rather than conservative.
        Portfolio(INITIAL_CAPITAL),
        SimulatedBroker(),
        PortfolioConstructor(sizer=ExplicitWeightSizer()),
        liquidate_unsignalled=True,
    )
    result = engine.run()

    engine_returns = result.returns()
    stage2 = engine_returns.rename("engine").to_frame().join(
        published.rename("published"), how="inner"
    ).dropna()
    stage2_passed = describe(
        (stage2["engine"] - stage2["published"]) * 1e4,
        "engine NAV returns vs published HML",
        args.tolerance_bps,
    )

    curve = result.equity_curve
    print(f"\n    engine diagnostics")
    print(f"      orders / fills   : {result.n_orders:,} / {result.n_fills:,}")
    print(f"      rejected         : {result.n_rejections:,}")
    print(f"      mean gross lev   : "
          f"{(curve['gross_exposure'] / curve['nav']).mean():.3f}x")
    print(f"      mean net lev     : "
          f"{(curve['net_exposure'] / curve['nav']).mean():+.2e}x")
    print(f"      total costs      : {result.fills['total_cost'].sum():.2f}")

    # --- Verdict ----------------------------------------------------------
    print("\n" + "=" * 72)
    if stage1_passed and stage2_passed:
        print("DONE CRITERION MET — engine reproduces published HML within tolerance")
    else:
        print("DONE CRITERION NOT MET")
    print("=" * 72)

    return 0 if (stage1_passed and stage2_passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
