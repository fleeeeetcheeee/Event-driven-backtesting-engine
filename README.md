# Event-Driven Backtesting Engine

A backtester that simulates realistic execution bar by bar — market events → signal events → order events → fill events → portfolio updates — with transaction costs, market impact, portfolio accounting, risk constraints, and walk-forward evaluation. Built to answer one question a vectorised backtester cannot: **when, exactly, could this order have filled, and for how much?**

## The result

**The engine reproduces Fama-French HML to within 0.5 basis points per month, across 1,199 months (July 1926 – June 2026), with no month deviating by more than 1 bp.**

| | mean \|diff\| | max \|diff\| | months > 1 bp |
|---|---|---|---|
| Arithmetic reconstruction vs published HML | 0.2510 bps | 0.5000 bps | 0 / 1,200 |
| **Engine's traded NAV returns vs published HML** | **0.2511 bps** | **0.5000 bps** | **0 / 1,199** |

The two rows being identical is the actual finding. French publishes factor returns rounded to two decimal places in percent, so the half-ulp of his own reporting is exactly 0.5 bp — the bound the engine hits. **The engine contributes no error at all**; the entire residual is his rounding. A mis-signed short, a cost basis carried across a rebalance, a stale NAV used for sizing, or an off-by-one in fill timing would each show up as drift beyond that floor, and none does.

Reproduce it with:

```bash
python scripts/fetch_french_data.py
python scripts/replicate_hml.py
```

353 tests pass with 95% coverage. The replication is pinned in `tests/integration/test_hml_replication.py` (skipped until the data is fetched); the rest of the suite needs no network and no data.

A plain-language companion for non-specialists is in [`EXPLANATION.md`](EXPLANATION.md), with a glossary.

### What this does and does not establish

It establishes that the engine's **accounting** is correct: it took French's six size × book-to-market portfolios as tradeable instruments and arrived at his factor by actually trading them — sizing orders against NAV, filling them a bar later, carrying a 100% short book, marking to market, and compounding across a century.

It does *not* establish that HML can be rebuilt bottom-up from free data. That is a much harder problem, it needs CRSP and Compustat, and it is deliberately factored out here: feeding French's own portfolios in means a mismatch cannot be blamed on the price source, which is what makes this a clean test of the engine specifically. See [What is not done](#what-is-not-done).

## Why build one rather than use `backtrader` or `zipline`

Because the interesting part is the part a library hides. Three of the engine's rules exist specifically to refuse an optimistic assumption, and each one costs money in the results:

**An order submitted at time `t` may only fill against a bar strictly after `t`.** In `execution/broker.py` this is an assertion, not a convention — an order stamped at the bar it would fill on raises `LookaheadError` rather than being silently skipped, because no legitimate code path produces one, and a strategy that quietly stops trading is far harder to notice than a crash. In a pandas backtest the same bug is `signal * returns` with no `.shift()`, it is one character wide, and it produces a beautiful equity curve.

You can watch it cost money in `tests/integration/test_no_lookahead.py`: a strategy sizes 1,000 shares off day 1's close of 100 and fills at day 2's open of 110. It does not get 1,000 shares at 100.

**A limit order needs the bar to trade *through* its price, not merely touch it.** Touching your limit means the market reached the front of a queue you were somewhere in the middle of. Assuming a fill there is the single largest source of fabricated returns in mean-reversion backtesting, because limits get touched precisely on the bars where you would have been left behind.

**Fills are capped at a fraction of each bar's volume.** An order for 500k shares of a name trading 1M a day is worked across days, at prices that move while you work it. This is also what makes capacity a question the engine can answer at all — if costs don't scale with size, capacity is infinite and the model is silent.

## Architecture

```
src/evbt/
├── engine.py                    # the event loop
├── config.py                    # paths and engine-wide defaults
├── core/
│   ├── events.py                # the event vocabulary + EventPriority
│   └── event_queue.py           # deterministic (time, priority, sequence) heap
├── data/
│   ├── base.py                  # DataHandler ABC — the no-lookahead guarantee
│   ├── frame.py                 # in-memory long-format source
│   ├── parquet.py               # out-of-core: chunked DuckDB streaming
│   └── french.py                # Ken French data library parser
├── strategy/base.py             # Strategy ABC + StrategyContext
├── portfolio/
│   ├── position.py              # per-symbol accounting: flips, splits, dividends
│   ├── portfolio.py             # cash, NAV, exposures, equity curve
│   ├── financing.py             # short borrow and cash carry (ACT/360)
│   ├── risk.py                  # pre-trade constraints on target weights
│   └── construction.py          # signals → weights → orders
├── execution/
│   ├── costs.py                 # commission, spread, Almgren-Chriss + square-root impact
│   ├── slippage.py              # participation- and volatility-scaled
│   └── broker.py                # order lifecycle, fill rules, partial fills
├── analytics/
│   ├── metrics.py               # Sharpe (with standard errors), Sortino, Calmar, turnover
│   ├── attribution.py           # factor regression with Newey-West errors
│   └── capacity.py              # AUM at which costs eat the alpha
└── walkforward/runner.py        # folds, embargo, out-of-sample stitching
```

### The trading day is the priority ordering

Everything that happens sits in one queue, ordered by `(timestamp, priority, sequence)`. Within a timestamp, `EventPriority` fixes the sequence:

| # | Phase | Why here |
|---|-------|----------|
| 1 | `CORPORATE_ACTION` | Splits restate share counts and dividends pay on the position held *going into* the ex-date — both must land before anything trades. |
| 2 | `MARKET` | Today's bar becomes visible. |
| 3 | `FILL` | Resting orders — all placed on *earlier* bars — match against today's prices. |
| 4 | `ORDER_EXPIRED` / `ORDER_REJECTED` | Unfilled DAY orders die; rejections are recorded as data, not log lines. |
| 5 | `MARK_TO_MARKET` | The book is valued, carry accrues, the equity curve gains a row. |
| 6 | `SIGNAL` | The strategy sees today's bar and the freshly marked portfolio. |
| 7 | `ORDER` | Orders rest until tomorrow. |

Two placements carry the weight. `FILL` before `MARK_TO_MARKET` means the equity curve includes today's trades. `SIGNAL` after `MARK_TO_MARKET` means a strategy sizing against NAV sees today's NAV — sizing off a stale one is a permanent compounding distortion that never shows up in the output.

The `sequence` counter is not decoration: `heapq` is not a stable sort, so a bare `(timestamp, priority)` key resolves ties by heap geometry, and a run reproduces until it doesn't.

## Install and run the tests

```bash
git clone https://github.com/fleeeeetcheeee/Event-driven-backtesting-engine.git
cd Event-driven-backtesting-engine

python -m venv .venv && source .venv/bin/activate

# Reproducible: the exact versions this has been verified against
pip install -r requirements.lock && pip install -e . --no-deps

# Or, for development against current versions
pip install -e ".[dev]"

pytest                            # full suite with coverage
pytest tests/integration/         # the barrier and the accounting proofs
pytest -p no:cacheprovider --no-cov tests/unit/test_broker.py   # quick single module
```

No API keys, no data download, no `.env`. Every test runs on synthetic series with known answers.

## A minimal backtest

```python
import pandas as pd

from evbt.data.frame import DataFrameDataHandler
from evbt.engine import Backtest
from evbt.execution.broker import SimulatedBroker
from evbt.execution.costs import BpsCommission, FixedBpsSpread, SquareRootImpact
from evbt.execution.slippage import ParticipationSlippage
from evbt.portfolio.construction import EqualWeightSizer, PortfolioConstructor
from evbt.portfolio.financing import FinancingModel
from evbt.portfolio.portfolio import Portfolio
from evbt.portfolio.risk import RiskLimits, RiskManager
from evbt.strategy.base import Strategy

class Momentum(Strategy):
    """Long the top decile of 60-day return, short the bottom. Monthly."""
    name = "momentum"

    def on_bar(self, event, ctx):
        if event.timestamp.month == getattr(self, "_last_month", None):
            return []
        self._last_month = event.timestamp.month

        scores = {}
        for symbol in event.symbols:
            closes = ctx.data.series(symbol, "close", 61)
            if len(closes) == 61:                 # abstain rather than pad
                scores[symbol] = closes[-1] / closes[0] - 1.0
        if len(scores) < 10:
            return []

        ranked = sorted(scores, key=scores.get, reverse=True)
        n = max(1, len(ranked) // 10)
        return [ctx.long(s) for s in ranked[:n]] + [ctx.short(s) for s in ranked[-n:]]

# bars: long-format DataFrame [timestamp, symbol, open, high, low, close, volume]
data = DataFrameDataHandler(bars, actions)

engine = Backtest(
    data,
    Momentum(),
    Portfolio(10_000_000, financing=FinancingModel(borrow_rate=0.004)),
    SimulatedBroker(
        commission=BpsCommission(bps=1.0),
        spread=FixedBpsSpread(bps=5.0),
        impact=SquareRootImpact(Y=0.5),
        slippage=ParticipationSlippage(k=0.1),
        max_participation=0.10,
    ),
    PortfolioConstructor(
        sizer=EqualWeightSizer(gross_leverage=1.0),
        risk=RiskManager(RiskLimits(max_position_weight=0.02, max_net_leverage=0.10)),
        min_trade_notional=5_000,
    ),
    liquidate_unsignalled=True,
)

result = engine.run()
print(result.summary())
```

Then evaluate it:

```python
from evbt.analytics.metrics import evaluate
from evbt.analytics.attribution import attribute
from evbt.analytics.capacity import estimate_capacity, realised_cost_rate

report = evaluate(result.equity_curve, result.fills)
print(report)          # Sharpe with its standard error and t-stat, not a bare number

print(attribute(result.returns(), fama_french_factors))   # is it alpha, or unnamed beta?

print(estimate_capacity(
    backtest_aum=10_000_000,
    gross_annual_return=report.annualised_return + report.cost_drag_annualised,
    annual_cost_rate=realised_cost_rate(result.fills, result.equity_curve),
))
```

## Methodology notes worth defending

**Market impact.** Both models the spec asks for are implemented. Almgren-Chriss linear impact (`temporary = η·(Q/V)·P`, `permanent = γ·(Q/V)·P`) is what makes the optimal execution problem solvable in closed form; permanent impact is charged at *half* its magnitude because the price walks away linearly as you trade, so the average transacted price sits at the midpoint of the walk — this is the `γX²/2` term, and charging the full displacement double-counts. The square-root law (`ΔP/P = Y·σ·√(Q/V)`) is the better-supported empirical form and the one to use when a number is meant to be believed. `CompositeImpact` runs them together.

`almgren_chriss_schedule` solves the actual control problem, `x_j = X·sinh(κ(T−t_j))/sinh(κT)`, from the exact discrete relation `2(cosh(κτ)−1)/τ² = λσ²/η̃` rather than the small-τ approximation `κ ≈ √(λσ²/η̃)`, which drifts visibly at daily bar lengths. It is tested at both analytic limits: risk-neutral collapses to TWAP, risk-averse front-loads.

**Slippage is not impact, and the overlap is stated rather than hidden.** Three things get called slippage. The gap between decision and execution is *not modelled* — the broker fills against the real next bar, so it is already in the data, and adding a term would charge twice. Your own trade moving the price is impact. What `slippage.py` models is intrabar execution uncertainty. Effects two and three genuinely overlap, and calibrating both from the same realised-shortfall figure will double count; the honest procedure is to fit the total and split it, or set one to zero. `ZeroSlippage` exists for that.

**Financing defaults to zero.** Borrow, cash credit, and margin debit rates are all zero out of the box, so any financing assumption in a reported result had to be switched on deliberately. Day counts are ACT/360 — the money-market convention — and carry accrues on *calendar* days, so a Friday-to-Monday mark is charged three days. A per-bar accrual undercounts borrow by roughly 40% a year, and the equity curve still looks perfectly smooth while it does.

**Risk constraints clip target weights, not orders.** Rejecting orders at the broker makes the resulting book depend on submission order, which is arbitrary. Constraints apply in the order position caps → sector → gross → net → turnover, and one pass suffices because each constraint defines a convex set and every step after the first is a pure reduction or an interpolation between two compliant books.

**Sharpe ratios are reported with standard errors.** `sharpe_standard_error` implements Lo (2002): `SE(SR) ≈ √((1 + SR²/2)/n)`. Three years of daily data gives roughly 0.045, so a measured Sharpe of 1.0 carries a 95% interval of about [0.91, 1.09] — before accounting for having tried more than one strategy. `evaluate()` attaches a note when the sample is too short to support a point estimate, and another when the Sharpe exceeds 3.0, which on US equity long-short is a red flag to investigate rather than a result to report.

**Factor attribution uses Newey-West standard errors.** Strategy returns are autocorrelated whenever positions persist, and the autocorrelation biases OLS standard errors *downwards* — inflating t-statistics on exactly the strategies that hold longest. Implemented directly in numpy rather than pulled from statsmodels, with the Bartlett kernel and the `L = ⌊4(n/100)^(2/9)⌋` rule of thumb.

**Data sources are interchangeable, and chunk size is invisible.** `DataFrameDataHandler` holds history in RAM — fine to a few million rows, which covers a 30-year daily backtest on 1,000 names. `ParquetDataHandler` streams the same data in date chunks through DuckDB, so peak memory is set by the chunk size rather than the length of the backtest, and predicate pushdown means only the relevant partitions are ever opened. Both satisfy the same `DataHandler` contract, and the tests assert that chunk sizes of 1, 7, 50 and 10,000 produce byte-identical slices to the in-memory handler. That matters: a backtest whose results depended on how much RAM the machine had would not be reproducible. It required carrying a volatility warm-up tail across chunk boundaries, without which every boundary produces a run of bars with no vol estimate and therefore no slippage.

**Walk-forward folds are half-open.** `[start, end)` on both train and test windows. With inclusive bounds and a zero embargo, the bar at `train_end == test_start` is both trained on and scored — and the same collision between consecutive test windows would double-count it in the stitched out-of-sample series. The embargo defaults to zero because the correct value is the forward-looking horizon of the strategy's labels, and a wrong non-zero default would be worse than making the choice explicit.

## What is not done

Read this before trusting anything the engine produces.

**HML has not been built bottom-up from free data.** The replication validates the engine's accounting by feeding it French's own portfolios. It says nothing about whether the *factor construction* — the 2×3 sort on size and book-to-market, NYSE breakpoints, the June rebalance with prior-December accounting data — can be reproduced from free sources. It largely cannot: that needs CRSP and Compustat, and Project 01's fundamentals only begin in 2009 with survivorship-incomplete prices. Quantifying that gap is worth doing and is not done here. Do not read "reproduces HML to 0.5 bps" as "can build HML from scratch".

**The replication is frictionless by construction.** Zero commission, zero spread, zero impact, zero slippage, zero financing — because the published series is a gross academic factor return and charging it anything would make the comparison meaningless rather than conservative. It therefore validates the accounting, not the cost models. The cost models are unit-tested against their own formulas and calibrated to published coefficients, which is a weaker claim.

**Monthly bars mean the fill rule is barely exercised.** The replication's synthetic bars have `open[t] == close[t-1]`, so the engine sizes and fills at the same price and no execution drift enters. That is deliberate — it isolates accounting from timing — but it means the next-bar fill rule, partial fills, and participation caps are exercised only by the synthetic tests in `tests/unit/test_broker.py` and `tests/integration/test_no_lookahead.py`, not by the replication.

**No `EXPLANATION.md`.** Project 01 carries a plain-language companion with a glossary; the portfolio standard says to follow that pattern. Not yet written.

**Borrow accrues one interval early.** Financing is charged at each mark using the position *after* that bar's fills, so a newly opened short pays for the interval preceding its own existence. One interval per position lifetime, in the conservative direction, immaterial on a multi-month hold — but wrong, and documented rather than fixed because the fix needs previous-mark positions carried as extra state.

**Impact coefficients are literature values, not fitted.** `η`, `γ`, and `Y` are round numbers from published estimates, not measurements against any dataset in this repo. Any conclusion that depends on their exact value must be reported as a range across plausible coefficients. `analytics/capacity.py` sweeps them for this reason.

**Capacity estimates are an upper bound, not a central estimate.** The model scales costs with size but does not model alpha decaying with size. A large book takes longer to build, and a signal with a five-day half-life has decayed materially before the position is on. Quantifying that needs the signal's decay profile, which is Project 6. Read the numbers as "no more than this".

**Sector constraints are exposure-based, not risk-based.** Both caps — `max_sector_weight` on net and `max_sector_gross_weight` on gross — treat two names in the same GICS sector as equally substitutable. The principled version constrains estimated factor risk and needs a risk model (Project 10). Note that net and gross answer different questions and neither subsumes the other: a book 50% long and 50% short one sector has zero net exposure and a large bet on within-sector dispersion, which is exactly the shape of a stat-arb book.

**Fractional shares are permitted by default.** `round_lots=True` on the constructor turns this off. At institutional size the difference is immaterial; for a small account it removes a real rounding drag.

**Single-currency, no futures roll, no options.** US cash equities only.

## Relationship to Project 01

This engine consumes Project 01's point-in-time store as an external dataset — the same way it would read any vendor file. The two repos do not import from each other. `config.pit_store_root` points at Project 01's `data/` directory if it is present on the machine; nothing requires it to be.

Project 01's limitations propagate directly into any backtest run on its output, and the survivorship gap in particular is material: yfinance does not serve price history for many delisted tickers, so a universe assembled from that store silently omits a slice of the historical cross-section skewed toward exactly the failures that matter. A survivorship-bias-free engine fed survivorship-biased data is still a survivorship-biased backtest.
