# Project 2 — Event-Driven Backtesting Engine

**Tier:** 0 (Infrastructure)
**Spec:** `ResearchToDo.md` → Part 3 → Tier 0 → Project 2
**Repo:** https://github.com/fleeeeetcheeee/Event-driven-backtesting-engine
**Status:** Engine core built and **verified** — 282 tests pass, 94% coverage, including
hand-computed end-to-end accounting and the no-lookahead barrier. The spec's **done criterion is
not yet met**: no published strategy has been reproduced. See [Open items](#open-items).

---

## Goal

An event-driven backtester that simulates realistic execution — bar events → signal events →
order events → fill events → portfolio updates — with transaction costs, slippage, portfolio
accounting, risk constraints, performance analytics, and walk-forward evaluation.

**Done criterion (from the spec):** reproduce the backtest of a simple documented strategy
(e.g. Fama-French HML) and match the published monthly returns to within a few basis points.

**Explicit prohibition (from the spec):** no vectorised pandas backtester. Event-driven only,
because vectorisation hides order timing, partial fills, and cash-constraint bugs.

---

## Build log

### 2026-08-12 — Scaffold, core engine, verified test suite

Entire session written as work happened. One sitting, nothing committed yet.

#### Repository setup

`git init` + `git remote add origin` against the pre-existing GitHub repo. No commit, no push —
per the workspace rule that every commit is approved first.

**Package named `evbt`.** Project 01's package is `tierzero`, which names the *tier* rather than
the project; reusing that pattern would collide the moment both projects are installed into one
environment. Considered and rejected: `backtest` (collides with the PyPI `backtesting` package
in any shared venv), `edbe` (initialism nobody can read back), `engine` (far too generic to
import cleanly). `evbt` is short, unambiguous at the import site — `from evbt.engine import
Backtest` — and collides with nothing.

#### Architecture as built

```
core/       events.py, event_queue.py     the event vocabulary and its ordering
data/       base.py, frame.py             streaming, and the no-lookahead guarantee
strategy/   base.py                       Strategy ABC + StrategyContext
portfolio/  position.py, portfolio.py     accounting
            financing.py                  borrow and cash carry
            risk.py                       pre-trade constraints
            construction.py               signals -> weights -> orders
execution/  costs.py, slippage.py         commission, spread, impact
            broker.py                     order lifecycle and fill rules
analytics/  metrics.py, attribution.py, capacity.py
walkforward/runner.py                     folds and out-of-sample stitching
engine.py                                 the loop
```

#### Decisions locked in, and why

**The event priority ordering is the design.** Events sharing a timestamp are ordered
`CORPORATE_ACTION → MARKET → FILL → expiry/rejection → MARK_TO_MARKET → SIGNAL → ORDER`. Two
placements carry real weight:

- *FILL before MARK_TO_MARKET* so the equity curve includes today's trades.
- *SIGNAL after MARK_TO_MARKET* so a strategy sizing against NAV sees today's NAV. Sizing off a
  stale NAV is a permanent compounding distortion that never shows up in the output.

**`MarkToMarketEvent` was added mid-build.** The first pass had the portfolio mark itself as a
side effect of handling `MarketEvent`. That put valuation *before* fills, so the equity curve
lagged the book by one bar. Rather than reordering the handler internals — which would have
buried the sequencing decision inside an engine method — valuation became its own event with its
own priority. The phase structure is now readable off `EventPriority` and inspectable in a trace.
This is the one structural rewrite of the session.

**The queue orders on `(timestamp, priority, sequence)`.** `heapq` is not a stable sort, so a
bare `(timestamp, priority)` key resolves ties by heap geometry — meaning a run reproduces until
it doesn't. The monotonic sequence counter makes ties resolve on insertion order and guarantees
the event object is never compared (which would raise, since events are not orderable).

**Scheduling an event into the past raises rather than warns.** An event inserted before the
clock is what lookahead looks like from inside the engine. It is otherwise invisible.

**The no-lookahead barrier raises on same-timestamp orders instead of skipping them.** An order
stamped at the bar it would fill on cannot arise from any legitimate path. Returning `False`
would hide the upstream bug while making the strategy merely look inactive — a strategy that
silently stops trading is much harder to notice than a crash.

**Limit orders require the bar to trade *through* the price, not merely touch it.** Touching
your limit means the market reached the front of a queue you were in the middle of. Assuming a
fill there is, as far as I can tell, the single largest source of fabricated returns in
mean-reversion backtests, because limits get touched precisely on the bars where you would have
been left behind. `require_trade_through=False` exists to demonstrate how much it flatters a
result.

**Prices stay unadjusted; splits and dividends replay as events.** Same convention as Project 01,
and for the same reason: exact cash accounting is impossible on an adjusted series. You cannot
credit a $0.24 dividend against a share count that has been retroactively rescaled by every
split since.

**Signals carry no size.** `SignalEvent` has a direction and a conviction; `PortfolioConstructor`
decides shares. Collapsing the two would make a capacity analysis impossible — the same signal
has to be runnable at $10M and $10B without editing the strategy.

**Risk constraints apply to target weights, not to orders.** Rejecting individual orders at the
broker makes the resulting book depend on submission order, which is arbitrary. Clipping the
target book first is deterministic and explainable. The application order (position caps →
sector → gross → net → turnover) works in a single pass because each constraint defines a convex
set and every step after the first is a pure reduction or an interpolation between two compliant
books — so no step ever needs revisiting. That argument is written out in `risk.py` and tested in
`test_risk.py::TestConstraintInteraction`.

**Financing defaults to zero.** Every borrow and cash-rate assumption in a reported result should
have been switched on by hand rather than inherited from a library's idea of a reasonable number.

**Almgren-Chriss implemented as both a cost model and an optimal schedule.** The cost model is
what the broker charges; `almgren_chriss_schedule` solves the actual control problem and is
tested at its two analytic limits — risk-neutral collapses to TWAP, risk-averse front-loads. The
spec's Part 4 lists deriving the AC schedule as an interview expectation, so it is implemented
from the discrete relation `2(cosh(κτ) − 1)/τ² = λσ²/η̃` rather than the small-τ approximation
`κ ≈ √(λσ²/η̃)`, which visibly drifts at daily bar lengths.

**Newey-West standard errors implemented by hand, not via statsmodels.** Thirty lines of numpy
against a dependency, for a formula the portfolio standard says must be derivable on a
whiteboard.

#### Two real bugs the tests caught

**1. Sharpe ratio of 7.3 × 10¹⁶ on a flat return series.** `pd.Series([0.001] * 100).std(ddof=1)`
is about `2e-19`, not `0.0`, so the `if sigma == 0` guard missed it entirely. Any strategy that
sits in cash, or holds one position through a flat stretch, hits this. Fixed with an explicit
`ZERO_VOL_TOLERANCE = 1e-12` and the same guard added to Sortino. Worth recording because the
failure mode is a *plausible-looking pipeline* producing an absurd number rather than crashing —
and because the naive `== 0` check is what I wrote first without thinking about it.

**2. A one-bar leak in walk-forward folds.** `restricted_to` was inclusive at both ends and
`generate_folds` set `test_start = train_end` when the embargo was zero, so the bar at that
instant was both trained on and scored. The same collision recurred between consecutive test
windows, which would additionally have double-counted that bar in the stitched out-of-sample
series. Fixed by making all fold windows half-open `[start, end)`. This is exactly the class of
error the project exists to prevent, it survived the initial write, and it was caught only by a
test that compared the windows directly rather than checking a summary statistic — no aggregate
number would have moved detectably.

#### Verification

First execution of the suite in a fresh `.venv` (Python 3.13.9, pandas 3.0.5, numpy 2.5.2).

**Result: 282 passed, 94% coverage (1,733 statements, 109 uncovered).**

The tests that matter most:

- `tests/integration/test_no_lookahead.py` — the barrier end-to-end. A strategy that reacts to a
  100% price spike is shown to fill at the *post*-spike price and earn exactly zero from it; a
  leaking engine would double its money. Also asserts that data accessors return only streamed
  bars under adversarial querying, and that repeated runs are bit-identical.
- `tests/integration/test_engine_accounting.py` — end-to-end against answers computed by hand in
  the docstrings. A dividend-and-split scenario lands on exactly 102,000 (the split is
  economically neutral; the dividend is the only return). A short book at 5% borrow lands on
  99,923.61, including the check that a Friday→Monday mark is charged three days of carry, not
  one — a per-bar accrual would undercount borrow by ~40% a year.
- `tests/unit/test_position.py::TestSignFlip` — long 100, sell 150. Realise on 100, open a short
  of 50 at the fill price. The naive `quantity += delta` carries the old basis across the flip
  and books P&L that never happened.

Smoke-checked the whole loop manually before writing tests, on a two-symbol trending series: the
engine sized 1,000 shares off day 1's close of 100 and filled at day 2's open of 101, taking cash
to −10,000. That is the barrier costing money in the expected direction, and it was the first
sign the ordering was right.

---

## Status against the done criterion

**Not met.** The engine is built and its internal arithmetic is verified, but the spec asks for
something stronger: reproduce a *published* return series to within a few basis points. Nothing
in the current test suite compares against an external source — every expected value is either
hand-computed or synthetic, which proves self-consistency and not correctness against reality.

Planned approach, in the order it should be attempted:

1. **Arithmetic check (no engine).** Download Ken French's 6 portfolios formed on size and
   book-to-market, plus his published HML. Reconstruct
   `HML = ½(SmallValue + BigValue) − ½(SmallGrowth + BigGrowth)` and confirm it matches his HML
   to rounding. This validates the *data* before the engine is involved, so a later mismatch
   cannot be blamed on it.
2. **Engine check.** Drive the engine with those 6 portfolios as tradeable instruments,
   rebalancing monthly to weights `(+½, +½, −½, −½)` with all costs set to zero. The engine's
   monthly return series must match French's published HML to a few bps. This is a genuine test
   of the event loop, order generation, fill handling, portfolio accounting, and return
   computation, with data quality factored out — any drift is an engine bug by construction.
3. **Bottom-up attempt, reported honestly.** Build HML from scratch on free data (Project 01's
   output) and report how far off it lands and why. Matching French bottom-up needs
   CRSP + Compustat; with survivorship-incomplete free prices and fundamentals starting in 2009,
   it will not reach a few bps. The *size of the gap* is the interesting result, and quantifying
   what free data costs is a stronger write-up than quietly substituting an easier target.

Step 2 is what the done criterion is really testing — whether the backtester is trustworthy — and
is the one that must pass. Steps 1 and 3 are the honest framing around it.

---

## Open items

1. **Done criterion unmet.** The three steps above. Requires a network fetch from Ken French's
   data library, which has not been attempted yet.
2. **No `EXPLANATION.md`.** Project 01 carries a plain-language companion with a glossary and the
   portfolio standard says to follow that pattern. Not written yet; should be, once the done
   criterion work settles what the headline result is.
3. **Borrow accrues one interval early.** Financing is charged at each mark for the elapsed
   interval using the position *after* that bar's fills, so a newly opened short is charged for
   the interval preceding its own existence. The error is a single interval per position
   lifetime, is in the conservative direction (overcharging), and is immaterial on a
   multi-month hold — but it is wrong, and it is documented rather than fixed because the fix
   needs the previous mark's positions carried as extra state.
4. **`config.py` is at 0% coverage.** Nothing imports it yet — paths are passed explicitly
   everywhere. It exists for the done-criterion scripts to use and is untested until they do.
5. **No Parquet data handler.** `DataFrameDataHandler` holds everything in memory, which covers a
   30-year daily backtest on 1,000 names (~7.5M rows) but not intraday. The `DataHandler`
   contract is designed so a chunked Parquet reader drops in behind it; not needed yet.
6. **Sector constraint binds on net, not gross.** An internally hedged sector passes the limit
   even when its gross exposure is large. The principled fix is a factor-model constraint
   (Project 10); this is the standard exposure-based approximation and is noted as such in
   `risk.py`.
7. **Impact coefficients are literature values, not fitted.** `eta`, `gamma`, and `Y` are round
   numbers from published estimates. Any result sensitive to their exact value must be reported
   as a range — `analytics/capacity.py` sweeps them for this reason, but no result has yet been
   produced that needs it.
8. **Nothing committed.** The working tree holds the full session's work, unstaged.
