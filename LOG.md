# Project 2 — Event-Driven Backtesting Engine

**Tier:** 0 (Infrastructure)
**Spec:** `ResearchToDo.md` → Part 3 → Tier 0 → Project 2
**Repo:** https://github.com/fleeeeetcheeee/Event-driven-backtesting-engine
**Status:** **Done criterion met.** The engine reproduces Fama-French HML to within 0.5 bps per
month over 1,199 months (1926-07 – 2026-06), with zero months deviating by more than 1 bp — and
its error distribution is indistinguishable from the pure arithmetic reconstruction's, meaning
the engine contributes no error beyond French's own 2-decimal rounding. 320 tests pass, 95%
coverage. See [Open items](#open-items) for what remains.

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

Entire session written as work happened. One sitting, committed as `99b7b82` at the end of it.

#### Repository setup

`git init` + `git remote add origin` against the pre-existing GitHub repo. No commit, no push —
per the workspace rule that every commit is approved first.

The local repo was initialised fresh, so its history was unrelated to the remote's, which already
carried an `Initial commit` (`554490b`) holding a `LICENSE` and a one-line README stub created
through the GitHub web UI. Reconciled by rebasing the session's two commits onto that initial
commit rather than force-pushing over it — the LICENSE is worth keeping, and discarding a commit
that exists on a remote is not something to do to avoid a two-minute conflict resolution. The
only conflict was add/add on `README.md`, resolved in favour of this project's version.

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

### 2026-08-13 — HML replication: the done criterion

Written as work happened.

#### Getting the data

`scripts/fetch_french_data.py` pulls `6_Portfolios_2x3_CSV.zip` and
`F-F_Research_Data_Factors_CSV.zip` from Ken French's library into `data/raw/fama_french/`,
writing bytes verbatim with an atomic `.tmp`-then-rename — same ingestion discipline as Project
01. Both files are gitignored; the repo ships the replication, not the data.

#### The parser was most of the work

French's format is worse than it looks. `6_Portfolios_2x3.csv` holds **ten** stacked tables in
one file: value- and equal-weighted returns, monthly and annual, plus firm counts, average market
cap, and four BE/ME variants. Each is introduced by a free-text title and a header row beginning
with a bare comma. Monthly and annual blocks share identical column names and are distinguished
only by the width of the date field (`202606` vs `2026`). Missing data is `-99.99` or `-999`.
Returns are in percent.

Line numbers are useless — French republishes monthly and every table shifts — so `french.py`
locates tables **structurally**: a `,`-leading header, then consecutive rows whose first field is
a 4- or 6-digit integer, with the title taken from the nearest preceding non-empty line. Verified
against the real file: it finds all ten tables and classifies periodicity correctly.

**The trap, and it is a real one.** The value-weighted and equal-weighted monthly tables are
adjacent and have identical column names. HML is built from the **value-weighted** set. Reading
the equal-weighted table by mistake yields a series that still correlates **0.93** with the real
HML and is wrong by **92 bps a month** — plausible enough to publish, entirely incorrect. So
`weighting` is a required argument with no default, and the wrongness is pinned as a control
assertion in `test_equal_weighted_is_a_different_and_wrong_factor`. Similarly `SMALL LoBM` is
low book-to-market, i.e. small-cap *growth*; reading it as value inverts the factor's sign, so
the columns are renamed to `SmallGrowth`/`SmallValue` on load rather than left as French's
labels.

#### Making the comparison exact rather than approximate

`returns_to_bars` converts French's monthly returns into OHLCV bars with `open[t] = close[t-1]`,
so the price series is continuous across bars. The engine sizes orders at bar `t`'s close and
fills them at bar `t+1`'s open — the *same price* — so no execution drift enters and any
deviation found is an accounting bug with nowhere to hide. Volume is set enormous so the
participation cap never binds. This is a validation harness, not a simulation of trading these
portfolios, and the README says so.

Added `FixedWeightPortfolio` to `strategy/base.py`: restates the target weights every bar. It
signals on *every* bar deliberately — a fixed-weight factor is *defined* as rebalanced each
period, and letting the weights drift between rebalances would be a different portfolio with a
different return.

#### Result

Both steps passed on the first run, which was not the expected outcome and is worth recording as
such. Max deviation is exactly 0.5000 bps in both — recognisably the half-ulp of French's
2-decimal reporting rather than an arbitrary number, which is what makes it credible: an
accounting bug would produce a messy bound, not the data's own precision floor.

The closed-form check behind it: a book holding weights `wᵢ` in assets returning `rᵢ`, funded
from cash, has NAV return exactly `Σ wᵢrᵢ`. For the HML weights that is the factor return. So
the engine had a known target and hit it.

#### One wrong assumption of mine, caught by a test

I asserted the book would be dollar-neutral with max |net leverage| under 5%, and it hit **26%**.
That was my expectation being wrong, not the engine. The equity curve is snapshotted at month
end — after a full month of drift, before the next rebalance fills — and the worst months are
July and August 1932 (HML +35.5% and +34.2%), September 1939, April 1933, and March 2020.
Exactly the months when value and growth diverge most. Median net leverage is 1.6%.

Rather than loosening the bound to whatever passed, I replaced it with a test that asserts the
*mechanism*: net drift must correlate with |HML| above 0.5. A book that had silently stopped
rebalancing would drift monotonically and show no such relationship, so the new test
distinguishes the two cases where a loosened threshold would not.

#### Verification

**320 passed, 95% coverage** (up from 282 / 94%). New: 25 parser tests against a synthetic
fixture in the real format — no network, no data directory needed — and 13 replication tests that
skip cleanly when the French files are absent, so a fresh clone still runs green.

### 2026-08-13 — Clearing the open items

Written as work happened, after the HML replication was committed (`e2e8c53`).

**Borrow accrual fixed (was open item 3).** Carry was charged at each mark against the book
*after* that bar's fills, so a short opened on day 1 paid for the night of day 0. Fixed by
snapshotting the short market values at each mark and accruing the following interval against
that basis — carry is owed on what was *held* through an interval, and that is the position which
existed at its start. Cash interest deliberately still uses the live balance: unlike a borrow,
which is contracted against a specific position held overnight, the cash balance is simply what
it is when interest is computed.

The integration test's hand calculation moved from 11 accrual days to 10, which is the fix
visible in a number. Added `test_no_carry_is_charged_before_the_position_exists` as a regression
pin.

**`ParquetDataHandler` added (was open item 6).** Streams from a Parquet store in date chunks via
DuckDB, so peak memory is set by the chunk size rather than by the length of the backtest. Reads
single files, plain directories, and Hive-partitioned `date=YYYY-MM-DD/` trees.

The property worth having is that chunking is *invisible*: `test_chunk_size_is_invisible_in_the_output`
runs chunk sizes of 1, 7, 50 and 10,000 and asserts every slice is identical to the in-memory
handler's. A backtest whose results depended on available RAM would not be reproducible. That
required carrying a warm-up tail across chunk boundaries so the rolling volatility window does not
restart — without it, every boundary produces a run of bars with no vol estimate and therefore no
slippage, purely as an artefact of the chunk size.

**A wrong assumption caught while writing it.** I wrote `PIT_STORE_COLUMNS` mapping Project 01's
schema from memory as `open_unadj` / `high_unadj` / `low_unadj` / `close_unadj`, symmetrically.
Reading that project's `price_cleaner.py` shows the schema is *asymmetric*: only `close` carries
the unadjusted/adjusted split; `open`, `high`, `low` and `volume` are stored raw under plain
names. The mapping I had written would have failed at query time against the real store. Corrected
and pinned in `test_pit_store_mapping_matches_project_01s_actual_schema`, with a note in the
module explaining the asymmetry — this is the second time in this project that assuming a schema
instead of reading it would have cost real time.

**Sector gross constraint added (was open item 7).** `max_sector_weight` binds on *net* exposure,
so a book 50% long and 50% short one sector passes it while carrying a large bet on within-sector
dispersion — a stat-arb book is exactly that shape. Added `max_sector_gross_weight` alongside it.
When both bind, the tighter scale factor governs, which satisfies both in one pass since each
constraint is homogeneous of degree one in the weights. Neither is a substitute for a factor-model
constraint (Project 10); both are the standard exposure-based approximations and the module says so.

**`config.py` tests added (was open item 4).** It went from 0% coverage to tested once the French
scripts started importing it. `trading_days_per_year` gets its own assertion because 252 vs 250 vs
260 makes Sharpe ratios silently incomparable across codebases.

**`EXPLANATION.md` written (was open item 2).** Plain-language companion for a non-expert, with a
glossary of every technical term, following Project 01's pattern. It states the same limitations
as the README rather than a softened version of them — in particular that reproducing HML to
0.5 bps does not mean HML can be built from scratch.

**Verification: 353 passed, 95% coverage** (up from 320 / 95%).

**Two open items closed as out of scope rather than done**, with reasoning in
[Open items](#open-items): bottom-up HML construction is Project 3's subject matter, not this
project's, and the frictionless-replication caveat is a property of the validation method rather
than a defect to fix.

---

## Status against the done criterion

**Met**, for the part of it that tests the engine. Steps 1 and 2 of the planned approach both
pass; step 3 (bottom-up construction on free data) remains deliberately unattempted — see
[Open items](#open-items).

| | mean \|diff\| | max \|diff\| | months > 1 bp |
|---|---|---|---|
| Step 1 — arithmetic reconstruction vs published HML | 0.2510 bps | 0.5000 bps | 0 / 1,200 |
| Step 2 — engine's traded NAV returns vs published HML | 0.2511 bps | 0.5000 bps | 0 / 1,199 |

The rows being identical is the result, not a coincidence. French publishes to two decimals in
percent, so the half-ulp of his own reporting is exactly 0.5 bp — the floor the engine sits on.
The engine adds nothing to it. That is asserted directly in
`test_engine_adds_no_error_beyond_french_rounding`, which compares the two error distributions
rather than each against a tolerance: a mis-signed short, a basis carried across a rebalance, a
stale NAV used for sizing, or an off-by-one in fill timing would each push the engine's error
above the arithmetic floor, and none does.

Step 2 is what the criterion is really asking — whether the backtester is trustworthy — because
the engine had to *trade* its way there: size orders against NAV, fill them a bar later, carry a
100% short book, mark to market, and compound across a century of monthly rebalances (4,800
orders, 4,796 fills, zero rejections).

---

## Open items

All the actionable items from 2026-08-12 are now closed — see the 2026-08-13 entry above. What
remains is split into standing limitations (properties of the approach, documented rather than
fixable) and work that belongs to a different project.

### Standing limitations

1. **The replication is frictionless and monthly.** All costs and financing are zero in it,
   because French's published series is a gross academic return and charging it anything would
   make the comparison meaningless rather than conservative. So it validates the *accounting*,
   not the cost models, which remain tested only against their own formulas. And because
   `open[t] == close[t-1]` by construction, the next-bar fill rule and partial fills are
   exercised only by the synthetic tests. Not a defect to fix — it is what isolating accounting
   from timing costs you, and the README says so plainly.
2. **Impact coefficients are literature values, not fitted.** `eta`, `gamma` and `Y` are round
   numbers from published estimates, not measurements against any dataset here. Any result
   sensitive to their exact value must be reported as a range; `analytics/capacity.py` sweeps
   them for that reason. Fitting them properly needs order-level execution data this project
   does not have and is unlikely to get.
3. **Capacity estimates are an upper bound.** Costs scale with size in the model, but alpha decay
   with size does not — a large book takes longer to build and the signal has decayed before the
   position is on. Quantifying that needs a signal decay profile, which is Project 6.
4. **Sector constraints are exposure-based, not risk-based.** Both the net and gross caps treat
   two names in the same GICS sector as equally substitutable. The principled version constrains
   estimated factor risk and needs a risk model — Project 10.

### Deferred to another project

5. **Bottom-up HML construction.** Building the six size × book-to-market portfolios from raw data
   — the 2×3 sort, NYSE breakpoints, the June rebalance on prior-December accounting data — is
   *the subject of Project 3* ("Fama-French factor replication"), not of a backtesting engine. It
   was listed here because the done criterion mentions HML, but the criterion tests whether the
   *backtester* is trustworthy, and that is settled. Reclassified rather than carried as debt.

   It also depends on infrastructure that does not exist yet: Project 01 is verified only on a
   smoke run (one EDGAR quarter, 40 tickers), its full bootstrap has never been executed, and its
   price coverage for delisted companies is materially incomplete. Matching French bottom-up needs
   CRSP + Compustat regardless.

   The standing caution: "reproduces HML to 0.5 bps" must never be read as "can build HML from
   scratch". README and `EXPLANATION.md` both say so explicitly.

### Repository state

6. ~~**Not pushed.**~~ **Closed** — both commits are on `origin/main`: `e2e8c53` (HML replication)
   and `dace4bb` (open-item follow-up). Working tree clean, nothing outstanding.
