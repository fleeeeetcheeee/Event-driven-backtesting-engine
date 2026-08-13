# What this project is, in plain language

The README is written for someone who already works in quantitative finance. This is the same system explained to someone who doesn't. No prior knowledge assumed; a glossary of every technical term is at the bottom.

---

## The problem

Suppose you have an idea for a trading strategy. *Buy companies that look cheap relative to their assets; sell the ones that look expensive.* Before risking real money, you want to know whether it would have worked in the past.

So you get historical prices, write some code, and simulate: on each day in history, what would my rules have told me to do, and what would have happened? You get an answer — a chart of hypothetical wealth over time. This is called a **backtest**.

Here is the uncomfortable fact about backtests: **most of them are wrong, and the wrong ones look exactly like the right ones.** There's no error message. The chart is smooth, the numbers are plausible, and the strategy loses money the moment real capital touches it.

This project builds a backtester designed around that fact — one whose structure makes the most common ways of being wrong difficult rather than easy.

---

## The single most common mistake

Imagine you're simulating a day of trading. Your code is looking at Tuesday. It calculates that a stock looks cheap. So it buys.

**At what price?**

The obvious answer — Tuesday's closing price — is wrong, and the reason is subtle. To know Tuesday's closing price, you have to wait until Tuesday's market close. But the decision "this looks cheap" was itself based on Tuesday's closing price. So you're buying at a price that only became knowable at the exact instant your decision was made. In reality, you'd place that order Tuesday evening and it would execute Wednesday morning, at whatever price Wednesday opened at — which nobody knew on Tuesday.

This is called **lookahead bias**: the simulation using information that wasn't actually available at the time it pretends to act.

It sounds like a small detail. It is not. Consider a stock that closes at \$100 on Tuesday and opens at \$110 on Wednesday because of overnight news. A backtest that buys at Tuesday's close captures that 10% jump. A real trader would have bought at \$110 and captured nothing. Repeat across thousands of trades and you have manufactured an enormous amount of profit out of nothing at all.

The insidious part: this bug is often *one character* of code. In the most common way people write backtests, forgetting a single instruction that shifts data by one day produces exactly this error, silently.

**How this project handles it.** The rule is enforced structurally: *an order created at a given moment can only execute against price data from strictly after that moment.* If any part of the system ever tries to break it, the program stops with an error rather than quietly producing a good-looking result. There is no configuration flag to turn it off.

There's a test in this repository that demonstrates it. Prices sit flat at \$100, then jump to \$200. A strategy watches for jumps and buys when it sees one. It buys at \$200 — after the jump — and earns nothing. A broken simulator would buy at \$100 and double its money.

---

## Two ways to build a backtester

**The fast way.** Load all the prices into a big table, add a column for "what my strategy said to do that day," multiply it by a column of daily price changes, and add it up. Thousands of times faster to run and about ten lines of code. It's called a *vectorised* backtest.

**The slow way.** Simulate time actually passing. Monday happens. Then Tuesday happens. Prices arrive, orders execute, cash moves, the strategy reacts, new orders are placed for Wednesday. This is called an **event-driven** backtest, and it's what this project builds.

The slow way is right, and the reason isn't performance — it's that the fast way *cannot represent* the things that go wrong in real trading:

- **Order timing.** In the table, "what I decided" and "what happened" are two columns on the same row. The one-day gap between them, which is where lookahead bias lives, has no place to exist.
- **Not getting your whole order.** If you want a million shares of a stock that only trades a million shares a day, you cannot have them today. Real orders get filled in pieces across days, at prices that keep moving while you work. In a table, you own the position instantly.
- **Running out of money.** In a table, nothing stops the "position" column from implying you spent money you didn't have. A simulation that quietly runs a negative bank balance is a simulation of a strategy with a free unlimited credit line.

The project specification this work follows rules out the fast approach explicitly, for exactly these reasons.

---

## How the simulation works

Everything that happens is an **event** in a queue, and the queue plays them in order. Within a single day, the order is fixed and each step depends on the one before it:

1. **Corporate actions.** Dividends get paid; stock splits change share counts. These come first because they affect the position you're carrying *into* the day.
2. **Prices arrive.** Today's prices become visible for the first time.
3. **Orders execute.** Orders placed on *earlier* days — never this one — get filled at today's prices.
4. **Unfilled orders expire.**
5. **The portfolio is valued.** Everything is priced at today's close; interest and borrowing costs are charged; a row is written to the wealth chart.
6. **The strategy looks around.** Only now does it see today's prices and its own updated position.
7. **New orders are placed.** They wait for tomorrow.

Two of those placements matter more than they look.

*Orders execute before the portfolio is valued* — otherwise today's wealth number wouldn't include today's trades.

*The strategy looks around after the portfolio is valued* — otherwise a strategy deciding "invest 5% of my money in this" would be calculating 5% of yesterday's money. That error is small each day, never visible in the output, and compounds forever.

---

## The costs that decide whether a strategy is real

A strategy that makes money on paper often loses money in practice, because trading isn't free. This simulator charges four separate things, kept separate because they behave differently and a good post-mortem needs to know which one killed you.

**Commission.** What your broker charges. The easy one — it's a contract, and you know the number.

**The spread.** At any moment there are two prices: what buyers will pay and what sellers will accept. The gap between them is the spread. Buy now and you pay the higher one; sell now and you get the lower one. Roughly speaking you lose half the spread every time you trade in a hurry.

Something important here: spreads get *wider* when markets are turbulent. And many strategies trade specifically *because* markets are turbulent. So they hit their worst trading costs exactly when they trade most. Assuming one fixed spread hides that connection entirely.

**Market impact — the one that decides how big you can get.** When you buy a lot of something, you push its price up. You're competing with everyone else who wants it. So the more you buy, the worse the average price you pay. Your own trading works against you.

This is why a strategy can be excellent with \$10 million and worthless with \$10 billion. The profitable idea doesn't change; the cost of acting on it grows with your size until it eats everything. That ceiling is called **capacity**, and this project estimates it.

The relationship isn't proportional. Empirically, cost grows roughly with the *square root* of how much you trade — so buying four times as much costs about twice as much per share, not four times. This is the "square-root law," one of the most stable empirical regularities in finance.

**Financing.** Two costs of simply holding positions overnight.

If you're betting a stock will *fall*, you sell shares you don't own — you borrow them first. That borrowing has a daily fee. And crucially, if the company pays a dividend while you're borrowing its shares, **you** pay it, to whoever lent them to you. A simulation that forgets this overstates every downward bet by the dividend.

In this project all financing costs default to **zero**, which sounds backwards but is deliberate: it means any financing assumption in a published result had to be typed in by hand rather than inherited silently from a library's guess.

---

## Proving it works

An engine that produces smooth, plausible, wrong numbers is the exact failure mode we're guarding against — so "it runs without crashing" proves nothing. Two kinds of evidence here.

**Arithmetic checkable by hand.** Small scenarios where the right answer is worked out on paper in the test itself. Buy 1,000 shares at \$100, collect a \$2 dividend, survive a 2-for-1 split, and the final wealth must be exactly \$102,000 — the split being economically neutral, the dividend the only source of gain. If the code disagrees with the paper, the code is wrong.

**Matching a published result computed by someone else.** This is the strong evidence.

There's a famous stock-market strategy called **HML** — buy cheap-looking companies, sell expensive-looking ones — defined precisely by two academics, Eugene Fama and Kenneth French, in the early 1990s. French publishes its month-by-month returns going back to 1926. It's one of the most scrutinised numbers in finance.

So: hand the simulator the same raw ingredients French uses, tell it to run that strategy, and see whether it arrives at his published answer.

**It does — across 1,199 months, nearly a century, with the largest single-month disagreement being 0.5 basis points.** A basis point is one hundredth of one percent. Half of one is roughly five parts in a million.

The reason that number is *exactly* 0.5 is the most convincing part. French publishes his returns rounded to two decimal places, so his own figures carry a rounding uncertainty of exactly 0.5 basis points. The simulator agrees with him as closely as his published precision permits, and no closer — because no closer is possible. **The engine contributes no error of its own.** Had anything been wrong — a bet recorded with the wrong sign, a purchase price carried over incorrectly, stale wealth used to size a trade, a one-day timing slip — the disagreement would be messy and larger. Instead it lands exactly on the floor set by the data.

That's the difference between "my code is self-consistent" and "my code agrees with reality."

---

## What this does *not* prove

Being clear about limits is part of the work, not an apology for it.

**The engine was given French's ingredients, not asked to grow them.** Building those six ingredient portfolios from scratch — sorting thousands of companies by size and cheapness, reconstructing decades-old accounting data, handling firms that went bankrupt or were acquired — is a separate and much harder problem needing expensive commercial databases. Reproducing HML to 0.5 basis points does **not** mean "can build HML from nothing."

**The test ran with all costs switched off.** French's published numbers are theoretical, before trading costs, so charging anything would have made the comparison meaningless. The test therefore proves the *bookkeeping* is right. It says nothing about whether the cost models are calibrated well — those are checked only against their own formulas.

**The comparison used monthly data, which barely exercises the timing rules.** The construction that makes the match exact also removes the price gap between deciding and trading. So the one-day-delay rule and partial-fill logic are proven by the synthetic tests, not by this comparison.

**The cost model numbers come from published research, not from measurement.** Nobody fitted them to data in this repository. Any conclusion sensitive to their exact values must be reported as a range, not a single number.

---

## Why this matters before anything else

This is the second of roughly twenty-five projects. It comes near the beginning deliberately: every later project — every trading idea, every statistical model — will be judged by running it through this simulator.

If the simulator is wrong, everything downstream inherits the error, and the error will look like a discovery. You'd conclude a bad idea was good and spend months building on sand. Hence the disproportionate effort spent proving this one is right before using it for anything.

---

## Glossary

**Backtest** — Simulating a trading strategy on historical data to see how it would have performed.

**Basis point (bp)** — One hundredth of one percent. 0.01%. 100 basis points = 1%.

**Bar** — One period's price summary: opening price, highest, lowest, closing price, and volume traded. A "daily bar" summarises one day.

**Capacity** — The largest amount of money a strategy can manage before its own trading costs consume its profits.

**Corporate action** — Something a company does that mechanically changes its shares: paying a dividend, splitting the stock.

**Dividend** — A cash payment a company makes to shareholders.

**Event-driven** — A simulation design where time advances step by step and each occurrence is handled in sequence, as opposed to processing all history at once as a table.

**Fill** — An order actually executing. Orders can be *partially* filled if there isn't enough trading volume to complete them at once.

**HML** — "High Minus Low." A standard reference strategy: buy companies that are cheap relative to their accounting book value, sell those that are expensive. Published monthly since 1926 by Kenneth French.

**Leverage** — Trading with more exposure than you have money. *Gross* leverage counts all positions regardless of direction; *net* counts them against each other.

**Lookahead bias** — The error of a simulation using information that wasn't available at the moment it pretends to act. The most common way backtests produce fake profits.

**Market impact** — The price movement caused by your own trading. Buying pushes prices up against you; selling pushes them down.

**NAV (Net Asset Value)** — Total wealth: cash plus the current market value of everything held, minus what's owed.

**Order** — An instruction to buy or sell. A *market* order says "at whatever the price is." A *limit* order says "only at this price or better."

**Point-in-time** — Data as it was actually known on a given date, rather than as it's known today after later corrections.

**Portfolio** — The full collection of positions and cash.

**Position** — A holding in one particular security. *Long* means you own it and profit if it rises. *Short* means you've borrowed and sold it, and profit if it falls.

**Rebalance** — Adjusting holdings back to target proportions, since prices drift them apart over time.

**Sharpe ratio** — Return earned per unit of risk taken. Higher is better. Around 1 is good for a real strategy; anything above 3 in a stock-market backtest is far more likely to be a bug than a discovery.

**Short selling** — Betting a price will fall, by borrowing shares, selling them, and buying them back later. You pay a borrowing fee, and you owe any dividends paid while you hold the position.

**Slippage** — The gap between the price you expected and the price you got.

**Spread** — The difference between the highest price a buyer will pay and the lowest a seller will accept.

**Stock split** — A company dividing existing shares into more shares. A 2-for-1 split doubles your share count and halves the price. Your wealth is unchanged; only the units change.

**Survivorship bias** — Studying only the companies that still exist today, silently excluding those that went bankrupt or were acquired. Makes historical results look far better than they were.

**Turnover** — How much trading a strategy does. High turnover means high costs.

**Volume** — How many shares changed hands. Determines how much you can trade without moving the price against yourself.

**Walk-forward** — Testing repeatedly on data that comes *after* the data used to make decisions, mimicking how a strategy would really be run.
