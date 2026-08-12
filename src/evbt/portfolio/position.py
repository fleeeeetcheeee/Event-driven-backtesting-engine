"""
Per-symbol position accounting.

Small module, disproportionate importance: three specific cases here are where
hand-rolled backtesters lose money that the real strategy would not have made,
or make money it would not have.

**1. Sign flips.** Holding +100 and selling 150 is not "quantity becomes -50".
It is: realise the P&L on 100 shares at the fill price, then open a *new* short
of 50 whose cost basis is the fill price. Code that just adds signed quantities
and keeps the old average cost carries a basis across the flip and produces
realised P&L that never happened. `apply_fill` handles the crossing explicitly.

**2. Average cost on partial closes.** Closing part of a position does not
change the average cost of what remains. Recomputing it on the way out is a
common slip and silently rewrites the basis of the residual.

**3. Shorts owe dividends.** Quantity is signed and dividends are credited as
`quantity * dividend_per_share`, so a short position is *debited* automatically.
Every real short seller pays the dividend to the lender; a backtest that skips
this overstates short-side returns by the dividend yield, which on a value or
low-beta short book is a large fraction of the whole result.

Cost basis uses a weighted average rather than FIFO lots. For pre-tax P&L —
which is what a backtest measures — the two are identical in total; they differ
only in how P&L is split between realised and unrealised along the way. FIFO
would be needed for tax-lot accounting, which is out of scope and is noted as
such in the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from evbt.core.events import FillEvent


@dataclass
class Position:
    """
    A holding in one symbol. `quantity` is signed: negative is short.

    `average_cost` is always a positive per-share price, for both longs and
    shorts. For a short it is the average price at which the shares were sold.
    """

    symbol: str
    quantity: float = 0.0
    average_cost: float = 0.0

    realized_pnl: float = 0.0          # gross of commission, so costs stay separable
    total_commission: float = 0.0
    dividends_received: float = 0.0    # negative when short: dividends owed
    borrow_cost_paid: float = 0.0

    # Last mark, kept so NAV can be computed without re-reading market data.
    last_price: Optional[float] = None

    # --- state ------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        # Exact zero comparison is safe here: quantity reaches zero only via the
        # explicit closing branch in `apply_fill`, which assigns 0.0 outright
        # rather than arriving there by subtraction.
        return self.quantity != 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0.0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0.0

    @property
    def cost_basis(self) -> float:
        """Signed capital committed. Negative for a short (proceeds received)."""
        return self.quantity * self.average_cost

    def market_value(self, price: Optional[float] = None) -> float:
        """
        Signed market value. Negative for a short — it is a liability, and
        summing signed market values into NAV is what makes the short side net
        out correctly against the cash its sale produced.
        """
        mark = price if price is not None else self.last_price
        if mark is None:
            return 0.0
        return self.quantity * mark

    def unrealized_pnl(self, price: Optional[float] = None) -> float:
        mark = price if price is not None else self.last_price
        if mark is None or not self.is_open:
            return 0.0
        return self.quantity * (mark - self.average_cost)

    def total_pnl(self, price: Optional[float] = None) -> float:
        """All-in P&L for this symbol: trading, carry, and costs."""
        return (
            self.realized_pnl
            + self.unrealized_pnl(price)
            + self.dividends_received
            - self.borrow_cost_paid
            - self.total_commission
        )

    # --- mutation ---------------------------------------------------------

    def apply_fill(self, fill: FillEvent) -> float:
        """
        Apply an execution. Returns the realised P&L produced by *this* fill.

        Three cases, in the order they are tested:
          - opening or adding in the same direction  -> weighted-average the cost
          - reducing without crossing zero           -> realise, basis unchanged
          - crossing zero                            -> realise the old side in
                                                        full, then open the new
                                                        side at the fill price
        """
        if fill.symbol != self.symbol:
            raise ValueError(
                f"fill for {fill.symbol} applied to position in {self.symbol}"
            )

        self.total_commission += fill.commission
        delta = fill.signed_quantity
        realized = 0.0

        if self.quantity == 0.0:
            self.quantity = delta
            self.average_cost = fill.fill_price

        elif (self.quantity > 0) == (delta > 0):
            # Adding to the existing side.
            total_qty = self.quantity + delta
            self.average_cost = (
                abs(self.quantity) * self.average_cost + abs(delta) * fill.fill_price
            ) / abs(total_qty)
            self.quantity = total_qty

        else:
            # Reducing or crossing. `closing` is how much of the existing
            # position this fill actually retires.
            closing = min(abs(delta), abs(self.quantity))
            direction = 1.0 if self.quantity > 0 else -1.0
            realized = closing * (fill.fill_price - self.average_cost) * direction
            self.realized_pnl += realized

            remaining = abs(delta) - closing
            if remaining > 0.0:
                # Crossed through zero: the residual is a brand-new position on
                # the other side, and its basis is this fill's price.
                self.quantity = remaining * (1.0 if delta > 0 else -1.0)
                self.average_cost = fill.fill_price
            else:
                self.quantity += delta
                if self.quantity == 0.0:
                    # Flat. Zero the basis so a stale price cannot leak into a
                    # later unrealised-P&L calculation.
                    self.average_cost = 0.0

        self.last_price = fill.fill_price
        return realized

    def apply_split(self, ratio: float) -> None:
        """
        A `ratio`-for-1 split: shares scale up, per-share basis scales down.

        Position *value* is invariant, which is the invariant worth asserting in
        a test: `quantity * average_cost` is unchanged, and so is
        `quantity * price` once the price series reflects the split.
        """
        if ratio <= 0:
            raise ValueError(f"split ratio must be positive, got {ratio}")
        self.quantity *= ratio
        self.average_cost /= ratio
        if self.last_price is not None:
            self.last_price /= ratio

    def apply_dividend(self, per_share: float) -> float:
        """
        Credit (or, when short, debit) a cash dividend. Returns the cash effect.

        The cost basis is deliberately left alone. Reducing basis by the
        dividend is a tax convention for return-of-capital distributions, not
        the accounting for an ordinary dividend, and doing it here would
        double-count the payment against realised P&L on the eventual sale.
        """
        cash = self.quantity * per_share
        self.dividends_received += cash
        return cash

    def accrue_borrow(self, cost: float) -> None:
        self.borrow_cost_paid += cost

    def mark(self, price: float) -> None:
        self.last_price = price
