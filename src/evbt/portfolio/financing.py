"""
Carry: what it costs to hold the book overnight, before anything trades.

Financing is the cost most often omitted from long-short backtests, and the
omission is not small. A dollar-neutral book at 1.0x gross runs a short book
equal to half its gross; at a general-collateral borrow of 30–50 bps that is
15–25 bps a year straight off the top, and specials — the names a value or
short-interest signal actually selects — run anywhere from 100 bps to several
hundred. A strategy whose edge is 200 bps a year is a different strategy once
you charge it.

Day counts use ACT/360, the money-market convention for financing. Not ACT/365:
the difference is 1.4% of the financing charge, which is small but free to get
right and the sort of thing a practitioner notices.

Defaults are all zero. That is deliberate — every financing assumption in a
reported result should have been switched on by hand, not inherited from a
library's idea of a reasonable rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FinancingModel:
    """
    Overnight carry on shorts and on the cash balance.

    Parameters
    ----------
    borrow_rate
        Annualised general-collateral borrow, charged on the absolute market
        value of short positions. 0.003 = 30 bps is a reasonable large-cap GC
        assumption; small caps and crowded shorts are much worse.
    hard_to_borrow
        Per-symbol overrides for specials. A name here is charged its own rate
        instead of `borrow_rate`.
    cash_credit_rate
        Earned on a positive cash balance (including short-sale proceeds, which
        a prime broker rebates). Left at zero by default, which understates the
        returns of a market-neutral book in a high-rate regime — an error in the
        conservative direction.
    cash_debit_rate
        Charged on a negative cash balance, i.e. margin borrowing.
    """

    borrow_rate: float = 0.0
    hard_to_borrow: dict[str, float] = field(default_factory=dict)
    cash_credit_rate: float = 0.0
    cash_debit_rate: float = 0.0
    day_count: float = 360.0

    def borrow_rate_for(self, symbol: str) -> float:
        return self.hard_to_borrow.get(symbol, self.borrow_rate)

    def borrow_charge(self, symbol: str, short_market_value: float, days: float) -> float:
        """
        Cost of borrowing one short position for `days`. Always non-negative.

        `short_market_value` is the position's signed market value, so it is
        negative for a short; longs are never charged.
        """
        if short_market_value >= 0.0 or days <= 0.0:
            return 0.0
        return abs(short_market_value) * self.borrow_rate_for(symbol) * days / self.day_count

    def cash_interest(self, cash: float, days: float) -> float:
        """
        Signed interest on the cash balance: positive is earned, negative paid.
        """
        if days <= 0.0:
            return 0.0
        rate = self.cash_credit_rate if cash >= 0.0 else self.cash_debit_rate
        return cash * rate * days / self.day_count
