"""
Portfolio state: cash, positions, NAV, and the equity curve.

This module is pure accounting. It decides nothing — no sizing, no risk checks,
no order generation. It applies fills, corporate actions, and carry to a book,
and it records what happened. Keeping it decision-free is what makes it
testable against hand-computed answers, which is how the arithmetic here is
actually verified.

NAV definition
--------------
    NAV = cash + sum over positions of (signed quantity x mark)

Short positions carry negative quantity, so their market value is negative and
nets against the cash their sale produced. A short sold at 100 and marked at 90
shows +100 cash and -90 market value: +10 of NAV, which is the gain. Getting
this right is why quantity is signed rather than paired with a direction flag.

Cash and short sales
--------------------
Short proceeds land in cash and stay there. That is how a prime-brokerage
account behaves; the proceeds are the lender's collateral, they sit in the
account, and the borrow fee is charged separately by `FinancingModel`. It also
means gross leverage — not the cash balance — is the binding constraint on a
long-short book, which is why leverage limits live in `portfolio.risk` and
matter more here than a cash check does.

What this module refuses to do
------------------------------
It will not let cash go negative unless margin is explicitly enabled. A
backtest that quietly runs a negative cash balance is a backtest of a strategy
with a free credit line, and it is one of the four bugs the project spec calls
out by name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from evbt.core.events import (
    CorporateActionEvent,
    CorporateActionType,
    FillEvent,
    MarkToMarketEvent,
    OrderEvent,
    OrderSide,
)
from evbt.portfolio.financing import FinancingModel
from evbt.portfolio.position import Position

log = logging.getLogger(__name__)


class InsufficientCashError(RuntimeError):
    """Raised when a fill would overdraw an account with margin disabled."""


@dataclass
class PortfolioSnapshot:
    """One row of the equity curve, recorded at each mark-to-market."""

    timestamp: datetime
    cash: float
    positions_value: float      # signed: longs positive, shorts negative
    nav: float
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float       # positive magnitude
    n_positions: int
    realized_pnl: float
    unrealized_pnl: float
    cumulative_commission: float
    cumulative_financing: float
    cumulative_dividends: float
    turnover_notional: float    # notional traded since the previous snapshot


class Portfolio:
    """
    The book. Consumes fills and corporate actions; produces NAV and history.
    """

    def __init__(
        self,
        initial_cash: float,
        *,
        financing: Optional[FinancingModel] = None,
        allow_margin: bool = True,
        name: str = "portfolio",
    ) -> None:
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {initial_cash}")

        self.name = name
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.financing = financing or FinancingModel()
        self.allow_margin = allow_margin

        self.positions: dict[str, Position] = {}
        self._marks: dict[str, float] = {}

        self.cumulative_commission = 0.0
        self.cumulative_financing = 0.0
        self.cumulative_dividends = 0.0
        self.realized_pnl = 0.0

        self.history: list[PortfolioSnapshot] = []
        self.fills: list[FillEvent] = []

        self._last_mark_time: Optional[datetime] = None
        self._turnover_since_snapshot = 0.0
        # Short market values as of the last mark. Financing for the interval
        # that follows is charged against this, not against the live book —
        # see `_snapshot_financing_basis`.
        self._financing_basis: dict[str, float] = {}

    # --- valuation ---------------------------------------------------------

    def position(self, symbol: str) -> Position:
        """The position in `symbol`, creating a flat one if it does not exist."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def quantity(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0.0

    @property
    def open_positions(self) -> dict[str, Position]:
        return {s: p for s, p in self.positions.items() if p.is_open}

    def mark_price(self, symbol: str) -> Optional[float]:
        return self._marks.get(symbol)

    @property
    def positions_value(self) -> float:
        """Signed total market value of all open positions."""
        return sum(
            p.market_value(self._marks.get(s)) for s, p in self.positions.items() if p.is_open
        )

    @property
    def nav(self) -> float:
        """Net asset value — the number the equity curve is built from."""
        return self.cash + self.positions_value

    @property
    def gross_exposure(self) -> float:
        return sum(
            abs(p.market_value(self._marks.get(s)))
            for s, p in self.positions.items()
            if p.is_open
        )

    @property
    def net_exposure(self) -> float:
        return self.positions_value

    @property
    def long_exposure(self) -> float:
        return sum(
            p.market_value(self._marks.get(s))
            for s, p in self.positions.items()
            if p.is_long
        )

    @property
    def short_exposure(self) -> float:
        """Positive magnitude of the short book."""
        return -sum(
            p.market_value(self._marks.get(s))
            for s, p in self.positions.items()
            if p.is_short
        )

    @property
    def gross_leverage(self) -> float:
        nav = self.nav
        return self.gross_exposure / nav if nav > 0 else 0.0

    @property
    def net_leverage(self) -> float:
        nav = self.nav
        return self.net_exposure / nav if nav > 0 else 0.0

    @property
    def unrealized_pnl(self) -> float:
        return sum(
            p.unrealized_pnl(self._marks.get(s))
            for s, p in self.positions.items()
            if p.is_open
        )

    # --- event handling ----------------------------------------------------

    def on_fill(self, fill: FillEvent) -> None:
        """
        Apply an execution: move cash, update the position, record it.

        Cash moves by `fill.cash_delta`, which is negative for a buy (you paid)
        and positive for a sell (you received), with commission always subtracted.
        """
        projected_cash = self.cash + fill.cash_delta
        if projected_cash < 0 and not self.allow_margin:
            raise InsufficientCashError(
                f"fill {fill.order_id} ({fill.side.value} {fill.quantity:g} "
                f"{fill.symbol} @ {fill.fill_price:.4f}) would take cash to "
                f"{projected_cash:.2f} with margin disabled"
            )

        position = self.position(fill.symbol)
        realized = position.apply_fill(fill)

        self.cash = projected_cash
        self.realized_pnl += realized
        self.cumulative_commission += fill.commission
        self._marks[fill.symbol] = fill.fill_price
        self._turnover_since_snapshot += fill.gross_value
        self.fills.append(fill)

    def on_corporate_action(self, event: CorporateActionEvent) -> None:
        """
        Apply a split or dividend to an existing position.

        No-ops when flat, which is the common case — corporate actions are
        streamed for the whole universe, not just what is held.
        """
        position = self.positions.get(event.symbol)
        if position is None or not position.is_open:
            # A split still has to restate any stale mark, or the next NAV
            # computed before the symbol's next bar arrives would be wrong by
            # the split ratio. Positions that are flat have nothing else to do.
            if event.action is CorporateActionType.SPLIT and event.symbol in self._marks:
                self._marks[event.symbol] /= event.split_ratio
            return

        if event.action is CorporateActionType.SPLIT:
            position.apply_split(event.split_ratio)
            if event.symbol in self._marks:
                self._marks[event.symbol] /= event.split_ratio
        else:
            cash = position.apply_dividend(event.cash_amount)
            self.cash += cash
            self.cumulative_dividends += cash

    def on_mark_to_market(self, event: MarkToMarketEvent) -> None:
        """
        Mark the book to the new bars, accrue carry, and snapshot.

        Carry for the interval that just elapsed is charged against the book as
        it stood at the *previous* mark, then today's prices are applied. Both
        halves of that ordering matter — see `_accrue_financing`.
        """
        self._accrue_financing(event.timestamp)

        for symbol, bar in event.bars.items():
            self._marks[symbol] = bar.close
            position = self.positions.get(symbol)
            if position is not None:
                position.mark(bar.close)

        self._last_mark_time = event.timestamp
        self._record_snapshot(event.timestamp)
        self._snapshot_financing_basis()

    def _snapshot_financing_basis(self) -> None:
        """
        Record the short book as it stands at this mark, to be financed over the
        interval that follows.

        Carry is owed on what was *held* through an interval, and the position
        held through `[t-1, t]` is the one that existed at `t-1` — not the one
        that exists at `t`, which already includes today's fills. Accruing
        against the current book charges a short opened this morning for the
        night before it existed, and credits nothing for one closed this
        morning that was in fact borrowed all night. Both errors are a single
        interval per position, but they are errors, and holding the basis
        explicitly is cheaper than explaining the discrepancy later.
        """
        self._financing_basis = {
            symbol: position.market_value(self._marks.get(symbol))
            for symbol, position in self.positions.items()
            if position.is_short
        }

    def _accrue_financing(self, now: datetime) -> None:
        if self._last_mark_time is None:
            return
        days = (now - self._last_mark_time).total_seconds() / 86400.0
        if days <= 0:
            return

        total = 0.0
        for symbol, market_value in self._financing_basis.items():
            charge = self.financing.borrow_charge(symbol, market_value, days)
            if charge:
                self.position(symbol).accrue_borrow(charge)
                total += charge

        # Cash interest uses the live balance rather than a snapshot: unlike a
        # borrow, which is contracted against a specific position held overnight,
        # the cash balance is what it is at the moment interest is computed.
        interest = self.financing.cash_interest(self.cash, days)
        self.cash += interest - total
        self.cumulative_financing += total - interest

    def _record_snapshot(self, timestamp: datetime) -> None:
        self.history.append(
            PortfolioSnapshot(
                timestamp=timestamp,
                cash=self.cash,
                positions_value=self.positions_value,
                nav=self.nav,
                gross_exposure=self.gross_exposure,
                net_exposure=self.net_exposure,
                long_exposure=self.long_exposure,
                short_exposure=self.short_exposure,
                n_positions=len(self.open_positions),
                realized_pnl=self.realized_pnl,
                unrealized_pnl=self.unrealized_pnl,
                cumulative_commission=self.cumulative_commission,
                cumulative_financing=self.cumulative_financing,
                cumulative_dividends=self.cumulative_dividends,
                turnover_notional=self._turnover_since_snapshot,
            )
        )
        self._turnover_since_snapshot = 0.0

    # --- pre-trade queries -------------------------------------------------

    def can_afford(self, order: OrderEvent, estimated_price: float) -> tuple[bool, str]:
        """
        Whether `order` is fundable at `estimated_price`. Returns (ok, reason).

        Called by the broker before an order is accepted. `estimated_price` is
        necessarily an estimate — the true fill price is a bar in the future,
        which is exactly the point — so this is a screen against obvious
        overdrafts, not a guarantee. The broker re-checks at fill time, where
        the price is known.
        """
        if self.allow_margin:
            return True, ""
        if order.side is OrderSide.SELL:
            return True, ""
        cost = order.quantity * estimated_price
        if cost > self.cash:
            return (
                False,
                f"insufficient cash: need {cost:,.2f}, have {self.cash:,.2f}",
            )
        return True, ""

    def target_quantity(self, symbol: str, target_weight: float) -> float:
        """
        Shares needed to hold `target_weight` of NAV in `symbol` at its last mark.

        Uses the last mark, which is the most recent *known* price — never a
        future one. The order this sizes will fill at a different price; that
        gap is real execution risk and the engine does not paper over it.
        """
        price = self._marks.get(symbol)
        if price is None or price <= 0:
            return 0.0
        return target_weight * self.nav / price

    # --- output ------------------------------------------------------------

    def equity_curve(self) -> pd.DataFrame:
        """The recorded history as a DataFrame indexed by timestamp."""
        if not self.history:
            return pd.DataFrame()
        frame = pd.DataFrame([vars(s) for s in self.history])
        return frame.set_index("timestamp").sort_index()

    def fills_frame(self) -> pd.DataFrame:
        """Every fill, for cost attribution and trade-level post-mortems."""
        if not self.fills:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "timestamp": f.timestamp,
                    "order_id": f.order_id,
                    "symbol": f.symbol,
                    "side": f.side.value,
                    "quantity": f.quantity,
                    "fill_price": f.fill_price,
                    "reference_price": f.reference_price,
                    "gross_value": f.gross_value,
                    "commission": f.commission,
                    "spread_cost": f.spread_cost,
                    "impact_cost": f.impact_cost,
                    "slippage_cost": f.slippage_cost,
                    "total_cost": f.total_cost,
                    "is_partial": f.is_partial,
                }
                for f in self.fills
            ]
        )

    def positions_frame(self) -> pd.DataFrame:
        """Current holdings, one row per open position."""
        rows = []
        for symbol, position in sorted(self.open_positions.items()):
            mark = self._marks.get(symbol)
            rows.append(
                {
                    "symbol": symbol,
                    "quantity": position.quantity,
                    "average_cost": position.average_cost,
                    "mark": mark,
                    "market_value": position.market_value(mark),
                    "weight": position.market_value(mark) / self.nav if self.nav else 0.0,
                    "unrealized_pnl": position.unrealized_pnl(mark),
                    "realized_pnl": position.realized_pnl,
                    "dividends_received": position.dividends_received,
                    "borrow_cost_paid": position.borrow_cost_paid,
                }
            )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"<Portfolio {self.name} nav={self.nav:,.2f} cash={self.cash:,.2f} "
            f"positions={len(self.open_positions)} gross={self.gross_leverage:.2f}x>"
        )
