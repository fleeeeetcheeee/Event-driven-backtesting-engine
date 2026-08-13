"""
Risk constraints: each limit in isolation, then the interaction ordering.

The ordering claim in `risk.py` — that later constraints never breach earlier
ones — is the part worth testing hardest, because it is the argument for why
one pass is enough.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from evbt.portfolio.risk import RiskLimits, RiskManager

T0 = datetime(2024, 1, 2)


class TestPositionCap:
    def test_clips_oversized_longs_and_shorts(self):
        risk = RiskManager(RiskLimits(max_position_weight=0.05))
        out = risk.apply({"A": 0.20, "B": -0.30, "C": 0.01}, timestamp=T0)

        assert out == pytest.approx({"A": 0.05, "B": -0.05, "C": 0.01})

    def test_records_every_clip(self):
        risk = RiskManager(RiskLimits(max_position_weight=0.05))
        risk.apply({"A": 0.20, "B": -0.30}, timestamp=T0)
        assert risk.summary()["max_position_weight"] == 2

    def test_compliant_book_is_untouched_and_silent(self):
        risk = RiskManager(RiskLimits(max_position_weight=0.05))
        out = risk.apply({"A": 0.04, "B": -0.05})
        assert out == pytest.approx({"A": 0.04, "B": -0.05})
        assert risk.violations == []


class TestGrossLeverage:
    def test_scales_the_whole_book_uniformly(self):
        risk = RiskManager(RiskLimits(max_gross_leverage=1.0))
        out = risk.apply({"A": 1.0, "B": -1.0}, timestamp=T0)

        assert sum(abs(w) for w in out.values()) == pytest.approx(1.0)
        # Relative sizing preserved: still equal and opposite.
        assert out["A"] == pytest.approx(0.5)
        assert out["B"] == pytest.approx(-0.5)

    def test_under_the_limit_is_untouched(self):
        risk = RiskManager(RiskLimits(max_gross_leverage=2.0))
        out = risk.apply({"A": 0.5, "B": -0.3})
        assert out == pytest.approx({"A": 0.5, "B": -0.3})


class TestNetLeverage:
    def test_reduces_the_dominant_side(self):
        # net = 0.6, cap 0.1 -> remove 0.5 from the long side (total 0.8)
        risk = RiskManager(RiskLimits(max_net_leverage=0.1))
        out = risk.apply({"A": 0.5, "B": 0.3, "C": -0.2}, timestamp=T0)

        assert sum(out.values()) == pytest.approx(0.1)
        assert out["C"] == pytest.approx(-0.2)  # the short side is left alone

    def test_reduction_preserves_relative_weights_within_the_side(self):
        risk = RiskManager(RiskLimits(max_net_leverage=0.1))
        out = risk.apply({"A": 0.5, "B": 0.3, "C": -0.2})
        assert out["A"] / out["B"] == pytest.approx(0.5 / 0.3)

    def test_handles_a_dominant_short_side(self):
        risk = RiskManager(RiskLimits(max_net_leverage=0.1))
        out = risk.apply({"A": 0.2, "B": -0.5, "C": -0.3})
        assert sum(out.values()) == pytest.approx(-0.1)

    def test_already_neutral_book_is_untouched(self):
        risk = RiskManager(RiskLimits(max_net_leverage=0.1))
        out = risk.apply({"A": 0.5, "B": -0.5})
        assert out == pytest.approx({"A": 0.5, "B": -0.5})


class TestSectorCap:
    SECTORS = {"A": "TECH", "B": "TECH", "C": "ENERGY"}

    def test_scales_an_offending_sector(self):
        risk = RiskManager(RiskLimits(max_sector_weight=0.20), sectors=self.SECTORS)
        out = risk.apply({"A": 0.3, "B": 0.3, "C": 0.1}, timestamp=T0)

        assert out["A"] + out["B"] == pytest.approx(0.20)
        assert out["C"] == pytest.approx(0.1)

    def test_internally_hedged_sector_passes_a_net_only_limit(self):
        """
        Net exposure is blind to this book: 50% long and 50% short the same
        sector nets to zero. That is the whole reason `max_sector_gross_weight`
        exists — see the next test.
        """
        risk = RiskManager(RiskLimits(max_sector_weight=0.20), sectors=self.SECTORS)
        out = risk.apply({"A": 0.5, "B": -0.5})
        assert out == pytest.approx({"A": 0.5, "B": -0.5})

    def test_gross_limit_catches_the_hedged_sector_that_net_misses(self):
        """
        A book 50% long and 50% short one sector carries a large bet on
        within-sector dispersion while showing zero net exposure. A stat-arb
        book is exactly this shape.
        """
        risk = RiskManager(
            RiskLimits(max_sector_gross_weight=0.40), sectors=self.SECTORS
        )
        out = risk.apply({"A": 0.5, "B": -0.5}, timestamp=T0)

        assert sum(abs(w) for w in out.values()) == pytest.approx(0.40)
        assert risk.summary()["max_sector_gross_weight"] == 1

    def test_gross_scaling_preserves_relative_sizing(self):
        risk = RiskManager(
            RiskLimits(max_sector_gross_weight=0.30), sectors=self.SECTORS
        )
        out = risk.apply({"A": 0.6, "B": -0.2})
        assert out["A"] / out["B"] == pytest.approx(0.6 / -0.2)

    def test_when_both_caps_bind_the_tighter_one_wins(self):
        # Net 0.6 against a 0.30 cap wants scale 0.5; gross 0.8 against a 0.20
        # cap wants scale 0.25. The gross limit is tighter and must govern.
        risk = RiskManager(
            RiskLimits(max_sector_weight=0.30, max_sector_gross_weight=0.20),
            sectors=self.SECTORS,
        )
        out = risk.apply({"A": 0.7, "B": -0.1}, timestamp=T0)

        assert sum(abs(w) for w in out.values()) == pytest.approx(0.20)
        assert abs(sum(out.values())) <= 0.30 + 1e-9

    def test_compliant_sector_is_untouched_under_both_caps(self):
        risk = RiskManager(
            RiskLimits(max_sector_weight=0.50, max_sector_gross_weight=0.50),
            sectors=self.SECTORS,
        )
        out = risk.apply({"A": 0.2, "B": -0.1})
        assert out == pytest.approx({"A": 0.2, "B": -0.1})
        assert risk.violations == []

    def test_unmapped_symbols_pool_into_unknown_rather_than_escaping(self):
        risk = RiskManager(RiskLimits(max_sector_weight=0.20), sectors={})
        out = risk.apply({"A": 0.3, "B": 0.3}, timestamp=T0)
        assert sum(out.values()) == pytest.approx(0.20)
        assert risk.summary()["max_sector_weight"] == 1


class TestTurnover:
    def test_interpolates_toward_the_target(self):
        # current flat, target gross 1.0 -> turnover 1.0, capped at 0.25
        risk = RiskManager(RiskLimits(max_turnover=0.25))
        out = risk.apply({"A": 0.5, "B": -0.5}, current={}, timestamp=T0)

        assert out["A"] == pytest.approx(0.125)
        assert out["B"] == pytest.approx(-0.125)

    def test_turnover_of_the_result_equals_the_cap(self):
        risk = RiskManager(RiskLimits(max_turnover=0.25))
        current = {"A": 0.2, "B": -0.1}
        out = risk.apply({"A": 0.8, "B": -0.6}, current=current)

        realised = sum(abs(out[s] - current.get(s, 0.0)) for s in out)
        assert realised == pytest.approx(0.25)

    def test_small_rebalance_is_untouched(self):
        risk = RiskManager(RiskLimits(max_turnover=0.50))
        out = risk.apply({"A": 0.3}, current={"A": 0.25})
        assert out["A"] == pytest.approx(0.3)

    def test_exiting_positions_counts_toward_turnover(self):
        risk = RiskManager(RiskLimits(max_turnover=0.10))
        out = risk.apply({}, current={"A": 0.5}, timestamp=T0)
        # Held at 0.5, target 0, allowed to move 0.10 of the way there.
        assert out["A"] == pytest.approx(0.4)


class TestConstraintInteraction:
    def test_later_constraints_do_not_breach_earlier_ones(self):
        """
        The argument that makes a single pass sufficient: each constraint
        defines a convex set, and every step after the first is a pure
        reduction or an interpolation between two compliant books.
        """
        limits = RiskLimits(
            max_position_weight=0.10,
            max_gross_leverage=1.0,
            max_net_leverage=0.05,
            max_sector_weight=0.30,
        )
        sectors = {s: ("TECH" if i < 6 else "ENERGY") for i, s in enumerate("ABCDEFGHIJ")}
        risk = RiskManager(limits, sectors=sectors)

        targets = {"A": 0.4, "B": 0.3, "C": 0.3, "D": 0.2, "E": 0.2, "F": 0.2,
                   "G": -0.3, "H": -0.3, "I": -0.2, "J": -0.1}
        out = risk.apply(targets, timestamp=T0)

        assert max(abs(w) for w in out.values()) <= 0.10 + 1e-9
        assert sum(abs(w) for w in out.values()) <= 1.0 + 1e-9
        assert abs(sum(out.values())) <= 0.05 + 1e-9
        for sector in ("TECH", "ENERGY"):
            net = sum(w for s, w in out.items() if sectors[s] == sector)
            assert abs(net) <= 0.30 + 1e-9

    def test_no_limits_means_no_changes(self):
        risk = RiskManager()
        targets = {"A": 5.0, "B": -3.0}
        assert risk.apply(targets) == pytest.approx(targets)
        assert risk.violations == []

    def test_negative_limits_rejected(self):
        with pytest.raises(ValueError):
            RiskLimits(max_gross_leverage=-1.0)

    def test_reset_clears_the_record(self):
        risk = RiskManager(RiskLimits(max_position_weight=0.01))
        risk.apply({"A": 0.5}, timestamp=T0)
        risk.reset()
        assert risk.summary() == {}
