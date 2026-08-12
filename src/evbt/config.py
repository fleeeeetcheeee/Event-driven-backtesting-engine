"""
Single source of every path and engine-wide default.

Same rule as Project 01: no data path is hardcoded anywhere else in the
codebase. Derived paths are computed in `__post_init__`.

Model *parameters* (commission rates, impact coefficients, risk limits) do not
live here — they live on the model objects themselves in `execution.costs` and
`portfolio.risk`, because they are part of the experiment being run and must be
recorded alongside its results. A backtest whose cost assumptions come from a
global mutable singleton is not reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class Config:
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    data_root: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_ROOT", str(_PROJECT_ROOT / "data")))
    )

    # Project 01's processed store, if it is available on this machine. The two
    # projects are separate repos and do not import from each other; this
    # engine reads Project 01's Parquet output as a plain external dataset,
    # exactly as it would read any vendor file.
    pit_store_root: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "PIT_STORE_ROOT",
                str(
                    _PROJECT_ROOT.parent
                    / "Project01-PointInTimeEquityDataPipeline"
                    / "data"
                ),
            )
        )
    )

    # --- Derived paths (set in __post_init__) ---
    raw_dir: Path = field(default=None)
    processed_dir: Path = field(default=None)
    results_dir: Path = field(default=None)
    fama_french_dir: Path = field(default=None)

    # --- Reference-data sources ---
    # Ken French's data library. Used by the done-criterion replication: the
    # engine has to reproduce a published return series, and this is the
    # canonical published series for equity factors.
    french_library_url: str = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    )

    # --- Simulation defaults ---
    # Trading days per year. Used only to annualise metrics; stated explicitly
    # because Sharpe ratios are silently incomparable across codebases that
    # disagree on it (252 vs 250 vs 260 moves a Sharpe by ~1%).
    trading_days_per_year: int = 252

    def __post_init__(self) -> None:
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"
        self.results_dir = self.project_root / "results"
        self.fama_french_dir = self.data_root / "raw" / "fama_french"


config = Config()
