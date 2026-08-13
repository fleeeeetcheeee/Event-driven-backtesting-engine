"""
Download the Ken French data library files the HML replication needs.

Two files:

  6_Portfolios_2x3_CSV.zip            the six size x book-to-market portfolios
  F-F_Research_Data_Factors_CSV.zip   the published Mkt-RF, SMB, HML, RF factors

The six portfolios are the *inputs* French builds HML from; the factor file is
the published answer. Having both is what makes the replication a real test
rather than a restatement — one is fed to the engine, the other is what the
engine's output has to match.

Files land in `data/raw/fama_french/` verbatim, unmodified, exactly as Project
01's ingestion layer does it. Parsing happens in `evbt.data.french`, not here.

Usage
-----
    python scripts/fetch_french_data.py
    python scripts/fetch_french_data.py --force    # re-download even if present
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evbt.config import config  # noqa: E402

FILES = {
    "6_Portfolios_2x3_CSV.zip": "six size x book-to-market portfolios",
    "F-F_Research_Data_Factors_CSV.zip": "published Fama-French factors",
}

# A descriptive User-Agent is courtesy rather than a hard requirement here, but
# identifying an automated client is the right default for any academic host.
USER_AGENT = "evbt-research (https://github.com/fleeeeetcheeee)"


def download(filename: str, destination: Path, *, force: bool = False) -> Path:
    target = destination / filename
    if target.exists() and not force:
        print(f"  {filename:<40} already present ({target.stat().st_size:,} bytes)")
        return target

    url = config.french_library_url + filename
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    response.raise_for_status()

    # Atomic write, same rule as Project 01's storage layer: a partial download
    # must never be visible as a complete file to whatever reads it next.
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(response.content)
    tmp.rename(target)

    print(f"  {filename:<40} downloaded ({len(response.content):,} bytes)")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    destination = config.fama_french_dir
    destination.mkdir(parents=True, exist_ok=True)

    print(f"Ken French data library -> {destination}")
    for filename, description in FILES.items():
        print(f"\n{description}")
        download(filename, destination, force=args.force)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
