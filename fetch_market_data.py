"""Fetch daily BTC market data using yfinance and save to Parquet.

Downloads daily OHLCV data for BTC-USD and writes to data/raw/btc_market_data.parquet
by default. Supports CLI argument overrides for output path. Can be imported and
invoked from CI/smoke-test harnesses.

Usage
-----
Fetch the default BTC time series:

    python scripts/fetch_market_data.py

Notes
-----
- Requires `yfinance` installed in the environment.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

# Default config
OUTPUT_DIR = Path("data/raw")
ASSET = "BTC-USD"
START_DATE = "2022-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Networked operation — do not auto-run in smoke-tests unless explicitly allowed
SMOKE_TEST_SAFE = False


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters
    ----------
    argv:
        List of command-line arguments for testing; defaults to sys.argv.
    """
    parser = argparse.ArgumentParser(
        prog="fetch_market_data",
        description="Fetch BTC market data via yfinance and save as Parquet",
    )
    parser.add_argument("--asset", default=ASSET, help="Ticker symbol to fetch, e.g. BTC-USD")
    parser.add_argument("--start", default=START_DATE, help="Start date for history (YYYY-MM-DD)")
    parser.add_argument("--end", default=END_DATE, help="End date for history (YYYY-MM-DD)")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "btc_market_data.parquet",
        help="Output Parquet file path",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def fetch_and_save(asset: str, start: str, end: str, out_path: Path) -> None:
    """Fetch asset history and persist it as Parquet.

    Parameters
    ----------
    asset:
        Ticker symbol to download from yfinance.
    start:
        Start date for the historical window.
    end:
        End date for the historical window.
    out_path:
        Destination path for the Parquet file.
    """
    logger.info("Fetching %s from %s to %s", asset, start, end)
    df = yf.download(asset, start=start, end=end, interval="1d", progress=False)

    if df.empty:
        raise RuntimeError("No data retrieved from yfinance.")

    df = df.reset_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    logger.info("Saved market data to %s", out_path)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for CLI invocation."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        fetch_and_save(args.asset, args.start, args.end, args.output)
    except Exception as exc:  # pragma: no cover - surface errors at top level
        logger.exception("Failed to fetch market data: %s", exc)
        raise


if __name__ == "__main__":
    main()