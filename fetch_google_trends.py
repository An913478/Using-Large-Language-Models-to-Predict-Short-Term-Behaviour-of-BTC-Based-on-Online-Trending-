"""Fetch Google Trends time series for Bitcoin and save to Parquet.

Queries Google Trends via pytrends library for Bitcoin keyword data and persists
results to data/raw/btc_google_trends.parquet by default. Supports CLI options
for custom keywords, timeframes, and output paths.

The script is intentionally conservative and exposes a few options so it
can be used in automated pipelines and unit-tested.

Usage examples
--------------
Fetch default Bitcoin series:

    python scripts/fetch_google_trends.py

Fetch a custom keyword and timeframe:

    python scripts/fetch_google_trends.py --keyword ethereum --timeframe "2020-01-01 2024-12-31" --output data/raw/eth_trends.parquet

Notes
-----
- Requires `pytrends` to be installed.
- The default timeframe is an inclusive date range string accepted by
  `pytrends`.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from pytrends.request import TrendReq

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters
    ----------
    argv:
        List of command-line arguments (for testing); defaults to sys.argv.
    """

    parser = argparse.ArgumentParser(
        prog="fetch_google_trends",
        description="Fetch Google Trends time series and save to Parquet",
    )
    parser.add_argument("--keyword", type=str, default="Bitcoin", help="Search keyword")
    parser.add_argument(
        "--timeframe",
        type=str,
        default="2022-01-01 2024-12-31",
        help="Timeframe string accepted by pytrends (e.g. 'YYYY-MM-DD YYYY-MM-DD')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/btc_google_trends.parquet"),
        help="Output Parquet file path",
    )
    parser.add_argument("--tz", type=int, default=0, help="Timezone offset for pytrends")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args(argv)


def fetch_and_save(keyword: str, timeframe: str, out_path: Path, tz: int = 0) -> None:
    """Fetch Google Trends data and persist it as a Parquet file.

    Parameters
    ----------
    keyword:
        Search keyword to query.
    timeframe:
        Timeframe string in the format expected by pytrends.
    out_path:
        Destination path for the Parquet file. Parent directory will be created
        if it does not exist.
    tz:
        Timezone offset passed to pytrends.TrendReq.

    Raises
    ------
    RuntimeError
        If no data is returned by Google Trends for the provided keyword/timeframe.
    """

    pytrends = TrendReq(hl="en-US", tz=tz)
    pytrends.build_payload([keyword], timeframe=timeframe)

    df = pytrends.interest_over_time()
    if df.empty:
        raise RuntimeError("No Google Trends data returned for keyword={keyword} timeframe={timeframe}.")

    # Drop the 'isPartial' column when present (pytrends artifact)
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)

    logger.info("Saved Google Trends data to %s", out_path)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for CLI invocation."""

    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        fetch_and_save(args.keyword, args.timeframe, args.output, tz=args.tz)
    except Exception as exc:  # pragma: no cover - surface errors at top level
        logger.exception("Failed to fetch Google Trends: %s", exc)
        raise


if __name__ == "__main__":
    main()