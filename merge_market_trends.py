"""Merge BTC market data with Google Trends into a single processed table.

Flattens MultiIndex columns produced by yfinance, standardizes dates, and merges
weekly Google Trends data to daily market dataframe. Writes output to
data/processed/btc_market_trends_merged.parquet. File-local operation (no network).
"""

from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

MARKET_FILE = RAW_DIR / "btc_market_data.parquet"
TRENDS_FILE = RAW_DIR / "btc_google_trends.parquet"
OUTPUT_FILE = PROCESSED_DIR / "btc_market_trends_merged.parquet"

# Safe for smoke tests
SMOKE_TEST_SAFE = True


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance-style columns whether MultiIndex or tuple-like strings."""

    new_cols = []

    for col in df.columns:
        # Case 1: real tuple / MultiIndex entry
        if isinstance(col, tuple):
            new_cols.append(col[0])

        # Case 2: string representation of tuple
        elif isinstance(col, str) and col.startswith("(") and col.endswith(")"):
            try:
                parsed = ast.literal_eval(col)
                if isinstance(parsed, tuple):
                    new_cols.append(parsed[0])
                else:
                    new_cols.append(col)
            except Exception:
                new_cols.append(col)

        else:
            new_cols.append(col)

    df.columns = new_cols
    return df


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for market/trends merging."""
    parser = argparse.ArgumentParser(
        prog="merge_market_trends",
        description="Merge BTC market data with Google Trends into a single processed table.",
    )
    parser.add_argument(
        "--market-file",
        type=Path,
        default=MARKET_FILE,
        help="Input BTC market parquet file",
    )
    parser.add_argument(
        "--trends-file",
        type=Path,
        default=TRENDS_FILE,
        help="Input Google Trends parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Output merged parquet file",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(
    market_file: Path = MARKET_FILE,
    trends_file: Path = TRENDS_FILE,
    output_file: Path = OUTPUT_FILE,
    verbose: bool = False,
) -> None:
    """Merge market and Google Trends data into a single processed table."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load market data
    market_df = pd.read_parquet(market_file)
    market_df = flatten_columns(market_df)

    logger.info("Market columns after flattening: %s", market_df.columns.tolist())

    # Standardise date column
    if "Date" not in market_df.columns:
        raise KeyError(f"'Date' column not found in market data. Columns: {market_df.columns.tolist()}")

    market_df["Date"] = pd.to_datetime(market_df["Date"])
    market_df = market_df.sort_values("Date")

    # Keep useful columns only
    keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    market_df = market_df[keep_cols]

    # Load Google Trends
    trends_df = pd.read_parquet(trends_file).reset_index()

    # Rename columns safely
    if len(trends_df.columns) != 2:
        raise ValueError(f"Unexpected Google Trends columns: {trends_df.columns.tolist()}")

    trends_df.columns = ["Date", "GoogleTrends"]
    trends_df["Date"] = pd.to_datetime(trends_df["Date"])
    trends_df = trends_df.sort_values("Date")

    # Merge on Date
    merged_df = pd.merge(market_df, trends_df, on="Date", how="left")

    # Forward-fill weekly trends to daily
    merged_df["GoogleTrends"] = merged_df["GoogleTrends"].ffill()

    # Drop any leading NaNs if present
    merged_df = merged_df.dropna(subset=["GoogleTrends"]).reset_index(drop=True)

    # Save output
    merged_df.to_parquet(output_file, index=False)

    logger.info("Saved merged dataset to: %s", output_file)
    logger.info("Columns: %s", merged_df.columns.tolist())
    logger.info("Rows: %d", len(merged_df))


def cli_main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    main(
        market_file=args.market_file,
        trends_file=args.trends_file,
        output_file=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    cli_main()