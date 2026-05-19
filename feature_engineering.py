"""Create base feature engineering for BTC market + trends.

This module computes standard price and attention-derived features used by
the downstream modelling pipeline. The behaviour mirrors the original
feature generation but is wrapped with a small CLI and docstrings so it
can be included in CI and documented on GitHub.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

INPUT_FILE = Path("data/processed/btc_market_trends_merged.parquet")
OUTPUT_FILE = Path("data/processed/btc_features.parquet")

# File-only transformation: safe for smoke-tests
SMOKE_TEST_SAFE = True


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute core BTC market, momentum, and trend features for modelling."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Core price features
    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility_7d"] = df["Return"].rolling(window=7).std()
    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["MA_14"] = df["Close"].rolling(window=14).mean()
    df["Volume_Change"] = df["Volume"].pct_change()

    # Momentum features
    df["Momentum_3d"] = df["Close"] - df["Close"].shift(3)
    df["Momentum_7d"] = df["Close"] - df["Close"].shift(7)

    # Google Trends feature
    df["Trends_Change"] = df["GoogleTrends"].pct_change()

    # Targets
    df["Target_Close_NextDay"] = df["Close"].shift(-1)
    df["Target_Return_NextDay"] = df["Return"].shift(-1)
    df["Target_Direction"] = (df["Target_Close_NextDay"] > df["Close"]).astype(int)

    # Drop NaNs caused by rolling windows and shifts
    df = df.dropna().reset_index(drop=True)
    return df


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for BTC feature generation."""
    parser = argparse.ArgumentParser(
        prog="feature_engineering",
        description="Compute BTC market feature engineering and save processed data.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help="Input merged market/trends parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Output feature-engineered parquet file",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Load raw market/trends data, generate features, and write processed parquet."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_parquet(args.input)
    out_df = build_features(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)

    logger.info("Saved engineered feature dataset to: %s", args.output)
    logger.info("Rows: %d", len(out_df))
    logger.info("Columns: %d", len(out_df.columns))
    logger.info("Columns list: %s", out_df.columns.tolist())


if __name__ == "__main__":
    main()