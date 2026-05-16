"""Expand contextual features by fetching multiple Google Trends keywords.

This script fetches additional attention keywords, merges them with the
market dataset and computes aggregate attention indices and derived
features. Network usage means it is not safe for blind smoke-test runs.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from pytrends.request import TrendReq

logger = logging.getLogger(__name__)

INPUT_FILE = Path("data/processed/btc_market_trends_merged.parquet")
OUTPUT_FILE = Path("data/processed/btc_features_context_expanded.parquet")

KEYWORDS = [
    "Bitcoin",
    "BTC price",
    "Bitcoin crash",
    "Bitcoin rally",
    "crypto market",
    "crypto news",
    "Ethereum",
]

TIMEFRAME = "2022-01-01 2024-12-31"

# Networked operation — not safe for automatic smoke-test runs
SMOKE_TEST_SAFE = False


def fetch_trends_batch(keywords, timeframe=TIMEFRAME):
    pytrends = TrendReq(hl="en-US", tz=0)
    pytrends.build_payload(keywords, timeframe=timeframe)
    df = pytrends.interest_over_time()

    if df.empty:
        raise RuntimeError(f"No Google Trends data returned for batch: {keywords}")

    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    return df


def fetch_all_trends(keywords):
    """Fetch keywords in small batches to reduce throttling errors."""
    all_dfs = []
    batch_size = 3

    for i in range(0, len(keywords), batch_size):
        batch = keywords[i : i + batch_size]
        logger.info("Fetching batch: %s", batch)

        batch_df = fetch_trends_batch(batch)
        batch_df = batch_df.reset_index()
        batch_df["date"] = pd.to_datetime(batch_df["date"])

        all_dfs.append(batch_df)

        # Pause slightly to reduce throttling risk
        time.sleep(2)

    merged = all_dfs[0]
    for df in all_dfs[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")

    # If duplicate dates appear, average them
    merged = merged.groupby("date", as_index=False).mean(numeric_only=True)
    merged = merged.sort_values("date").reset_index(drop=True)

    return merged


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for expanded context feature expansion."""
    parser = argparse.ArgumentParser(
        prog="context_feature_expansion",
        description="Fetch expanded context Google Trends and merge with market data.",
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
        help="Output expanded context features parquet file",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for CLI invocation."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load BTC market + base trends dataset
    df = pd.read_parquet(args.input).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Fetch expanded trends data
    trends_df = fetch_all_trends(KEYWORDS)
    trends_df = trends_df.rename(columns={"date": "Date"})

    # Merge onto daily BTC market data
    merged = pd.merge(df, trends_df, on="Date", how="left")

    # Forward-fill each keyword separately after merge
    for kw in KEYWORDS:
        if kw in merged.columns:
            merged[kw] = merged[kw].ffill()

    # Keep rows where at least the main keyword exists
    merged = merged.dropna(subset=["Bitcoin"]).reset_index(drop=True)

    # =========================
    # Expanded contextual features
    # =========================
    merged["AttentionIndex_Mean"] = merged[KEYWORDS].mean(axis=1)
    merged["AttentionIndex_Max"] = merged[KEYWORDS].max(axis=1)
    merged["AttentionIndex_Std"] = merged[KEYWORDS].std(axis=1)

    # Forward-fill aggregate contextual features too
    merged["AttentionIndex_Mean"] = merged["AttentionIndex_Mean"].ffill()
    merged["AttentionIndex_Max"] = merged["AttentionIndex_Max"].ffill()
    merged["AttentionIndex_Std"] = merged["AttentionIndex_Std"].ffill()

    merged["AttentionIndex_Change"] = merged["AttentionIndex_Mean"].pct_change()
    merged["AttentionIndex_MA_7"] = merged["AttentionIndex_Mean"].rolling(7).mean()
    merged["AttentionIndex_MA_14"] = merged["AttentionIndex_Mean"].rolling(14).mean()

    merged["AttentionVolatility_7d"] = merged["AttentionIndex_Mean"].rolling(7).std()
    merged["AttentionVolatility_14d"] = merged["AttentionIndex_Mean"].rolling(14).std()

    merged["AttentionSpike"] = (
        merged["AttentionIndex_Mean"]
        > merged["AttentionIndex_MA_7"] + merged["AttentionVolatility_7d"]
    ).astype(int)

    for kw in KEYWORDS:
        safe_name = kw.replace(" ", "_")
        merged[f"{safe_name}_Change"] = merged[kw].pct_change()

    # =========================
    # Rebuild market features
    # =========================
    merged["Return"] = merged["Close"].pct_change()
    merged["LogReturn"] = np.log(merged["Close"] / merged["Close"].shift(1))
    merged["Volatility_7d"] = merged["Return"].rolling(window=7).std()
    merged["MA_7"] = merged["Close"].rolling(window=7).mean()
    merged["MA_14"] = merged["Close"].rolling(window=14).mean()
    merged["Volume_Change"] = merged["Volume"].pct_change()
    merged["Momentum_3d"] = merged["Close"] - merged["Close"].shift(3)
    merged["Momentum_7d"] = merged["Close"] - merged["Close"].shift(7)

    # Targets
    merged["Target_Close_NextDay"] = merged["Close"].shift(-1)
    merged["Target_Return_NextDay"] = merged["Return"].shift(-1)
    merged["Target_Direction"] = (
        merged["Target_Close_NextDay"] > merged["Close"]
    ).astype(int)

    # Instead of dropping everything, only drop rows where core derived features
    # or targets are unavailable
    merged = merged.dropna(
        subset=[
            "Return",
            "LogReturn",
            "Volatility_7d",
            "MA_7",
            "MA_14",
            "Volume_Change",
            "Momentum_3d",
            "Momentum_7d",
            "AttentionIndex_Change",
            "AttentionIndex_MA_7",
            "AttentionIndex_MA_14",
            "AttentionVolatility_7d",
            "AttentionVolatility_14d",
            "Target_Close_NextDay",
            "Target_Return_NextDay",
        ]
    ).reset_index(drop=True)

    merged.to_parquet(args.output, index=False)

    logger.info("Saved expanded contextual dataset to: %s", args.output)
    logger.info("Rows: %d", len(merged))
    logger.info("Columns: %d", len(merged.columns))
    logger.info("Columns: %s", merged.columns.tolist())
    logger.info("Head:\n%s", merged.head())


if __name__ == "__main__":
    main()