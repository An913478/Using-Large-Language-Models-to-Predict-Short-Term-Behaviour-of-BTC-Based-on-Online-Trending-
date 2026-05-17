"""Build phase 1+ feature dataset from BTC data and LLM market features.

This script reads a baseline Parquet dataset, adds derived return / volatility / trend
/ momentum / volume / LLM intensity features, constructs multi-horizon targets,
cleans the final table, and saves the result to a processed Parquet file.

The file layout is intentionally simple and reusable from the command line.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "processed" / "btc_final_features_with_llm.parquet"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "btc_features_phase1_plus.parquet"


def add_base_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add raw return and log-return features."""
    if "Return" not in df.columns:
        df["Return"] = df["Close"].pct_change()

    if "LogReturn" not in df.columns:
        df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling volatility and volatility ratio features."""
    df["Volatility_3d"] = df["Return"].rolling(3).std()
    df["Volatility_7d"] = df["Return"].rolling(7).std()
    df["Volatility_14d"] = df["Return"].rolling(14).std()
    df["Volatility_30d"] = df["Return"].rolling(30).std()

    df["Volatility_Ratio_7_30"] = df["Volatility_7d"] / (df["Volatility_30d"] + 1e-8)
    df["Volatility_Change_7d"] = df["Volatility_7d"].pct_change()

    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving average, trend ratio, and price-vs-MA features."""
    df["MA_3"] = df["Close"].rolling(3).mean()
    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_14"] = df["Close"].rolling(14).mean()
    df["MA_30"] = df["Close"].rolling(30).mean()

    df["Trend_Ratio_3_7"] = df["MA_3"] / (df["MA_7"] + 1e-8)
    df["Trend_Ratio_7_14"] = df["MA_7"] / (df["MA_14"] + 1e-8)
    df["Trend_Ratio_14_30"] = df["MA_14"] / (df["MA_30"] + 1e-8)

    df["Price_vs_MA7"] = df["Close"] / (df["MA_7"] + 1e-8)
    df["Price_vs_MA14"] = df["Close"] / (df["MA_14"] + 1e-8)
    df["Price_vs_MA30"] = df["Close"] / (df["MA_30"] + 1e-8)

    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum and rolling return summary features."""
    df["Momentum_1d"] = df["Close"] - df["Close"].shift(1)
    df["Momentum_3d"] = df["Close"] - df["Close"].shift(3)
    df["Momentum_7d"] = df["Close"] - df["Close"].shift(7)
    df["Momentum_14d"] = df["Close"] - df["Close"].shift(14)

    df["Return_Sum_3d"] = df["Return"].rolling(3).sum()
    df["Return_Sum_7d"] = df["Return"].rolling(7).sum()
    df["Return_Sum_14d"] = df["Return"].rolling(14).sum()

    df["Return_Mean_3d"] = df["Return"].rolling(3).mean()
    df["Return_Mean_7d"] = df["Return"].rolling(7).mean()
    df["Return_Mean_14d"] = df["Return"].rolling(14).mean()

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-derived features when volume exists."""
    if "Volume" not in df.columns:
        return df

    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_MA_7"] = df["Volume"].rolling(7).mean()
    df["Volume_MA_14"] = df["Volume"].rolling(14).mean()
    df["Volume_Ratio_7_14"] = df["Volume_MA_7"] / (df["Volume_MA_14"] + 1e-8)
    df["Volume_vs_MA7"] = df["Volume"] / (df["Volume_MA_7"] + 1e-8)

    return df


def add_llm_intensity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived news intensity and sentiment trend features."""
    possible_cols = [
        "news_count",
        "article_count",
        "btc_news_count",
        "Marketaux_News_Count",
        "LLM_News_Count",
    ]

    found = [c for c in possible_cols if c in df.columns]
    if found:
        base_col = found[0]
        df["News_Count_7d"] = df[base_col].rolling(7).sum()
        df["News_Count_14d"] = df[base_col].rolling(14).sum()
        df["News_Count_Change"] = df[base_col].pct_change()

    sentiment_cols = [
        "avg_sentiment",
        "btc_entity_sentiment",
        "sentiment_score",
        "LLM_Sentiment",
    ]
    found_sent = [c for c in sentiment_cols if c in df.columns]
    if found_sent:
        sentiment_col = found_sent[0]
        df["Sentiment_MA_3"] = df[sentiment_col].rolling(3).mean()
        df["Sentiment_MA_7"] = df[sentiment_col].rolling(7).mean()
        df["Sentiment_Change"] = df[sentiment_col].diff()

    return df


def add_multi_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create multi-horizon target and direction labels."""
    for horizon in [1, 3, 7]:
        df[f"Target_Close_{horizon}d"] = df["Close"].shift(-horizon)
        df[f"Target_Return_{horizon}d"] = (df["Close"].shift(-horizon) / df["Close"]) - 1.0
        df[f"Target_Direction_{horizon}d"] = (df[f"Target_Return_{horizon}d"] > 0).astype(int)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset, fill gaps, and drop rows missing target values."""
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    text_cols = [c for c in ["dominant_event_type", "daily_summary", "rationale"] if c in df.columns]
    for c in text_cols:
        df[c] = df[c].fillna("")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    target_cols = [c for c in df.columns if c.startswith("Target_")]
    df = df.dropna(subset=target_cols).reset_index(drop=True)

    return df


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="feature_engineering_phase1_plus",
        description="Generate phase 1+ BTC features from the final LLM-enhanced dataset.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help="Input Parquet dataset path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Output Parquet path for the phase 1+ dataset.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Main entry point for the feature engineering script."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    input_path = args.input
    output_path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path).copy()
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("Input dataset must contain at least 'Date' and 'Close' columns.")

    df["Date"] = pd.to_datetime(df["Date"])

    df = add_base_returns(df)
    df = add_volatility_features(df)
    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_volume_features(df)
    df = add_llm_intensity_features(df)
    df = add_multi_horizon_targets(df)
    df = clean_dataset(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logging.info("Saved Phase 1+ dataset to: %s", output_path)
    logging.info("Rows: %d", len(df))
    logging.info("Columns: %d", len(df.columns))


if __name__ == "__main__":
    main()
