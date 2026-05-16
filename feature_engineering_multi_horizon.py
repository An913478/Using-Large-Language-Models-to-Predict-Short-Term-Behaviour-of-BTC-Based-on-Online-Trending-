"""Build multi-horizon targets and volatility features for model training.

Loads market and LLM base features, creates multi-horizon targets (1d, 3d, 7d)
for returns and direction, adds volatility and rolling average features, and
cleans dataset. File-based aggregation, safe for smoke tests.
"""

import argparse
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

# File-only operation: safe for smoke-tests
SMOKE_TEST_SAFE = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "btc_final_features_with_llm.parquet")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "btc_features_multi_horizon.parquet")


def add_multi_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    horizons = [1, 3, 7]

    for h in horizons:
        df[f"Target_Close_{h}d"] = df["Close"].shift(-h)
        df[f"Target_Return_{h}d"] = (df["Close"].shift(-h) / df["Close"]) - 1.0
        df[f"Target_Direction_{h}d"] = (df[f"Target_Return_{h}d"] > 0).astype(int)

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Return" not in df.columns:
        df["Return"] = df["Close"].pct_change()

    df["Volatility_7d"] = df["Return"].rolling(7).std()
    df["Volatility_14d"] = df["Return"].rolling(14).std()

    # Optional extras that often help
    df["Return_Mean_3d"] = df["Return"].rolling(3).mean()
    df["Return_Mean_7d"] = df["Return"].rolling(7).mean()
    df["Volume_MA_7"] = df["Volume"].rolling(7).mean() if "Volume" in df.columns else np.nan
    df["Volume_MA_14"] = df["Volume"].rolling(14).mean() if "Volume" in df.columns else np.nan

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)

    # replace infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    # keep text columns as text
    text_cols = [c for c in ["dominant_event_type", "daily_summary", "rationale"] if c in df.columns]
    for c in text_cols:
        df[c] = df[c].fillna("")

    # numeric columns only: forward/back fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # drop final rows with missing future targets
    target_cols = [c for c in df.columns if c.startswith("Target_")]
    df = df.dropna(subset=target_cols).reset_index(drop=True)

    return df


def main(input_file: str = INPUT_FILE, output_file: str = OUTPUT_FILE) -> None:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_parquet(input_file).copy()

    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in input dataset.")
    if "Close" not in df.columns:
        raise ValueError("Expected a 'Close' column in input dataset.")

    df["Date"] = pd.to_datetime(df["Date"])

    # base return if missing
    if "Return" not in df.columns:
        df["Return"] = df["Close"].pct_change()

    df = add_volatility_features(df)
    df = add_multi_horizon_targets(df)
    df = clean_dataset(df)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_parquet(output_file, index=False)

    logger.info(f"Saved multi-horizon dataset to: {output_file}")
    logger.info(f"Rows: {len(df)}")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info("Created target columns:")
    for c in [col for col in df.columns if col.startswith("Target_")]:
        logger.info(f"  - {c}")

    logger.info("Added volatility/context columns:")
    for c in ["Volatility_7d", "Volatility_14d", "Return_Mean_3d", "Return_Mean_7d", "Volume_MA_7", "Volume_MA_14"]:
        if c in df.columns:
            logger.info(f"  - {c}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-horizon targets and volatility features.")
    p.add_argument("--input", default=INPUT_FILE, help="Input features parquet file")
    p.add_argument("--output", default=OUTPUT_FILE, help="Output features parquet file")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.output)