"""Merge BTC market features with OpenAI-extracted LLM features.

Loads base market data and OpenAI LLM features, performs data cleaning and
merging, assigns provider metadata, and writes final dataset to parquet.
File-only operation, safe for smoke tests.
"""

import argparse
import logging
import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_FEATURES_FILE = os.path.join(BASE_DIR, "data", "processed", "btc_features_phase1_plus.parquet")
LLM_FILE = os.path.join(BASE_DIR, "data", "processed", "btc_marketaux_llm_features_openai.parquet")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "btc_final_features_with_llm_openai.parquet")

# File-only operation: safe for smoke-tests
SMOKE_TEST_SAFE = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


LLM_NUMERIC_COLS = [
    "llm_sentiment_score",
    "llm_market_impact_score",
    "llm_uncertainty_score",
    "llm_adoption_score",
    "llm_regulation_score",
    "llm_macro_score",
    "llm_technical_score",
]

LLM_TEXT_COLS = [
    "llm_event_type",
    "llm_summary",
    "llm_rationale",
]


def main() -> None:
    if not os.path.exists(BASE_FEATURES_FILE):
        raise FileNotFoundError(f"Base features file not found: {BASE_FEATURES_FILE}")

    if not os.path.exists(LLM_FILE):
        raise FileNotFoundError(f"LLM feature file not found: {LLM_FILE}")

    base_df = pd.read_parquet(BASE_FEATURES_FILE).copy()
    llm_df = pd.read_parquet(LLM_FILE).copy()

    if "Date" not in base_df.columns:
        raise ValueError("Base dataset must contain 'Date' column.")
    if "Date" not in llm_df.columns:
        raise ValueError("LLM dataset must contain 'Date' column.")

    base_df["Date"] = pd.to_datetime(base_df["Date"]).dt.normalize()
    llm_df["Date"] = pd.to_datetime(llm_df["Date"]).dt.normalize()

    keep_cols = ["Date"] + [c for c in LLM_NUMERIC_COLS + LLM_TEXT_COLS if c in llm_df.columns]
    llm_df = llm_df[keep_cols].copy()

    for col in LLM_NUMERIC_COLS:
        if col not in llm_df.columns:
            llm_df[col] = 0.0

    for col in LLM_TEXT_COLS:
        if col not in llm_df.columns:
            llm_df[col] = ""

    llm_df = llm_df.groupby("Date", as_index=False).first()

    merged_df = base_df.merge(llm_df, on="Date", how="left")

    for col in LLM_NUMERIC_COLS:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").fillna(0.0)

    for col in LLM_TEXT_COLS:
        merged_df[col] = merged_df[col].fillna("").astype(str)

    merged_df["llm_provider"] = "openai"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    merged_df.to_parquet(OUTPUT_FILE, index=False)

    logger.info("Saved merged dataset: %s", OUTPUT_FILE)
    logger.info("Rows: %d", len(merged_df))
    logger.info("Added columns: %s", LLM_NUMERIC_COLS + LLM_TEXT_COLS + ["llm_provider"])
    logger.info("Head:\n%s", merged_df[["Date"] + LLM_NUMERIC_COLS[:3] + ["llm_event_type", "llm_provider"]].head())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge market + OpenAI LLM features and write final parquet.")
    p.add_argument("--base", default=BASE_FEATURES_FILE, help="Base features parquet file")
    p.add_argument("--llm", default=LLM_FILE, help="OpenAI LLM features parquet file")
    p.add_argument("--output", default=OUTPUT_FILE, help="Output parquet file path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    BASE_FEATURES_FILE = args.base
    LLM_FILE = args.llm
    OUTPUT_FILE = args.output
    main()