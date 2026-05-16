"""Merge BTC market features with Gemini-extracted LLM numeric and text features.

Joins base market data with Gemini LLM features, normalizes numeric columns,
assigns provider metadata, and writes the merged dataset to parquet.
"""

import argparse
import logging
import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_FEATURES_FILE = os.path.join(BASE_DIR, "data", "processed", "btc_features_phase1_plus.parquet")
LLM_FILE = os.path.join(BASE_DIR, "data", "processed", "btc_marketaux_llm_features_gemini.parquet")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "btc_final_features_with_llm_gemini.parquet")

# File-only operation: safe for smoke-tests
SMOKE_TEST_SAFE = True


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


def main(input_file: str = BASE_FEATURES_FILE, llm_file: str = LLM_FILE, output_file: str = OUTPUT_FILE, verbose: bool = False) -> None:
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Base features file not found: {input_file}")

    if not os.path.exists(llm_file):
        raise FileNotFoundError(f"LLM feature file not found: {llm_file}")

    base_df = pd.read_parquet(input_file).copy()
    llm_df = pd.read_parquet(llm_file).copy()

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

    merged_df["llm_provider"] = "gemini"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_df.to_parquet(output_file, index=False)

    logger = logging.getLogger(__name__)
    logger.info("Saved merged dataset: %s", output_file)
    logger.info("Rows: %d", len(merged_df))
    logger.info("Added columns: %s", LLM_NUMERIC_COLS + LLM_TEXT_COLS + ["llm_provider"])
    logger.info("Head:\n%s", merged_df[["Date"] + LLM_NUMERIC_COLS[:3] + ["llm_event_type", "llm_provider"]].head())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge market + Gemini LLM features and write final parquet.")
    parser.add_argument("--base", type=str, default=BASE_FEATURES_FILE, help="Base features parquet file")
    parser.add_argument("--llm", type=str, default=LLM_FILE, help="Gemini LLM features parquet file")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output parquet file path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(input_file=args.base, llm_file=args.llm, output_file=args.output, verbose=args.verbose)