"""Merge market/context features with LLM-derived features.

Loads existing market/context feature parquet and LLM-derived feature
parquet, performs cleaning and event-dummy creation, and writes the
final merged parquet. Exposes a small CLI to override file paths for
use in pipelines and marks the module as smoke-test safe (file-only).
"""

from pathlib import Path
import argparse
import logging
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MARKET_FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "btc_features_context_expanded.parquet"
LLM_FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "btc_marketaux_llm_features.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "btc_final_features_with_llm.parquet"

# file-only operation: safe for smoke-tests
SMOKE_TEST_SAFE = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def add_event_dummies(df: pd.DataFrame) -> pd.DataFrame:
    event_types = [
        "macro", "regulation", "etf", "exchange", "security",
        "adoption", "mining", "technical", "other"
    ]

    df["dominant_event_type"] = (
        df["dominant_event_type"]
        .fillna("other")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    for event in event_types:
        df[f"Event_{event}"] = (df["dominant_event_type"] == event).astype(int)

    return df


def main():
    if not MARKET_FEATURES_FILE.exists():
        raise FileNotFoundError(f"Missing market/context feature file: {MARKET_FEATURES_FILE}")

    if not LLM_FEATURES_FILE.exists():
        raise FileNotFoundError(f"Missing LLM feature file: {LLM_FEATURES_FILE}")

    market_df = pd.read_parquet(MARKET_FEATURES_FILE).copy()
    llm_df = pd.read_parquet(LLM_FEATURES_FILE).copy()

    market_df["Date"] = pd.to_datetime(market_df["Date"])
    llm_df["Date"] = pd.to_datetime(llm_df["Date"])

    llm_keep_cols = [
        "Date",
        "ArticleCount",
        "UniqueSourceCount",
        "MeanRelevanceScore",
        "MeanBTCEntitySentiment",
        "sentiment_score",
        "bullish_score",
        "bearish_score",
        "uncertainty_score",
        "market_impact_score",
        "dominant_event_type",
        "mentions_etf",
        "mentions_regulation",
        "mentions_exchange",
        "mentions_hack_or_security",
        "mentions_institutional_adoption",
        "daily_summary",
        "rationale",
    ]

    missing_cols = [col for col in llm_keep_cols if col not in llm_df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected LLM columns: {missing_cols}")

    llm_df = llm_df[llm_keep_cols].copy()

    numeric_cols = [
        "ArticleCount",
        "UniqueSourceCount",
        "MeanRelevanceScore",
        "MeanBTCEntitySentiment",
        "sentiment_score",
        "bullish_score",
        "bearish_score",
        "uncertainty_score",
        "market_impact_score",
        "mentions_etf",
        "mentions_regulation",
        "mentions_exchange",
        "mentions_hack_or_security",
        "mentions_institutional_adoption",
    ]

    text_cols = [
        "dominant_event_type",
        "daily_summary",
        "rationale",
    ]

    # Clean LLM dataframe before merge
    for col in numeric_cols:
        llm_df[col] = pd.to_numeric(llm_df[col], errors="coerce")

    for col in text_cols:
        llm_df[col] = llm_df[col].fillna("").astype(str)

    merged_df = pd.merge(market_df, llm_df, on="Date", how="left")

    # Fill numeric columns after merge
    for col in numeric_cols:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").fillna(0.0)

    # Fill text columns after merge
    if "dominant_event_type" in merged_df.columns:
        merged_df["dominant_event_type"] = (
            merged_df["dominant_event_type"]
            .fillna("other")
            .astype(str)
        )

    if "daily_summary" in merged_df.columns:
        merged_df["daily_summary"] = merged_df["daily_summary"].fillna("").astype(str)

    if "rationale" in merged_df.columns:
        merged_df["rationale"] = merged_df["rationale"].fillna("").astype(str)

    merged_df = add_event_dummies(merged_df)

    event_cols = [col for col in merged_df.columns if col.startswith("Event_")]
    for col in event_cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").fillna(0).astype(int)

    merged_df = merged_df.sort_values("Date").reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(OUTPUT_FILE, index=False)
    logger.info("Saved merged final dataset: %s", OUTPUT_FILE)
    logger.info("Rows: %d", len(merged_df))
    logger.info("Columns: %d", len(merged_df.columns))
    logger.info("LLM-related columns included: %s", llm_keep_cols + event_cols)
    logger.info("Dtypes of text columns:\n%s", merged_df[text_cols].dtypes)
    logger.info("Head:\n%s", merged_df.head())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge market + LLM features and write final parquet.")
    p.add_argument("--market", default=str(MARKET_FEATURES_FILE), help="Market/context features parquet path")
    p.add_argument("--llm", default=str(LLM_FEATURES_FILE), help="LLM features parquet path")
    p.add_argument("--output", default=str(OUTPUT_FILE), help="Output parquet path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    MARKET_FEATURES_FILE = Path(args.market)
    LLM_FEATURES_FILE = Path(args.llm)
    OUTPUT_FILE = Path(args.output)
    main()

