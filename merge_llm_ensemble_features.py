"""Create an ensemble LLM features dataset from all three providers.

This script loads OpenAI, Claude, and Gemini LLM feature parquet files,
computes ensemble statistics for numeric scores, combines text fields into a
single summary/rationale, and writes the merged ensemble dataset to parquet.
It is a file-only aggregation step and is safe for smoke tests.
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

BASE_FEATURES_FILE = os.path.join(DATA_DIR, "btc_features_phase1_plus.parquet")

OPENAI_FILE = os.path.join(DATA_DIR, "btc_marketaux_llm_features_openai.parquet")
CLAUDE_FILE = os.path.join(DATA_DIR, "btc_marketaux_llm_features_claude.parquet")
GEMINI_FILE = os.path.join(DATA_DIR, "btc_marketaux_llm_features_gemini.parquet")

OUTPUT_FILE = os.path.join(DATA_DIR, "btc_final_features_with_llm_ensemble.parquet")

# File-only operation: safe for smoke-tests
SMOKE_TEST_SAFE = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


NUMERIC_COLS = [
    "llm_sentiment_score",
    "llm_market_impact_score",
    "llm_uncertainty_score",
    "llm_adoption_score",
    "llm_regulation_score",
    "llm_macro_score",
    "llm_technical_score",
]

TEXT_COLS = [
    "llm_event_type",
    "llm_summary",
    "llm_rationale",
]


def load_provider_file(path: str, provider_name: str) -> pd.DataFrame:
    """Load and normalize a single provider's LLM feature file.

    Args:
        path: Path to the provider parquet file.
        provider_name: A short provider identifier used for column renaming.

    Returns:
        A DataFrame containing the provider's normalized numeric and text LLM
        features, with the provider prefix applied to each column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {provider_name} file: {path}")

    df = pd.read_parquet(path).copy()

    if "Date" not in df.columns:
        raise ValueError(f"{provider_name} file must contain Date column.")

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    keep_cols = ["Date"] + [c for c in NUMERIC_COLS + TEXT_COLS if c in df.columns]
    df = df[keep_cols].copy()

    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0

    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = ""

    df = df.groupby("Date", as_index=False).first()

    rename_map = {col: f"{provider_name}_{col}" for col in NUMERIC_COLS + TEXT_COLS}
    df = df.rename(columns=rename_map)

    return df


def most_common_event_type(row: pd.Series) -> str:
    """Choose the most common event type across providers for one row.

    Args:
        row: A DataFrame row containing provider-specific event type columns.

    Returns:
        The provider event type with the highest count, or 'other' if none exist.
    """
    vals = [
        row.get("openai_llm_event_type", ""),
        row.get("claude_llm_event_type", ""),
        row.get("gemini_llm_event_type", ""),
    ]
    vals = [str(v).strip() for v in vals if str(v).strip() != ""]
    if not vals:
        return "other"

    counts = pd.Series(vals).value_counts()
    return str(counts.index[0])


def combine_text(row: pd.Series, suffix: str) -> str:
    """Concatenate provider text values into a single ensemble string.

    Args:
        row: A DataFrame row containing provider-specific text columns.
        suffix: The text field suffix to combine (e.g. 'llm_summary').

    Returns:
        A combined string containing each provider's non-empty text value.
    """
    parts = []
    for provider in ["openai", "claude", "gemini"]:
        col = f"{provider}_{suffix}"
        val = str(row.get(col, "")).strip()
        if val:
            parts.append(f"[{provider}] {val}")
    return " | ".join(parts)


def main() -> None:
    """Run the ensemble merge pipeline and write the final parquet output.

    This function performs the full merge sequence: load base features,
    load each provider dataset, normalize columns, compute ensemble statistics,
    aggregate text fields, create one-hot event dummies, and save the final
    ensemble parquet file.
    """
    if not os.path.exists(BASE_FEATURES_FILE):
        raise FileNotFoundError(f"Base features file not found: {BASE_FEATURES_FILE}")

    base_df = pd.read_parquet(BASE_FEATURES_FILE).copy()
    if "Date" not in base_df.columns:
        raise ValueError("Base features file must contain Date column.")

    base_df["Date"] = pd.to_datetime(base_df["Date"]).dt.normalize()

    openai_df = load_provider_file(OPENAI_FILE, "openai")
    claude_df = load_provider_file(CLAUDE_FILE, "claude")
    gemini_df = load_provider_file(GEMINI_FILE, "gemini")

    merged = base_df.merge(openai_df, on="Date", how="left")
    merged = merged.merge(claude_df, on="Date", how="left")
    merged = merged.merge(gemini_df, on="Date", how="left")

    for provider in ["openai", "claude", "gemini"]:
        for col in NUMERIC_COLS:
            merged[f"{provider}_{col}"] = pd.to_numeric(
                merged[f"{provider}_{col}"], errors="coerce"
            ).fillna(0.0)

        for col in TEXT_COLS:
            merged[f"{provider}_{col}"] = merged[f"{provider}_{col}"].fillna("").astype(str)

    for col in NUMERIC_COLS:
        provider_cols = [f"openai_{col}", f"claude_{col}", f"gemini_{col}"]
        merged[f"ensemble_{col}_mean"] = merged[provider_cols].mean(axis=1)
        merged[f"ensemble_{col}_std"] = merged[provider_cols].std(axis=1).fillna(0.0)
        merged[f"ensemble_{col}_min"] = merged[provider_cols].min(axis=1)
        merged[f"ensemble_{col}_max"] = merged[provider_cols].max(axis=1)

    merged["ensemble_llm_event_type"] = merged.apply(most_common_event_type, axis=1)
    merged["ensemble_llm_summary"] = merged.apply(lambda row: combine_text(row, "llm_summary"), axis=1)
    merged["ensemble_llm_rationale"] = merged.apply(lambda row: combine_text(row, "llm_rationale"), axis=1)

    event_dummies = pd.get_dummies(
        merged["ensemble_llm_event_type"],
        prefix="ensemble_event",
        dtype=int,
    )
    merged = pd.concat([merged, event_dummies], axis=1)

    merged["llm_provider"] = "ensemble"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    merged.to_parquet(OUTPUT_FILE, index=False)

    added_numeric = []
    for col in NUMERIC_COLS:
        added_numeric.extend(
            [
                f"ensemble_{col}_mean",
                f"ensemble_{col}_std",
                f"ensemble_{col}_min",
                f"ensemble_{col}_max",
            ]
        )

    logger.info("Saved ensemble dataset: %s", OUTPUT_FILE)
    logger.info("Rows: %d", len(merged))
    logger.info("Columns: %d", len(merged.columns))
    logger.info("Example ensemble columns: %s", added_numeric[:8] + ["ensemble_llm_event_type", "llm_provider"])

    preview_cols = [
        "Date",
        "ensemble_llm_sentiment_score_mean",
        "ensemble_llm_market_impact_score_mean",
        "ensemble_llm_uncertainty_score_mean",
        "ensemble_llm_event_type",
        "llm_provider",
    ]
    preview_cols = [c for c in preview_cols if c in merged.columns]
    logger.info("Head:\n%s", merged[preview_cols].head())


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ensemble merge script."""
    p = argparse.ArgumentParser(description="Build ensemble LLM features from all three providers.")
    p.add_argument("--base", default=BASE_FEATURES_FILE, help="Base features parquet file")
    p.add_argument("--openai", default=OPENAI_FILE, help="OpenAI LLM features parquet")
    p.add_argument("--claude", default=CLAUDE_FILE, help="Claude LLM features parquet")
    p.add_argument("--gemini", default=GEMINI_FILE, help="Gemini LLM features parquet")
    p.add_argument("--output", default=OUTPUT_FILE, help="Output parquet file path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    BASE_FEATURES_FILE = args.base
    OPENAI_FILE = args.openai
    CLAUDE_FILE = args.claude
    GEMINI_FILE = args.gemini
    OUTPUT_FILE = args.output
    main()