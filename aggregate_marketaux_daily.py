"""Aggregate MarketAux raw articles into daily summaries.

Reads raw MarketAux articles parquet from fetch_marketaux_news.py, creates
daily-level summaries with text concatenation, source counts, and sentiment
statistics. File-based and deterministic, safe for smoke tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "btc_marketaux_news.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "btc_marketaux_daily.parquet"

# Safe for smoke tests (no network)
SMOKE_TEST_SAFE = True


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").strip()


def build_daily_text(group: pd.DataFrame) -> str:
    """Concatenate title/description/snippet/source fields into a single daily string."""

    chunks = []

    for _, row in group.iterrows():
        title = clean_text(row.get("title"))
        description = clean_text(row.get("description"))
        snippet = clean_text(row.get("snippet"))
        source = clean_text(row.get("source"))

        text_parts = []
        if title:
            text_parts.append(f"Title: {title}")
        if description:
            text_parts.append(f"Description: {description}")
        if snippet:
            text_parts.append(f"Snippet: {snippet}")
        if source:
            text_parts.append(f"Source: {source}")

        if text_parts:
            chunks.append(" | ".join(text_parts))

    return "\n".join(chunks)


def main(input_file: Path = INPUT_FILE, output_file: Path = OUTPUT_FILE, verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_file).copy()

    if df.empty:
        raise RuntimeError("Marketaux raw dataset is empty.")

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"]).copy()

    df["Date"] = df["published_at"].dt.date
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["snippet"] = df["snippet"].fillna("")
    df["source"] = df["source"].fillna("")

    daily_rows = []

    for date_value, group in df.groupby("Date"):
        group = group.sort_values("published_at").reset_index(drop=True)

        daily_text = build_daily_text(group)

        daily_rows.append({
            "Date": pd.to_datetime(date_value),
            "ArticleCount": int(len(group)),
            "UniqueSourceCount": int(group["source"].nunique()),
            "MeanRelevanceScore": float(group["relevance_score"].dropna().mean()) if group["relevance_score"].notna().any() else None,
            "MeanBTCEntitySentiment": float(group["btc_entity_sentiment"].dropna().mean()) if group["btc_entity_sentiment"].notna().any() else None,
            "DailyNewsText": daily_text,
        })

    daily_df = pd.DataFrame(daily_rows).sort_values("Date").reset_index(drop=True)

    if daily_df.empty:
        raise RuntimeError("No daily rows were produced.")

    daily_df.to_parquet(output_file, index=False)

    logger.info("Saved daily Marketaux dataset: %s", output_file)
    logger.info("Rows: %d", len(daily_df))
    logger.info("Columns: %s", daily_df.columns.tolist())
    logger.info("Head:\n%s", daily_df.head())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate MarketAux raw articles into daily summaries.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE, help="Input raw MarketAux parquet file")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Output daily summary parquet file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(input_file=args.input, output_file=args.output, verbose=args.verbose)