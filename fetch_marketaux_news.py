"""Fetch MarketAux news articles for BTC and save a parquet checkpoint.

This module wraps the MarketAux API with robust retrying and checkpointing
behaviour. It is intended for offline data acquisition steps in the
pipeline and therefore is networked — it is not marked as safe for the
smoke-test runner by default.
"""

from __future__ import annotations

import argparse
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Project paths
# -----------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)

API_KEY = os.getenv("MARKETAUX_API_KEY")

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "btc_marketaux_news.parquet"
CHECKPOINT_FILE = OUTPUT_DIR / "btc_marketaux_news_checkpoint.parquet"

# Networked operation — not safe for blind smoke-test execution
SMOKE_TEST_SAFE = False

# -----------------------------------------------------
# Config
# -----------------------------------------------------

START_DATE = "2022-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

LANGUAGE = "en"

LIMIT = 100
MAX_PAGES_PER_WINDOW = 8
WINDOW_DAYS = 30

REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
SLEEP_BETWEEN_REQUESTS = 1.0

# BTC specific filters
SYMBOLS = "CC:BTC"
ENTITY_TYPES = "cryptocurrency"
FILTER_ENTITIES = True
MUST_HAVE_ENTITIES = True
GROUP_SIMILAR = True

# -----------------------------------------------------
# Helper: date windows
# -----------------------------------------------------

def daterange_windows(start_date: str, end_date: str, window_days: int):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    current = start
    while current <= end:
        window_end = min(current + timedelta(days=window_days - 1), end)
        yield current.isoformat(), window_end.isoformat()
        current = window_end + timedelta(days=1)

# -----------------------------------------------------
# Robust request wrapper
# -----------------------------------------------------

def safe_request(url, params, retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT):

    last_exception = None

    for attempt in range(1, retries + 1):

        try:

            response = requests.get(url, params=params, timeout=timeout)

            if response.status_code == 200:
                return response

            if response.status_code == 429:
                logger.warning("Rate limit hit (attempt %d), sleeping...", attempt)
                time.sleep(10)
                continue

            if response.status_code == 402:
                raise RuntimeError("Marketaux quota exceeded or subscription limit reached")

            logger.warning("Unexpected status %d", response.status_code)
            logger.warning(response.text[:300])
            time.sleep(3 * attempt)

        except requests.exceptions.ReadTimeout as e:
            last_exception = e
            logger.warning("Timeout (attempt %d) retrying...", attempt)
            time.sleep(5 * attempt)

        except requests.exceptions.ConnectionError as e:
            last_exception = e
            logger.warning("Connection error (attempt %d) retrying...", attempt)
            time.sleep(5 * attempt)

        except requests.exceptions.RequestException as e:
            last_exception = e
            logger.warning("Request exception: %s", e)
    if last_exception:
        raise last_exception

    raise RuntimeError("Marketaux request failed after retries")

# -----------------------------------------------------
# Parse article into flat structure
# -----------------------------------------------------

def parse_article(article):

    entities = article.get("entities", []) or []

    btc_entities = [e for e in entities if e.get("symbol") == "CC:BTC"]

    btc_sentiments = [
        e.get("sentiment_score")
        for e in btc_entities
        if e.get("sentiment_score") is not None
    ]

    mentioned_symbols = []
    for e in entities:
        symbol = e.get("symbol")
        if symbol:
            mentioned_symbols.append(symbol)

    return {

        "uuid": article.get("uuid"),

        "published_at": article.get("published_at"),

        "title": article.get("title"),

        "description": article.get("description"),

        "snippet": article.get("snippet"),

        "source": article.get("source"),

        "url": article.get("url"),

        "language": article.get("language"),

        "relevance_score": article.get("relevance_score"),

        "entity_count": len(entities),

        "btc_entity_count": len(btc_entities),

        "btc_entity_sentiment": (
            sum(btc_sentiments) / len(btc_sentiments)
            if btc_sentiments else None
        ),

        "mentioned_symbols": (
            ",".join(sorted(set(mentioned_symbols)))
            if mentioned_symbols else None
        )
    }

# -----------------------------------------------------
# Save checkpoint
# -----------------------------------------------------

def save_checkpoint(rows, checkpoint_file: Path) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)

    if df.empty:
        return

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    df = df.drop_duplicates(subset=["url"])
    df = df.sort_values("published_at").reset_index(drop=True)

    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(checkpoint_file, index=False)

    logger.info("Checkpoint saved (%d rows)", len(df))

# -----------------------------------------------------
# API parameter builder
# -----------------------------------------------------

def build_params(win_start, win_end, page, limit):
    return {
        "api_token": API_KEY,
        "symbols": SYMBOLS,
        "entity_types": ENTITY_TYPES,
        "filter_entities": str(FILTER_ENTITIES).lower(),
        "must_have_entities": str(MUST_HAVE_ENTITIES).lower(),
        "group_similar": str(GROUP_SIMILAR).lower(),
        "language": LANGUAGE,
        "limit": limit,
        "page": page,
        "published_after": win_start,
        "published_before": win_end,
    }

# -----------------------------------------------------
# Main fetch function
# -----------------------------------------------------

def fetch_news(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    checkpoint_file: Path = CHECKPOINT_FILE,
    window_days: int = WINDOW_DAYS,
    max_pages: int = MAX_PAGES_PER_WINDOW,
    limit: int = LIMIT,
) -> pd.DataFrame:
    if not API_KEY:
        raise RuntimeError(f"MARKETAUX_API_KEY not found in .env at {ENV_PATH}")

    url = "https://api.marketaux.com/v1/news/all"

    logger.info("Using .env file: %s", ENV_PATH)
    logger.info("API key loaded: %s", "Yes" if API_KEY else "No")

    all_rows = []

    for win_start, win_end in daterange_windows(start_date, end_date, window_days):
        logger.info("Processing window %s → %s", win_start, win_end)

        for page in range(1, max_pages + 1):
            params = build_params(win_start, win_end, page, limit)

            try:
                response = safe_request(url, params)
            except RuntimeError as e:
                if "quota exceeded" in str(e).lower():
                    logger.warning("Quota limit reached. Saving partial data.")
                    save_checkpoint(all_rows, checkpoint_file)
                    return pd.DataFrame(all_rows)
                save_checkpoint(all_rows, checkpoint_file)
                raise

            data = response.json()
            articles = data.get("data", []) or []

            logger.info("Page %d: %d articles", page, len(articles))
            if not articles:
                break

            for article in articles:
                all_rows.append(parse_article(article))

            save_checkpoint(all_rows, checkpoint_file)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No articles retrieved")

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    df = df.drop_duplicates(subset=["url"])
    df = df.sort_values("published_at").reset_index(drop=True)

    return df

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for MarketAux news fetching.

    Parameters
    ----------
    argv:
        List of command-line arguments for testing; defaults to sys.argv.
    """
    parser = argparse.ArgumentParser(
        prog="fetch_marketaux_news",
        description="Fetch MarketAux BTC news articles and checkpoint results.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Output parquet file path")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_FILE, help="Checkpoint parquet file path")
    parser.add_argument("--start", default=START_DATE, help="Start date for fetching articles (YYYY-MM-DD)")
    parser.add_argument("--end", default=END_DATE, help="End date for fetching articles (YYYY-MM-DD)")
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS, help="Window size in days for paged MarketAux requests")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_PER_WINDOW, help="Max pages per date window")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Max articles per page request")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for CLI invocation."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_news(
        start_date=args.start,
        end_date=args.end,
        checkpoint_file=args.checkpoint,
        window_days=args.window_days,
        max_pages=args.max_pages,
        limit=args.limit,
    )

    if df.empty:
        raise RuntimeError("No rows collected")

    df.to_parquet(args.output, index=False)

    logger.info("Saved dataset: %s", args.output)
    logger.info("Rows: %d", len(df))
    logger.info("Columns: %s", df.columns.tolist())
    logger.info("Head:\n%s", df.head())

# -----------------------------------------------------

if __name__ == "__main__":
    main()