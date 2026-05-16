"""Extract daily LLM features from MarketAux daily text using Gemini.

This module queries Google's Gemini API (via the `google-genai` client)
to extract structured features from daily news text. It writes periodic
checkpoints so the process can be resumed.

This is a networked operation and is NOT smoke-test safe.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Optional

from google import genai

from llm_feature_extraction_shared import (
    SYSTEM_PROMPT,
    build_user_prompt,
    default_empty_features,
    extract_json_from_text,
    load_daily_dataset,
    load_env,
    load_existing_checkpoint,
    make_result_row,
    merge_with_existing,
    normalise_feature_dict,
    output_paths,
    print_progress,
    retry_sleep,
    save_checkpoint,
    should_skip_text,
)

logger = logging.getLogger(__name__)

# Networked operation — not safe for smoke-tests
SMOKE_TEST_SAFE = False

PROVIDER = "gemini"
MODEL_NAME = os.getenv("GEMINI_LLM_COMPARE_MODEL", "gemini-2.5-flash")
MAX_ATTEMPTS = 4
CHECKPOINT_EVERY = 25


def call_gemini(client: genai.Client, date_str: str, daily_text: str, model_name: str) -> Dict:
    prompt = build_user_prompt(date_str=date_str, daily_text=daily_text)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(model=model_name, contents=full_prompt)

            text = (getattr(response, "text", None) or "").strip()
            data = extract_json_from_text(text)
            return normalise_feature_dict(data)

        except Exception as e:
            last_error = e
            logger.warning("[%s] attempt %d failed for %s: %s", PROVIDER, attempt, date_str, e)
            if attempt < MAX_ATTEMPTS:
                retry_sleep(attempt)

    raise RuntimeError(f"[{PROVIDER}] failed after {MAX_ATTEMPTS} attempts for {date_str}: {last_error}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for Gemini LLM feature extraction."""
    p = argparse.ArgumentParser(description="Extract LLM features using Gemini")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    p.add_argument("--limit", type=int, default=0, help="Process only first N days (for testing)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the Gemini provider CLI."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    load_env()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)

    daily_df = load_daily_dataset()
    existing_df = load_existing_checkpoint(PROVIDER)
    work_df = merge_with_existing(daily_df, existing_df)

    rows = []
    if not existing_df.empty:
        rows.extend(existing_df.to_dict(orient="records"))

    todo_df = work_df[~work_df["done"]].copy().reset_index(drop=True)
    total = len(todo_df)

    logger.info("[%s] model: %s", PROVIDER, args.model)
    logger.info("[%s] total days to process: %d", PROVIDER, total)

    if args.limit and args.limit > 0:
        todo_df = todo_df.iloc[: args.limit]
        total = len(todo_df)

    for i, (_, row) in enumerate(todo_df.iterrows(), start=1):
        date_value = row["Date"]
        date_str = date_value.strftime("%Y-%m-%d")
        daily_text = row["daily_text"]

        print_progress(i, total, date_str, PROVIDER)

        if should_skip_text(daily_text):
            features = default_empty_features()
        else:
            features = call_gemini(client, date_str, daily_text, args.model)

        rows.append(make_result_row(date_value, features, PROVIDER))

        if i % args.checkpoint_every == 0:
            save_checkpoint(rows, PROVIDER)

    save_checkpoint(rows, PROVIDER)
    parquet_path, _ = output_paths(PROVIDER)
    logger.info("[%s] complete: %s", PROVIDER, parquet_path)


if __name__ == "__main__":
    main()