"""Create a weighted LLM-provider ensemble and save final features.

Loads base market features and per-provider LLM feature files, computes weighted
ensemble statistics for numeric scores and concatenated text summaries, derives
provider weights from selection fold performance, and saves final ensemble dataset.
"""

import argparse
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

BASE_FEATURES_FILE = os.path.join(DATA_DIR, "btc_features_phase1_plus.parquet")

OPENAI_FILE = os.path.join(DATA_DIR, "btc_marketaux_llm_features_openai.parquet")
CLAUDE_FILE = os.path.join(DATA_DIR, "btc_marketaux_llm_features_claude.parquet")
GEMINI_FILE = os.path.join(DATA_DIR, "btc_marketaux_llm_features_gemini.parquet")

OUTPUT_FILE = os.path.join(DATA_DIR, "btc_final_features_with_llm_weighted_ensemble.parquet")


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

# Mark safe for smoke-tests (file-only, no network or heavy training)
SMOKE_TEST_SAFE = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROVIDER_FOLD_METRICS = ROOT / "results" / "llm_comparison" / "fold_metrics_llm_compare.csv"
WEIGHT_OUTPUT_DIR = ROOT / "results" / "ensemble_llm"
WEIGHT_FOLDS = list(range(1, 8))  # folds 1--7 only


def compute_provider_weights_from_selection_folds(
    metrics_path: Path = PROVIDER_FOLD_METRICS,
    output_path: Path | None = None,
    horizon: int | str | None = None,
) -> pd.DataFrame:
    """
    Compute frozen provider weights using folds 1--7 only.

    The final reported folds must not be used to estimate these weights.
    Weights are inverse-RMSE weights, normalised to sum to one.
    """
    df = pd.read_csv(metrics_path)

    required_cols = {"provider", "fold", "rmse_lstm"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {metrics_path}: {missing}")

    df = df[df["fold"].isin(WEIGHT_FOLDS)].copy()

    horizon_key = None
    if horizon is not None:
        horizon_key = f"{horizon}d" if isinstance(horizon, int) else str(horizon)
        if "horizon" not in df.columns:
            raise ValueError(
                "A horizon was requested, but the metrics file has no 'horizon' column."
            )
        df = df[df["horizon"] == horizon_key].copy()

    if df.empty:
        raise ValueError("No provider metrics found for folds 1--7.")

    provider_scores = (
        df.groupby("provider", as_index=False)["rmse_lstm"]
        .mean()
        .rename(columns={"rmse_lstm": "selection_rmse"})
    )

    provider_scores["inverse_error"] = 1.0 / provider_scores["selection_rmse"]
    provider_scores["weight"] = (
        provider_scores["inverse_error"] / provider_scores["inverse_error"].sum()
    )

    provider_scores["weight_source_folds"] = "1--7"
    provider_scores["weighting_rule"] = "inverse_selection_rmse"

    if output_path is None:
        output_path = WEIGHT_OUTPUT_DIR / (
            f"provider_weights_folds_1_7_h{horizon_key}.csv"
            if horizon_key is not None
            else "provider_weights_folds_1_7.csv"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    provider_scores.to_csv(output_path, index=False)
    logger.info("Saved frozen provider weights: %s", output_path)

    return provider_scores


def load_provider_file(path: str, provider_name: str) -> pd.DataFrame:
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
    return df.rename(columns=rename_map)


def weighted_event_type(row: pd.Series, weights: dict) -> str:
    scores = {}
    for provider, weight in weights.items():
        event_val = str(row.get(f"{provider}_llm_event_type", "")).strip()
        if event_val:
            scores[event_val] = scores.get(event_val, 0.0) + weight

    if not scores:
        return "other"

    return max(scores.items(), key=lambda x: x[1])[0]


def weighted_text_concat(row: pd.Series, suffix: str, weights: dict) -> str:
    parts = []
    for provider, weight in sorted(weights.items(), key=lambda x: -x[1]):
        val = str(row.get(f"{provider}_{suffix}", "")).strip()
        if val:
            parts.append(f"[{provider}:{weight:.2f}] {val}")
    return " | ".join(parts)


def main() -> None:
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

    horizon_weights: dict[str, dict[str, float]] = {}
    for horizon in [1, 3, 7]:
        provider_weight_df = compute_provider_weights_from_selection_folds(
            horizon=horizon,
            output_path=WEIGHT_OUTPUT_DIR / f"provider_weights_folds_1_7_h{horizon}.csv",
        )
        horizon_weights[f"{horizon}d"] = provider_weight_df.set_index("provider")["weight"].to_dict()

    for horizon, weights in horizon_weights.items():
        for col in NUMERIC_COLS:
            merged[f"weighted_{horizon}_{col}"] = (
                weights["openai"] * merged[f"openai_{col}"]
                + weights["claude"] * merged[f"claude_{col}"]
                + weights["gemini"] * merged[f"gemini_{col}"]
            )

            provider_cols = [f"openai_{col}", f"claude_{col}", f"gemini_{col}"]
            merged[f"weighted_{horizon}_{col}_std"] = merged[provider_cols].std(axis=1).fillna(0.0)
            merged[f"weighted_{horizon}_{col}_range"] = (
                merged[provider_cols].max(axis=1) - merged[provider_cols].min(axis=1)
            )

        merged[f"weighted_{horizon}_llm_event_type"] = merged.apply(
            lambda row: weighted_event_type(row, weights), axis=1
        )
        merged[f"weighted_{horizon}_llm_summary"] = merged.apply(
            lambda row: weighted_text_concat(row, "llm_summary", weights), axis=1
        )
        merged[f"weighted_{horizon}_llm_rationale"] = merged.apply(
            lambda row: weighted_text_concat(row, "llm_rationale", weights), axis=1
        )

        dummies = pd.get_dummies(
            merged[f"weighted_{horizon}_llm_event_type"],
            prefix=f"weighted_{horizon}_event",
            dtype=int,
        )
        merged = pd.concat([merged, dummies], axis=1)

    merged["llm_provider"] = "weighted_ensemble"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    merged.to_parquet(OUTPUT_FILE, index=False)

    logger.info("Saved weighted ensemble dataset: %s", OUTPUT_FILE)
    logger.info("Rows: %d", len(merged))
    logger.info("Columns: %d", len(merged.columns))

    preview_cols = [
        "Date",
        "weighted_1d_llm_sentiment_score",
        "weighted_3d_llm_sentiment_score",
        "weighted_7d_llm_sentiment_score",
        "weighted_1d_llm_event_type",
        "weighted_3d_llm_event_type",
        "weighted_7d_llm_event_type",
        "llm_provider",
    ]
    preview_cols = [c for c in preview_cols if c in merged.columns]
    logger.info("\nHead:\n%s", merged[preview_cols].head())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build weighted LLM-provider ensemble features and save to parquet."
    )
    p.add_argument("--base", default=BASE_FEATURES_FILE, help="Base features parquet file")
    p.add_argument("--openai", default=OPENAI_FILE, help="OpenAI LLM features parquet")
    p.add_argument("--claude", default=CLAUDE_FILE, help="Claude LLM features parquet")
    p.add_argument("--gemini", default=GEMINI_FILE, help="Gemini LLM features parquet")
    p.add_argument("--output", default=OUTPUT_FILE, help="Output parquet file path")
    p.add_argument(
        "--metrics", default=str(PROVIDER_FOLD_METRICS), help="Fold metrics CSV for weight computation"
    )
    p.add_argument(
        "--weight-outdir",
        default=str(WEIGHT_OUTPUT_DIR),
        help="Directory to write provider weight CSVs",
    )
    p.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 3, 7],
        help="Horizons (days) to compute weighted features for",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Override module constants with CLI args for this run
    BASE_FEATURES_FILE = args.base
    OPENAI_FILE = args.openai
    CLAUDE_FILE = args.claude
    GEMINI_FILE = args.gemini
    OUTPUT_FILE = args.output
    PROVIDER_FOLD_METRICS = Path(args.metrics)
    WEIGHT_OUTPUT_DIR = Path(args.weight_outdir)

    # call main
    main()


if __name__ == "__main__":
    main()