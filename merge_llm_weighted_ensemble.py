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
RANDOM_SEED = 42  # Default seed for reproducibility; override via --seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROVIDER_FOLD_METRICS = ROOT / "results" / "llm_comparison" / "fold_metrics_llm_compare.csv"
WEIGHT_OUTPUT_DIR = ROOT / "results" / "ensemble_llm"
WEIGHT_FOLDS = list(range(1, 8))  # folds 1--7 only
CONFIRMATION_FOLDS = [8, 9]


def compute_provider_weights_from_selection_folds(
    metrics_path: Path = PROVIDER_FOLD_METRICS,
    output_path: Path | None = None,
    horizon: int | str | None = None,
) -> pd.DataFrame:
    """Compute frozen provider weights using folds 1--7 only.

    CRITICAL: This function requires a fold-level metrics CSV file (with 'fold' column).
    It will raise an error if the metrics file does not contain fold-level data.
    The final reported test folds (8–9) must not be used to estimate these weights.
    Weights are inverse-RMSE weights, normalised to sum to one.
    """
    df = pd.read_csv(metrics_path)

    required_cols = {"provider", "fold", "rmse_lstm"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Metrics file must contain fold-level data with columns {required_cols}.\n"
            f"Missing: {missing}\n"
            f"File: {metrics_path}\n"
            f"Ensure you pass --metrics with a fold-level file (e.g., fold_metrics_llm_compare.csv), "
            f"not a summary file."
        )

    logger.info("Read metrics file: %s", metrics_path)
    logger.info("Unique folds in file: %s", sorted(df["fold"].unique()))

    df_selection = df[df["fold"].isin(WEIGHT_FOLDS)].copy()
    logger.info("Filtering to selection folds: %s", WEIGHT_FOLDS)
    logger.info("Rows after filtering: %d (out of %d total)", len(df_selection), len(df))

    if df_selection.empty:
        raise ValueError(
            f"No provider metrics found for selection folds {WEIGHT_FOLDS}. "
            f"File may not contain those fold numbers or may be a summary file."
        )

    df = df_selection
    
    horizon_key = None
    if horizon is not None:
        horizon_key = f"{horizon}d" if isinstance(horizon, int) else str(horizon)
        if "horizon" not in df.columns:
            raise ValueError(
                "A horizon was requested, but the metrics file has no 'horizon' column."
            )
        df = df[df["horizon"] == horizon_key].copy()

    if df.empty:
        raise ValueError("No provider metrics found for selection folds 1--7 and horizon.")


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


def confirm_weights_on_folds(
    weights_df: pd.DataFrame,
    metrics_path: Path = PROVIDER_FOLD_METRICS,
    confirm_folds: list[int] | None = None,
    horizon: int | str | None = None,
) -> pd.DataFrame:
    """Validate frozen weights on confirmation folds (e.g. folds 8-9).

    This uses provider fold RMSEs on the confirmation folds to compute an
    approximate weighted MSE and compares it to the best single-provider RMSE.
    Returns a small DataFrame summarising confirmation statistics per provider
    and the ensemble weighted estimate.
    """
    if confirm_folds is None:
        confirm_folds = CONFIRMATION_FOLDS

    df = pd.read_csv(metrics_path)
    if horizon is not None:
        horizon_key = f"{horizon}d" if isinstance(horizon, int) else str(horizon)
        if "horizon" not in df.columns:
            raise ValueError("Metrics file missing 'horizon' column for confirmation step")
        df = df[df["horizon"] == horizon_key].copy()

    df = df[df["fold"].isin(confirm_folds)].copy()
    if df.empty:
        raise ValueError("No metrics found for confirmation folds: %s" % confirm_folds)

    # compute per-provider mean MSE on confirmation folds
    df["mse_lstm"] = df["rmse_lstm"] ** 2
    confirm_scores = df.groupby("provider", as_index=False)["mse_lstm"].mean()
    confirm_scores["rmse_confirm"] = confirm_scores["mse_lstm"].apply(lambda x: float(np.sqrt(x)))

    # assemble weights
    w = weights_df.set_index("provider")["weight"].to_dict()

    # approximate ensemble mse as weighted average of provider MSEs
    provider_mse_map = dict(zip(confirm_scores["provider"], confirm_scores["mse_lstm"]))
    weighted_mse = sum(w.get(p, 0.0) * provider_mse_map.get(p, 0.0) for p in provider_mse_map)
    weighted_rmse = float(np.sqrt(weighted_mse))

    best_provider_rmse = float(confirm_scores.loc[confirm_scores["rmse_confirm"].idxmin(), "rmse_confirm"])

    summary = pd.DataFrame([
        {
            "weighted_rmse_confirm": weighted_rmse,
            "best_provider_rmse_confirm": best_provider_rmse,
            "confirmed": weighted_rmse <= best_provider_rmse * 1.02,  # allow 2% tolerance
        }
    ])

    result = confirm_scores.merge(weights_df[ ["provider", "weight"] ], on="provider", how="left")
    result = result[["provider", "weight", "rmse_confirm"]].sort_values("provider")
    # attach ensemble summary as extra rows
    ensemble_row = pd.DataFrame(
        {"provider": ["__ensemble__"], "weight": [np.nan], "rmse_confirm": [weighted_rmse]}
    )
    result = pd.concat([result, ensemble_row], ignore_index=True)
    result["confirmed"] = summary.at[0, "confirmed"]
    result.attrs["summary"] = summary.to_dict(orient="records")[0]
    return result


def load_provider_file(path: str, provider_name: str) -> pd.DataFrame:
    """Load and normalize a per-provider LLM feature parquet file.

    This function ensures the provider dataset has a normalized Date index,
    fills any missing numeric/text feature columns with safe defaults, and
    prefixes all imported feature columns with the provider name.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {provider_name} file: {path}")

    df = pd.read_parquet(path).copy()

    if "Date" not in df.columns:
        raise ValueError(f"{provider_name} file must contain Date column.")

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    keep_cols = ["Date"] + [c for c in NUMERIC_COLS + TEXT_COLS if c in df.columns]
    df = df[keep_cols].copy()

    # ensure every provider file has the same numeric/text columns
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
    """Choose the highest-weighted event type across provider outputs.

    Provider event type strings are aggregated by their provider weights, and
    the event label with the highest total weight is selected for the ensemble.
    """
    scores = {}
    for provider, weight in weights.items():
        event_val = str(row.get(f"{provider}_llm_event_type", "")).strip()
        if event_val:
            scores[event_val] = scores.get(event_val, 0.0) + weight

    if not scores:
        return "other"

    return max(scores.items(), key=lambda x: x[1])[0]


def weighted_text_concat(row: pd.Series, suffix: str, weights: dict) -> str:
    """Concatenate provider text fields in descending weight order.

    Each provider's text is annotated with its weight so that the final
    ensemble summary reflects relative provider importance.
    """
    parts = []
    for provider, weight in sorted(weights.items(), key=lambda x: -x[1]):
        val = str(row.get(f"{provider}_{suffix}", "")).strip()
        if val:
            parts.append(f"[{provider}:{weight:.2f}] {val}")
    return " | ".join(parts)


def main(require_confirmation: bool = True) -> None:
    """Build weighted ensemble features from base and provider LLM data.

    Configurable via CLI arguments:
      --base: Base market features parquet file path.
      --openai, --claude, --gemini: Per-provider LLM feature files.
      --output: Output weighted ensemble parquet file.
      --metrics: Fold metrics CSV for computing provider weights.
      --weight-outdir: Directory to write provider weight files.
      --horizons: Target horizons (days) as space-separated integers (default: 1 3 7).
      --require-confirmation: Validate weights on confirmation folds.
      --seed: Random seed for reproducibility (default: 42).

    Loads the market base dataset and per-provider LLM feature sets, computes
    provider weights from selection folds, optionally validates the weights on
    confirmation folds, and writes the final weighted ensemble parquet file.
    """
    # Validate metrics file is fold-level before proceeding
    if not os.path.exists(PROVIDER_FOLD_METRICS):
        raise FileNotFoundError(
            f"Metrics file not found: {PROVIDER_FOLD_METRICS}\n"
            f"Pass --metrics with a fold-level CSV file (e.g., fold_metrics_llm_compare.csv)."
        )
    
    try:
        metrics_df_check = pd.read_csv(PROVIDER_FOLD_METRICS)
        if "fold" not in metrics_df_check.columns:
            raise ValueError(
                f"Metrics file does not contain 'fold' column.\n"
                f"File: {PROVIDER_FOLD_METRICS}\n"
                f"This appears to be a summary file, not fold-level data.\n"
                f"Ensure --metrics points to a fold-level metrics CSV."
            )
        logger.info("✓ Metrics file validated: contains fold-level data")
        logger.info("  Folds present: %s", sorted(metrics_df_check["fold"].unique()))
    except Exception as e:
        raise RuntimeError(f"Failed to validate metrics file: {e}")
    
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

    # standarize missing provider data before feature engineering
    for provider in ["openai", "claude", "gemini"]:
        for col in NUMERIC_COLS:
            merged[f"{provider}_{col}"] = pd.to_numeric(
                merged[f"{provider}_{col}"], errors="coerce"
            ).fillna(0.0)
        for col in TEXT_COLS:
            merged[f"{provider}_{col}"] = merged[f"{provider}_{col}"].fillna("").astype(str)

    horizon_weights: dict[str, dict[str, float]] = {}
    confirmation_reports = {}
    for horizon in [1, 3, 7]:
        provider_weight_df = compute_provider_weights_from_selection_folds(
            horizon=horizon,
            output_path=WEIGHT_OUTPUT_DIR / f"provider_weights_folds_1_7_h{horizon}.csv",
        )
        horizon_weights[f"{horizon}d"] = provider_weight_df.set_index("provider")["weight"].to_dict()

        # Optionally confirm weights on reserved folds (e.g. folds 8-9)
        if require_confirmation:
            try:
                confirm_df = confirm_weights_on_folds(
                    provider_weight_df, metrics_path=PROVIDER_FOLD_METRICS, confirm_folds=CONFIRMATION_FOLDS, horizon=horizon
                )
                confirmation_reports[f"{horizon}d"] = confirm_df
                conf_out = WEIGHT_OUTPUT_DIR / f"provider_weights_confirmation_h{horizon}.csv"
                confirm_df.to_csv(conf_out, index=False)
                logger.info("Wrote confirmation report: %s", conf_out)
                if not bool(confirm_df.get("confirmed", True).iloc[0]):
                    logger.warning(
                        "Weight confirmation failed for horizon %sd: weighted RMSE worse than best provider on confirmation folds.",
                        horizon,
                    )
            except Exception as exc:
                logger.exception("Weight confirmation step failed for horizon %s: %s", horizon, exc)

    # Compute weighted features and auxiliary indicators for each target horizon
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

    logger.info("="*60)
    logger.info("✓ COMPLETED: Weighted Ensemble Dataset")
    logger.info("Provider weights were computed from SELECTION FOLDS: %s", WEIGHT_FOLDS)
    logger.info("Confirmation folds (reserved for validation): %s", CONFIRMATION_FOLDS)
    logger.info("="*60)
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
        description="Build weighted LLM-provider ensemble features and save to parquet. All input/output paths are configurable."
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
    p.add_argument(
        "--require-confirmation",
        action="store_true",
        help="Require that computed provider weights are confirmed on reserved folds (8-9)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility (affects NumPy random ops if any).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    seed = args.seed
    np.random.seed(seed)

    # Override module constants with CLI args for this run
    BASE_FEATURES_FILE = args.base
    OPENAI_FILE = args.openai
    CLAUDE_FILE = args.claude
    GEMINI_FILE = args.gemini
    OUTPUT_FILE = args.output
    PROVIDER_FOLD_METRICS = Path(args.metrics)
    WEIGHT_OUTPUT_DIR = Path(args.weight_outdir)

    logger.info("=== Configuration ===")
    logger.info("Seed: %d", seed)
    logger.info("Base features: %s", BASE_FEATURES_FILE)
    logger.info("Output: %s", OUTPUT_FILE)
    logger.info("Metrics: %s", PROVIDER_FOLD_METRICS)
    logger.info("Weight output dir: %s", WEIGHT_OUTPUT_DIR)
    logger.info("Require confirmation: %s", args.require_confirmation)
    logger.info("="*50)

    # call main
    main(require_confirmation=args.require_confirmation)