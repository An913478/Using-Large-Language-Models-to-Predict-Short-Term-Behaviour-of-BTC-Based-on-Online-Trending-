#!/usr/bin/env python3
"""
Modality ablation for the BTC forecasting report.

Purpose
-------
Tests whether each information channel adds forecast value under the same
walk-forward, fold-local-scaling protocol:

    market only
    market + Google Trends
    market + LLM/news
    external only: Google Trends + LLM/news
    all modalities

This is designed as an extra robustness/diagnostic result. It uses a fast
Ridge regression sequence model so it can be run quickly without retraining all
neural models. Treat it as a feature-signal ablation, not as a replacement for
the main model-family comparison.

Typical command
---------------
python scripts/run_modality_ablation.py \
  --input data/processed/btc_final_features_with_llm_weighted_ensemble.parquet \
  --outdir results/modality_ablation \
  --sequence_length 30 \
  --horizons 1 3 7
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Column handling
# -----------------------------

DATE_CANDIDATES = ["Date", "date", "Datetime", "datetime", "timestamp", "Timestamp"]
CLOSE_CANDIDATES = ["Close", "close", "Adj Close", "Adj_Close", "close_price", "Close_BTC"]

NON_FEATURE_PATTERNS = [
    r"^target", r"target_", r"actual", r"predicted", r"prediction", r"fold",
    r"direction", r"label", r"text", r"title", r"url", r"source", r"summary",
    r"content", r"article_id", r"published", r"provider$", r"model$",
]

LLM_PATTERNS = [
    "llm", "sentiment", "impact", "uncert", "confidence", "news",
    "article", "openai", "gemini", "claude", "ensemble", "disagreement",
    "semantic", "narrative", "bullish", "bearish", "risk_score",
    "market_impact", "relevance",
]

TREND_PATTERNS = [
    "google", "pytrends", "search", "search_interest", "gt_",
    "bitcoin_crash", "bitcoin_rally", "btc_price", "crypto_market",
    "crypto_news", "ethereum", "attention",
]

MARKET_HINTS = [
    "open", "high", "low", "close", "volume", "return", "logreturn", "volatility",
    "ma_", "moving", "momentum", "price", "ohlcv", "drawdown", "trend_ratio",
]


def find_first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def read_table(path: Path) -> pd.DataFrame:
    """Read a parquet or CSV table from disk into a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if path.suffix.lower() in [".csv", ".txt"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}. Use .parquet or .csv")


def is_non_feature_column(col: str) -> bool:
    """Detect columns that should be excluded from feature groups."""
    c = col.lower()
    return any(re.search(pattern, c) for pattern in NON_FEATURE_PATTERNS)


def contains_any(col: str, patterns: Iterable[str]) -> bool:
    """Return True if a column name contains any of the provided string patterns."""
    c = col.lower()
    return any(p.lower() in c for p in patterns)


def is_google_trends_column(col: str) -> bool:
    """Avoid classifying market Trend_Ratio columns as Google Trends."""
    c = col.lower()
    if "trend_ratio" in c or "price_trend" in c:
        return False
    if contains_any(c, TREND_PATTERNS):
        return True
    # Allow columns such as Trends_Bitcoin or trend_bitcoin, but not generic market trend ratios.
    if c.startswith("trend_") or c.startswith("trends_"):
        return True
    return False


def is_llm_column(col: str) -> bool:
    return contains_any(col, LLM_PATTERNS)


def feature_groups(df: pd.DataFrame, date_col: Optional[str], close_col: Optional[str]) -> Dict[str, List[str]]:
    """Partition numeric columns into market, Google Trends, and LLM/news groups."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = []
    for col in numeric_cols:
        if date_col and col == date_col:
            continue
        if is_non_feature_column(col):
            continue
        candidates.append(col)

    llm = [c for c in candidates if is_llm_column(c)]
    trends = [c for c in candidates if is_google_trends_column(c)]
    llm_set, trend_set = set(llm), set(trends)

    # Market is the remaining numeric feature block. This deliberately includes engineered
    # price, return, volatility, volume and momentum variables.
    market = [c for c in candidates if c not in llm_set and c not in trend_set]

    # Keep Close as a predictor only if it was already in the table and not excluded.
    # Target construction uses future Close but the current close is date-valid.
    return {
        "market": market,
        "google_trends": trends,
        "llm_news": llm,
    }


# -----------------------------
# Target and sequence construction
# -----------------------------

def ensure_targets(df: pd.DataFrame, close_col: str, horizons: Sequence[int]) -> pd.DataFrame:
    """Add missing multi-horizon target return columns to the DataFrame."""
    df = df.copy()
    for h in horizons:
        target_col = f"Target_Return_{h}d"
        if target_col not in df.columns:
            df[target_col] = df[close_col].shift(-h) / df[close_col] - 1.0
    return df


def make_sequences(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    date_col: Optional[str],
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp | str | int]]:
    """Build supervised flattened sequence examples from features and target."""
    cols = list(feature_cols) + [target_col]
    if date_col:
        cols.append(date_col)

    sub = df[cols].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan)

    # Preserve chronological order; remove rows that cannot form valid supervised examples.
    feature_values = sub[list(feature_cols)].to_numpy(dtype=float)
    y_values = sub[target_col].to_numpy(dtype=float)
    dates = sub[date_col].tolist() if date_col else list(range(len(sub)))

    X_out, y_out, d_out = [], [], []
    for i in range(sequence_length - 1, len(sub)):
        y_i = y_values[i]
        x_i = feature_values[i - sequence_length + 1 : i + 1, :]
        if not np.isfinite(y_i):
            continue
        if not np.isfinite(x_i).all():
            continue
        X_out.append(x_i.reshape(-1))
        y_out.append(float(y_i))
        d_out.append(dates[i])

    if not X_out:
        return np.empty((0, len(feature_cols) * sequence_length)), np.empty((0,)), []
    return np.asarray(X_out, dtype=float), np.asarray(y_out, dtype=float), d_out


# -----------------------------
# Walk-forward modelling
# -----------------------------

def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """Compute regression and direction metrics for a prediction array."""
    if len(y_true) == 0:
        return {}
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    actual_dir = y_true > 0
    pred_dir = y_pred > 0
    return {
        f"{prefix}n": int(len(y_true)),
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}f1": f1_score(actual_dir, pred_dir, zero_division=0),
        f"{prefix}accuracy": accuracy_score(actual_dir, pred_dir),
    }


def walk_forward_predict(
    X: np.ndarray,
    y: np.ndarray,
    dates: Sequence[pd.Timestamp | str | int],
    *,
    initial_train_ratio: float,
    test_window: int,
    gap: int,
    ridge_alpha: float,
    min_train: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute fold-local walk-forward Ridge predictions and return metrics."""
    n = len(y)
    if n < min_train + test_window:
        raise ValueError(f"Not enough supervised rows ({n}) for min_train={min_train} and test_window={test_window}")

    initial_train_end = int(math.floor(initial_train_ratio * n))
    initial_train_end = max(initial_train_end, min_train)

    prediction_rows = []
    fold_rows = []
    fold = 0

    train_end = initial_train_end
    while train_end < n:
        test_start = train_end + gap
        test_end = min(test_start + test_window, n)
        if test_start >= n or test_end <= test_start:
            break

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        model = Ridge(alpha=ridge_alpha, random_state=0)
        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)

        naive = np.zeros_like(pred)

        fold_model_metrics = metric_dict(y[test_idx], pred)
        fold_naive_metrics = metric_dict(y[test_idx], naive, prefix="naive_")

        fold_rows.append({
            "Fold": fold + 1,
            "Train_End_Index": int(train_end),
            "Test_Start_Index": int(test_start),
            "Test_End_Index": int(test_end),
            **fold_model_metrics,
            **fold_naive_metrics,
        })

        for idx, y_i, p_i in zip(test_idx, y[test_idx], pred):
            prediction_rows.append({
                "Date": dates[idx],
                "Fold": fold + 1,
                "Actual_Return": float(y_i),
                "Predicted_Return": float(p_i),
                "Naive_Predicted_Return": 0.0,
            })

        fold += 1
        train_end += test_window

    return pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows)


def summarise_predictions(preds: pd.DataFrame) -> Dict[str, float]:
    """Summarize prediction outcomes and compute skill metrics relative to naive zero."""
    y = preds["Actual_Return"].to_numpy(float)
    p = preds["Predicted_Return"].to_numpy(float)
    naive = np.zeros_like(p)

    model = metric_dict(y, p)
    naive_metrics = metric_dict(y, naive, prefix="naive_")

    rmse_skill = 1.0 - model["rmse"] / naive_metrics["naive_rmse"] if naive_metrics["naive_rmse"] else np.nan
    mae_skill = 1.0 - model["mae"] / naive_metrics["naive_mae"] if naive_metrics["naive_mae"] else np.nan

    return {
        **model,
        **naive_metrics,
        "rmse_skill": rmse_skill,
        "mae_skill": mae_skill,
        "rmse_skill_pct": 100.0 * rmse_skill,
        "mae_skill_pct": 100.0 * mae_skill,
    }


# -----------------------------
# Plotting and table export
# -----------------------------

def save_rmse_skill_heatmap(summary: pd.DataFrame, outdir: Path) -> None:
    """Create a heatmap of RMSE skill across horizons and feature sets."""
    if summary.empty:
        return

    pivot = summary.pivot(index="Horizon", columns="Feature_Set", values="rmse_skill_pct")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(max(9.0, 1.6 * len(pivot.columns)), 6.0))
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=12)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{h}-day" for h in pivot.index], fontsize=12)
    ax.set_title("Benchmark-relative RMSE skill by modality", fontsize=16, fontweight="bold")
    ax.set_xlabel("Feature set", fontsize=14)
    ax.set_ylabel("Forecast horizon", fontsize=14)
    ax.set_ylim(pivot.shape[0] - 0.5, -0.5)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("RMSE skill vs naive (%)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    vmax = float(np.nanmax(pivot.to_numpy()))
    vmin = float(np.nanmin(pivot.to_numpy()))
    threshold = vmin + (vmax - vmin) * 0.52

    for i in range(pivot.shape[0]):
        row = pivot.iloc[i].to_numpy()
        best_j = int(np.nanargmax(row)) if np.isfinite(row).any() else -1
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                color = "white" if val < threshold else "black"
                weight = "bold" if j == best_j else "normal"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight=weight,
                    color=color,
                )

    fig.tight_layout()
    fig.savefig(outdir / "modality_ablation_rmse_skill_heatmap.png", dpi=300)
    fig.savefig(outdir / "modality_ablation_rmse_skill_heatmap.pdf")
    plt.close(fig)


def save_summary_tex(summary: pd.DataFrame, outdir: Path) -> None:
    """Write a LaTeX summary table reporting modality ablation results."""
    if summary.empty:
        return
    display = summary.copy()
    display["RMSE"] = display["rmse"].map(lambda x: f"{x:.4f}")
    display["MAE"] = display["mae"].map(lambda x: f"{x:.4f}")
    display["F1"] = display["f1"].map(lambda x: f"{x:.4f}")
    display["RMSE skill"] = display["rmse_skill_pct"].map(lambda x: f"{x:.1f}\\%")
    display["MAE skill"] = display["mae_skill_pct"].map(lambda x: f"{x:.1f}\\%")
    display = display[["Horizon", "Feature_Set", "n", "RMSE", "MAE", "F1", "RMSE skill", "MAE skill"]]
    tex = display.to_latex(index=False, escape=False, caption="Modality ablation results under walk-forward validation.", label="tab:modality_ablation_results")
    (outdir / "modality_ablation_summary_table.tex").write_text(tex)


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modality ablation under walk-forward validation.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/btc_final_features_with_llm_weighted_ensemble.parquet"))
    parser.add_argument("--outdir", type=Path, default=Path("results/modality_ablation"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 7])
    parser.add_argument("--sequence_length", type=int, default=30)
    parser.add_argument("--initial_train_ratio", type=float, default=0.60)
    parser.add_argument("--test_window", type=int, default=60)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--min_train", type=int, default=120)
    parser.add_argument("--date_col", type=str, default=None)
    parser.add_argument("--close_col", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Run modality ablation experiments and export predictions, folds, and summary tables."""
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = read_table(args.input)

    date_col = args.date_col or find_first_existing(df.columns, DATE_CANDIDATES)
    close_col = args.close_col or find_first_existing(df.columns, CLOSE_CANDIDATES)
    if close_col is None:
        raise ValueError(
            "Could not find a close-price column. Pass --close_col explicitly. "
            f"Available columns include: {list(df.columns)[:30]}"
        )

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="ignore")
        df = df.sort_values(date_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df = ensure_targets(df, close_col, args.horizons)
    groups = feature_groups(df, date_col, close_col)

    # Define ablation sets. Empty groups are automatically skipped.
    set_defs = {
        "market_only": groups["market"],
        "market_plus_google_trends": groups["market"] + groups["google_trends"],
        "market_plus_llm_news": groups["market"] + groups["llm_news"],
        "external_only_trends_llm": groups["google_trends"] + groups["llm_news"],
        "all_modalities": groups["market"] + groups["google_trends"] + groups["llm_news"],
    }

    # Remove duplicate columns while preserving order.
    for key, cols in list(set_defs.items()):
        seen = set()
        unique = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        set_defs[key] = unique

    group_summary = pd.DataFrame([
        {"Group": "market", "Columns": len(groups["market"])},
        {"Group": "google_trends", "Columns": len(groups["google_trends"])},
        {"Group": "llm_news", "Columns": len(groups["llm_news"])},
    ])
    group_summary.to_csv(args.outdir / "modality_group_column_counts.csv", index=False)

    # Save exact column group membership for audit.
    with (args.outdir / "modality_group_columns.txt").open("w") as f:
        for group, cols in groups.items():
            f.write(f"\n[{group}] {len(cols)} columns\n")
            for col in cols:
                f.write(f"  - {col}\n")

    all_predictions = []
    all_fold_metrics = []
    all_summaries = []

    for horizon in args.horizons:
        target_col = f"Target_Return_{horizon}d"
        for feature_set, cols in set_defs.items():
            if len(cols) == 0:
                continue

            X, y, dates = make_sequences(df, cols, target_col, date_col, args.sequence_length)
            if len(y) < args.min_train + max(10, args.test_window // 2):
                print(f"Skipping {feature_set}, {horizon}-day: only {len(y)} valid examples.")
                continue

            preds, fold_metrics = walk_forward_predict(
                X, y, dates,
                initial_train_ratio=args.initial_train_ratio,
                test_window=args.test_window,
                gap=args.gap,
                ridge_alpha=args.ridge_alpha,
                min_train=args.min_train,
            )

            preds["Horizon"] = horizon
            preds["Feature_Set"] = feature_set
            preds["Feature_Count"] = len(cols)
            fold_metrics["Horizon"] = horizon
            fold_metrics["Feature_Set"] = feature_set
            fold_metrics["Feature_Count"] = len(cols)

            summary = summarise_predictions(preds)
            summary.update({
                "Horizon": horizon,
                "Feature_Set": feature_set,
                "Feature_Count": len(cols),
                "Input_File": str(args.input),
                "Sequence_Length": args.sequence_length,
                "Initial_Train_Ratio": args.initial_train_ratio,
                "Test_Window": args.test_window,
                "Gap": args.gap,
                "Model": f"Ridge(alpha={args.ridge_alpha}) on flattened lookback sequence",
            })

            all_predictions.append(preds)
            all_fold_metrics.append(fold_metrics)
            all_summaries.append(summary)
            print(f"Finished {feature_set}, {horizon}-day: RMSE={summary['rmse']:.4f}, skill={summary['rmse_skill_pct']:.1f}%")

    if not all_summaries:
        raise RuntimeError("No ablation runs completed. Check feature groups and input table.")

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    folds_df = pd.concat(all_fold_metrics, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)

    ordered_cols = [
        "Horizon", "Feature_Set", "Feature_Count", "n", "rmse", "mae", "f1", "accuracy",
        "naive_rmse", "naive_mae", "rmse_skill_pct", "mae_skill_pct",
        "Sequence_Length", "Initial_Train_Ratio", "Test_Window", "Gap", "Model", "Input_File",
    ]
    summary_df = summary_df[[c for c in ordered_cols if c in summary_df.columns]]

    predictions_df.to_csv(args.outdir / "modality_ablation_predictions.csv", index=False)
    folds_df.to_csv(args.outdir / "modality_ablation_fold_metrics.csv", index=False)
    summary_df.to_csv(args.outdir / "modality_ablation_summary.csv", index=False)

    save_rmse_skill_heatmap(summary_df, args.outdir)
    save_summary_tex(summary_df, args.outdir)

    print("\nSaved:")
    print(f"  {args.outdir / 'modality_ablation_summary.csv'}")
    print(f"  {args.outdir / 'modality_ablation_predictions.csv'}")
    print(f"  {args.outdir / 'modality_ablation_rmse_skill_heatmap.png'}")
    print(f"  {args.outdir / 'modality_ablation_summary_table.tex'}")


if __name__ == "__main__":
    main()
