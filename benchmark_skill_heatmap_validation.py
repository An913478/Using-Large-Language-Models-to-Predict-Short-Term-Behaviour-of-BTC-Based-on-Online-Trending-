#!/usr/bin/env python3
"""
Benchmark-relative skill heatmap with significance overlay.

This script:
1. Loads one or more out-of-sample prediction CSV files.
2. Standardises common prediction-column names.
3. Compares each model against the naive zero-return benchmark.
4. Computes benchmark-relative RMSE skill:
       Skill_RMSE = 1 - RMSE_model / RMSE_naive
5. Runs horizon-aware Diebold-Mariano-style paired tests using Newey-West variance.
6. Computes moving-block bootstrap confidence intervals for RMSE skill.
7. Creates a heatmap with significance markers.
8. Writes CSV and LaTeX validation tables for the Results section.

Expected prediction fields, with flexible matching:

If a file has no model column, pass the model name using:
    path/to/file.csv::Model Name
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)
SMOKE_TEST_SAFE = True


# -----------------------------
# Column handling
# -----------------------------

COLUMN_CANDIDATES = {
    "date": [
        "Date", "date", "prediction_date", "Prediction_Date", "timestamp", "Timestamp"
    ],
    "fold": [
        "Fold", "fold", "fold_id", "Fold_ID", "walkforward_fold", "split"
    ],
    "horizon": [
        "Horizon", "horizon", "forecast_horizon", "Forecast_Horizon", "h", "H"
    ],
    "model": [
        "Model", "model", "model_name", "Model_Name", "model_family",
        "Model_Family", "configuration", "Configuration", "strategy", "Strategy"
    ],
    "actual": [
        "Actual_Return", "actual_return", "actual", "Actual",
        "y_true", "Y_true", "target", "Target", "realised_return",
        "realized_return", "Realised_Return", "Realized_Return"
    ],
    "predicted": [
        "Predicted_Return",
        "predicted_return",
        "predicted_return_lstm",
        "predicted_return_transformer",
        "predicted_return_attention",
        "predicted_return_attention_context",
        "predicted_return_dcn",
        "predicted_return_conv1d",
        "predicted_return_cnn",
        "predicted_return_convolutional",
        "predicted_return_llm",
        "predicted_return_multi_selector",
        "predicted_return_model_selector",
        "prediction",
        "Prediction",
        "y_pred",
        "Y_pred",
        "forecast",
        "Forecast",
        "predicted",
        "predicted_return_naive",
    ],
}


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    """Find a column by exact candidate match, then case-insensitive match."""
    cols = list(df.columns)

    for c in candidates:
        if c in cols:
            return c

    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    if required:
        raise ValueError(
            f"Could not find required column. Tried: {list(candidates)}. "
            f"Available columns: {cols}"
        )

    return None


def parse_horizon(value) -> int:
    """Convert horizon labels such as 1, '1', '1d', 'horizon_1d' into integer days."""
    if pd.isna(value):
        raise ValueError("Missing horizon value")

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, float) and value.is_integer():
        return int(value)

    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError(f"Could not parse horizon from value: {value}")

    return int(match.group(1))


def parse_prediction_arg(arg: str) -> tuple[Path, Optional[str]]:
    """
    Parse prediction argument.

    Accepts:
        path/to/file.csv
        path/to/file.csv::Model Name
    """
    if "::" in arg:
        path_str, model_name = arg.split("::", 1)
        return Path(path_str), model_name.strip()
    return Path(arg), None


def _find_column(df, candidates, required=True, purpose="column"):
    """
    Return the first matching column from a list of possible names.
    Matching is case-insensitive.
    """
    lower_to_original = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        key = str(candidate).lower()
        if key in lower_to_original:
            return lower_to_original[key]

    if required:
        raise ValueError(
            f"Could not identify {purpose}. "
            f"Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def _normalise_horizon(value):
    """
    Convert horizon values such as 1, '1', '1d', '1-day', 'horizon_1d'
    into integer horizons: 1, 3, 7.
    """
    if pd.isna(value):
        return value

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)

    text = str(value).strip().lower()
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))

    return value


def _choose_prediction_column(df, fallback_model=None):
    """
    Select the model prediction column from a prediction CSV.

    Handles files with columns such as:
      predicted_return
      Predicted_Return
      predicted_return_lstm
      predicted_return_transformer
      predicted_return_dcn
      predicted_return_naive

    The naive prediction column is not selected unless the fallback model name
    explicitly contains 'naive'.
    """
    columns = list(df.columns)
    lower_to_original = {str(c).lower(): c for c in columns}

    # Direct canonical options first
    direct_candidates = [
        "Predicted_Return",
        "predicted_return",
        "prediction",
        "pred",
        "y_pred",
        "forecast",
        "forecast_return",
    ]

    for candidate in direct_candidates:
        key = candidate.lower()
        if key in lower_to_original:
            return lower_to_original[key]

    model_text = (fallback_model or "").lower()

    model_specific_candidates = []

    if "lstm" in model_text:
        model_specific_candidates.extend([
            "predicted_return_lstm",
            "lstm_predicted_return",
            "lstm_prediction",
            "pred_lstm",
        ])

    if "transformer" in model_text:
        model_specific_candidates.extend([
            "predicted_return_transformer",
            "transformer_predicted_return",
            "transformer_prediction",
            "pred_transformer",
        ])

    if "attention" in model_text:
        model_specific_candidates.extend([
            "predicted_return_attention_context",
            "predicted_return_attention",
            "attention_context_predicted_return",
            "attention_predicted_return",
            "pred_attention",
        ])

    if "dcn" in model_text or "conv" in model_text or "convolution" in model_text:
        model_specific_candidates.extend([
            "predicted_return_dcn",
            "predicted_return_conv1d",
            "predicted_return_cnn",
            "predicted_return_convolutional",
            "dcn_predicted_return",
            "conv1d_predicted_return",
            "cnn_predicted_return",
            "pred_dcn",
        ])

    if "llm" in model_text or "selector" in model_text:
        model_specific_candidates.extend([
            "predicted_return_llm",
            "predicted_return_multi_selector",
            "predicted_return_model_selector",
            "multi_selector_predicted_return",
            "model_selector_predicted_return",
            "pred_llm",
        ])

    if "weighted" in model_text or "ensemble" in model_text:
        model_specific_candidates.extend([
            "predicted_return_weighted_ensemble",
            "predicted_return_ensemble",
            "weighted_ensemble_predicted_return",
            "ensemble_predicted_return",
            "pred_ensemble",
        ])

    for candidate in model_specific_candidates:
        key = candidate.lower()
        if key in lower_to_original:
            return lower_to_original[key]

    # Fallback: choose the only non-naive predicted_return column
    predicted_return_cols = [
        c for c in columns
        if str(c).lower().startswith("predicted_return")
    ]

    non_naive_cols = [
        c for c in predicted_return_cols
        if "naive" not in str(c).lower()
    ]

    if len(non_naive_cols) == 1:
        return non_naive_cols[0]

    if len(predicted_return_cols) == 1:
        return predicted_return_cols[0]

    raise ValueError(
        f"Could not identify model prediction column for fallback model "
        f"'{fallback_model}'. Available columns: {list(df.columns)}. "
        f"Candidate prediction columns found: {predicted_return_cols}"
    )


def normalise_prediction_file(path, fallback_model=None):
    """
    Read one prediction CSV and return a canonical long-format frame.

    Required canonical output columns:
      Date
      Horizon
      Fold
      Model
      Actual_Return
      Predicted_Return

    Optional canonical output columns:
      Naive_Predicted_Return
      Actual_Direction
      Predicted_Direction
    """
    path = Path(path)
    df = pd.read_csv(path)

    date_col = _find_column(
        df,
        ["Date", "date", "datetime", "timestamp"],
        required=True,
        purpose="date column",
    )

    horizon_col = _find_column(
        df,
        ["Horizon", "horizon", "forecast_horizon", "h"],
        required=True,
        purpose="horizon column",
    )

    fold_col = _find_column(
        df,
        ["Fold", "fold", "fold_id", "walk_forward_fold", "split", "split_id"],
        required=False,
        purpose="fold column",
    )

    actual_col = _find_column(
        df,
        [
            "Actual_Return",
            "actual_return",
            "actual",
            "realised_return",
            "realized_return",
            "y_true",
            "target",
            "Target_Return",
        ],
        required=True,
        purpose="actual return column",
    )

    pred_col = _choose_prediction_column(df, fallback_model=fallback_model)

    naive_col = _find_column(
        df,
        [
            "Naive_Predicted_Return",
            "Predicted_Return_Naive",
            "predicted_return_naive",
            "naive_predicted_return",
            "naive_prediction",
            "pred_naive",
        ],
        required=False,
        purpose="naive prediction column",
    )

    actual_dir_col = _find_column(
        df,
        [
            "Actual_Direction",
            "actual_direction",
            "realised_direction",
            "realized_direction",
            "y_true_direction",
        ],
        required=False,
        purpose="actual direction column",
    )

    pred_dir_col = None
    pred_dir_candidates = [
        "Predicted_Direction",
        "predicted_direction",
        "predicted_direction_lstm",
        "predicted_direction_transformer",
        "predicted_direction_dcn",
        "predicted_direction_conv1d",
        "predicted_direction_attention_context",
        "predicted_direction_llm",
        "predicted_direction_naive",
    ]

    model_text = (fallback_model or "").lower()
    if "lstm" in model_text:
        pred_dir_candidates.insert(0, "predicted_direction_lstm")
    elif "transformer" in model_text:
        pred_dir_candidates.insert(0, "predicted_direction_transformer")
    elif "dcn" in model_text or "conv" in model_text:
        pred_dir_candidates.insert(0, "predicted_direction_dcn")
    elif "attention" in model_text:
        pred_dir_candidates.insert(0, "predicted_direction_attention_context")
    elif "llm" in model_text or "selector" in model_text:
        pred_dir_candidates.insert(0, "predicted_direction_llm")

    pred_dir_col = _find_column(
        df,
        pred_dir_candidates,
        required=False,
        purpose="predicted direction column",
    )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["horizon"] = df[horizon_col].apply(_normalise_horizon)

    if fold_col is not None:
        out["fold"] = df[fold_col]
    else:
        out["fold"] = 0

    out["model"] = fallback_model if fallback_model else path.stem
    out["actual_return"] = pd.to_numeric(df[actual_col], errors="coerce")
    out["predicted_return"] = pd.to_numeric(df[pred_col], errors="coerce")

    if naive_col is not None:
        out["Naive_Predicted_Return"] = pd.to_numeric(df[naive_col], errors="coerce")
    else:
        # Zero-return benchmark used in the dissertation.
        out["Naive_Predicted_Return"] = 0.0

    if actual_dir_col is not None:
        out["Actual_Direction"] = df[actual_dir_col]

    if pred_dir_col is not None:
        out["Predicted_Direction"] = df[pred_dir_col]

    out = out.dropna(subset=["date", "horizon", "actual_return", "predicted_return"])

    if out.empty:
        raise ValueError(
            f"After normalising {path}, no valid prediction rows remained. "
            f"Check date, horizon, actual and prediction columns."
        )

    return out

# -----------------------------
# Metrics
# -----------------------------

def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def f1_direction(y: np.ndarray, yhat: np.ndarray) -> float:
    actual_pos = y > 0
    pred_pos = yhat > 0

    tp = np.sum(actual_pos & pred_pos)
    fp = np.sum(~actual_pos & pred_pos)
    fn = np.sum(actual_pos & ~pred_pos)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def normal_cdf(x: float) -> float:
    """Standard normal CDF using erf; avoids requiring scipy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def diebold_mariano_test(
    y: np.ndarray,
    yhat_model: np.ndarray,
    yhat_benchmark: np.ndarray,
    horizon: int,
    loss: str = "squared",
) -> dict:
    """
    Diebold-Mariano-style paired predictive accuracy test.

    d_t = L(e_model) - L(e_benchmark)

    Positive mean loss difference means the model is worse than benchmark.
    Newey-West lag is set to horizon - 1 to account for overlapping h-step labels.
    """
    e_model = y - yhat_model
    e_bench = y - yhat_benchmark

    if loss == "squared":
        d = e_model ** 2 - e_bench ** 2
    elif loss == "absolute":
        d = np.abs(e_model) - np.abs(e_bench)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)

    if n < 5:
        return {
            "dm_stat": np.nan,
            "p_value": np.nan,
            "mean_loss_diff": np.nan,
            "direction": "insufficient_data",
        }

    d_bar = float(np.mean(d))
    centred = d - d_bar
    lag = max(0, int(horizon) - 1)

    # Newey-West variance of the sample mean.
    gamma0 = float(np.mean(centred * centred))
    long_run_var = gamma0

    max_lag = min(lag, n - 1)
    for j in range(1, max_lag + 1):
        gamma_j = float(np.mean(centred[j:] * centred[:-j]))
        weight = 1.0 - j / (max_lag + 1.0)
        long_run_var += 2.0 * weight * gamma_j

    var_mean = long_run_var / n

    if var_mean <= 0 or not np.isfinite(var_mean):
        dm_stat = np.nan
        p_value = np.nan
    else:
        dm_stat = d_bar / math.sqrt(var_mean)
        p_value = 2.0 * (1.0 - normal_cdf(abs(dm_stat)))

    if d_bar < 0:
        direction = "model_better"
    elif d_bar > 0:
        direction = "model_worse"
    else:
        direction = "no_difference"

    return {
        "dm_stat": dm_stat,
        "p_value": p_value,
        "mean_loss_diff": d_bar,
        "direction": direction,
    }


def circular_moving_block_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Generate circular moving-block bootstrap indices."""
    starts = rng.integers(0, n, size=math.ceil(n / block_len))
    idx = []

    for s in starts:
        block = [(s + j) % n for j in range(block_len)]
        idx.extend(block)

    return np.asarray(idx[:n], dtype=int)


def bootstrap_skill_ci(
    y: np.ndarray,
    yhat_model: np.ndarray,
    horizon: int,
    n_boot: int = 2000,
    block_len: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """
    Moving-block bootstrap CI for RMSE skill.

    Skill_RMSE = 1 - RMSE_model / RMSE_naive
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    if block_len is None:
        block_len = max(int(horizon), int(round(math.sqrt(n))))

    skill_values = []

    for _ in range(n_boot):
        idx = circular_moving_block_indices(n, block_len, rng)
        y_b = y[idx]
        pred_b = yhat_model[idx]
        naive_b = np.zeros_like(y_b)

        rmse_model = rmse(y_b, pred_b)
        rmse_naive = rmse(y_b, naive_b)

        if rmse_naive > 0:
            skill_values.append(1.0 - rmse_model / rmse_naive)

    skill_values = np.asarray(skill_values, dtype=float)
    skill_values = skill_values[np.isfinite(skill_values)]

    if len(skill_values) == 0:
        return {
            "skill_ci_low": np.nan,
            "skill_ci_high": np.nan,
            "block_len": block_len,
        }

    return {
        "skill_ci_low": float(np.percentile(skill_values, 2.5)),
        "skill_ci_high": float(np.percentile(skill_values, 97.5)),
        "block_len": block_len,
    }


def fold_loss_differences(group: pd.DataFrame) -> pd.DataFrame:
    """Compute fold-wise RMSE and MAE differences against zero-return naive benchmark."""
    rows = []

    for fold, fdf in group.groupby("fold"):
        y = fdf["actual_return"].to_numpy(dtype=float)
        pred = fdf["predicted_return"].to_numpy(dtype=float)
        naive = np.zeros_like(y)

        rows.append({
            "model": fdf["model"].iloc[0],
            "horizon": int(fdf["horizon"].iloc[0]),
            "fold": fold,
            "rmse_model": rmse(y, pred),
            "rmse_naive": rmse(y, naive),
            "delta_rmse": rmse(y, pred) - rmse(y, naive),
            "mae_model": mae(y, pred),
            "mae_naive": mae(y, naive),
            "delta_mae": mae(y, pred) - mae(y, naive),
            "model_better_rmse": rmse(y, pred) < rmse(y, naive),
            "model_better_mae": mae(y, pred) < mae(y, naive),
        })

    return pd.DataFrame(rows)


def evaluate_group(group: pd.DataFrame, n_boot: int, alpha: float) -> dict:
    """Compute pooled metrics, DM tests, bootstrap CI and significance marker."""
    group = group.sort_values(["date", "fold"], na_position="last")

    y = group["actual_return"].to_numpy(dtype=float)
    pred = group["predicted_return"].to_numpy(dtype=float)
    naive = np.zeros_like(y)
    horizon = int(group["horizon"].iloc[0])
    model = str(group["model"].iloc[0])

    rmse_model = rmse(y, pred)
    rmse_naive = rmse(y, naive)
    mae_model = mae(y, pred)
    mae_naive = mae(y, naive)
    f1_model = f1_direction(y, pred)

    skill_rmse = 1.0 - rmse_model / rmse_naive if rmse_naive > 0 else np.nan
    skill_mae = 1.0 - mae_model / mae_naive if mae_naive > 0 else np.nan

    dm_squared = diebold_mariano_test(y, pred, naive, horizon=horizon, loss="squared")
    dm_absolute = diebold_mariano_test(y, pred, naive, horizon=horizon, loss="absolute")
    boot = bootstrap_skill_ci(y, pred, horizon=horizon, n_boot=n_boot)

    # Significance marker is based on squared-loss DM test because the heatmap uses RMSE skill.
    p = dm_squared["p_value"]
    direction = dm_squared["direction"]

    if np.isfinite(p) and p < alpha and direction == "model_better":
        marker = "*"
        significance = "significantly_better"
    elif np.isfinite(p) and p < alpha and direction == "model_worse":
        marker = "\u2020"  # dagger
        significance = "significantly_worse"
    else:
        marker = ""
        significance = "not_significant"

    return {
        "model": model,
        "horizon": horizon,
        "n": int(len(group)),
        "rmse_model": rmse_model,
        "rmse_naive": rmse_naive,
        "skill_rmse": skill_rmse,
        "mae_model": mae_model,
        "mae_naive": mae_naive,
        "skill_mae": skill_mae,
        "f1_model": f1_model,
        "dm_squared_stat": dm_squared["dm_stat"],
        "dm_squared_p": dm_squared["p_value"],
        "dm_squared_direction": dm_squared["direction"],
        "dm_absolute_stat": dm_absolute["dm_stat"],
        "dm_absolute_p": dm_absolute["p_value"],
        "dm_absolute_direction": dm_absolute["direction"],
        "skill_rmse_ci_low": boot["skill_ci_low"],
        "skill_rmse_ci_high": boot["skill_ci_high"],
        "bootstrap_block_len": boot["block_len"],
        "significance": significance,
        "marker": marker,
    }


# -----------------------------
# Plotting and LaTeX output
# -----------------------------

def plot_heatmap(summary: pd.DataFrame, figdir: Path, alpha: float) -> None:
    """Create benchmark skill heatmap with significance markers."""
    figdir.mkdir(parents=True, exist_ok=True)

    pivot = summary.pivot_table(
        index="model",
        columns="horizon",
        values="skill_rmse",
        aggfunc="first",
    )

    # Keep horizons in natural order if present.
    horizon_order = [h for h in [1, 3, 7] if h in pivot.columns]
    other_horizons = [h for h in pivot.columns if h not in horizon_order]
    pivot = pivot[horizon_order + other_horizons]

    marker_pivot = summary.pivot_table(
        index="model",
        columns="horizon",
        values="marker",
        aggfunc="first",
    ).reindex(index=pivot.index, columns=pivot.columns)

    values = pivot.to_numpy(dtype=float)

    if np.all(~np.isfinite(values)):
        raise ValueError("No finite skill values available for heatmap.")

    max_abs = float(np.nanmax(np.abs(values)))
    if max_abs == 0:
        max_abs = 0.01

    fig_width = max(10.0, 1.6 * len(pivot.columns) + 4.5)
    fig_height = max(6.5, 0.85 * len(pivot.index) + 3.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-max_abs, vmax=max_abs)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{int(h)}-day" for h in pivot.columns], fontsize=14, fontweight="bold")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=14)

    ax.set_xlabel("Forecast horizon", fontsize=16, fontweight="bold")
    ax.set_ylabel("Model / configuration", fontsize=16, fontweight="bold")

    ax.set_title(
        "Benchmark-relative RMSE skill against naive zero-return baseline",
        fontsize=20,
        fontweight="bold",
        pad=14,
    )

    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    threshold = -max_abs * 0.18
    for i in range(values.shape[0]):
        row = values[i, :]
        best_j = int(np.nanargmax(row)) if np.isfinite(row).any() else -1
        for j in range(values.shape[1]):
            val = values[i, j]
            marker = marker_pivot.iloc[i, j] if not pd.isna(marker_pivot.iloc[i, j]) else ""

            if np.isfinite(val):
                label = f"{val * 100:.1f}%{marker}"
                text_color = "white" if val < threshold else "black"
                font_weight = "bold" if j == best_j else "normal"
            else:
                label = "NA"
                text_color = "black"
                font_weight = "normal"

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=16,
                fontweight=font_weight,
                color=text_color,
            )

            if marker:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        linewidth=2.0,
                        edgecolor="#000000",
                    )
                )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(
        r"RMSE skill: $1-\mathrm{RMSE}_{model}/\mathrm{RMSE}_{naive}$",
        fontsize=16,
        fontweight="bold",
    )
    cbar.ax.tick_params(labelsize=12)

    note = (
        f"* significant improvement vs naive; \u2020 significant underperformance vs naive "
        f"(Diebold-Mariano-style paired test, alpha={alpha})."
    )
    fig.text(0.5, 0.01, note, ha="center", fontsize=12, color="#222222")

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    fig.savefig(figdir / "benchmark_skill_heatmap.pdf", bbox_inches="tight")
    fig.savefig(figdir / "benchmark_skill_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_latex_validation_table(summary: pd.DataFrame, outpath: Path) -> None:
    """
    Write a compact LaTeX table for the Results section.

    The table is intentionally compact. Full CSV outputs remain in results/statistical_validation/.
    """
    rows = []

    for _, row in summary.sort_values(["horizon", "model"]).iterrows():
        model = str(row["model"])
        horizon = int(row["horizon"])

        skill_pct = row["skill_rmse"] * 100.0
        ci_low = row["skill_rmse_ci_low"] * 100.0
        ci_high = row["skill_rmse_ci_high"] * 100.0

        p_val = row["dm_squared_p"]
        if pd.isna(p_val):
            p_text = "--"
        elif p_val < 0.001:
            p_text = "$<0.001$"
        else:
            p_text = f"{p_val:.3f}"

        sig = str(row["significance"]).replace("_", " ")

        rows.append(
            f"{horizon}-day & {model} & "
            f"{skill_pct:.1f}\\% & "
            f"[{ci_low:.1f}\\%, {ci_high:.1f}\\%] & "
            f"{p_text} & {sig} \\\\"
        )

    latex = r"""
\begin{table}[!htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.18}
\caption{Formal validation of benchmark-relative \gls{rmse} skill. Skill is defined as \(1-\mathrm{RMSE}_{model}/\mathrm{RMSE}_{naive}\); negative values indicate underperformance against the naive zero-return benchmark. Confidence intervals use moving-block bootstrap resampling.}
\label{tab:benchmark_skill_validation}
\begin{tabularx}{\textwidth}{L{0.13\textwidth}L{0.31\textwidth}>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>L{0.18\textwidth}}
\toprule
\textbf{Horizon} & \textbf{Model / configuration} & \textbf{\gls{rmse} skill} & \textbf{95\% bootstrap CI} & \textbf{DM \(p\)} & \textbf{Result} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
    outpath.write_text(latex.strip() + "\n", encoding="utf-8")


# -----------------------------
# Main entrypoint and CLI
# -----------------------------

def main(args: argparse.Namespace) -> None:
    # -----------------------------
    # Prepare output directories and load inputs
    # -----------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.figdir.mkdir(parents=True, exist_ok=True)

    frames = []

    for pred_arg in args.prediction:
        path, fallback_model = parse_prediction_arg(pred_arg)

        if not path.exists():
            if args.ignore_missing:
                logger.warning("Missing file skipped: %s", path)
                continue
            raise FileNotFoundError(path)

        logger.info("Loading %s", path)
        frames.append(normalise_prediction_file(path, fallback_model=fallback_model))

    if not frames:
        raise RuntimeError("No prediction files were loaded.")

    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions.dropna(subset=["actual_return", "predicted_return", "horizon", "model"])

    if predictions.empty:
        raise RuntimeError("Prediction table is empty after cleaning.")

    predictions.to_csv(args.outdir / "benchmark_validation_predictions_long.csv", index=False)

    summary_rows = []
    fold_rows = []

    for (model, horizon), group in predictions.groupby(["model", "horizon"]):
        if len(group) < 5:
            logger.warning("Skipping %s, horizon=%s: too few rows.", model, horizon)
            continue

        summary_rows.append(evaluate_group(group, n_boot=args.n_boot, alpha=args.alpha))

        if "fold" in group.columns:
            fold_rows.append(fold_loss_differences(group))

    summary = pd.DataFrame(summary_rows).sort_values(["horizon", "model"])
    summary.to_csv(args.outdir / "benchmark_skill_validation.csv", index=False)

    if fold_rows:
        fold_df = pd.concat(fold_rows, ignore_index=True)
        fold_df.to_csv(args.outdir / "fold_wise_loss_differences.csv", index=False)

        fold_summary = (
            fold_df.groupby(["model", "horizon"])
            .agg(
                folds=("fold", "nunique"),
                mean_delta_rmse=("delta_rmse", "mean"),
                sd_delta_rmse=("delta_rmse", "std"),
                folds_model_better_rmse=("model_better_rmse", "sum"),
                mean_delta_mae=("delta_mae", "mean"),
                sd_delta_mae=("delta_mae", "std"),
                folds_model_better_mae=("model_better_mae", "sum"),
            )
            .reset_index()
        )
        fold_summary.to_csv(args.outdir / "fold_wise_loss_difference_summary.csv", index=False)

    plot_heatmap(summary, figdir=args.figdir, alpha=args.alpha)

    write_latex_validation_table(
        summary,
        args.outdir / "benchmark_skill_validation_table.tex",
    )

    logger.info("Outputs written:")
    logger.info("  Figure PDF: %s", args.figdir / "benchmark_skill_heatmap.pdf")
    logger.info("  Figure PNG: %s", args.figdir / "benchmark_skill_heatmap.png")
    logger.info("  Summary CSV: %s", args.outdir / "benchmark_skill_validation.csv")
    logger.info("  Fold-wise CSV: %s", args.outdir / "fold_wise_loss_differences.csv")
    logger.info("  LaTeX table: %s", args.outdir / "benchmark_skill_validation_table.tex")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark skill heatmap validation for bitcoin forecasting models."
        ),
    )
    parser.add_argument(
        "--prediction",
        nargs="+",
        required=True,
        help=(
            "One or more prediction CSV files. If a file lacks a model column, "
            "use path/to/file.csv::Model Name"
        ),
    )
    parser.add_argument("--figdir", type=Path, default=Path("figures/results"), help="Output figure directory")
    parser.add_argument("--outdir", type=Path, default=Path("results/statistical_validation"), help="Output CSV/LaTeX directory")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for confidence intervals")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of moving-block bootstrap resamples.")
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Skip missing prediction files instead of raising an error.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)