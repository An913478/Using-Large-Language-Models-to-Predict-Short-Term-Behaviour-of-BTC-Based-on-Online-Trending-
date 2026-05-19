"""Summarize model predictions under embargo sensitivity experiments.

Loads prediction CSVs, computes RMSE and MAE skill metrics relative to
naive zero-return baseline, and exports CSV and LaTeX summary tables.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
SMOKE_TEST_SAFE = True


# -----------------------------
# Helper functions
# -----------------------------
def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find a matching column name from a list of candidate names, case-insensitive."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise ValueError(f"Could not find any of these columns: {candidates}. Available: {list(df.columns)}")


# -----------------------------
# Scoring helpers
# -----------------------------
def summarise_prediction_file(path: Path, model_name: str, horizon: int, gap: int) -> dict:
    """Compute skill metrics for a single prediction CSV under embargo sensitivity."""
    df = pd.read_csv(path)

    actual_col = find_col(df, ["Actual_Return", "actual_return", "Actual", "y_true", "target", "Target_Return"])
    pred_col = find_col(df, ["Predicted_Return", "predicted_return", "Prediction", "y_pred", "pred", "predicted_return_lstm", "predicted_return_dcn"])

    y = df[actual_col].astype(float).to_numpy()
    yhat = df[pred_col].astype(float).to_numpy()

    valid = np.isfinite(y) & np.isfinite(yhat)
    y = y[valid]
    yhat = yhat[valid]

    # Limit to 540 predictions for consistency with main report
    max_n = 540
    if len(y) > max_n:
        y = y[-max_n:]
        yhat = yhat[-max_n:]

    err_model = y - yhat
    err_naive = y

    model_rmse = float(np.sqrt(np.mean(err_model ** 2)))
    naive_rmse = float(np.sqrt(np.mean(err_naive ** 2)))
    rmse_skill = float(1.0 - model_rmse / naive_rmse)

    model_mae = float(np.mean(np.abs(err_model)))
    naive_mae = float(np.mean(np.abs(err_naive)))
    mae_skill = float(1.0 - model_mae / naive_mae)

    return {
        "Horizon": f"{horizon}-day",
        "Selected model": model_name,
        "Embargo": gap,
        "N": int(len(y)),
        "Model RMSE": model_rmse,
        "Naive RMSE": naive_rmse,
        "RMSE skill": rmse_skill,
        "Model MAE": model_mae,
        "Naive MAE": naive_mae,
        "MAE skill": mae_skill,
    }


# -----------------------------
# Main entrypoint
# -----------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise embargo sensitivity experiment prediction outputs.")
    parser.add_argument("--dcn3", type=Path, required=True, help="Path to DCN 3-day prediction CSV")
    parser.add_argument("--lstm7", type=Path, required=True, help="Path to LSTM 7-day prediction CSV")
    parser.add_argument("--outdir", type=Path, default=Path("results/embargo_sensitivity"), help="Directory to write summary outputs")
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    """Summarise embargo sensitivity prediction results and export CSV/LaTeX tables."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # -----------------------------
    # Prepare output folder and compute file summaries
    # -----------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = [
        summarise_prediction_file(
            args.dcn3,
            model_name="Convolutional / DCN baseline",
            horizon=3,
            gap=2,
        ),
        summarise_prediction_file(
            args.lstm7,
            model_name="LSTM baseline",
            horizon=7,
            gap=6,
        ),
    ]

    summary = pd.DataFrame(rows)
    summary.to_csv(args.outdir / "embargo_sensitivity_summary.csv", index=False)

    display_df = summary.copy()
    for col in ["Model RMSE", "Naive RMSE", "Model MAE", "Naive MAE"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    for col in ["RMSE skill", "MAE skill"]:
        display_df[col] = display_df[col].map(lambda x: f"{100*x:.1f}\\%")

    latex = display_df.to_latex(
        index=False,
        escape=False,
        column_format="llrrrrrrrr",
    )
    (args.outdir / "embargo_sensitivity_summary_table.tex").write_text(latex)

    logger.info("Summary:\n%s", summary)


if __name__ == "__main__":
    args = parse_args()
    main(args)
