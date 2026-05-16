#!/usr/bin/env python3
"""Fold-wise non-parametric robustness check using Wilcoxon signed-rank test.

Computes fold-level RMSE and MAE for each model, compares against benchmark
using Wilcoxon signed-rank and sign tests. Outputs metrics and test results
to results/validation/fold_wilcoxon_validation.csv.

Required prediction columns:
- Date
- Horizon
- Model
- Fold
- Actual_Return
- Predicted_Return

Output:
- results/validation/fold_wilcoxon_validation.csv
- results/validation/fold_metric_differences.csv
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)
SMOKE_TEST_SAFE = True


# -----------------------------
# Required schema and constants
# -----------------------------
REQUIRED_COLS = {"Date", "Horizon", "Model", "Fold", "Actual_Return", "Predicted_Return"}


def normalise_horizon(x: object) -> str:
    s = str(x).strip()
    if s in {"1", "1d", "1-day", "1 day"}:
        return "1-day"
    if s in {"3", "3d", "3-day", "3 day"}:
        return "3-day"
    if s in {"7", "7d", "7-day", "7 day"}:
        return "7-day"
    return s


def rmse(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(a - p)))


def sign_test_two_sided(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    nz = diffs[diffs != 0]
    n = len(nz)
    if n == 0:
        return float("nan")
    n_pos = int(np.sum(nz > 0))
    n_neg = int(np.sum(nz < 0))
    k = min(n_pos, n_neg)
    # Exact two-sided binomial test under p=0.5
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * p)


# -----------------------------
# Command-line arguments
# -----------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--out_summary_csv", default="results/validation/fold_wilcoxon_validation.csv")
    p.add_argument("--out_fold_diffs_csv", default="results/validation/fold_metric_differences.csv")
    p.add_argument("--benchmark", default="zero", choices=["zero", "model"])
    p.add_argument("--benchmark_model", default=None)
    return p.parse_args(argv)


# -----------------------------
# Main entrypoint
# -----------------------------
def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = pd.read_csv(args.predictions_csv)
    df.columns = [c.strip() for c in df.columns]

    lower_map = {c.lower(): c for c in df.columns}
    for required_col in REQUIRED_COLS:
        lower_key = required_col.lower()
        if required_col not in df.columns and lower_key in lower_map:
            df[required_col] = df[lower_map[lower_key]]

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Horizon"] = df["Horizon"].map(normalise_horizon)

    fold_rows = []
    summary_rows = []

    for (horizon, model), g_model in df.groupby(["Horizon", "Model"], sort=True):
        if args.benchmark == "model" and model == args.benchmark_model:
            continue

        diffs_by_metric = {"rmse": [], "mae": []}

        for fold, g_fold in g_model.groupby("Fold", sort=True):
            g_fold = g_fold.sort_values("Date").copy()
            actual = g_fold["Actual_Return"].to_numpy(dtype=float)
            pred_model = g_fold["Predicted_Return"].to_numpy(dtype=float)

            if args.benchmark == "zero":
                pred_bench = np.zeros_like(actual)
            else:
                g_bench = df[
                    (df["Horizon"] == horizon)
                    & (df["Model"] == args.benchmark_model)
                    & (df["Fold"] == fold)
                ][["Date", "Predicted_Return"]].rename(
                    columns={"Predicted_Return": "Predicted_Return_Benchmark"}
                )
                merged = g_fold.merge(g_bench, on="Date", how="inner")
                if merged.empty:
                    continue
                actual = merged["Actual_Return"].to_numpy(dtype=float)
                pred_model = merged["Predicted_Return"].to_numpy(dtype=float)
                pred_bench = merged["Predicted_Return_Benchmark"].to_numpy(dtype=float)

            rmse_model = rmse(actual, pred_model)
            rmse_bench = rmse(actual, pred_bench)
            mae_model = mae(actual, pred_model)
            mae_bench = mae(actual, pred_bench)

            rmse_diff = rmse_model - rmse_bench
            mae_diff = mae_model - mae_bench

            diffs_by_metric["rmse"].append(rmse_diff)
            diffs_by_metric["mae"].append(mae_diff)

            fold_rows.append(
                {
                    "Horizon": horizon,
                    "Model": model,
                    "Fold": fold,
                    "RMSE_Diff": rmse_diff,
                    "MAE_Diff": mae_diff,
                }
            )

        for metric, diffs in diffs_by_metric.items():
            diffs = np.asarray(diffs, dtype=float)
            if len(diffs) == 0:
                continue

            nz = diffs[diffs != 0]
            if len(nz) >= 1:
                try:
                    wil_stat, wil_p = stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    wil_stat, wil_p = np.nan, np.nan
            else:
                wil_stat, wil_p = np.nan, np.nan

            sign_p = sign_test_two_sided(diffs)

            mean_diff = float(np.mean(diffs))
            median_diff = float(np.median(diffs))

            if median_diff < 0:
                favours = "model"
            elif median_diff > 0:
                favours = "benchmark"
            else:
                favours = "tie"

            summary_rows.append(
                {
                    "Horizon": horizon,
                    "Model": model,
                    "Metric": metric,
                    "N_folds": len(diffs),
                    "Mean_Fold_Diff": mean_diff,
                    "Median_Fold_Diff": median_diff,
                    "Wilcoxon_stat": wil_stat,
                    "Wilcoxon_pvalue": wil_p,
                    "Sign_Positive": int(np.sum(diffs > 0)),
                    "Sign_Negative": int(np.sum(diffs < 0)),
                    "SignTest_pvalue": sign_p,
                    "Wilcoxon_Favours": favours,
                }
            )

    out_diffs = pd.DataFrame(fold_rows).sort_values(["Horizon", "Model", "Fold"]).reset_index(drop=True)
    out_summary = pd.DataFrame(summary_rows).sort_values(["Horizon", "Model", "Metric"]).reset_index(drop=True)

    Path(args.out_fold_diffs_csv).parent.mkdir(parents=True, exist_ok=True)
    out_diffs.to_csv(args.out_fold_diffs_csv, index=False)
    out_summary.to_csv(args.out_summary_csv, index=False)

    logger.info("Wrote %s", args.out_fold_diffs_csv)
    logger.info("Wrote %s", args.out_summary_csv)


if __name__ == "__main__":
    args = parse_args()
    main(args)
