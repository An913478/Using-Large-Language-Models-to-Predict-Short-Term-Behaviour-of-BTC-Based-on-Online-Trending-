#!/usr/bin/env python3
"""
aggregate_repeated_seed_results.py
===================================

Aggregate results from repeated seed runs into final CSVs.

Outputs:
- all_repeated_seed_predictions.csv: All prediction rows from all seeds, horizons, and models.
- seed_level_metrics.csv: One row per seed per selected configuration.
- repeated_seed_summary.csv: Mean ± standard deviation across seeds for each selected configuration.
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def main(args: argparse.Namespace) -> None:
    """Collect and aggregate predictions and metrics from seed subfolders into summary CSVs."""
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    all_metrics = []

    # Collect from subfolders
    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue

        pred_file = subdir / "predictions_multi_horizon.csv"  # For LSTM
        if not pred_file.exists():
            pred_file = subdir / "predictions_return_direction_dcn.csv"  # For DCN

        metrics_file = subdir / "summary_multi_horizon.csv"
        if not metrics_file.exists():
            metrics_file = subdir / "summary_return_direction_dcn.csv"

        if pred_file.exists():
            pred_df = pd.read_csv(pred_file)
            # Add configuration column
            h = pred_df['horizon'].iloc[0].replace('d', '')
            if "predicted_return_lstm" in pred_df.columns:
                pred_df["model"] = "lstm"
                pred_df["configuration"] = f"{h}_lstm"
            elif "predicted_return_dcn" in pred_df.columns:
                pred_df["model"] = "dcn"
                pred_df["configuration"] = f"{h}_dcn"
            all_predictions.append(pred_df)

        if metrics_file.exists():
            metrics_df = pd.read_csv(metrics_file)
            # Add configuration
            h = metrics_df['horizon'].iloc[0].replace('d', '')
            if "rmse_lstm_mean" in metrics_df.columns:
                metrics_df["model"] = "lstm"
                metrics_df["configuration"] = f"{h}_lstm"
                metrics_df = metrics_df.rename(columns={"rmse_lstm_mean": "rmse_mean", "mae_lstm_mean": "mae_mean", "f1_lstm_mean": "f1_mean"})
            elif "rmse_dcn_mean" in metrics_df.columns:
                metrics_df["model"] = "dcn"
                metrics_df["configuration"] = f"{h}_dcn"
                metrics_df = metrics_df.rename(columns={"rmse_dcn_mean": "rmse_mean", "mae_dcn_mean": "mae_mean", "f1_dcn_mean": "f1_mean"})
            all_metrics.append(metrics_df)

    # Concatenate
    if all_predictions:
        all_pred_df = pd.concat(all_predictions, ignore_index=True)
        all_pred_path = out_dir / "all_repeated_seed_predictions.csv"
        all_pred_df.to_csv(all_pred_path, index=False)
        print(f"Saved {all_pred_path}")

    if all_metrics:
        seed_level_df = pd.concat(all_metrics, ignore_index=True)
        seed_level_path = out_dir / "seed_level_metrics.csv"
        seed_level_df.to_csv(seed_level_path, index=False)
        print(f"Saved {seed_level_path}")

        # Compute summary with mean ± std
        summary_df = seed_level_df.groupby("configuration").agg(
            rmse_mean=("rmse_mean", "mean"),
            rmse_std=("rmse_mean", "std"),
            mae_mean=("mae_mean", "mean"),
            mae_std=("mae_mean", "std"),
            f1_mean=("f1_mean", "mean"),
            f1_std=("f1_mean", "std"),
            num_seeds=("seed", "count"),
        ).reset_index()

        summary_path = out_dir / "repeated_seed_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved {summary_path}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for aggregating repeated seed results."""
    parser = argparse.ArgumentParser(description="Aggregate repeated seed results.")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing seed subfolders.")
    parser.add_argument("--out", type=str, required=True, help="Output directory for aggregated CSVs.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(args)
