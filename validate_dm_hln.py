#!/usr/bin/env python3
"""
Diebold-Mariano predictive-accuracy test with Harvey-Leybourne-Newbold correction.

Compares each model against a benchmark on the same dated forecasts.

Default benchmark:
- zero-return benchmark

Required prediction columns:
- Date
- Horizon
- Model
- Actual_Return
- Predicted_Return

Optional:
- Fold

Output:
- results/validation/dm_hln_validation.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import stats


REQUIRED_COLS = {"Date", "Horizon", "Model", "Actual_Return", "Predicted_Return"}


def normalise_horizon(x: object) -> str:
    s = str(x).strip()
    if s in {"1", "1d", "1-day", "1 day"}:
        return "1-day"
    if s in {"3", "3d", "3-day", "3 day"}:
        return "3-day"
    if s in {"7", "7d", "7-day", "7 day"}:
        return "7-day"
    return s


def horizon_to_int(h: str) -> int:
    return int(str(h).split("-")[0].replace("d", "").strip())


def loss_fn(actual: np.ndarray, pred: np.ndarray, loss: str) -> np.ndarray:
    err = actual - pred
    if loss == "mse":
        return err ** 2
    if loss == "mae":
        return np.abs(err)
    raise ValueError("loss must be 'mse' or 'mae'")


def long_run_variance(d: np.ndarray, max_lag: int) -> float:
    d = np.asarray(d, dtype=float)
    T = len(d)
    if T < 2:
        return float("nan")
    x = d - d.mean()
    gamma0 = float(np.dot(x, x) / T)
    var = gamma0
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(x[lag:], x[:-lag]) / T)
        var += 2.0 * cov
    return var


def dm_hln_test(d: np.ndarray, horizon_int: int) -> Tuple[float, float]:
    """
    d_t = loss_model - loss_benchmark
    Negative mean(d) favours the model
    Positive mean(d) favours the benchmark
    """
    d = np.asarray(d, dtype=float)
    T = len(d)
    if T < 5:
        return float("nan"), float("nan")

    q = max(horizon_int - 1, 0)
    lrv = long_run_variance(d, q)
    if not np.isfinite(lrv) or lrv <= 0:
        return float("nan"), float("nan")

    mean_d = d.mean()
    dm = mean_d / math.sqrt(lrv / T)

    h = horizon_int
    hln_factor = math.sqrt((T + 1 - 2 * h + (h * (h - 1) / T)) / T)
    dm_hln = dm * hln_factor
    pvalue = 2.0 * stats.t.sf(abs(dm_hln), df=T - 1)
    return dm_hln, pvalue


def build_benchmark_series(df_model: pd.DataFrame, benchmark_mode: str, benchmark_model: str | None, full_df: pd.DataFrame) -> np.ndarray:
    if benchmark_mode == "zero":
        return np.zeros(len(df_model), dtype=float)

    if benchmark_mode == "model":
        if benchmark_model is None:
            raise ValueError("--benchmark_model must be provided when --benchmark model is used")

        df_bench = (
            full_df[
                (full_df["Model"] == benchmark_model)
                & (full_df["Horizon"] == df_model["Horizon"].iloc[0])
            ][["Date", "Predicted_Return"]]
            .rename(columns={"Predicted_Return": "Predicted_Return_Benchmark"})
        )

        merged = df_model.merge(df_bench, on="Date", how="inner")
        if merged.empty:
            raise ValueError(
                f"No overlapping dates between model={df_model['Model'].iloc[0]} "
                f"and benchmark_model={benchmark_model} for horizon={df_model['Horizon'].iloc[0]}"
            )
        return merged["Predicted_Return_Benchmark"].to_numpy(dtype=float)

    raise ValueError("benchmark_mode must be 'zero' or 'model'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--out_csv", default="results/validation/dm_hln_validation.csv")
    p.add_argument("--benchmark", default="zero", choices=["zero", "model"])
    p.add_argument("--benchmark_model", default=None)
    p.add_argument("--losses", nargs="+", default=["mse", "mae"], choices=["mse", "mae"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
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

    rows = []

    for (horizon, model), g in df.groupby(["Horizon", "Model"], sort=True):
        if args.benchmark == "model" and model == args.benchmark_model:
            continue

        g = g.sort_values("Date").copy()
        actual = g["Actual_Return"].to_numpy(dtype=float)
        pred_model = g["Predicted_Return"].to_numpy(dtype=float)

        if args.benchmark == "zero":
            pred_bench = np.zeros_like(actual)
        else:
            bench_df = (
                df[(df["Model"] == args.benchmark_model) & (df["Horizon"] == horizon)]
                .sort_values("Date")[["Date", "Predicted_Return"]]
                .rename(columns={"Predicted_Return": "Predicted_Return_Benchmark"})
            )
            merged = g.merge(bench_df, on="Date", how="inner")
            if merged.empty:
                continue
            actual = merged["Actual_Return"].to_numpy(dtype=float)
            pred_model = merged["Predicted_Return"].to_numpy(dtype=float)
            pred_bench = merged["Predicted_Return_Benchmark"].to_numpy(dtype=float)

        h_int = horizon_to_int(horizon)

        for loss in args.losses:
            model_loss = loss_fn(actual, pred_model, loss=loss)
            bench_loss = loss_fn(actual, pred_bench, loss=loss)
            d = model_loss - bench_loss

            stat, pval = dm_hln_test(d, horizon_int=h_int)
            mean_loss_diff = float(np.mean(d))

            if np.isnan(mean_loss_diff):
                favours = "undetermined"
            elif mean_loss_diff < 0:
                favours = "model"
            elif mean_loss_diff > 0:
                favours = "benchmark"
            else:
                favours = "tie"

            rows.append(
                {
                    "Horizon": horizon,
                    "Model": model,
                    "Metric": "rmse" if loss == "mse" else "mae",
                    "Loss_Function": loss,
                    "N_obs": len(d),
                    "Lag_Used": max(h_int - 1, 0),
                    "Mean_Loss_Diff": mean_loss_diff,
                    "DM_stat": stat,
                    "DM_pvalue": pval,
                    "DM_Favours": favours,
                }
            )

    out = pd.DataFrame(rows).sort_values(["Horizon", "Model", "Metric"]).reset_index(drop=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
