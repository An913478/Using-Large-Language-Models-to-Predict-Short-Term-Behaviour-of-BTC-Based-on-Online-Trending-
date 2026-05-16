#!/usr/bin/env python3
"""Moving-block bootstrap confidence intervals for benchmark-relative forecasting skill.

Computes dependence-aware confidence intervals for RMSE and MAE skill metrics
relative to zero-return benchmark. Outputs bootstrap summaries to
results/validation/bootstrap_skill_summary.csv.

Required prediction columns:
- Date
- Horizon
- Model
- Actual_Return
- Predicted_Return

Output:
- results/validation/bootstrap_skill_summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
SMOKE_TEST_SAFE = True


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


def rmse(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(a - p)))


def moving_block_indices(T: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    starts = np.arange(0, T - block_length + 1)
    idx = []
    while len(idx) < T:
        s = int(rng.choice(starts))
        idx.extend(range(s, s + block_length))
    return np.asarray(idx[:T], dtype=int)


def bootstrap_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(np.quantile(x, alpha / 2))
    hi = float(np.quantile(x, 1 - alpha / 2))
    return lo, hi


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--out_csv", default="results/validation/bootstrap_skill_summary.csv")
    p.add_argument("--benchmark", default="zero", choices=["zero", "model"])
    p.add_argument("--benchmark_model", default=None)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--block_length", type=int, default=0,
                   help="If 0, uses max(horizon, ceil(T^(1/3))).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.predictions_csv)
    df.columns = [c.strip() for c in df.columns]

    lower_map = {c.lower(): c for c in df.columns}
    missing = []
    for req in REQUIRED_COLS:
        if req.lower() not in lower_map:
            missing.append(req)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for req in REQUIRED_COLS:
        df[req] = df[lower_map[req.lower()]]

    df["Date"] = pd.to_datetime(df["Date"])
    df["Horizon"] = df["Horizon"].map(normalise_horizon)

    rows = []

    for (horizon, model), g_model in df.groupby(["Horizon", "Model"], sort=True):
        if args.benchmark == "model" and model == args.benchmark_model:
            continue

        g_model = g_model.sort_values("Date").copy()
        actual = g_model["Actual_Return"].to_numpy(dtype=float)
        pred_model = g_model["Predicted_Return"].to_numpy(dtype=float)

        if args.benchmark == "zero":
            pred_bench = np.zeros_like(actual)
        else:
            g_bench = (
                df[(df["Horizon"] == horizon) & (df["Model"] == args.benchmark_model)]
                .sort_values("Date")[["Date", "Predicted_Return"]]
                .rename(columns={"Predicted_Return": "Predicted_Return_Benchmark"})
            )
            merged = g_model.merge(g_bench, on="Date", how="inner")
            if merged.empty:
                continue
            actual = merged["Actual_Return"].to_numpy(dtype=float)
            pred_model = merged["Predicted_Return"].to_numpy(dtype=float)
            pred_bench = merged["Predicted_Return_Benchmark"].to_numpy(dtype=float)

        T = len(actual)
        h_int = horizon_to_int(horizon)
        block_length = args.block_length if args.block_length > 0 else max(h_int, int(np.ceil(T ** (1 / 3))))

        rmse_model = rmse(actual, pred_model)
        rmse_bench = rmse(actual, pred_bench)
        mae_model = mae(actual, pred_model)
        mae_bench = mae(actual, pred_bench)

        rmse_skill = 1.0 - (rmse_model / rmse_bench)
        mae_skill = 1.0 - (mae_model / mae_bench)

        boot_rmse_skill = np.empty(args.n_boot, dtype=float)
        boot_mae_skill = np.empty(args.n_boot, dtype=float)

        for b in range(args.n_boot):
            idx = moving_block_indices(T, block_length, rng)
            a_b = actual[idx]
            pm_b = pred_model[idx]
            pb_b = pred_bench[idx]

            rmse_m_b = rmse(a_b, pm_b)
            rmse_n_b = rmse(a_b, pb_b)
            mae_m_b = mae(a_b, pm_b)
            mae_n_b = mae(a_b, pb_b)

            boot_rmse_skill[b] = 1.0 - (rmse_m_b / rmse_n_b)
            boot_mae_skill[b] = 1.0 - (mae_m_b / mae_n_b)

        rmse_lo, rmse_hi = bootstrap_ci(boot_rmse_skill)
        mae_lo, mae_hi = bootstrap_ci(boot_mae_skill)

        rows.append(
            {
                "Horizon": horizon,
                "Model": model,
                "N_obs": T,
                "Block_Length": block_length,
                "N_Boot": args.n_boot,
                "RMSE_Model": rmse_model,
                "RMSE_Naive": rmse_bench,
                "RMSE_Skill": rmse_skill,
                "RMSE_Skill_CI_Low": rmse_lo,
                "RMSE_Skill_CI_High": rmse_hi,
                "MAE_Model": mae_model,
                "MAE_Naive": mae_bench,
                "MAE_Skill": mae_skill,
                "MAE_Skill_CI_Low": mae_lo,
                "MAE_Skill_CI_High": mae_hi,
            }
        )

    out = pd.DataFrame(rows).sort_values(["Horizon", "Model"]).reset_index(drop=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    logger.info("Wrote %s", args.out_csv)


if __name__ == "__main__":
    args = parse_args()
    main(args)
