"""Evaluate uncertainty-filtered trading strategies and export performance metrics.

Loads predictions with uncertainty estimates, applies confidence-aware filters,
backtests trading strategies, and computes risk-adjusted performance metrics.
Exports detailed performance outputs.
"""

import os
import argparse
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
SMOKE_TEST_SAFE = True


# -----------------------------
# Project paths and default output
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "results", "uncertainty_trading_evaluation")
DEFAULT_FEATURES_FILE = os.path.join(
    BASE_DIR, "data", "processed", "btc_final_features_with_llm_uncertainty.parquet"
)


# -----------------------------
# CLI arguments
# -----------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty-filtered trading strategies."
    )

    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions CSV, e.g. results/final_best_model/predictions_final_best.csv",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=DEFAULT_FEATURES_FILE,
        help="Path to uncertainty feature parquet file.",
    )
    parser.add_argument(
        "--horizon",
        type=str,
        required=True,
        choices=["1d", "3d", "7d"],
        help="Forecast horizon to evaluate.",
    )
    parser.add_argument(
        "--transaction_cost",
        type=float,
        default=0.001,
        help="Cost per unit turnover, e.g. 0.001 = 0.1%%",
    )
    parser.add_argument(
        "--prediction_thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.002, 0.005],
        help="Predicted return thresholds.",
    )
    parser.add_argument(
        "--disagreement_quantiles",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75],
        help="Quantiles used to define low-disagreement filters.",
    )
    parser.add_argument(
        "--annualization_factor",
        type=float,
        default=365.0,
        help="Annualization factor for BTC daily data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )

    return parser.parse_args(argv)


# -----------------------------
# Performance utility functions
# -----------------------------
def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def annualized_return(strategy_returns: pd.Series, annualization_factor: float) -> float:
    strategy_returns = strategy_returns.dropna()
    if len(strategy_returns) == 0:
        return 0.0

    total_return = float((1.0 + strategy_returns).prod())
    periods = len(strategy_returns)

    if total_return <= 0:
        return -1.0

    return float(total_return ** (annualization_factor / periods) - 1.0)


def annualized_volatility(strategy_returns: pd.Series, annualization_factor: float) -> float:
    strategy_returns = strategy_returns.dropna()
    if len(strategy_returns) <= 1:
        return 0.0
    return float(strategy_returns.std(ddof=1) * np.sqrt(annualization_factor))


def sharpe_ratio(strategy_returns: pd.Series, annualization_factor: float) -> float:
    vol = annualized_volatility(strategy_returns, annualization_factor)
    if vol == 0:
        return 0.0
    ann_ret = annualized_return(strategy_returns, annualization_factor)
    return float(ann_ret / vol)


# -----------------------------
# Prediction and feature loading
# -----------------------------
def load_predictions(predictions_path: str, horizon: str) -> pd.DataFrame:
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df = pd.read_csv(predictions_path)
    if "Date" not in df.columns:
        raise ValueError("Predictions file must contain Date column.")
    if "horizon" not in df.columns:
        raise ValueError("Predictions file must contain horizon column.")

    required = {"actual_return", "predicted_return_lstm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df[df["horizon"] == horizon].copy()

    if df.empty:
        raise ValueError(f"No prediction rows found for horizon {horizon}.")

    df = df.sort_values(["Date"]).reset_index(drop=True)
    return df


# -----------------------------
# Uncertainty feature preparation
# -----------------------------
def load_uncertainty_features(features_path: str, horizon: str) -> pd.DataFrame:
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")

    df = pd.read_parquet(features_path).copy()
    if "Date" not in df.columns:
        raise ValueError("Features file must contain Date column.")

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    candidates = [
        f"{horizon}_disagreement_index",
        f"{horizon}_agreement_index",
        f"{horizon}_consensus_strength",
        f"{horizon}_disagreement_sentiment",
        f"{horizon}_disagreement_impact",
        f"{horizon}_disagreement_uncertainty",
        "llm_disagreement_index",
        "llm_agreement_index",
        "llm_consensus_strength",
        "llm_llm_sentiment_score_std",
        "llm_llm_market_impact_score_std",
        "llm_llm_uncertainty_score_std",
    ]
    keep = ["Date"] + [c for c in candidates if c in df.columns]
    if len(keep) == 1:
        raise ValueError("No uncertainty feature columns found in features file.")

    out = df[keep].copy()
    out = out.groupby("Date", as_index=False).first()
    return out


def build_merged_frame(pred_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.merge(feat_df, on="Date", how="left")
    df = df.sort_values("Date").reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c != "Date" and pd.api.types.is_numeric_dtype(df[c])]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    if df[numeric_cols].isna().any().any():
        df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    return df


# -----------------------------
# Strategy performance construction
# -----------------------------
def compute_strategy_frame(
    df: pd.DataFrame,
    position: pd.Series,
    transaction_cost: float,
) -> pd.DataFrame:
    out = df.copy()
    out["position"] = position.astype(float)
    out["gross_strategy_return"] = out["position"] * out["actual_return"]

    prev_position = out["position"].shift(1).fillna(0.0)
    out["turnover"] = (out["position"] - prev_position).abs()
    out["transaction_cost"] = transaction_cost * out["turnover"]
    out["net_strategy_return"] = out["gross_strategy_return"] - out["transaction_cost"]

    out["equity_curve"] = (1.0 + out["net_strategy_return"]).cumprod()
    out["buy_hold_equity_curve"] = (1.0 + out["actual_return"]).cumprod()

    running_max = out["equity_curve"].cummax()
    out["drawdown"] = out["equity_curve"] / running_max - 1.0

    bh_running_max = out["buy_hold_equity_curve"].cummax()
    out["buy_hold_drawdown"] = out["buy_hold_equity_curve"] / bh_running_max - 1.0

    return out


def summarize_strategy(
    strategy_df: pd.DataFrame,
    strategy_name: str,
    annualization_factor: float,
) -> Dict[str, float]:
    net_returns = strategy_df["net_strategy_return"]
    actual_returns = strategy_df["actual_return"]
    active_mask = strategy_df["position"] != 0

    if active_mask.sum() > 0:
        hit_rate = float(
            (np.sign(strategy_df.loc[active_mask, "position"]) == np.sign(strategy_df.loc[active_mask, "actual_return"])).mean()
        )
    else:
        hit_rate = 0.0

    return {
        "strategy": strategy_name,
        "n_periods": int(len(strategy_df)),
        "n_active_periods": int(active_mask.sum()),
        "trade_count_proxy": int((strategy_df["turnover"] > 0).sum()),
        "avg_turnover": float(strategy_df["turnover"].mean()),
        "cumulative_return": float(strategy_df["equity_curve"].iloc[-1] - 1.0),
        "annualized_return": annualized_return(net_returns, annualization_factor),
        "annualized_volatility": annualized_volatility(net_returns, annualization_factor),
        "sharpe_ratio": sharpe_ratio(net_returns, annualization_factor),
        "max_drawdown": max_drawdown(strategy_df["equity_curve"]),
        "mean_daily_return": float(net_returns.mean()),
        "std_daily_return": float(net_returns.std(ddof=1)) if len(net_returns) > 1 else 0.0,
        "hit_rate_active_periods": hit_rate,
        "buy_hold_cumulative_return": float(strategy_df["buy_hold_equity_curve"].iloc[-1] - 1.0),
        "buy_hold_annualized_return": annualized_return(actual_returns, annualization_factor),
        "buy_hold_annualized_volatility": annualized_volatility(actual_returns, annualization_factor),
        "buy_hold_sharpe_ratio": sharpe_ratio(actual_returns, annualization_factor),
        "buy_hold_max_drawdown": max_drawdown(strategy_df["buy_hold_equity_curve"]),
    }


def get_best_available_columns(df: pd.DataFrame, horizon: str) -> Tuple[str, str, str]:
    disagreement_candidates = [
        f"{horizon}_disagreement_index",
        "llm_disagreement_index",
        f"{horizon}_disagreement_sentiment",
        "llm_llm_sentiment_score_std",
    ]
    agreement_candidates = [
        f"{horizon}_agreement_index",
        "llm_agreement_index",
    ]
    consensus_candidates = [
        f"{horizon}_consensus_strength",
        "llm_consensus_strength",
    ]

    def first_existing(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    disagreement_col = first_existing(disagreement_candidates)
    agreement_col = first_existing(agreement_candidates)
    consensus_col = first_existing(consensus_candidates)

    if disagreement_col is None:
        raise ValueError("No disagreement column found.")
    if agreement_col is None:
        agreement_col = disagreement_col
    if consensus_col is None:
        consensus_col = disagreement_col

    return disagreement_col, agreement_col, consensus_col


# -----------------------------
# Uncertainty-filtered strategy generation
# -----------------------------
def build_uncertainty_strategies(
    df: pd.DataFrame,
    horizon: str,
    prediction_thresholds: List[float],
    disagreement_quantiles: List[float],
    transaction_cost: float,
    annualization_factor: float,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}

    disagreement_col, agreement_col, consensus_col = get_best_available_columns(df, horizon)
    disagreement_series = df[disagreement_col].astype(float)
    agreement_series = df[agreement_col].astype(float)
    consensus_series = df[consensus_col].astype(float)

    pred = df["predicted_return_lstm"].astype(float)

    for q in disagreement_quantiles:
        cutoff = float(disagreement_series.quantile(q))
        low_disagreement_mask = disagreement_series <= cutoff

        for pth in prediction_thresholds:
            # Long-only filtered
            pos_long = np.where((pred > pth) & low_disagreement_mask, 1.0, 0.0)
            name_long = f"unc_long_q{int(q*100):02d}_pth{pth:.3f}"
            long_df = compute_strategy_frame(df, pd.Series(pos_long, index=df.index), transaction_cost)
            results[name_long] = {
                "frame": long_df,
                "summary": summarize_strategy(long_df, name_long, annualization_factor),
            }

            # Long-short filtered
            pos_ls = np.where(
                (pred > pth) & low_disagreement_mask,
                1.0,
                np.where((pred < -pth) & low_disagreement_mask, -1.0, 0.0),
            )
            name_ls = f"unc_longshort_q{int(q*100):02d}_pth{pth:.3f}"
            ls_df = compute_strategy_frame(df, pd.Series(pos_ls, index=df.index), transaction_cost)
            results[name_ls] = {
                "frame": ls_df,
                "summary": summarize_strategy(ls_df, name_ls, annualization_factor),
            }

            # Confidence-weighted long-only
            strength = (1.0 - disagreement_series / max(disagreement_series.max(), 1e-9)).clip(lower=0.0, upper=1.0)
            pos_weighted_long = np.where((pred > pth) & low_disagreement_mask, strength, 0.0)
            name_wlong = f"unc_weighted_long_q{int(q*100):02d}_pth{pth:.3f}"
            wlong_df = compute_strategy_frame(df, pd.Series(pos_weighted_long, index=df.index), transaction_cost)
            results[name_wlong] = {
                "frame": wlong_df,
                "summary": summarize_strategy(wlong_df, name_wlong, annualization_factor),
            }

            # Consensus-weighted long-short
            cons = consensus_series.copy().astype(float)
            cons = cons.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            cons = cons.clip(lower=0.0, upper=1.0) if cons.max() <= 1.5 else (cons / max(cons.max(), 1e-9)).clip(0.0, 1.0)

            sign = np.where(pred > pth, 1.0, np.where(pred < -pth, -1.0, 0.0))
            pos_consensus = np.where(low_disagreement_mask, sign * cons, 0.0)
            name_cons = f"unc_consensus_longshort_q{int(q*100):02d}_pth{pth:.3f}"
            cons_df = compute_strategy_frame(df, pd.Series(pos_consensus, index=df.index), transaction_cost)
            results[name_cons] = {
                "frame": cons_df,
                "summary": summarize_strategy(cons_df, name_cons, annualization_factor),
            }

    # Baseline long-only from your previous best style
    baseline_long = np.where(pred > 0, 1.0, 0.0)
    baseline_name = "baseline_lstm_long_only"
    baseline_df = compute_strategy_frame(df, pd.Series(baseline_long, index=df.index), transaction_cost)
    results[baseline_name] = {
        "frame": baseline_df,
        "summary": summarize_strategy(baseline_df, baseline_name, annualization_factor),
    }

    return results


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_equity_curves(strategy_results: Dict[str, pd.DataFrame], output_path: str, title: str, top_n: int = 8) -> None:
    summaries = []
    for name, frame in strategy_results.items():
        ret = float(frame["equity_curve"].iloc[-1] - 1.0)
        summaries.append((name, ret))
    top_names = [x[0] for x in sorted(summaries, key=lambda z: z[1], reverse=True)[:top_n]]

    plt.figure(figsize=(12, 7))
    buy_hold_plotted = False

    for strategy_name in top_names:
        df = strategy_results[strategy_name]
        plt.plot(df["Date"], df["equity_curve"], label=strategy_name)
        if not buy_hold_plotted:
            plt.plot(df["Date"], df["buy_hold_equity_curve"], linestyle="--", label="buy_and_hold")
            buy_hold_plotted = True

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity Curve")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_drawdowns(strategy_results: Dict[str, pd.DataFrame], output_path: str, title: str, top_n: int = 8) -> None:
    summaries = []
    for name, frame in strategy_results.items():
        ret = float(frame["equity_curve"].iloc[-1] - 1.0)
        summaries.append((name, ret))
    top_names = [x[0] for x in sorted(summaries, key=lambda z: z[1], reverse=True)[:top_n]]

    plt.figure(figsize=(12, 7))
    buy_hold_plotted = False

    for strategy_name in top_names:
        df = strategy_results[strategy_name]
        plt.plot(df["Date"], df["drawdown"], label=strategy_name)
        if not buy_hold_plotted:
            plt.plot(df["Date"], df["buy_hold_drawdown"], linestyle="--", label="buy_and_hold")
            buy_hold_plotted = True

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("SCRIPT STARTED")

    pred_df = load_predictions(args.predictions, args.horizon)
    feat_df = load_uncertainty_features(args.features, args.horizon)
    df = build_merged_frame(pred_df, feat_df)

    output_dir = os.path.join(args.output_dir, f"horizon_{args.horizon}")
    os.makedirs(output_dir, exist_ok=True)

    strategy_results = build_uncertainty_strategies(
        df=df,
        horizon=args.horizon,
        prediction_thresholds=sorted(set(args.prediction_thresholds)),
        disagreement_quantiles=sorted(set(args.disagreement_quantiles)),
        transaction_cost=args.transaction_cost,
        annualization_factor=args.annualization_factor,
    )

    metrics = [payload["summary"] for payload in strategy_results.values()]
    metrics_df = pd.DataFrame(metrics).sort_values(
        by=["sharpe_ratio", "cumulative_return"],
        ascending=False,
    ).reset_index(drop=True)

    metrics_csv = os.path.join(output_dir, "uncertainty_trading_metrics.csv")
    metrics_json = os.path.join(output_dir, "uncertainty_trading_metrics.json")
    metrics_df.to_csv(metrics_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2, default=str)

    frames_dir = os.path.join(output_dir, "strategy_frames")
    os.makedirs(frames_dir, exist_ok=True)
    for strategy_name, payload in strategy_results.items():
        payload["frame"].to_csv(os.path.join(frames_dir, f"{strategy_name}.csv"), index=False)

    equity_plot = os.path.join(output_dir, "uncertainty_equity_curves.png")
    drawdown_plot = os.path.join(output_dir, "uncertainty_drawdowns.png")

    plot_equity_curves(
        {k: v["frame"] for k, v in strategy_results.items()},
        equity_plot,
        f"Uncertainty-Filtered Equity Curves ({args.horizon}, cost={args.transaction_cost})",
    )
    plot_drawdowns(
        {k: v["frame"] for k, v in strategy_results.items()},
        drawdown_plot,
        f"Uncertainty-Filtered Drawdowns ({args.horizon}, cost={args.transaction_cost})",
    )

    logger.info("===== UNCERTAINTY TRADING SUMMARY =====")
    logger.info("%s", metrics_df.head(15))

    if not metrics_df.empty:
        logger.info("Best strategy by Sharpe: %s", metrics_df.iloc[0].to_dict())

    logger.info("Saved metrics to: %s", metrics_csv)
    logger.info("Saved metrics JSON to: %s", metrics_json)
    logger.info("Saved strategy frames to: %s", frames_dir)
    logger.info("Saved equity plot to: %s", equity_plot)
    logger.info("Saved drawdown plot to: %s", drawdown_plot)


if __name__ == "__main__":
    args = parse_args()
    main(args)