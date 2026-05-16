"""Evaluate trading strategies derived from prediction CSVs.

Loads model predictions, backtests trading strategies with configurable entry
and exit thresholds, computes performance metrics (returns, Sharpe ratio, Sortino),
and exports performance summaries.
"""

import os
import argparse
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
SMOKE_TEST_SAFE = True


# -----------------------------
# Project paths and default output
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "results", "trading_evaluation")


# -----------------------------
# CLI arguments
# -----------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trading strategies from prediction CSVs.")

    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions CSV, e.g. results/final_best_model/predictions_final_best.csv",
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default=None,
        help="Filter to a single horizon, e.g. 1d, 3d, 7d",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Optional provider filter for files that contain provider column, e.g. openai, claude, gemini",
    )
    parser.add_argument(
        "--transaction_cost",
        type=float,
        default=0.001,
        help="Cost applied per unit turnover, e.g. 0.001 = 0.1%%",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.002, 0.005],
        help="Prediction thresholds for threshold strategies",
    )
    parser.add_argument(
        "--annualization_factor",
        type=float,
        default=365.0,
        help="Annualization factor. For BTC daily data, 365 is reasonable.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save metrics and plots",
    )

    return parser.parse_args(argv)


# -----------------------------
# Helper utilities
# -----------------------------
def safe_name(value: Optional[str]) -> str:
    if value is None or value == "":
        return "all"
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


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


def build_positions_from_threshold(predicted_returns: pd.Series, threshold: float) -> pd.Series:
    positions = np.where(predicted_returns > threshold, 1, np.where(predicted_returns < -threshold, -1, 0))
    return pd.Series(positions, index=predicted_returns.index, dtype=float)


# -----------------------------
# Strategy performance construction
# -----------------------------
def compute_strategy_frame(
    df: pd.DataFrame,
    position_col: str,
    actual_return_col: str,
    transaction_cost: float,
) -> pd.DataFrame:
    out = df.copy()

    out["position"] = out[position_col].astype(float)
    out["actual_return"] = out[actual_return_col].astype(float)

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
        active_hit_rate = float(
            (np.sign(strategy_df.loc[active_mask, "position"]) == np.sign(strategy_df.loc[active_mask, "actual_return"])).mean()
        )
    else:
        active_hit_rate = 0.0

    summary = {
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
        "hit_rate_active_periods": active_hit_rate,
        "buy_hold_cumulative_return": float(strategy_df["buy_hold_equity_curve"].iloc[-1] - 1.0),
        "buy_hold_annualized_return": annualized_return(actual_returns, annualization_factor),
        "buy_hold_annualized_volatility": annualized_volatility(actual_returns, annualization_factor),
        "buy_hold_sharpe_ratio": sharpe_ratio(actual_returns, annualization_factor),
        "buy_hold_max_drawdown": max_drawdown(strategy_df["buy_hold_equity_curve"]),
    }
    return summary


# -----------------------------
# Plotting utilities
# -----------------------------
def plot_equity_curves(
    strategy_results: Dict[str, pd.DataFrame],
    output_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(11, 6))

    buy_hold_plotted = False
    for strategy_name, df in strategy_results.items():
        plt.plot(df["Date"], df["equity_curve"], label=strategy_name)
        if not buy_hold_plotted:
            plt.plot(df["Date"], df["buy_hold_equity_curve"], linestyle="--", label="buy_and_hold")
            buy_hold_plotted = True

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_drawdowns(
    strategy_results: Dict[str, pd.DataFrame],
    output_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(11, 6))

    buy_hold_plotted = False
    for strategy_name, df in strategy_results.items():
        plt.plot(df["Date"], df["drawdown"], label=strategy_name)
        if not buy_hold_plotted:
            plt.plot(df["Date"], df["buy_hold_drawdown"], linestyle="--", label="buy_and_hold")
            buy_hold_plotted = True

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Data loading and filtering
# -----------------------------
def load_and_filter_predictions(
    predictions_path: str,
    horizon: Optional[str],
    provider: Optional[str],
) -> pd.DataFrame:
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df = pd.read_csv(predictions_path)
    if "Date" not in df.columns:
        raise ValueError("Predictions file must contain Date column.")

    df["Date"] = pd.to_datetime(df["Date"])

    if horizon is not None:
        if "horizon" not in df.columns:
            raise ValueError("This predictions file does not contain a horizon column.")
        df = df[df["horizon"] == horizon].copy()

    if provider is not None:
        if "provider" not in df.columns:
            raise ValueError("This predictions file does not contain a provider column.")
        df = df[df["provider"].astype(str).str.lower() == provider.lower()].copy()

    if df.empty:
        raise ValueError("No rows left after filtering predictions.")

    sort_cols: List[str] = ["Date"]
    if "fold" in df.columns:
        sort_cols = ["fold", "Date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    required_cols = {"actual_return", "predicted_return_lstm"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing required columns: {missing}")

    return df


# -----------------------------
# Strategy generation and evaluation
# -----------------------------
def evaluate_all_strategies(
    df: pd.DataFrame,
    thresholds: List[float],
    transaction_cost: float,
    annualization_factor: float,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}

    base = df.copy()

    # Strategy 1: sign-based always in market
    sign_positions = np.where(base["predicted_return_lstm"] > 0, 1.0, -1.0)
    base["position_sign"] = sign_positions
    sign_df = compute_strategy_frame(base, "position_sign", "actual_return", transaction_cost)
    results["lstm_sign"] = {
        "frame": sign_df,
        "summary": summarize_strategy(sign_df, "lstm_sign", annualization_factor),
    }

    # Strategy 2: long-only on positive predictions
    long_only_positions = np.where(base["predicted_return_lstm"] > 0, 1.0, 0.0)
    base["position_long_only"] = long_only_positions
    long_only_df = compute_strategy_frame(base, "position_long_only", "actual_return", transaction_cost)
    results["lstm_long_only"] = {
        "frame": long_only_df,
        "summary": summarize_strategy(long_only_df, "lstm_long_only", annualization_factor),
    }

    # Strategy 3: threshold strategies
    for threshold in thresholds:
        col = f"position_threshold_{str(threshold).replace('.', 'p')}"
        strat_name = f"lstm_threshold_{threshold:.3f}"

        base[col] = build_positions_from_threshold(base["predicted_return_lstm"], threshold)
        strat_df = compute_strategy_frame(base, col, "actual_return", transaction_cost)

        results[strat_name] = {
            "frame": strat_df,
            "summary": summarize_strategy(strat_df, strat_name, annualization_factor),
        }

    return results


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    predictions_path = args.predictions
    horizon = args.horizon
    provider = args.provider
    transaction_cost = args.transaction_cost
    thresholds = sorted(set(args.thresholds))
    annualization_factor = args.annualization_factor

    df = load_and_filter_predictions(predictions_path, horizon, provider)

    tag = f"horizon_{safe_name(horizon)}__provider_{safe_name(provider)}"
    output_dir = os.path.join(args.output_dir, tag)
    os.makedirs(output_dir, exist_ok=True)

    strategy_results = evaluate_all_strategies(
        df=df,
        thresholds=thresholds,
        transaction_cost=transaction_cost,
        annualization_factor=annualization_factor,
    )

    summaries = [v["summary"] for v in strategy_results.values()]
    metrics_df = pd.DataFrame(summaries).sort_values(
        by=["sharpe_ratio", "cumulative_return"],
        ascending=False,
    ).reset_index(drop=True)

    metrics_csv = os.path.join(output_dir, "trading_metrics.csv")
    metrics_json = os.path.join(output_dir, "trading_metrics.json")
    metrics_df.to_csv(metrics_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2, default=str)

    frame_output_dir = os.path.join(output_dir, "strategy_frames")
    os.makedirs(frame_output_dir, exist_ok=True)
    for strategy_name, payload in strategy_results.items():
        strategy_path = os.path.join(frame_output_dir, f"{strategy_name}.csv")
        payload["frame"].to_csv(strategy_path, index=False)

    title_suffix = f"horizon={safe_name(horizon)}, provider={safe_name(provider)}, cost={transaction_cost}"
    equity_plot = os.path.join(output_dir, "equity_curves.png")
    drawdown_plot = os.path.join(output_dir, "drawdowns.png")

    plot_equity_curves(
        {k: v["frame"] for k, v in strategy_results.items()},
        equity_plot,
        f"Equity Curves ({title_suffix})",
    )

    plot_drawdowns(
        {k: v["frame"] for k, v in strategy_results.items()},
        drawdown_plot,
        f"Drawdowns ({title_suffix})",
    )

    logger.info("===== TRADING STRATEGY SUMMARY =====")
    logger.info("%s", metrics_df)

    if not metrics_df.empty:
        best_row = metrics_df.iloc[0]
        logger.info("Best strategy by Sharpe ratio: %s", best_row.to_dict())

    logger.info("Saved trading metrics to: %s", metrics_csv)
    logger.info("Saved trading metrics JSON to: %s", metrics_json)
    logger.info("Saved strategy frames to: %s", frame_output_dir)
    logger.info("Saved equity curve plot to: %s", equity_plot)
    logger.info("Saved drawdown plot to: %s", drawdown_plot)


if __name__ == "__main__":
    args = parse_args()
    main(args)