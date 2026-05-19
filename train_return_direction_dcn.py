"""Train DCN/LSTM models for multi-horizon BTC return and direction forecasting.

Implements multi-horizon (1d/3d/7d) walk-forward validation with LSTM and DCN
architectures for both return regression and direction classification. Trains
models on folds 1–9, produces fold metrics, and aggregates summary statistics
per horizon.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler

# Defaults
INPUT_FILE = "data/processed/btc_features.parquet"
RESULTS_DIR = "results/return_direction_dcn"

SEQUENCE_LENGTH = 14
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
RANDOM_SEED = 42  # Set for reproducibility; override via --seed

INITIAL_TRAIN_RATIO = 0.60
TEST_WINDOW = 60
STEP_SIZE = 60
MAX_FOLDS = 9
EMBARGO_GAP = 0  # Days to reserve before each test window; override via --embargo-gap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =========================
# Models
# =========================

class LSTMRegressor(nn.Module):
    """LSTM encoder for return regression."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class DCNRegressor(nn.Module):
    """Deep Cross Network (DCN) for return regression."""

    def __init__(self, input_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class LSTMClassifier(nn.Module):
    """LSTM encoder for direction classification."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class DCNClassifier(nn.Module):
    """Deep Cross Network (DCN) for direction classification."""

    def __init__(self, input_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# =========================
# Helper Functions
# =========================


def create_sequences(
    features: np.ndarray, targets: np.ndarray, dates: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences for LSTM/DCN training.

    Args:
        features: Feature matrix [n_samples, n_features].
        targets: Target vector [n_samples].
        dates: Date array [n_samples].
        seq_len: Sequence length.

    Returns:
        X, y, dates_out: Sequences, targets, and corresponding dates.
    """
    X, y, d = [], [], []
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len])
        d.append(dates[i + seq_len])
    return np.array(X), np.array(y), np.array(d)


def train_regression_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Train regression model and make predictions.

    Args:
        model: PyTorch model (LSTM or DCN regressor).
        X_train: Training sequences [n_train, seq_len, n_features].
        y_train: Training targets [n_train].
        X_test: Test sequences [n_test, seq_len, n_features].

    Returns:
        Predictions on test set [n_test, 1].
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        epoch_loss = 0.0

        for i in range(0, X_train_t.size(0), BATCH_SIZE):
            idx = permutation[i : i + BATCH_SIZE]
            batch_x = X_train_t[idx]
            batch_y = y_train_t[idx]

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        logger.debug("    Epoch %d/%d, Loss: %.6f", epoch + 1, EPOCHS, epoch_loss)

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()

    return preds


def train_classification_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train classification model and make predictions.

    Args:
        model: PyTorch model (LSTM or DCN classifier).
        X_train: Training sequences [n_train, seq_len, n_features].
        y_train: Training targets [n_train].
        X_test: Test sequences [n_test, seq_len, n_features].

    Returns:
        (predictions, probabilities): Both arrays [n_test].
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        epoch_loss = 0.0

        for i in range(0, X_train_t.size(0), BATCH_SIZE):
            idx = permutation[i : i + BATCH_SIZE]
            batch_x = X_train_t[idx]
            batch_y = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        logger.debug("    Epoch %d/%d, Loss: %.6f", epoch + 1, EPOCHS, epoch_loss)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)

    return preds, probs


# =========================
# Per-Horizon Training
# =========================


def run_horizon(
    horizon_days: int,
    df: pd.DataFrame,
    feature_cols: List[str],
    task: str,
    results_dir: str,
    max_folds: int = MAX_FOLDS,
) -> Dict:
    """Run walk-forward training for a single horizon.

    Args:
        horizon_days: Days ahead to forecast (1, 3, 7, etc.).
        df: DataFrame with features and targets.
        feature_cols: List of feature column names.
        task: Either "regression" or "classification".
        results_dir: Output directory for results.
        max_folds: Maximum number of walk-forward folds.

    Returns:
        Summary dict with mean metrics.
    """
    logger.info("\n=== Horizon: %dd, Task: %s ===", horizon_days, task)

    target_col = f"Target_{task.split('_')[0].title()}_{horizon_days}d"
    if target_col not in df.columns:
        logger.warning("Target column %s not found; skipping horizon %dd", target_col, horizon_days)
        return None

    dates = pd.to_datetime(df["Date"]).values
    features_raw = df[feature_cols].values
    target_raw = df[target_col].values.reshape(-1, 1)

    os.makedirs(results_dir, exist_ok=True)

    initial_train_size = int(len(df) * INITIAL_TRAIN_RATIO)
    summaries = []
    all_preds = []

    fold_num = 1
    train_end = initial_train_size

    while train_end + TEST_WINDOW <= len(df) and fold_num <= max_folds:
        test_end = train_end + TEST_WINDOW
        logger.info("  Fold %d: train_end=%d, test_end=%d", fold_num, train_end, test_end)

        # Scale features and targets per fold
        feature_scaler = MinMaxScaler()
        feature_scaler.fit(features_raw[:train_end])
        features_scaled = feature_scaler.transform(features_raw)

        if task == "regression":
            target_scaler = MinMaxScaler()
            target_scaler.fit(target_raw[:train_end])
            target_scaled = target_scaler.transform(target_raw)
        else:
            target_scaler = None
            target_scaled = target_raw

        X_all, y_all, d_all = create_sequences(
            features_scaled, target_scaled, dates, SEQUENCE_LENGTH
        )

        seq_train_end = train_end - SEQUENCE_LENGTH
        seq_test_end = test_end - SEQUENCE_LENGTH

        if seq_test_end - seq_train_end <= 0:
            break

        X_train = X_all[:seq_train_end]
        y_train = y_all[:seq_train_end]
        X_test = X_all[seq_train_end:seq_test_end]
        y_test = y_all[seq_train_end:seq_test_end]
        d_test = d_all[seq_train_end:seq_test_end]

        if len(X_test) == 0:
            break

        fold_metrics = {}

        if task == "regression":
            y_train = y_train.astype(np.float32)
            y_test = y_test.astype(np.float32)

            # Train LSTM
            logger.info("    Training LSTM regressor...")
            lstm_model = LSTMRegressor(input_size=X_train.shape[2]).to(DEVICE)
            lstm_preds_scaled = train_regression_model(lstm_model, X_train, y_train, X_test)

            # Train DCN
            logger.info("    Training DCN regressor...")
            dcn_model = DCNRegressor(input_size=X_train.shape[2]).to(DEVICE)
            dcn_preds_scaled = train_regression_model(dcn_model, X_train, y_train, X_test)

            # Inverse transform
            y_test_rescaled = target_scaler.inverse_transform(y_test)
            lstm_preds = target_scaler.inverse_transform(lstm_preds_scaled)
            dcn_preds = target_scaler.inverse_transform(dcn_preds_scaled)
            naive_preds = np.zeros_like(y_test_rescaled)

            # Metrics
            rmse_lstm = math.sqrt(mean_squared_error(y_test_rescaled, lstm_preds))
            mae_lstm = mean_absolute_error(y_test_rescaled, lstm_preds)
            rmse_dcn = math.sqrt(mean_squared_error(y_test_rescaled, dcn_preds))
            mae_dcn = mean_absolute_error(y_test_rescaled, dcn_preds)
            rmse_naive = math.sqrt(mean_squared_error(y_test_rescaled, naive_preds))
            mae_naive = mean_absolute_error(y_test_rescaled, naive_preds)

            fold_metrics = {
                "fold": fold_num,
                "horizon": horizon_days,
                "rmse_lstm": rmse_lstm,
                "mae_lstm": mae_lstm,
                "rmse_dcn": rmse_dcn,
                "mae_dcn": mae_dcn,
                "rmse_naive": rmse_naive,
                "mae_naive": mae_naive,
            }

            logger.info("    LSTM  -> RMSE: %.6f, MAE: %.6f", rmse_lstm, mae_lstm)
            logger.info("    DCN   -> RMSE: %.6f, MAE: %.6f", rmse_dcn, mae_dcn)
            logger.info("    Naive -> RMSE: %.6f, MAE: %.6f", rmse_naive, mae_naive)

            pred_df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(d_test),
                    "Horizon": horizon_days,
                    "Actual": y_test_rescaled.flatten(),
                    "LSTM": lstm_preds.flatten(),
                    "DCN": dcn_preds.flatten(),
                    "Naive": naive_preds.flatten(),
                    "Fold": fold_num,
                }
            )

        else:  # classification
            y_train = y_train.astype(np.float32).flatten()
            y_test = y_test.astype(int).flatten()

            # Train LSTM
            logger.info("    Training LSTM classifier...")
            lstm_model = LSTMClassifier(input_size=X_train.shape[2]).to(DEVICE)
            lstm_preds, lstm_probs = train_classification_model(
                lstm_model, X_train, y_train, X_test
            )

            # Train DCN
            logger.info("    Training DCN classifier...")
            dcn_model = DCNClassifier(input_size=X_train.shape[2]).to(DEVICE)
            dcn_preds, dcn_probs = train_classification_model(
                dcn_model, X_train, y_train, X_test
            )

            naive_preds = np.ones_like(y_test)

            # Metrics
            acc_lstm = accuracy_score(y_test, lstm_preds)
            prec_lstm = precision_score(y_test, lstm_preds, zero_division=0)
            rec_lstm = recall_score(y_test, lstm_preds, zero_division=0)
            f1_lstm = f1_score(y_test, lstm_preds, zero_division=0)

            acc_dcn = accuracy_score(y_test, dcn_preds)
            prec_dcn = precision_score(y_test, dcn_preds, zero_division=0)
            rec_dcn = recall_score(y_test, dcn_preds, zero_division=0)
            f1_dcn = f1_score(y_test, dcn_preds, zero_division=0)

            acc_naive = accuracy_score(y_test, naive_preds)
            prec_naive = precision_score(y_test, naive_preds, zero_division=0)
            rec_naive = recall_score(y_test, naive_preds, zero_division=0)
            f1_naive = f1_score(y_test, naive_preds, zero_division=0)

            fold_metrics = {
                "fold": fold_num,
                "horizon": horizon_days,
                "acc_lstm": acc_lstm,
                "prec_lstm": prec_lstm,
                "rec_lstm": rec_lstm,
                "f1_lstm": f1_lstm,
                "acc_dcn": acc_dcn,
                "prec_dcn": prec_dcn,
                "rec_dcn": rec_dcn,
                "f1_dcn": f1_dcn,
                "acc_naive": acc_naive,
                "prec_naive": prec_naive,
                "rec_naive": rec_naive,
                "f1_naive": f1_naive,
            }

            logger.info("    LSTM  -> ACC: %.4f, F1: %.4f", acc_lstm, f1_lstm)
            logger.info("    DCN   -> ACC: %.4f, F1: %.4f", acc_dcn, f1_dcn)
            logger.info("    Naive -> ACC: %.4f, F1: %.4f", acc_naive, f1_naive)

            pred_df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(d_test),
                    "Horizon": horizon_days,
                    "Actual": y_test.flatten(),
                    "LSTM": lstm_preds.flatten(),
                    "DCN": dcn_preds.flatten(),
                    "Naive": naive_preds.flatten(),
                    "LSTM_Prob": lstm_probs.flatten(),
                    "DCN_Prob": dcn_probs.flatten(),
                    "Fold": fold_num,
                }
            )

        summaries.append(fold_metrics)
        all_preds.append(pred_df)

        train_end += STEP_SIZE
        fold_num += 1

    if not summaries:
        logger.warning("No folds generated for horizon %dd task %s", horizon_days, task)
        return None

    metrics_df = pd.DataFrame(summaries)
    preds_df = pd.concat(all_preds, ignore_index=True)

    # Save fold results
    task_name = "regression" if task == "regression" else "direction"
    metrics_file = Path(results_dir) / f"fold_metrics_{task_name}_{horizon_days}d.csv"
    preds_file = Path(results_dir) / f"predictions_{task_name}_{horizon_days}d.csv"

    metrics_df.to_csv(metrics_file, index=False)
    preds_df.to_csv(preds_file, index=False)

    logger.info("  Saved: %s", metrics_file)
    logger.info("  Saved: %s", preds_file)

    # Compute summary
    summary = {"horizon": horizon_days, "task": task, "num_folds": len(metrics_df)}

    if task == "regression":
        summary.update(
            {
                "rmse_lstm_mean": metrics_df["rmse_lstm"].mean(),
                "mae_lstm_mean": metrics_df["mae_lstm"].mean(),
                "rmse_dcn_mean": metrics_df["rmse_dcn"].mean(),
                "mae_dcn_mean": metrics_df["mae_dcn"].mean(),
                "rmse_naive_mean": metrics_df["rmse_naive"].mean(),
                "mae_naive_mean": metrics_df["mae_naive"].mean(),
            }
        )
    else:
        summary.update(
            {
                "acc_lstm_mean": metrics_df["acc_lstm"].mean(),
                "f1_lstm_mean": metrics_df["f1_lstm"].mean(),
                "acc_dcn_mean": metrics_df["acc_dcn"].mean(),
                "f1_dcn_mean": metrics_df["f1_dcn"].mean(),
                "acc_naive_mean": metrics_df["acc_naive"].mean(),
                "f1_naive_mean": metrics_df["f1_naive"].mean(),
            }
        )

    return summary


# =========================
# Main
# =========================


def main(argv: List[str] = None) -> None:
    """Main entry point for multi-horizon DCN/LSTM training.

    Configurable parameters:
      --seed: Random seed for reproducibility (PyTorch and NumPy).
      --embargo-gap: Reserved period (days) between train and test windows.
      --max-folds: Maximum number of walk-forward folds.
      Other standard args: --input, --results-dir, --verbose.

    Args:
        argv: Command-line arguments.
    """
    args = parse_args(argv)

    input_file = args.input
    results_dir = args.results_dir
    verbose = args.verbose
    seed = args.seed
    embargo_gap = args.embargo_gap

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("Configuration: seed=%d, embargo_gap=%d", seed, embargo_gap)

    os.makedirs(results_dir, exist_ok=True)

    logger.info("Loading dataset: %s", input_file)
    df = pd.read_parquet(input_file).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    logger.info("Rows: %d, Columns: %d", len(df), len(df.columns))

    feature_cols = [
        "Close",
        "Volume",
        "GoogleTrends",
        "Return",
        "LogReturn",
        "Volatility_7d",
        "MA_7",
        "MA_14",
        "Volume_Change",
        "Momentum_3d",
        "Momentum_7d",
        "Trends_Change",
    ]

    horizons = [1, 3, 7]
    all_summaries = []

    # Regression: Return forecasting
    for horizon in horizons:
        summary = run_horizon(
            horizon_days=horizon,
            df=df,
            feature_cols=feature_cols,
            task="regression",
            results_dir=results_dir,
            max_folds=args.max_folds,
        )
        if summary:
            all_summaries.append(summary)

    # Classification: Direction forecasting
    for horizon in horizons:
        summary = run_horizon(
            horizon_days=horizon,
            df=df,
            feature_cols=feature_cols,
            task="classification",
            results_dir=results_dir,
            max_folds=args.max_folds,
        )
        if summary:
            all_summaries.append(summary)
    
    # Append configuration metadata to output
    config_note = f"Generated with: seed={seed}, embargo_gap={embargo_gap}, max_folds={args.max_folds}"
    summary_df["_config_note"] = config_note

    # Save overall summary
    summary_df = pd.DataFrame(all_summaries)
    summary_csv = Path(results_dir) / "summary_return_direction_dcn.csv"
    summary_json = Path(results_dir) / "summary_return_direction_dcn.json"

    summary_df.to_csv(summary_csv, index=False)
    summary_dict = [s for s in all_summaries]
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    logger.info("\nFinal Summary:")
    logger.info("\n%s", summary_df)
    logger.info("\n=== Configuration Note ===")
    logger.info("Results generated with: seed=%d, embargo_gap=%d, max_folds=%d", seed, embargo_gap, args.max_folds)
    logger.info("Saved to:")
    logger.info("  CSV: %s", summary_csv)
    logger.info("  JSON: %s", summary_json)


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments (default: sys.argv).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train multi-horizon DCN/LSTM return and direction forecasting models."
    )
    parser.add_argument("--input", default=INPUT_FILE, help="Input parquet file path.")
    parser.add_argument(
        "--results-dir", default=RESULTS_DIR, help="Results output directory."
    )
    parser.add_argument("--max-folds", type=int, default=MAX_FOLDS, help="Max walk-forward folds.")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility (affects torch, numpy, etc.).",
    )
    parser.add_argument(
        "--embargo-gap",
        type=int,
        default=EMBARGO_GAP,
        help="Days to reserve between training and test window (embargo period).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    # Launch the canonical multi-horizon DCN baseline.
    main()