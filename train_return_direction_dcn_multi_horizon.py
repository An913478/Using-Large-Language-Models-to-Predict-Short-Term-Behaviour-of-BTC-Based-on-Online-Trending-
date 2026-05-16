"""
train_return_direction_dcn_multi_horizon.py
===========================================

Purpose
-------
Train and evaluate a Conv1D / dilated-convolution sequence baseline for direct
1-day, 3-day and 7-day BTC return forecasting.

Outputs
-------
results/return_direction_dcn/summary_return_direction_dcn.csv
results/return_direction_dcn/fold_metrics_return_direction_dcn.csv
results/return_direction_dcn/predictions_return_direction_dcn.csv
results/return_direction_dcn/summary_return_direction_dcn.json

Design
------
- Uses the same horizon set as the rest of the project: 1d, 3d, 7d.
- Uses expanding walk-forward validation.
- Fits scaling statistics on each training fold only.
- Trains a causal dilated Conv1D model using Huber loss.
- Reports RMSE, MAE, accuracy and F1 so it can be added to the Results tables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ======================================================
# Default configuration
# ======================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_CANDIDATES = [
    ROOT / "data" / "processed" / "btc_features_multi_horizon.parquet",
    ROOT / "data" / "processed" / "btc_features_phase1_plus.parquet",
    ROOT / "data" / "processed" / "btc_final_features_with_llm_uncertainty.parquet",
    ROOT / "data" / "processed" / "btc_final_features_with_llm_ensemble.parquet",
    ROOT / "data" / "processed" / "btc_final_features_with_llm.parquet",
]

RESULTS_DIR = ROOT / "results" / "return_direction_dcn"

HORIZONS = [1, 3, 7]
SEQUENCE_LENGTH = 30
INITIAL_TRAIN_RATIO = 0.60
TEST_WINDOW = 60
GAP = 0
MAX_FOLDS = 9

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_CHANNELS = 64
DROPOUT = 0.20
SEED = 42


# ======================================================
# Reproducibility and device
# ======================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic settings can be slower, but make runs easier to reproduce.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set Python, NumPy and PyTorch random seeds.
    This makes each seed run reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ======================================================
# Data loading and target construction
# ======================================================

def find_input_file(user_input: str | None) -> Path:
    if user_input is not None:
        path = Path(user_input)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    for path in DEFAULT_INPUT_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No input feature table found. Checked:\n"
        + "\n".join(str(p) for p in DEFAULT_INPUT_CANDIDATES)
        + "\n\nPass an explicit file using --input."
    )


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported input format: {path.suffix}")


def ensure_targets(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """
    Ensure Target_Return_* and Target_Direction_* columns exist.
    """
    if "Close" not in df.columns:
        raise ValueError("Input table must contain a 'Close' column to construct targets.")

    df = df.copy()

    for h in horizons:
        ret_col = f"Target_Return_{h}d"
        dir_col = f"Target_Direction_{h}d"

        if ret_col not in df.columns:
            df[ret_col] = (df["Close"].shift(-h) / df["Close"]) - 1.0

        if dir_col not in df.columns:
            df[dir_col] = (df[ret_col] > 0).astype(int)

    return df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select numeric predictor columns and remove reference, text and target columns.

    This keeps the script robust across project feature-table variants.
    """
    excluded_prefixes = (
        "Target_",
    )

    excluded_exact = {
        "Date",
        "date",
        "Close",          # retained as reference/target source, not model input
        "Target",
        "fold",
        "Fold",
    }

    excluded_contains = (
        "summary",
        "rationale",
        "text",
        "title",
        "description",
        "url",
        "provider",
        "article",
        "symbol",
        "ticker",
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols: list[str] = []
    for col in numeric_cols:
        if col in excluded_exact:
            continue
        if any(col.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if any(token.lower() in col.lower() for token in excluded_contains):
            continue
        if col.startswith("Target_Close_"):
            continue
        feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No usable numeric feature columns were found.")

    return feature_cols


def clean_for_horizon(df: pd.DataFrame, feature_cols: list[str], horizon: int) -> pd.DataFrame:
    """
    Keep rows with complete features and the selected horizon target.
    """
    ret_col = f"Target_Return_{horizon}d"
    dir_col = f"Target_Direction_{horizon}d"

    use_cols = ["Date", ret_col, dir_col] + feature_cols
    out = df[use_cols].copy()

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[ret_col, dir_col])
    out[feature_cols] = out[feature_cols].ffill()
    out = out.dropna(subset=feature_cols)
    out = out.sort_values("Date").reset_index(drop=True)

    return out


# ======================================================
# Sequence and fold construction
# ======================================================

def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_return_col: str,
    target_direction_col: str,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a row-wise table into sequence tensors.

    Returns
    -------
    X : shape (N, L, F)
    y_return : shape (N,)
    y_direction : shape (N,)
    dates : shape (N,)
    """
    X_values = df[feature_cols].to_numpy(dtype=np.float32)
    y_return_values = df[target_return_col].to_numpy(dtype=np.float32)
    y_direction_values = df[target_direction_col].to_numpy(dtype=np.int64)
    date_values = pd.to_datetime(df["Date"]).to_numpy()

    X_seq = []
    y_ret = []
    y_dir = []
    dates = []

    for end_idx in range(sequence_length - 1, len(df)):
        start_idx = end_idx - sequence_length + 1
        X_seq.append(X_values[start_idx:end_idx + 1])
        y_ret.append(y_return_values[end_idx])
        y_dir.append(y_direction_values[end_idx])
        dates.append(date_values[end_idx])

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_ret, dtype=np.float32),
        np.asarray(y_dir, dtype=np.int64),
        np.asarray(dates),
    )


def walk_forward_splits(
    n_samples: int,
    initial_train_ratio: float,
    test_window: int,
    gap: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """
    Expanding-window walk-forward splits.
    """
    initial_train_end = int(math.floor(initial_train_ratio * n_samples))

    folds = []
    fold_id = 1
    train_end = initial_train_end

    while True:
        test_start = train_end + gap
        test_end = min(test_start + test_window, n_samples)

        if test_start >= n_samples:
            break
        if test_end <= test_start:
            break
        if fold_id > MAX_FOLDS:
            break

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) == 0 or len(test_idx) == 0:
            break

        folds.append((fold_id, train_idx, test_idx))

        fold_id += 1
        train_end += test_window

    return folds


# ======================================================
# DCN model
# ======================================================

class CausalConv1d(nn.Module):
    """
    1D causal convolution.

    Input shape:  (batch, channels, sequence_length)
    Output shape: (batch, out_channels, sequence_length)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)


class DilatedConvNetRegressor(nn.Module):
    """
    Dilated Conv1D / DCN return-regression model.
    """

    def __init__(
        self,
        n_features: int,
        hidden_channels: int = 64,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.net = nn.Sequential(
            CausalConv1d(n_features, hidden_channels, kernel_size=3, dilation=1),
            nn.ReLU(),
            nn.Dropout(dropout),

            CausalConv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=2),
            nn.ReLU(),
            nn.Dropout(dropout),

            CausalConv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (batch, sequence_length, features)
        """
        x = x.transpose(1, 2)          # (batch, features, sequence_length)
        h = self.net(x)               # (batch, hidden_channels, sequence_length)
        last = h[:, :, -1]            # final historical representation
        out = self.head(last).squeeze(-1)
        return out


# ======================================================
# Training and evaluation
# ======================================================

def scale_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a StandardScaler on training sequences only, then apply to train and test.

    The scaler is fitted on flattened time-feature rows from the training fold.
    """
    n_train, seq_len, n_features = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    train_flat = X_train.reshape(-1, n_features)
    scaler.fit(train_flat)

    X_train_scaled = scaler.transform(train_flat).reshape(n_train, seq_len, n_features)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(n_test, seq_len, n_features)

    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32)


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[DilatedConvNetRegressor, np.ndarray]:
    """
    Train one DCN fold and return predictions on the test set.
    """
    set_seed(seed)

    model = DilatedConvNetRegressor(
        n_features=X_train.shape[-1],
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )

    model.train()
    for _epoch in range(EPOCHS):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        test_pred = model(X_test_t).detach().cpu().numpy()

    return model, test_pred


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred_dir = (y_pred > 0).astype(int)
    true_dir = (y_true > 0).astype(int)

    return {
        "rmse_dcn": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae_dcn": mean_absolute_error(y_true, y_pred),
        "acc_dcn": accuracy_score(true_dir, pred_dir),
        "f1_dcn": f1_score(true_dir, pred_dir, zero_division=0),
    }


# ======================================================
# Main pipeline
# ======================================================

def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)

    input_file = find_input_file(args.input)
    device = get_device()

    print(f"Input file: {input_file}")
    print(f"Device: {device}")

    df = read_table(input_file)

    if "Date" not in df.columns:
        raise ValueError("Input table must contain a 'Date' column.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = ensure_targets(df, HORIZONS)

    feature_cols = select_feature_columns(df)

    print(f"Selected {len(feature_cols)} feature columns.")
    print(f"First 10 features: {feature_cols[:10]}")

    if args.output_dir:
        RESULTS_DIR_LOCAL = Path(args.output_dir)
    else:
        RESULTS_DIR_LOCAL = RESULTS_DIR
    RESULTS_DIR_LOCAL.mkdir(parents=True, exist_ok=True)

    horizons_to_run = HORIZONS if args.horizon is None else [args.horizon]

    all_fold_metrics: list[dict] = []
    all_predictions: list[dict] = []

    for horizon in horizons_to_run:
        print(f"\n=== Horizon: {horizon}d ===")

        ret_col = f"Target_Return_{horizon}d"
        dir_col = f"Target_Direction_{horizon}d"

        h_df = clean_for_horizon(df, feature_cols, horizon)

        X, y_return, y_direction, dates = make_sequences(
            h_df,
            feature_cols,
            ret_col,
            dir_col,
            args.sequence_length,
        )

        folds = walk_forward_splits(
            n_samples=len(X),
            initial_train_ratio=args.initial_train_ratio,
            test_window=args.test_window,
            gap=args.gap,
        )

        print(f"Samples: {len(X)} | Folds: {len(folds)}")

        for fold_id, train_idx, test_idx in folds:
            X_train_raw = X[train_idx]
            y_train = y_return[train_idx]
            X_test_raw = X[test_idx]
            y_test = y_return[test_idx]

            X_train, X_test = scale_fold(X_train_raw, X_test_raw)

            _model, y_pred = train_one_fold(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                device=device,
                seed=args.seed + fold_id + horizon,
            )

            metrics = evaluate_predictions(y_test, y_pred)

            fold_record = {
                "horizon": f"{horizon}d",
                "fold": fold_id,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "seed": args.seed,
                **metrics,
            }
            all_fold_metrics.append(fold_record)

            for local_i, global_i in enumerate(test_idx):
                all_predictions.append({
                    "Date": pd.Timestamp(dates[global_i]).strftime("%Y-%m-%d"),
                    "horizon": f"{horizon}d",
                    "fold": fold_id,
                    "actual_return": float(y_return[global_i]),
                    "predicted_return_dcn": float(y_pred[local_i]),
                    "actual_direction": int(y_direction[global_i]),
                    "predicted_direction_dcn": int(y_pred[local_i] > 0),
                    "seed": args.seed,
                })

            print(
                f"Fold {fold_id:02d}: "
                f"RMSE={metrics['rmse_dcn']:.4f}, "
                f"MAE={metrics['mae_dcn']:.4f}, "
                f"F1={metrics['f1_dcn']:.4f}"
            )

    fold_df = pd.DataFrame(all_fold_metrics)
    pred_df = pd.DataFrame(all_predictions)

    summary_records = []
    for horizon, group in fold_df.groupby("horizon", sort=False):
        summary_records.append({
            "horizon": horizon,
            "rmse_dcn_mean": group["rmse_dcn"].mean(),
            "mae_dcn_mean": group["mae_dcn"].mean(),
            "acc_dcn_mean": group["acc_dcn"].mean(),
            "f1_dcn_mean": group["f1_dcn"].mean(),
            "num_folds": int(group["fold"].nunique()),
            "seed": args.seed,
        })

    summary_df = pd.DataFrame(summary_records)

    fold_path = RESULTS_DIR_LOCAL / "fold_metrics_return_direction_dcn.csv"
    pred_path = RESULTS_DIR_LOCAL / "predictions_return_direction_dcn.csv"
    summary_path = RESULTS_DIR_LOCAL / "summary_return_direction_dcn.csv"
    summary_json_path = RESULTS_DIR_LOCAL / "summary_return_direction_dcn.json"

    fold_df.to_csv(fold_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    print("\nSaved outputs:")
    print(f"  {fold_path}")
    print(f"  {pred_path}")
    print(f"  {summary_path}")
    print(f"  {summary_json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DCN / dilated Conv1D baseline for multi-horizon BTC forecasting."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional input CSV/Parquet feature table. If omitted, common project paths are searched.",
    )

    parser.add_argument(
        "--sequence_length",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Sequence lookback length.",
    )

    parser.add_argument(
        "--initial_train_ratio",
        type=float,
        default=INITIAL_TRAIN_RATIO,
        help="Initial expanding-window training ratio.",
    )

    parser.add_argument(
        "--test_window",
        type=int,
        default=TEST_WINDOW,
        help="Out-of-sample test window length per fold.",
    )

    parser.add_argument(
        "--gap",
        type=int,
        default=GAP,
        help="Optional temporal gap between training and test windows.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results. If not provided, uses default.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        choices=[1, 3, 7],
        help="Specific horizon to train (1, 3, or 7). If not provided, trains all.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())