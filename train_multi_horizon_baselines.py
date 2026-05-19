"""Train baseline LSTM models on multi-horizon BTC features.

Performs walk-forward training on multi-horizon targets (1d, 3d, 7d), computes
regression and direction metrics, and saves predictions, summary metrics, and
JSON output. Supports CLI options for output directory and reproducibility.
"""

import logging
import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "btc_features_multi_horizon.parquet")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "multi_horizon")

SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

INITIAL_TRAIN_RATIO = 0.60
DEFAULT_TEST_WINDOW = 60

MAX_FOLDS = 9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model training: unsafe for smoke-tests
SMOKE_TEST_SAFE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set deterministic random seeds for Python, NumPy, and PyTorch.

    This function configures the environment so that the same seed produces
    repeatable results across CPU and GPU runs where determinism is available.
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


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for a minimal reproducible PyTorch run.

    This helper is a lighter-weight alternative to :func:`set_global_seed`
    and can be used when full deterministic configuration is not required.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    """Simple PyTorch dataset wrapper for sequence data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    """LSTM-based regression model for return forecasting."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


@dataclass
class FoldResult:
    horizon: str
    fold: int
    rmse_lstm: float
    mae_lstm: float
    rmse_naive: float
    mae_naive: float
    acc_lstm: float
    f1_lstm: float
    acc_naive: float
    f1_naive: float
    seed: int


def build_feature_list(df: pd.DataFrame) -> List[str]:
    """Build a list of numeric feature columns for model training.

    Excludes date and all target columns so that only predictive inputs remain.
    """
    excluded = {
        "Date",
        "Target_Close_1d",
        "Target_Return_1d",
        "Target_Direction_1d",
        "Target_Close_3d",
        "Target_Return_3d",
        "Target_Direction_3d",
        "Target_Close_7d",
        "Target_Return_7d",
        "Target_Direction_7d",
        "Target_Close_NextDay",
        "Target_Return_NextDay",
        "Target_Direction",
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in excluded]
    if not features:
        raise ValueError("No numeric feature columns found after exclusions.")
    return features


def make_sequences(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    dates: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert time series arrays into rolling sequences for LSTM input.

    Each returned sample contains a historical window of features and the
    corresponding target at the next time step.
    """
    X_seq, y_seq, d_seq = [], [], []
    for i in range(seq_len, len(feature_array)):
        X_seq.append(feature_array[i - seq_len : i])
        y_seq.append(target_array[i])
        d_seq.append(dates[i])
    return np.array(X_seq), np.array(y_seq), np.array(d_seq)


def train_lstm_model(X_train: np.ndarray, y_train: np.ndarray, input_size: int, seed: int) -> LSTMRegressor:
    """Train an LSTM regressor on a single fold and return the trained model.

    This function sets up a seeded data loader, trains for a fixed number of
    epochs, and logs epoch-level loss during training.
    """
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    dataset = SequenceDataset(X_train, y_train)

    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        logger.info("      Epoch %d/%d Loss: %.6f", epoch + 1, EPOCHS, np.mean(losses))

    return model


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute regression metrics for continuous return predictions."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def evaluate_direction(y_true_reg: np.ndarray, y_pred_reg: np.ndarray) -> Tuple[float, float]:
    """Compute binary direction metrics from continuous return forecasts."""
    y_true_dir = (y_true_reg > 0).astype(int)
    y_pred_dir = (y_pred_reg > 0).astype(int)

    acc = float(accuracy_score(y_true_dir, y_pred_dir))
    f1 = float(f1_score(y_true_dir, y_pred_dir, zero_division=0))
    return acc, f1


def get_walk_forward_splits(
    n_samples: int,
    initial_train_ratio: float = INITIAL_TRAIN_RATIO,
    test_window: int = DEFAULT_TEST_WINDOW,
    gap: int = 0,
) -> List[Tuple[int, int]]:
    """Generate walk-forward train/test index splits for sequential data.

    Each fold uses all data up to the training end, with an optional embargo gap
    before the test window. The test window moves forward until the end of data.
    """
    if n_samples < 100:
        return []

    initial_train_end = int(n_samples * initial_train_ratio)
    test_window = min(test_window, max(20, (n_samples - initial_train_end) // 3))

    if initial_train_end + gap >= n_samples:
        return []

    splits = []
    train_end = initial_train_end
    while train_end + gap < n_samples:
        test_start = train_end + gap
        test_end = min(test_start + test_window, n_samples)

        if test_start >= test_end:
            break

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

        train_end += test_window

    return splits


def run_horizon(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizon_name: str,
    target_return_col: str,
    seed: int,
    gap: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run walk-forward training and evaluation for a single target horizon."""
    logger.info("\n==================== HORIZON %s ====================", horizon_name)

    df_local = df[["Date"] + feature_cols + [target_return_col]].copy()
    logger.info("Rows before cleaning: %d", len(df_local))

    df_local = df_local.dropna(subset=[target_return_col]).reset_index(drop=True)
    logger.info("Rows after dropping missing target: %d", len(df_local))

    feature_df = df_local[feature_cols].replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.ffill().bfill()

    valid_mask = feature_df.notna().all(axis=1)
    df_local = df_local.loc[valid_mask].reset_index(drop=True)
    feature_df = feature_df.loc[valid_mask].reset_index(drop=True)

    logger.info("Rows after feature cleaning: %d", len(df_local))

    target_return = df_local[target_return_col].astype(float).values
    dates = pd.to_datetime(df_local["Date"]).values
    X_all = feature_df.to_numpy(dtype=np.float32)

    X_seq, y_seq, d_seq = make_sequences(X_all, target_return, dates, SEQUENCE_LENGTH)

    logger.info("Sequence count for %s: %d", horizon_name, len(X_seq))

    splits = get_walk_forward_splits(
        n_samples=len(X_seq),
        initial_train_ratio=INITIAL_TRAIN_RATIO,
        test_window=DEFAULT_TEST_WINDOW,
        gap=gap,
    )
    logger.info("Number of walk-forward folds for %s (gap=%d): %d", horizon_name, gap, len(splits))

    if not splits:
        raise ValueError(f"No valid walk-forward splits generated for horizon {horizon_name}.")

    fold_results = []
    pred_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        if fold_idx > MAX_FOLDS:
            break
        train_end = int(train_idx[-1]) + 1
        test_start = int(test_idx[0])
        test_end = int(test_idx[-1]) + 1
        logger.info("\nFold %d: train_end=%d, test_start=%d, test_end=%d", fold_idx, train_end, test_start, test_end)

        X_train_raw = X_seq[train_idx]
        y_train = y_seq[train_idx]
        X_test_raw = X_seq[test_idx]
        y_test = y_seq[test_idx]
        d_test = d_seq[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(
            X_train_raw.reshape(-1, X_train_raw.shape[-1])
        ).reshape(X_train_raw.shape)
        X_test_scaled = scaler.transform(
            X_test_raw.reshape(-1, X_test_raw.shape[-1])
        ).reshape(X_test_raw.shape)

        model = train_lstm_model(X_train_scaled, y_train, X_train_scaled.shape[-1], seed)

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
            y_pred_lstm = model(X_test_tensor).cpu().numpy()

        y_pred_naive = np.zeros_like(y_test)

        rmse_lstm, mae_lstm = evaluate_regression(y_test, y_pred_lstm)
        rmse_naive, mae_naive = evaluate_regression(y_test, y_pred_naive)

        acc_lstm, f1_lstm = evaluate_direction(y_test, y_pred_lstm)
        acc_naive, f1_naive = evaluate_direction(y_test, y_pred_naive)

        logger.info("    LSTM  -> RMSE: %.6f, MAE: %.6f, ACC: %.4f, F1: %.4f", rmse_lstm, mae_lstm, acc_lstm, f1_lstm)
        logger.info("    Naive -> RMSE: %.6f, MAE: %.6f, ACC: %.4f, F1: %.4f", rmse_naive, mae_naive, acc_naive, f1_naive)

        fold_results.append(
            FoldResult(
                horizon=horizon_name,
                fold=fold_idx,
                rmse_lstm=rmse_lstm,
                mae_lstm=mae_lstm,
                rmse_naive=rmse_naive,
                mae_naive=mae_naive,
                acc_lstm=acc_lstm,
                f1_lstm=f1_lstm,
                acc_naive=acc_naive,
                f1_naive=f1_naive,
                seed=seed,
            )
        )

        for i in range(len(y_test)):
            pred_rows.append(
                {
                    "Date": pd.to_datetime(d_test[i]),
                    "horizon": horizon_name,
                    "fold": fold_idx,
                    "actual_return": float(y_test[i]),
                    "predicted_return_lstm": float(y_pred_lstm[i]),
                    "predicted_return_naive": float(y_pred_naive[i]),
                    "actual_direction": int(y_test[i] > 0),
                    "predicted_direction_lstm": int(y_pred_lstm[i] > 0),
                    "predicted_direction_naive": int(y_pred_naive[i] > 0),
                    "seed": seed,
                }
            )

    fold_df = pd.DataFrame([vars(r) for r in fold_results])

    summary_df = (
        fold_df.groupby("horizon", as_index=False)
        .agg(
            rmse_lstm_mean=("rmse_lstm", "mean"),
            mae_lstm_mean=("mae_lstm", "mean"),
            rmse_naive_mean=("rmse_naive", "mean"),
            mae_naive_mean=("mae_naive", "mean"),
            acc_lstm_mean=("acc_lstm", "mean"),
            f1_lstm_mean=("f1_lstm", "mean"),
            acc_naive_mean=("acc_naive", "mean"),
            f1_naive_mean=("f1_naive", "mean"),
            num_folds=("fold", "count"),
        )
    )
    summary_df["seed"] = seed

    preds_df = pd.DataFrame(pred_rows)
    return fold_df, summary_df, preds_df


def main(args: argparse.Namespace) -> None:
    """Execute the full multi-horizon baseline training workflow."""
    set_global_seed(args.seed)

    # -----------------------------
    # Prepare output directory and dataset path
    # -----------------------------
    if args.output_dir is None:
        results_dir_local = RESULTS_DIR
    else:
        results_dir_local = args.output_dir
    os.makedirs(results_dir_local, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    logger.info("Dataset loaded")
    logger.info("Rows: %d", len(df))
    logger.info("Columns: %d", len(df.columns))

    feature_cols = build_feature_list(df)
    logger.info("\nNumber of feature columns used: %d", len(feature_cols))

    horizon_map = {
        "1d": "Target_Return_1d",
        "3d": "Target_Return_3d",
        "7d": "Target_Return_7d",
    }

    if args.horizon:
        horizon_map = {args.horizon: horizon_map[args.horizon]}

    all_fold_dfs = []
    all_summary_dfs = []
    all_preds_dfs = []

    # -----------------------------
    # Train one model per horizon and collect results
    # -----------------------------
    for horizon_name, target_col in horizon_map.items():
        fold_df, summary_df, preds_df = run_horizon(
            df=df,
            feature_cols=feature_cols,
            horizon_name=horizon_name,
            target_return_col=target_col,
            seed=args.seed,
            gap=args.gap,
        )
        all_fold_dfs.append(fold_df)
        all_summary_dfs.append(summary_df)
        all_preds_dfs.append(preds_df)

    # -----------------------------
    # Aggregate results across horizons and save outputs
    # -----------------------------
    fold_results_df = pd.concat(all_fold_dfs, ignore_index=True)
    summary_results_df = pd.concat(all_summary_dfs, ignore_index=True)
    predictions_df = pd.concat(all_preds_dfs, ignore_index=True)

    fold_path = os.path.join(results_dir_local, "fold_metrics_multi_horizon.csv")
    summary_path = os.path.join(results_dir_local, "summary_multi_horizon.csv")
    preds_path = os.path.join(results_dir_local, "predictions_multi_horizon.csv")
    json_path = os.path.join(results_dir_local, "summary_multi_horizon.json")

    fold_results_df.to_csv(fold_path, index=False)
    summary_results_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(preds_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results_df.to_dict(orient="records"), f, indent=2, default=str)

    logger.info("\n===== FINAL SUMMARY =====")
    logger.info("\n%s", summary_results_df)

    logger.info("\nSaved fold metrics to: %s", fold_path)
    logger.info("Saved summary metrics to: %s", summary_path)
    logger.info("Saved predictions to: %s", preds_path)
    logger.info("Saved JSON summary to: %s", json_path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM baselines for multi-horizon BTC forecasting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to write metrics and predictions.")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm"], help="Baseline model architecture to train.")
    parser.add_argument("--horizon", type=str, default=None, choices=["1d", "3d", "7d"], help="Optional single horizon to run.")
    parser.add_argument("--gap", type=int, default=0, help="Embargo gap in samples between train and test windows.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(args)