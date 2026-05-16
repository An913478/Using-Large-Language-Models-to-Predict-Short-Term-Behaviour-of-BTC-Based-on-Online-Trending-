"""Train a multi-horizon Transformer model on BTC features.

Loads market features, trains a Transformer architecture with positional encoding
for walk-forward validation across 1d, 3d, and 7d return targets, computes
regression and direction metrics, and exports predictions and summaries.
"""

import argparse
import logging
import os
import json
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "btc_features_phase1_plus.parquet")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "multi_horizon_transformer")

SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1

INITIAL_TRAIN_RATIO = 0.60
DEFAULT_TEST_WINDOW = 60

# Model training: unsafe for smoke-tests
SMOKE_TEST_SAFE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.regressor(x)
        return x.squeeze(-1)


@dataclass
class FoldResult:
    horizon: str
    fold: int
    rmse_transformer: float
    mae_transformer: float
    rmse_naive: float
    mae_naive: float
    acc_transformer: float
    f1_transformer: float
    acc_naive: float
    f1_naive: float


def build_feature_list(df: pd.DataFrame) -> List[str]:
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
    X_seq, y_seq, d_seq = [], [], []
    for i in range(seq_len, len(feature_array)):
        X_seq.append(feature_array[i - seq_len : i])
        y_seq.append(target_array[i])
        d_seq.append(dates[i])
    return np.array(X_seq), np.array(y_seq), np.array(d_seq)


def train_transformer_model(X_train: np.ndarray, y_train: np.ndarray, input_size: int) -> TransformerRegressor:
    model = TransformerRegressor(
        input_size=input_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(DEVICE)

    dataset = SequenceDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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

        logger.info(f"      Epoch {epoch + 1}/{EPOCHS} Loss: {np.mean(losses):.6f}")

    return model


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def evaluate_direction(y_true_reg: np.ndarray, y_pred_reg: np.ndarray) -> Tuple[float, float]:
    y_true_dir = (y_true_reg > 0).astype(int)
    y_pred_dir = (y_pred_reg > 0).astype(int)

    acc = float(accuracy_score(y_true_dir, y_pred_dir))
    f1 = float(f1_score(y_true_dir, y_pred_dir, zero_division=0))
    return acc, f1


def get_walk_forward_splits(n_samples: int) -> List[Tuple[int, int]]:
    if n_samples < 100:
        return []

    initial_train_end = int(n_samples * INITIAL_TRAIN_RATIO)
    test_window = min(DEFAULT_TEST_WINDOW, max(20, (n_samples - initial_train_end) // 3))

    if initial_train_end + test_window > n_samples:
        return []

    splits = []
    train_end = initial_train_end
    while train_end + test_window <= n_samples:
        test_end = train_end + test_window
        splits.append((train_end, test_end))
        train_end = test_end
    return splits


def run_horizon(df: pd.DataFrame, feature_cols: List[str], horizon_name: str, target_return_col: str):
    logger.info(f"\n==================== HORIZON {horizon_name} ====================")

    df_local = df[["Date"] + feature_cols + [target_return_col]].copy()
    df_local = df_local.dropna(subset=[target_return_col]).reset_index(drop=True)

    feature_df = df_local[feature_cols].replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.ffill().bfill()

    valid_mask = feature_df.notna().all(axis=1)
    df_local = df_local.loc[valid_mask].reset_index(drop=True)
    feature_df = feature_df.loc[valid_mask].reset_index(drop=True)

    target_return = df_local[target_return_col].astype(float).values
    dates = pd.to_datetime(df_local["Date"]).values
    X_all = feature_df.to_numpy(dtype=np.float32)

    X_seq, y_seq, d_seq = make_sequences(X_all, target_return, dates, SEQUENCE_LENGTH)

    logger.info(f"Rows after cleaning: {len(df_local)}")
    logger.info(f"Sequence count for {horizon_name}: {len(X_seq)}")

    splits = get_walk_forward_splits(len(X_seq))
    logger.info(f"Number of walk-forward folds for {horizon_name}: {len(splits)}")

    if not splits:
        raise ValueError(f"No valid walk-forward splits generated for horizon {horizon_name}.")

    fold_results = []
    pred_rows = []

    for fold_idx, (train_end, test_end) in enumerate(splits, start=1):
        logger.info(f"\nFold {fold_idx}: train_end={train_end}, test_end={test_end}")

        X_train_raw = X_seq[:train_end]
        y_train_raw = y_seq[:train_end]
        X_test_raw = X_seq[train_end:test_end]
        y_test = y_seq[train_end:test_end]
        d_test = d_seq[train_end:test_end]

        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(
            X_train_raw.reshape(-1, X_train_raw.shape[-1])
        ).reshape(X_train_raw.shape)
        X_test_scaled = feature_scaler.transform(
            X_test_raw.reshape(-1, X_test_raw.shape[-1])
        ).reshape(X_test_raw.shape)

        target_scaler = StandardScaler()
        y_train = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()

        model = train_transformer_model(X_train_scaled, y_train, X_train_scaled.shape[-1])

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
            y_pred_scaled = model(X_test_tensor).cpu().numpy()

        y_pred_transformer = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_pred_naive = np.zeros_like(y_test)

        rmse_tr, mae_tr = evaluate_regression(y_test, y_pred_transformer)
        rmse_nv, mae_nv = evaluate_regression(y_test, y_pred_naive)

        acc_tr, f1_tr = evaluate_direction(y_test, y_pred_transformer)
        acc_nv, f1_nv = evaluate_direction(y_test, y_pred_naive)

        logger.info(f"    Transformer -> RMSE: {rmse_tr:.6f}, MAE: {mae_tr:.6f}, ACC: {acc_tr:.4f}, F1: {f1_tr:.4f}")
        logger.info(f"    Naive       -> RMSE: {rmse_nv:.6f}, MAE: {mae_nv:.6f}, ACC: {acc_nv:.4f}, F1: {f1_nv:.4f}")

        fold_results.append(
            FoldResult(
                horizon=horizon_name,
                fold=fold_idx,
                rmse_transformer=rmse_tr,
                mae_transformer=mae_tr,
                rmse_naive=rmse_nv,
                mae_naive=mae_nv,
                acc_transformer=acc_tr,
                f1_transformer=f1_tr,
                acc_naive=acc_nv,
                f1_naive=f1_nv,
            )
        )

        for i in range(len(y_test)):
            pred_rows.append(
                {
                    "Date": pd.to_datetime(d_test[i]),
                    "horizon": horizon_name,
                    "fold": fold_idx,
                    "actual_return": float(y_test[i]),
                    "predicted_return_transformer": float(y_pred_transformer[i]),
                    "predicted_return_naive": float(y_pred_naive[i]),
                    "actual_direction": int(y_test[i] > 0),
                    "predicted_direction_transformer": int(y_pred_transformer[i] > 0),
                    "predicted_direction_naive": int(y_pred_naive[i] > 0),
                }
            )

    fold_df = pd.DataFrame([vars(r) for r in fold_results])

    summary_df = (
        fold_df.groupby("horizon", as_index=False)
        .agg(
            rmse_transformer_mean=("rmse_transformer", "mean"),
            mae_transformer_mean=("mae_transformer", "mean"),
            rmse_naive_mean=("rmse_naive", "mean"),
            mae_naive_mean=("mae_naive", "mean"),
            acc_transformer_mean=("acc_transformer", "mean"),
            f1_transformer_mean=("f1_transformer", "mean"),
            acc_naive_mean=("acc_naive", "mean"),
            f1_naive_mean=("f1_naive", "mean"),
            num_folds=("fold", "count"),
        )
    )

    preds_df = pd.DataFrame(pred_rows)
    return fold_df, summary_df, preds_df


def main(input_file: str = INPUT_FILE, output_dir: str = RESULTS_DIR, verbose: bool = False) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_parquet(INPUT_FILE).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    logger.info("Dataset loaded")
    logger.info(f"Rows: {len(df)}")
    logger.info(f"Columns: {len(df.columns)}")

    feature_cols = build_feature_list(df)
    logger.info(f"Number of feature columns used: {len(feature_cols)}")

    horizon_map = {
        "1d": "Target_Return_1d",
        "3d": "Target_Return_3d",
        "7d": "Target_Return_7d",
    }

    all_fold_dfs = []
    all_summary_dfs = []
    all_preds_dfs = []

    for horizon_name, target_col in horizon_map.items():
        fold_df, summary_df, preds_df = run_horizon(df, feature_cols, horizon_name, target_col)
        all_fold_dfs.append(fold_df)
        all_summary_dfs.append(summary_df)
        all_preds_dfs.append(preds_df)

    fold_results_df = pd.concat(all_fold_dfs, ignore_index=True)
    summary_results_df = pd.concat(all_summary_dfs, ignore_index=True)
    predictions_df = pd.concat(all_preds_dfs, ignore_index=True)

    fold_path = os.path.join(output_dir, "fold_metrics_multi_horizon_transformer.csv")
    summary_path = os.path.join(output_dir, "summary_multi_horizon_transformer.csv")
    preds_path = os.path.join(output_dir, "predictions_multi_horizon_transformer.csv")
    json_path = os.path.join(output_dir, "summary_multi_horizon_transformer.json")

    fold_results_df.to_csv(fold_path, index=False)
    summary_results_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(preds_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results_df.to_dict(orient="records"), f, indent=2, default=str)

    logger.info("===== FINAL SUMMARY =====")
    logger.info("%s", summary_results_df)
    logger.info("Saved fold metrics to: %s", fold_path)
    logger.info("Saved summary metrics to: %s", summary_path)
    logger.info("Saved predictions to: %s", preds_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-horizon transformer model with walk-forward evaluation.")
    parser.add_argument("--input", type=str, default=INPUT_FILE, help="Input features parquet file")
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR, help="Directory where results are saved")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(input_file=args.input, output_dir=args.output_dir, verbose=args.verbose)