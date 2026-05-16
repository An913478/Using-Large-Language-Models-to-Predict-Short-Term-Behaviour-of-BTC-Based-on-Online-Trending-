"""Train the final best LLM-augmented BTC forecasting model.

Loads the uncertainty-enhanced LLM feature dataset, trains an LSTM regressor
with multi-horizon walk-forward validation, and exports fold metrics, predictions,
and JSON summaries.
"""

import argparse
import logging
import os
import json
import random
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
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "btc_final_features_with_llm_uncertainty.parquet")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "final_best_model")

SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 96
NUM_LAYERS = 2
DROPOUT = 0.25

INITIAL_TRAIN_RATIO = 0.60
DEFAULT_TEST_WINDOW = 60

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model training: unsafe for smoke-tests
SMOKE_TEST_SAFE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        pred = self.head(last_hidden)
        return pred.squeeze(-1)


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


def build_feature_list(df: pd.DataFrame, horizon_name: str) -> List[str]:
    excluded = {
        "Date",
        "llm_provider",
        "openai_llm_event_type",
        "openai_llm_summary",
        "openai_llm_rationale",
        "claude_llm_event_type",
        "claude_llm_summary",
        "claude_llm_rationale",
        "gemini_llm_event_type",
        "gemini_llm_summary",
        "gemini_llm_rationale",
        "1d_selected_llm_event_type",
        "1d_selected_llm_summary",
        "1d_selected_llm_rationale",
        "3d_selected_llm_event_type",
        "3d_selected_llm_summary",
        "3d_selected_llm_rationale",
        "7d_selected_llm_event_type",
        "7d_selected_llm_summary",
        "7d_selected_llm_rationale",
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
    base_feature_cols = [c for c in numeric_cols if c not in excluded]

    horizon_specific_cols = [
        c for c in base_feature_cols
        if not c.startswith("1d_") and not c.startswith("3d_") and not c.startswith("7d_")
    ]

    chosen_prefix = f"{horizon_name}_"
    chosen_horizon_cols = [c for c in base_feature_cols if c.startswith(chosen_prefix)]

    feature_cols = horizon_specific_cols + chosen_horizon_cols
    feature_cols = sorted(list(dict.fromkeys(feature_cols)))

    if not feature_cols:
        raise ValueError(f"No usable numeric feature columns found for {horizon_name}.")

    return feature_cols


def make_sequences(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    dates: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_seq = []
    y_seq = []
    d_seq = []

    for i in range(seq_len, len(feature_array)):
        X_seq.append(feature_array[i - seq_len:i])
        y_seq.append(target_array[i])
        d_seq.append(dates[i])

    return np.array(X_seq), np.array(y_seq), np.array(d_seq)


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


def train_lstm_model(X_train: np.ndarray, y_train: np.ndarray, input_size: int) -> LSTMRegressor:
    model = LSTMRegressor(input_size=input_size).to(DEVICE)

    dataset = SequenceDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        losses = []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        logger.info("      Epoch %d/%d Loss: %.6f", epoch + 1, EPOCHS, np.mean(losses))

    return model


def run_horizon(
    df: pd.DataFrame,
    horizon_name: str,
    target_return_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = build_feature_list(df, horizon_name)

    logger.info("\n==================== HORIZON %s ====================", horizon_name)
    logger.info("Feature count for %s: %d", horizon_name, len(feature_cols))    # -----------------------------
    # Prepare horizon-specific dataset and targets
    # -----------------------------
    use_cols = ["Date"] + feature_cols + [target_return_col]
    df_local = df[use_cols].copy()
    logger.info("Rows before cleaning: %d", len(df_local))

    df_local = df_local.dropna(subset=[target_return_col]).reset_index(drop=True)
    logger.info("Rows after dropping missing target: %d", len(df_local))

    feature_df = df_local[feature_cols].replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.ffill().bfill()

    valid_mask = feature_df.notna().all(axis=1)
    df_local = df_local.loc[valid_mask].reset_index(drop=True)
    feature_df = feature_df.loc[valid_mask].reset_index(drop=True)
    logger.info("Rows after feature cleaning: %d", len(df_local))

    X_all = feature_df.to_numpy(dtype=np.float32)
    y_all = df_local[target_return_col].astype(float).values
    d_all = pd.to_datetime(df_local["Date"]).values

    X_seq, y_seq, d_seq = make_sequences(X_all, y_all, d_all, SEQUENCE_LENGTH)
    logger.info("Sequence count for %s: %d", horizon_name, len(X_seq))

    splits = get_walk_forward_splits(len(X_seq))
    logger.info("Number of walk-forward folds for %s: %d", horizon_name, len(splits))

    if not splits:
        raise ValueError(f"No valid walk-forward splits generated for horizon {horizon_name}.")

    fold_results = []
    pred_rows = []

    for fold_idx, (train_end, test_end) in enumerate(splits, start=1):
        logger.info("\nFold %d: train_end=%d, test_end=%d", fold_idx, train_end, test_end)

        X_train_raw = X_seq[:train_end]
        y_train_raw = y_seq[:train_end]
        X_test_raw = X_seq[train_end:test_end]
        y_test = y_seq[train_end:test_end]
        d_test = d_seq[train_end:test_end]

        feature_scaler = StandardScaler()
        X_train = feature_scaler.fit_transform(
            X_train_raw.reshape(-1, X_train_raw.shape[-1])
        ).reshape(X_train_raw.shape)
        X_test = feature_scaler.transform(
            X_test_raw.reshape(-1, X_test_raw.shape[-1])
        ).reshape(X_test_raw.shape)

        target_scaler = StandardScaler()
        y_train = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()

        model = train_lstm_model(X_train, y_train, X_train.shape[-1])

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            y_pred_scaled = model(X_test_tensor).cpu().numpy()

        y_pred_lstm = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_pred_naive = np.zeros_like(y_test)

        rmse_lstm, mae_lstm = evaluate_regression(y_test, y_pred_lstm)
        rmse_naive, mae_naive = evaluate_regression(y_test, y_pred_naive)

        acc_lstm, f1_lstm = evaluate_direction(y_test, y_pred_lstm)
        acc_naive, f1_naive = evaluate_direction(y_test, y_pred_naive)

        logger.info(
            "    Final model -> RMSE: %.6f, MAE: %.6f, ACC: %.4f, F1: %.4f",
            rmse_lstm,
            mae_lstm,
            acc_lstm,
            f1_lstm,
        )
        logger.info(
            "    Naive       -> RMSE: %.6f, MAE: %.6f, ACC: %.4f, F1: %.4f",
            rmse_naive,
            mae_naive,
            acc_naive,
            f1_naive,
        )

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

    preds_df = pd.DataFrame(pred_rows)
    return fold_df, summary_df, preds_df


def main() -> None:
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing upgraded dataset: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    logger.info("Loaded final upgraded dataset")
    logger.info("Rows: %d", len(df))
    logger.info("Columns: %d", len(df.columns))
    # -----------------------------
    # Train one model for each forecast horizon
    # -----------------------------

    horizon_map = {
        "1d": "Target_Return_1d",
        "3d": "Target_Return_3d",
        "7d": "Target_Return_7d",
    }

    all_fold_dfs = []
    all_summary_dfs = []
    all_preds_dfs = []

    for horizon_name, target_col in horizon_map.items():
        fold_df, summary_df, preds_df = run_horizon(
            df=df,
            horizon_name=horizon_name,
            target_return_col=target_col,
        )
        all_fold_dfs.append(fold_df)
        all_summary_dfs.append(summary_df)
        all_preds_dfs.append(preds_df)

    # -----------------------------
    # Aggregate fold results and save all output files
    # -----------------------------
    fold_results_df = pd.concat(all_fold_dfs, ignore_index=True)
    summary_results_df = pd.concat(all_summary_dfs, ignore_index=True)
    predictions_df = pd.concat(all_preds_dfs, ignore_index=True)

    fold_path = os.path.join(RESULTS_DIR, "fold_metrics_final_best.csv")
    summary_path = os.path.join(RESULTS_DIR, "summary_final_best.csv")
    preds_path = os.path.join(RESULTS_DIR, "predictions_final_best.csv")
    json_path = os.path.join(RESULTS_DIR, "summary_final_best.json")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the final best model with LLM uncertainty features.")
    parser.add_argument("--input", default=INPUT_FILE, help="Input parquet file")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Results directory")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    INPUT_FILE = args.input
    RESULTS_DIR = args.results_dir
    SEED = args.seed

    main()