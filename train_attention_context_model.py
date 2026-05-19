"""Train an attention-context LSTM model on BTC market and context features.

Loads market and context feature data, trains an LSTM model with walk-forward
validation across 1d, 3d, and 7d horizons, computes regression and direction
metrics, and exports fold metrics, summary tables, and predictions.
"""

import argparse
import json
import logging
import os
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
RESULTS_DIR = os.path.join(BASE_DIR, "results", "attention_context")

SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3

MARKET_HIDDEN_SIZE = 64
MARKET_NUM_LAYERS = 2
MARKET_DROPOUT = 0.2

CONTEXT_HIDDEN_SIZE = 32
FUSION_HIDDEN_SIZE = 64

INITIAL_TRAIN_RATIO = 0.60
DEFAULT_TEST_WINDOW = 60
MAX_FOLDS = 9

# Model training: unsafe for smoke-tests
SMOKE_TEST_SAFE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AttentionContextDataset(Dataset):
    """PyTorch dataset for paired market and context feature sequences."""

    def __init__(self, X_market: np.ndarray, X_context: np.ndarray, y: np.ndarray):
        self.X_market = torch.tensor(X_market, dtype=torch.float32)
        self.X_context = torch.tensor(X_context, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X_market)

    def __getitem__(self, idx: int):
        return self.X_market[idx], self.X_context[idx], self.y[idx]


class AttentionContextModel(nn.Module):
    def __init__(
        self,
        market_input_size: int,
        context_input_size: int,
        market_hidden_size: int = MARKET_HIDDEN_SIZE,
        market_num_layers: int = MARKET_NUM_LAYERS,
        market_dropout: float = MARKET_DROPOUT,
        context_hidden_size: int = CONTEXT_HIDDEN_SIZE,
        fusion_hidden_size: int = FUSION_HIDDEN_SIZE,
    ):
        super().__init__()

        self.market_lstm = nn.LSTM(
            input_size=market_input_size,
            hidden_size=market_hidden_size,
            num_layers=market_num_layers,
            batch_first=True,
            dropout=market_dropout if market_num_layers > 1 else 0.0,
        )

        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_size, context_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(context_hidden_size, context_hidden_size),
            nn.ReLU(),
        )

        self.context_attention = nn.Sequential(
            nn.Linear(context_hidden_size, context_hidden_size),
            nn.Tanh(),
            nn.Linear(context_hidden_size, market_hidden_size),
            nn.Sigmoid(),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(market_hidden_size + context_hidden_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden_size // 2, 1),
        )

    def forward(self, x_market: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        market_out, _ = self.market_lstm(x_market)
        market_hidden = market_out[:, -1, :]

        context_hidden = self.context_encoder(x_context)
        attention_gate = self.context_attention(context_hidden)

        attended_market = market_hidden * attention_gate
        fused = torch.cat([attended_market, context_hidden], dim=1)

        output = self.fusion_head(fused)
        return output.squeeze(-1)


@dataclass
class FoldResult:
    horizon: str
    fold: int
    rmse_attention: float
    mae_attention: float
    rmse_naive: float
    mae_naive: float
    acc_attention: float
    f1_attention: float
    acc_naive: float
    f1_naive: float


def classify_feature_groups(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split numeric columns into market and context groups for attention training."""
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

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]

    context_keywords = [
        "GoogleTrends",
        "Attention",
        "Bitcoin",
        "BTC",
        "crypto market",
        "crypto news",
        "Ethereum",
        "news_count",
        "article_count",
        "sentiment",
        "Event_",
        "LLM",
        "dominant",
        "Summary",
        "Marketaux",
    ]

    market_features = []
    context_features = []

    for col in numeric_cols:
        if any(keyword.lower() in col.lower() for keyword in context_keywords):
            context_features.append(col)
        else:
            market_features.append(col)

    if len(market_features) == 0:
        raise ValueError("No market features identified.")
    if len(context_features) == 0:
        raise ValueError("No context features identified.")

    return market_features, context_features


def make_sequences(
    X_market: np.ndarray,
    X_context: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct training sequences from market/context features and targets."""
    X_market_seq = []
    X_context_seq = []
    y_seq = []
    d_seq = []

    for i in range(seq_len, len(X_market)):
        X_market_seq.append(X_market[i - seq_len : i])
        X_context_seq.append(X_context[i])
        y_seq.append(y[i])
        d_seq.append(dates[i])

    return (
        np.array(X_market_seq),
        np.array(X_context_seq),
        np.array(y_seq),
        np.array(d_seq),
    )


def get_walk_forward_splits(n_samples: int) -> List[Tuple[int, int]]:
    """Generate simple walk-forward train/test boundaries for time series samples."""
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
        if len(splits) >= MAX_FOLDS:
            break
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


def train_attention_model(
    X_market_train: np.ndarray,
    X_context_train: np.ndarray,
    y_train: np.ndarray,
    market_input_size: int,
    context_input_size: int,
) -> AttentionContextModel:
    """Train the attention-context model for one fold."""
    model = AttentionContextModel(
        market_input_size=market_input_size,
        context_input_size=context_input_size,
    ).to(DEVICE)

    dataset = AttentionContextDataset(X_market_train, X_context_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        losses = []

        for xb_market, xb_context, yb in loader:
            xb_market = xb_market.to(DEVICE)
            xb_context = xb_context.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb_market, xb_context)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        logger.info(f"      Epoch {epoch + 1}/{EPOCHS} Loss: {np.mean(losses):.6f}")

    return model


def run_horizon(
    df: pd.DataFrame,
    market_features: List[str],
    context_features: List[str],
    horizon_name: str,
    target_return_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(f"\n==================== HORIZON {horizon_name} ====================")

    use_cols = ["Date"] + market_features + context_features + [target_return_col]
    df_local = df[use_cols].copy()
    logger.info(f"Rows before cleaning: {len(df_local)}")

    df_local = df_local.dropna(subset=[target_return_col]).reset_index(drop=True)
    logger.info(f"Rows after dropping missing target: {len(df_local)}")

    feature_cols = market_features + context_features
    feature_df = df_local[feature_cols].replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.ffill().bfill()

    valid_mask = feature_df.notna().all(axis=1)
    df_local = df_local.loc[valid_mask].reset_index(drop=True)
    logger.info(f"Rows after feature cleaning: {len(df_local)}")

    X_market_all = df_local[market_features].to_numpy(dtype=np.float32)
    X_context_all = df_local[context_features].to_numpy(dtype=np.float32)
    y_all = df_local[target_return_col].astype(float).values
    d_all = pd.to_datetime(df_local["Date"]).values

    X_market_seq, X_context_seq, y_seq, d_seq = make_sequences(
        X_market_all,
        X_context_all,
        y_all,
        d_all,
        SEQUENCE_LENGTH,
    )

    logger.info(f"Sequence count for {horizon_name}: {len(X_market_seq)}")

    splits = get_walk_forward_splits(len(X_market_seq))
    logger.info(f"Number of walk-forward folds for {horizon_name}: {len(splits)}")

    if not splits:
        raise ValueError(f"No valid walk-forward splits generated for horizon {horizon_name}.")

    fold_results = []
    pred_rows = []

    for fold_idx, (train_end, test_end) in enumerate(splits, start=1):
        logger.info(f"\nFold {fold_idx}: train_end={train_end}, test_end={test_end}")

        X_market_train_raw = X_market_seq[:train_end]
        X_context_train_raw = X_context_seq[:train_end]
        y_train_raw = y_seq[:train_end]

        X_market_test_raw = X_market_seq[train_end:test_end]
        X_context_test_raw = X_context_seq[train_end:test_end]
        y_test = y_seq[train_end:test_end]
        d_test = d_seq[train_end:test_end]

        market_scaler = StandardScaler()
        X_market_train = market_scaler.fit_transform(
            X_market_train_raw.reshape(-1, X_market_train_raw.shape[-1])
        ).reshape(X_market_train_raw.shape)
        X_market_test = market_scaler.transform(
            X_market_test_raw.reshape(-1, X_market_test_raw.shape[-1])
        ).reshape(X_market_test_raw.shape)

        context_scaler = StandardScaler()
        X_context_train = context_scaler.fit_transform(X_context_train_raw)
        X_context_test = context_scaler.transform(X_context_test_raw)

        target_scaler = StandardScaler()
        y_train = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()

        model = train_attention_model(
            X_market_train=X_market_train,
            X_context_train=X_context_train,
            y_train=y_train,
            market_input_size=X_market_train.shape[-1],
            context_input_size=X_context_train.shape[-1],
        )

        model.eval()
        with torch.no_grad():
            X_market_test_tensor = torch.tensor(X_market_test, dtype=torch.float32).to(DEVICE)
            X_context_test_tensor = torch.tensor(X_context_test, dtype=torch.float32).to(DEVICE)
            y_pred_scaled = model(X_market_test_tensor, X_context_test_tensor).cpu().numpy()

        y_pred_attention = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_pred_naive = np.zeros_like(y_test)

        rmse_attention, mae_attention = evaluate_regression(y_test, y_pred_attention)
        rmse_naive, mae_naive = evaluate_regression(y_test, y_pred_naive)

        acc_attention, f1_attention = evaluate_direction(y_test, y_pred_attention)
        acc_naive, f1_naive = evaluate_direction(y_test, y_pred_naive)

        logger.info(
            f"    Attention -> RMSE: {rmse_attention:.6f}, MAE: {mae_attention:.6f}, "
            f"ACC: {acc_attention:.4f}, F1: {f1_attention:.4f}"
        )
        logger.info(
            f"    Naive     -> RMSE: {rmse_naive:.6f}, MAE: {mae_naive:.6f}, "
            f"ACC: {acc_naive:.4f}, F1: {f1_naive:.4f}"
        )

        fold_results.append(
            FoldResult(
                horizon=horizon_name,
                fold=fold_idx,
                rmse_attention=rmse_attention,
                mae_attention=mae_attention,
                rmse_naive=rmse_naive,
                mae_naive=mae_naive,
                acc_attention=acc_attention,
                f1_attention=f1_attention,
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
                    "predicted_return_attention": float(y_pred_attention[i]),
                    "predicted_return_naive": float(y_pred_naive[i]),
                    "actual_direction": int(y_test[i] > 0),
                    "predicted_direction_attention": int(y_pred_attention[i] > 0),
                    "predicted_direction_naive": int(y_pred_naive[i] > 0),
                }
            )

    fold_df = pd.DataFrame([vars(r) for r in fold_results])

    summary_df = (
        fold_df.groupby("horizon", as_index=False)
        .agg(
            rmse_attention_mean=("rmse_attention", "mean"),
            mae_attention_mean=("mae_attention", "mean"),
            rmse_naive_mean=("rmse_naive", "mean"),
            mae_naive_mean=("mae_naive", "mean"),
            acc_attention_mean=("acc_attention", "mean"),
            f1_attention_mean=("f1_attention", "mean"),
            acc_naive_mean=("acc_naive", "mean"),
            f1_naive_mean=("f1_naive", "mean"),
            num_folds=("fold", "count"),
        )
    )

    preds_df = pd.DataFrame(pred_rows)
    return fold_df, summary_df, preds_df


def main(input_file: str = INPUT_FILE, output_dir: str = RESULTS_DIR, verbose: bool = False) -> None:
    """Run the end-to-end attention-context training workflow and save results."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_parquet(input_file).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    logger.info("Dataset loaded")
    logger.info(f"Rows: {len(df)}")
    logger.info(f"Columns: {len(df.columns)}")

    market_features, context_features = classify_feature_groups(df)

    logger.info(f"\nNumber of market features: {len(market_features)}")
    logger.info(f"Number of context features: {len(context_features)}")

    logger.info("\nSample market features:")
    logger.info(f"{market_features[:10]}")

    logger.info("\nSample context features:")
    logger.info(f"{context_features[:10]}")

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
            market_features=market_features,
            context_features=context_features,
            horizon_name=horizon_name,
            target_return_col=target_col,
        )
        all_fold_dfs.append(fold_df)
        all_summary_dfs.append(summary_df)
        all_preds_dfs.append(preds_df)

    fold_results_df = pd.concat(all_fold_dfs, ignore_index=True)
    summary_results_df = pd.concat(all_summary_dfs, ignore_index=True)
    predictions_df = pd.concat(all_preds_dfs, ignore_index=True)

    fold_path = os.path.join(output_dir, "fold_metrics_attention_context.csv")
    summary_path = os.path.join(output_dir, "summary_attention_context.csv")
    preds_path = os.path.join(output_dir, "predictions_attention_context.csv")
    json_path = os.path.join(output_dir, "summary_attention_context.json")

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
    logger.info("Saved JSON summary to: %s", json_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train attention-context BTC model with walk-forward evaluation.")
    parser.add_argument("--input", type=str, default=INPUT_FILE, help="Input features parquet file")
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR, help="Directory where results will be saved")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(input_file=args.input, output_dir=args.output_dir, verbose=args.verbose)