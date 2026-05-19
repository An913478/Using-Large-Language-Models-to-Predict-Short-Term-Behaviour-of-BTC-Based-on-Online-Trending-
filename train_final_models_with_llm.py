"""Train final LSTM baseline models on BTC features enriched with LLM signals.

This script runs walk-forward experiments over several feature subsets,
trains an LSTM regressor for next-day return prediction, and exports fold-level
metrics, direction metrics, and predictions for each feature set.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "btc_final_features_with_llm.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"

SEQUENCE_LENGTH = 14
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

INITIAL_TRAIN_RATIO = 0.60
TEST_WINDOW = 60
STEP_SIZE = 60

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMRegressor(nn.Module):
    """Simple LSTM regression model for next-day return forecasting."""

    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def create_sequences(features, targets, seq_len):
    """Convert raw feature and target arrays into rolling sequences.

    Each sample is a historical window of length ``seq_len`` and the target at
    the next time step.
    """
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X), np.array(y)


def train_regressor(model, X_train, y_train, X_test):
    """Train the LSTM regressor on one fold and return test predictions.

    Uses batch training with Adam and returns predictions on the held-out test set.
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
            idx = permutation[i:i + BATCH_SIZE]
            batch_x = X_train_t[idx]
            batch_y = y_train_t[idx]

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"      Epoch {epoch + 1}/{EPOCHS} Loss: {epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()

    return preds


def run_walkforward(df, feature_cols, label):
    """Run a walk-forward experiment for one feature set.

    Constructs rolling sequences, trains an LSTM on each fold, and records both
    regression and direction metrics for the model and a naive zero baseline.
    """
    print(f"\n=== Running walk-forward: {label} ===")

    features = df[feature_cols].values
    target_return = df["Target_Return_NextDay"].values.reshape(-1, 1)
    target_direction = df["Target_Direction"].values.reshape(-1, 1)
    dates = pd.to_datetime(df["Date"]).reset_index(drop=True)

    initial_train_size = int(len(df) * INITIAL_TRAIN_RATIO)

    regression_rows = []
    direction_rows = []
    prediction_rows = []

    train_end = initial_train_size
    fold = 1

    while train_end + TEST_WINDOW <= len(df):
        print(f"\nFold {fold}: train_end={train_end}, test_end={train_end + TEST_WINDOW}")

        test_end = train_end + TEST_WINDOW

        # Scale features and returns using only the training portion of each fold.
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        feature_scaler.fit(features[:train_end])
        target_scaler.fit(target_return[:train_end])

        X_scaled = feature_scaler.transform(features)
        y_scaled = target_scaler.transform(target_return)

        X_all, y_all = create_sequences(X_scaled, y_scaled, SEQUENCE_LENGTH)
        _, y_dir_all = create_sequences(X_scaled, target_direction, SEQUENCE_LENGTH)

        seq_train_end = train_end - SEQUENCE_LENGTH
        seq_test_end = test_end - SEQUENCE_LENGTH

        X_train = X_all[:seq_train_end]
        X_test = X_all[seq_train_end:seq_test_end]

        y_train = y_all[:seq_train_end]
        y_test = y_all[seq_train_end:seq_test_end]
        y_dir_test = y_dir_all[seq_train_end:seq_test_end].astype(int).flatten()

        # Dates aligned to targets after rolling sequence creation.
        test_dates = dates.iloc[train_end:test_end].reset_index(drop=True)

        model = LSTMRegressor(input_size=X_train.shape[2]).to(DEVICE)
        preds_scaled = train_regressor(model, X_train, y_train, X_test)

        y_test_rescaled = target_scaler.inverse_transform(y_test).flatten()
        preds_rescaled = target_scaler.inverse_transform(preds_scaled).flatten()

        naive_preds = np.zeros_like(y_test_rescaled)

        rmse_lstm = math.sqrt(mean_squared_error(y_test_rescaled, preds_rescaled))
        mae_lstm = mean_absolute_error(y_test_rescaled, preds_rescaled)

        rmse_naive = math.sqrt(mean_squared_error(y_test_rescaled, naive_preds))
        mae_naive = mean_absolute_error(y_test_rescaled, naive_preds)

        pred_dir = (preds_rescaled > 0).astype(int)
        naive_dir = np.zeros_like(pred_dir)

        acc_lstm = accuracy_score(y_dir_test, pred_dir)
        f1_lstm = f1_score(y_dir_test, pred_dir, zero_division=0)

        acc_naive = accuracy_score(y_dir_test, naive_dir)
        f1_naive = f1_score(y_dir_test, naive_dir, zero_division=0)

        print(f"    LSTM  -> RMSE: {rmse_lstm:.6f}, MAE: {mae_lstm:.6f}, ACC: {acc_lstm:.4f}, F1: {f1_lstm:.4f}")
        print(f"    Naive -> RMSE: {rmse_naive:.6f}, MAE: {mae_naive:.6f}, ACC: {acc_naive:.4f}, F1: {f1_naive:.4f}")

        regression_rows.append({
            "fold": fold,
            "rmse_lstm": rmse_lstm,
            "mae_lstm": mae_lstm,
            "rmse_naive": rmse_naive,
            "mae_naive": mae_naive,
        })

        direction_rows.append({
            "fold": fold,
            "acc_lstm": acc_lstm,
            "f1_lstm": f1_lstm,
            "acc_naive": acc_naive,
            "f1_naive": f1_naive,
        })

        fold_pred_df = pd.DataFrame({
            "Date": test_dates,
            "fold": fold,
            "model_set": label,
            "actual_return": y_test_rescaled,
            "predicted_return_lstm": preds_rescaled,
            "predicted_return_naive": naive_preds,
            "actual_direction": y_dir_test,
            "predicted_direction_lstm": pred_dir,
            "predicted_direction_naive": naive_dir,
        })
        prediction_rows.append(fold_pred_df)

        train_end += STEP_SIZE
        fold += 1

    reg_df = pd.DataFrame(regression_rows)
    dir_df = pd.DataFrame(direction_rows)
    pred_df = pd.concat(prediction_rows, ignore_index=True)

    summary = {
        "model_set": label,
        "rmse_lstm_mean": reg_df["rmse_lstm"].mean(),
        "mae_lstm_mean": reg_df["mae_lstm"].mean(),
        "rmse_naive_mean": reg_df["rmse_naive"].mean(),
        "mae_naive_mean": reg_df["mae_naive"].mean(),
        "acc_lstm_mean": dir_df["acc_lstm"].mean(),
        "f1_lstm_mean": dir_df["f1_lstm"].mean(),
        "acc_naive_mean": dir_df["acc_naive"].mean(),
        "f1_naive_mean": dir_df["f1_naive"].mean(),
        "num_folds": len(reg_df),
    }

    return reg_df, dir_df, pred_df, summary


def main():
    """Load data, run walk-forward experiments for each feature set, and save results."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_FILE).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    market_only_cols = [
        "Close", "Volume",
        "Return", "LogReturn", "Volatility_7d",
        "MA_7", "MA_14", "Volume_Change",
        "Momentum_3d", "Momentum_7d"
    ]

    market_trends_cols = market_only_cols + [
        "GoogleTrends",
        "Bitcoin", "BTC price", "Bitcoin crash", "Bitcoin rally",
        "crypto market", "crypto news", "Ethereum",
        "AttentionIndex_Mean", "AttentionIndex_Max", "AttentionIndex_Std",
        "AttentionIndex_Change", "AttentionIndex_MA_7", "AttentionIndex_MA_14",
        "AttentionVolatility_7d", "AttentionVolatility_14d", "AttentionSpike",
    ]

    market_trends_llm_cols = market_trends_cols + [
        "ArticleCount", "UniqueSourceCount", "MeanRelevanceScore", "MeanBTCEntitySentiment",
        "sentiment_score", "bullish_score", "bearish_score", "uncertainty_score", "market_impact_score",
        "mentions_etf", "mentions_regulation", "mentions_exchange",
        "mentions_hack_or_security", "mentions_institutional_adoption",
        "Event_macro", "Event_regulation", "Event_etf", "Event_exchange",
        "Event_security", "Event_adoption", "Event_mining", "Event_technical", "Event_other"
    ]

    summaries = []
    all_predictions = []

    for label, cols in [
        ("market_only", market_only_cols),
        ("market_plus_trends", market_trends_cols),
        ("market_plus_trends_plus_llm", market_trends_llm_cols),
    ]:
        reg_df, dir_df, pred_df, summary = run_walkforward(df, cols, label)

        reg_df.to_csv(RESULTS_DIR / f"{label}_regression_metrics.csv", index=False)
        dir_df.to_csv(RESULTS_DIR / f"{label}_direction_metrics.csv", index=False)
        pred_df.to_csv(RESULTS_DIR / f"{label}_predictions.csv", index=False)

        summaries.append(summary)
        all_predictions.append(pred_df)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULTS_DIR / "final_model_comparison_summary.csv", index=False)

    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    combined_predictions.to_csv(RESULTS_DIR / "all_model_predictions.csv", index=False)

    with open(RESULTS_DIR / "final_model_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("\n=== FINAL SUMMARY ===")
    print(summary_df)


if __name__ == "__main__":
    main()