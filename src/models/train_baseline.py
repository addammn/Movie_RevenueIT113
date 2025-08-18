# src/models/train_baseline.py
from __future__ import annotations
import os, json, math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

SEED = 42
TEST_SIZE = 0.2

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def regression_report(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

def main():
    # --- Load processed features & target from Step 3 ---
    X = pd.read_csv(DATA_DIR / "X_features.csv")
    y = pd.read_csv(DATA_DIR / "y_target.csv")["revenue"]

    assert len(X) == len(y), "X and y must have same length"

    # Train/valid split (fixed seed for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # --- 0) Naive baseline: predict the training-mean revenue ---
    naive_mean = float(y_train.mean())
    y_pred_naive = np.full_like(y_test.values, fill_value=naive_mean, dtype=float)
    res_naive = regression_report(y_test.values, y_pred_naive)
    results.append({"model": "naive_mean_revenue", **res_naive})

    # --- 1) Linear Regression on RAW revenue ---
    lin_raw = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LinearRegression(n_jobs=None))  # n_jobs not used by LinearRegression, kept simple
    ])
    lin_raw.fit(X_train, y_train)
    y_pred_lin_raw = lin_raw.predict(X_test)
    res_lin_raw = regression_report(y_test.values, y_pred_lin_raw)
    results.append({"model": "linear_raw_revenue", **res_lin_raw})
    joblib.dump(lin_raw, MODELS_DIR / "linear_raw.pkl")

    # --- 2) Linear Regression on LOG revenue ---
    # Train on log1p target, then expm1 predictions back to original units for metrics.
    y_train_log = np.log1p(y_train.clip(lower=0))
    lin_log = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LinearRegression())
    ])
    lin_log.fit(X_train, y_train_log)
    y_pred_log = lin_log.predict(X_test)
    y_pred_lin_log_back = np.expm1(y_pred_log)
    res_lin_log = regression_report(y_test.values, y_pred_lin_log_back)
    results.append({"model": "linear_log_revenue_expm1_back", **res_lin_log})
    joblib.dump(lin_log, MODELS_DIR / "linear_log.pkl")

    # --- (Optional) 3) Quick tree baseline (often strong without much tuning) ---
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
        max_depth=None,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    res_rf = regression_report(y_test.values, y_pred_rf)
    results.append({"model": "rf_raw_revenue_300", **res_rf})
    joblib.dump(rf, MODELS_DIR / "rf_raw.pkl")

    # --- Save results table ---
    df_results = pd.DataFrame(results)
    df_results.to_csv(REPORTS_DIR / "baseline_results.csv", index=False)

    # --- Save predictions for error analysis / plots later ---
    preds_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred_naive": y_pred_naive,
        "y_pred_linear_raw": y_pred_lin_raw,
        "y_pred_linear_log_back": y_pred_lin_log_back,
        "y_pred_rf": y_pred_rf,
    })
    preds_df.to_csv(REPORTS_DIR / "baseline_predictions.csv", index=False)

    # Also dump a short JSON summary for quick reading
    summary = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": results,
        "features_used": list(X.columns),
        "seed": SEED,
        "test_size": TEST_SIZE,
    }
    with open(REPORTS_DIR / "baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("✅ Baseline training complete.")
    print(df_results.sort_values("RMSE"))

if __name__ == "__main__":
    main()
