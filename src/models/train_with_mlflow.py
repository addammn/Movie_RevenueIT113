# src/models/train_with_mlflow.py
from __future__ import annotations
import os, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED = 42
TEST_SIZE = 0.2

DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
ARTIFACT_TMP = Path("reports/_tmp_artifacts")
ARTIFACT_TMP.mkdir(parents=True, exist_ok=True)

def rmse(y_true, y_pred): 
    return math.sqrt(mean_squared_error(y_true, y_pred))

def regression_report(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

def save_residual_plot(y_true, y_pred, outpath: Path, title: str):
    plt.figure(figsize=(8,5))
    residuals = y_true - y_pred
    plt.hist(residuals, bins=60)
    plt.title(f"Residuals – {title}")
    plt.xlabel("y_true - y_pred")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()

def run_linear_raw(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="linear_raw", nested=True):
        mlflow.set_tags({"stage": "baseline", "target": "revenue"})
        # Model
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Metrics
        mets = regression_report(y_test.values, y_pred)
        for k, v in mets.items(): mlflow.log_metric(k, v)
        # Artifacts
        preds = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
        pth = ARTIFACT_TMP / "preds_linear_raw.csv"
        preds.to_csv(pth, index=False)
        mlflow.log_artifact(str(pth), artifact_path="predictions")
        img = ARTIFACT_TMP / "residuals_linear_raw.png"
        save_residual_plot(y_test.values, y_pred, img, "Linear (raw)")
        mlflow.log_artifact(str(img), artifact_path="plots")
        # Model
        mlflow.sklearn.log_model(model, artifact_path="model")

        return {"model": "linear_raw", **mets}

def run_linear_log(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="linear_log_revenue", nested=True):
        mlflow.set_tags({"stage": "baseline", "target": "log1p(revenue)"})
        # Train on log1p target
        y_train_log = np.log1p(y_train.clip(lower=0))
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])
        model.fit(X_train, y_train_log)
        y_pred_back = np.expm1(model.predict(X_test))
        mets = regression_report(y_test.values, y_pred_back)
        for k, v in mets.items(): mlflow.log_metric(k, v)
        preds = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred_back})
        pth = ARTIFACT_TMP / "preds_linear_log_back.csv"
        preds.to_csv(pth, index=False)
        mlflow.log_artifact(str(pth), artifact_path="predictions")
        img = ARTIFACT_TMP / "residuals_linear_log_back.png"
        save_residual_plot(y_test.values, y_pred_back, img, "Linear (log->back)")
        mlflow.log_artifact(str(img), artifact_path="plots")
        mlflow.sklearn.log_model(model, artifact_path="model")

        return {"model": "linear_log_revenue_expm1_back", **mets}

def run_rf(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="rf_raw", nested=True):
        mlflow.set_tags({"stage": "baseline", "target": "revenue"})
        rf = RandomForestRegressor(
            n_estimators=300, random_state=SEED, n_jobs=-1, max_depth=None
        )
        mlflow.log_params({"n_estimators": 300, "max_depth": None})
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        mets = regression_report(y_test.values, y_pred)
        for k, v in mets.items(): mlflow.log_metric(k, v)
        preds = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
        pth = ARTIFACT_TMP / "preds_rf_raw.csv"
        preds.to_csv(pth, index=False)
        mlflow.log_artifact(str(pth), artifact_path="predictions")
        img = ARTIFACT_TMP / "residuals_rf.png"
        save_residual_plot(y_test.values, y_pred, img, "RandomForest")
        mlflow.log_artifact(str(img), artifact_path="plots")
        mlflow.sklearn.log_model(rf, artifact_path="model")

        return {"model": "rf_raw_revenue_300", **mets}

def main():
    # ---------- Setup tracking ----------
    mlflow.set_tracking_uri("file:./mlruns")  # local folder
    mlflow.set_experiment("Movie_RevenueIT113")

    # ---------- Load processed data ----------
    X = pd.read_csv(DATA_DIR / "X_features.csv")
    y = pd.read_csv(DATA_DIR / "y_target.csv")["revenue"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # ---------- Parent run to group baselines ----------
    with mlflow.start_run(run_name="baseline_suite") as parent:
        mlflow.set_tags({
            "project": "Movie_RevenueIT113",
            "step": "baseline_with_tracking",
            "seed": str(SEED),
            "test_size": str(TEST_SIZE),
        })
        # Log data/feature specs as artifacts for traceability
        for p in ["reports/feature_spec.json", "reports/data_prep_decisions.md"]:
            if Path(p).exists():
                mlflow.log_artifact(p, artifact_path="specs")

        results = []
        results.append(run_linear_raw(X_train, y_train, X_test, y_test))
        results.append(run_linear_log(X_train, y_train, X_test, y_test))
        results.append(run_rf(X_train, y_train, X_test, y_test))

        # Save a consolidated results table as an artifact of the parent run
        df = pd.DataFrame(results)
        out = ARTIFACT_TMP / "mlflow_baseline_results.csv"
        df.to_csv(out, index=False)
        mlflow.log_artifact(str(out), artifact_path="results")

        print("✅ Logged runs to MLflow. Results:")
        print(df.sort_values("RMSE"))

if __name__ == "__main__":
    main()
