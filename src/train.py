"""Train the from-scratch linear regression model on the Pune housing data.

Run from the project root:
    python -m src.train --csv data/raw/pune_houses.csv

Saves:
    models/weights.npz       trained w, b
    models/scaler.npz        mu, sigma
    models/feature_columns.json   ordered list of feature names
    data/processed/pune_clean.csv  cleaned dataset (used by the Streamlit app)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .linear_regression import LinearRegressionGD
from .preprocess import build_feature_matrix, clean, load_raw
from .scaler import ZScoreScaler


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to raw house data CSV")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--top-localities", type=int, default=30)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    processed_dir = project_root / "data" / "processed"
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} ...")
    raw = load_raw(args.csv)
    print(f"Raw rows: {len(raw)}")

    cleaned = clean(raw, top_n_localities=args.top_localities)
    print(f"Cleaned rows: {len(cleaned)}")
    cleaned.to_csv(processed_dir / "pune_clean.csv", index=False)

    X, y, feature_names = build_feature_matrix(cleaned)
    print(f"Feature matrix: {X.shape}, features: {len(feature_names)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = ZScoreScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"\nTraining LinearRegressionGD (alpha={args.alpha}, iters={args.iters}) ...")
    model = LinearRegressionGD(alpha=args.alpha, n_iters=args.iters, verbose=True).fit(X_train_s, y_train)

    train_m = metrics(y_train, model.predict(X_train_s))
    test_m = metrics(y_test, model.predict(X_test_s))
    print("\n=== From-scratch model ===")
    print(f"  Train -> MAE={train_m['mae']:.2f}L  RMSE={train_m['rmse']:.2f}L  R^2={train_m['r2']:.4f}")
    print(f"  Test  -> MAE={test_m['mae']:.2f}L  RMSE={test_m['rmse']:.2f}L  R^2={test_m['r2']:.4f}")

    model.save(str(models_dir / "weights.npz"))
    scaler.save(str(models_dir / "scaler.npz"))
    with open(models_dir / "feature_columns.json", "w") as f:
        json.dump(feature_names, f)

    print(f"\nSaved model + scaler + feature columns to {models_dir}/")


if __name__ == "__main__":
    main()
