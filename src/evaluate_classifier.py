"""Sanity-check the from-scratch classifier against sklearn.LogisticRegression.

Run:
    python -m src.evaluate_classifier
    python -m src.evaluate_classifier --price-threshold 75

Trains both models on the same scaled split and prints a side-by-side table of
accuracy / precision / recall / F1. The two models won't be byte-identical
(sklearn uses L-BFGS; we use batch gradient descent), but the metrics should
match within a small margin — the proof that the from-scratch implementation
is correct.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .logistic_regression import LogisticRegressionGD
from .preprocess import build_feature_matrix
from .scaler import ZScoreScaler
from .train import train_test_split
from .train_classifier import classification_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-threshold", type=float, default=100.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--lam", type=float, default=0.1)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    processed_csv = project_root / "data" / "processed" / "pune_clean.csv"
    feature_cols_path = project_root / "models" / "feature_columns.json"

    if not processed_csv.exists() or not feature_cols_path.exists():
        print("Run `python -m src.train` first to produce the prerequisites.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(processed_csv)
    y = (df["price"].to_numpy(dtype=float) > args.price_threshold).astype(int)
    with open(feature_cols_path) as f:
        feature_columns = json.load(f)
    X, _, _ = build_feature_matrix(df, feature_columns=feature_columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y.astype(float))
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    scaler = ZScoreScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    ours = LogisticRegressionGD(
        alpha=args.alpha, n_iters=args.iters, lam=args.lam
    ).fit(X_train_s, y_train.astype(float))

    # sklearn's `C` is the INVERSE of regularization strength; map our lam -> C.
    # When lam == 0 we pass a very large C to approximate "no regularization".
    # sklearn defaults to L2 regularization, so we don't pass `penalty` here
    # (that argument was deprecated in sklearn 1.8).
    sk_C = 1.0 / args.lam if args.lam > 0 else 1e12
    theirs = LogisticRegression(C=sk_C, solver="lbfgs", max_iter=2000)
    theirs.fit(X_train_s, y_train)

    ours_m = classification_metrics(y_test, ours.predict(X_test_s))
    theirs_m = classification_metrics(y_test, theirs.predict(X_test_s))

    print(f"{'':28s}  {'Accuracy':>9s}  {'Precision':>10s}  {'Recall':>8s}  {'F1':>6s}")
    print(f"{'From-scratch (GD + L2)':28s}  {ours_m['accuracy']:>9.4f}  "
          f"{ours_m['precision']:>10.4f}  {ours_m['recall']:>8.4f}  {ours_m['f1']:>6.4f}")
    print(f"{'sklearn (L-BFGS + L2)':28s}  {theirs_m['accuracy']:>9.4f}  "
          f"{theirs_m['precision']:>10.4f}  {theirs_m['recall']:>8.4f}  {theirs_m['f1']:>6.4f}")


if __name__ == "__main__":
    main()
