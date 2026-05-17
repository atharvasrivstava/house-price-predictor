"""Sanity-check the from-scratch model against scikit-learn's LinearRegression.

Run:
    python -m src.evaluate --csv data/raw/pune_houses.csv

The two models won't be identical (sklearn solves the normal equation in
closed form; we run batch gradient descent), but on a well-scaled dataset
the metrics should match within a small margin. That parity is the proof
that the from-scratch implementation is correct.
"""

from __future__ import annotations

import argparse

from sklearn.linear_model import LinearRegression

from .linear_regression import LinearRegressionGD
from .preprocess import build_feature_matrix, clean, load_raw
from .scaler import ZScoreScaler
from .train import metrics, train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=2000)
    args = parser.parse_args()

    raw = load_raw(args.csv)
    cleaned = clean(raw)
    X, y, _ = build_feature_matrix(cleaned)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = ZScoreScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    ours = LinearRegressionGD(alpha=args.alpha, n_iters=args.iters).fit(X_train_s, y_train)
    theirs = LinearRegression().fit(X_train_s, y_train)

    ours_m = metrics(y_test, ours.predict(X_test_s))
    theirs_m = metrics(y_test, theirs.predict(X_test_s))

    print(f"{'':25s}  {'MAE (L)':>10s}  {'RMSE (L)':>10s}  {'R^2':>8s}")
    print(f"{'From-scratch (GD)':25s}  {ours_m['mae']:>10.3f}  {ours_m['rmse']:>10.3f}  {ours_m['r2']:>8.4f}")
    print(f"{'sklearn (normal eq.)':25s}  {theirs_m['mae']:>10.3f}  {theirs_m['rmse']:>10.3f}  {theirs_m['r2']:>8.4f}")


if __name__ == "__main__":
    main()
