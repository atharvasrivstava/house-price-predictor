"""Train the from-scratch logistic regression classifier.

Predicts whether a Pune house listing is "expensive" (price > threshold lakhs)
versus "affordable". Must be run AFTER `python -m src.train`, because it reads
the cleaned dataset and the feature-column ordering produced by the regressor.

Run from the project root:
    python -m src.train_classifier
    python -m src.train_classifier --price-threshold 75 --lam 0.5

Saves:
    models/classifier_weights.npz   trained w, b
    models/classifier_scaler.npz    mu, sigma (separate from the regressor's)
    models/classifier_meta.json     {price_threshold, decision_threshold, lam, ...}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .logistic_regression import LogisticRegressionGD
from .preprocess import build_feature_matrix
from .scaler import ZScoreScaler
from .train import train_test_split


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-threshold", type=float, default=100.0,
                        help="Lakhs; listings above this are labeled Expensive (1).")
    parser.add_argument("--decision-threshold", type=float, default=0.5,
                        help="Probability cutoff for class assignment at inference.")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--lam", type=float, default=0.1, help="L2 regularization strength.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    processed_csv = project_root / "data" / "processed" / "pune_clean.csv"
    models_dir = project_root / "models"
    feature_cols_path = models_dir / "feature_columns.json"

    if not processed_csv.exists() or not feature_cols_path.exists():
        print(
            "Missing prerequisites. Run the regressor first:\n"
            "    python -m src.train --csv data/raw/pune_houses.csv\n"
            f"Expected files: {processed_csv} and {feature_cols_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading cleaned dataset from {processed_csv} ...")
    df = pd.read_csv(processed_csv)
    print(f"Rows: {len(df)}")

    # Build labels BEFORE dropping price from the feature frame.
    y = (df["price"].to_numpy(dtype=float) > args.price_threshold).astype(int)
    n_pos = int(y.sum())
    print(f"Label: price > {args.price_threshold} L  ->  positives: {n_pos} / {len(y)} "
          f"({100.0 * n_pos / len(y):.1f}%)")

    # Use the SAME feature columns the regressor was trained on. We pass
    # feature_columns to build_feature_matrix so the column order is locked.
    with open(feature_cols_path) as f:
        feature_columns = json.load(f)

    # build_feature_matrix drops 'price' internally if present
    X, _, _ = build_feature_matrix(df, feature_columns=feature_columns)
    print(f"Feature matrix: {X.shape} (columns aligned with regressor)")

    X_train, X_test, y_train, y_test = train_test_split(X, y.astype(float))
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    scaler = ZScoreScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"\nTraining LogisticRegressionGD "
          f"(alpha={args.alpha}, iters={args.iters}, lam={args.lam}) ...")
    model = LogisticRegressionGD(
        alpha=args.alpha, n_iters=args.iters, lam=args.lam, verbose=True
    ).fit(X_train_s, y_train.astype(float))

    train_m = classification_metrics(
        y_train, model.predict(X_train_s, args.decision_threshold)
    )
    test_m = classification_metrics(
        y_test, model.predict(X_test_s, args.decision_threshold)
    )
    print("\n=== From-scratch classifier ===")
    print(f"  Train -> acc={train_m['accuracy']:.4f}  prec={train_m['precision']:.4f}  "
          f"rec={train_m['recall']:.4f}  f1={train_m['f1']:.4f}")
    print(f"  Test  -> acc={test_m['accuracy']:.4f}  prec={test_m['precision']:.4f}  "
          f"rec={test_m['recall']:.4f}  f1={test_m['f1']:.4f}")
    print(f"  Test confusion: TP={test_m['tp']} TN={test_m['tn']} "
          f"FP={test_m['fp']} FN={test_m['fn']}")

    model.save(str(models_dir / "classifier_weights.npz"))
    scaler.save(str(models_dir / "classifier_scaler.npz"))
    meta = {
        "price_threshold": args.price_threshold,
        "decision_threshold": args.decision_threshold,
        "lam": args.lam,
        "alpha": args.alpha,
        "iters": args.iters,
    }
    with open(models_dir / "classifier_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved classifier artifacts to {models_dir}/")


if __name__ == "__main__":
    main()
