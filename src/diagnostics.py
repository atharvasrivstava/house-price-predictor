"""Generate the four diagnostic plots that visually demonstrate the concepts.

Run:
    python -m src.diagnostics --csv data/raw/pune_houses.csv

Outputs to notebooks/figures/:
    1_cost_convergence.png       cost J(w,b) vs iteration for chosen alpha
    2_learning_rates.png         overlay of cost curves for several alphas
    3_scaling_impact.png         convergence with vs without feature scaling
    4_pred_vs_actual.png         scatter of predicted vs actual test prices

These plots are the artifacts to put in the README and LinkedIn post.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .linear_regression import LinearRegressionGD
from .preprocess import build_feature_matrix, clean, load_raw
from .scaler import ZScoreScaler
from .train import train_test_split


def plot_cost_convergence(model: LinearRegressionGD, out_path: Path):
    plt.figure(figsize=(7, 4))
    plt.plot(model.cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(w, b)")
    plt.title(f"Cost convergence (alpha={model.alpha})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_learning_rates(X_train, y_train, out_path: Path, alphas=(0.001, 0.01, 0.1, 1.0), n_iters=500):
    plt.figure(figsize=(7, 4))
    for a in alphas:
        m = LinearRegressionGD(alpha=a, n_iters=n_iters).fit(X_train, y_train)
        hist = np.array(m.cost_history)
        # Clip exploding curves for plot readability
        hist = np.clip(hist, None, 1e12)
        plt.plot(hist, label=f"alpha = {a}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(w, b)")
    plt.yscale("log")
    plt.title("Effect of learning rate on convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_scaling_impact(X_train, y_train, out_path: Path, alpha=0.01, n_iters=500):
    scaler = ZScoreScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)

    # Unscaled training needs a much smaller alpha to not diverge; we use the
    # same alpha for both to make the visualization honest about WHY scaling
    # matters (raw features make GD unstable at sensible learning rates).
    m_unscaled = LinearRegressionGD(alpha=alpha, n_iters=n_iters).fit(X_train, y_train)
    m_scaled = LinearRegressionGD(alpha=alpha, n_iters=n_iters).fit(X_scaled, y_train)

    plt.figure(figsize=(7, 4))
    h_un = np.clip(np.array(m_unscaled.cost_history), None, 1e15)
    h_sc = np.array(m_scaled.cost_history)
    plt.plot(h_un, label="Unscaled features")
    plt.plot(h_sc, label="Z-score scaled features")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(w, b)")
    plt.yscale("log")
    plt.title(f"Feature scaling impact on convergence (alpha={alpha})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_pred_vs_actual(model, X_test, y_test, out_path: Path):
    y_pred = model.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, s=12)
    lo, hi = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", label="y = x")
    plt.xlabel("Actual price (lakhs)")
    plt.ylabel("Predicted price (lakhs)")
    plt.title("Predicted vs actual (test set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=2000)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    fig_dir = project_root / "notebooks" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.csv)
    cleaned = clean(raw)
    X, y, _ = build_feature_matrix(cleaned)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = ZScoreScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegressionGD(alpha=args.alpha, n_iters=args.iters).fit(X_train_s, y_train)

    print("Plotting...")
    plot_cost_convergence(model, fig_dir / "1_cost_convergence.png")
    plot_learning_rates(X_train_s, y_train, fig_dir / "2_learning_rates.png")
    plot_scaling_impact(X_train, y_train, fig_dir / "3_scaling_impact.png")
    plot_pred_vs_actual(model, X_test_s, y_test, fig_dir / "4_pred_vs_actual.png")
    print(f"Saved plots to {fig_dir}/")


if __name__ == "__main__":
    main()
