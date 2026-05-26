"""Logistic regression from scratch — vectorized batch gradient descent with L2.

Every line corresponds directly to a concept from Andrew Ng's ML Specialization
(Course 1, Week 3):

    Sigmoid:           g(z) = 1 / (1 + e^-z)
    Model (proba):     f(x) = g(w . x + b)              (vectorized: g(X @ w + b))
    BCE cost + L2:     J(w,b) = -(1/m) Σ[y log(p) + (1-y) log(1-p)]
                              + (lam / (2m)) Σ w_j^2     (bias b is NOT regularized)
    Gradients:         dJ/dw = (1/m) X.T @ (p - y) + (lam/m) w
                       dJ/db = (1/m) Σ (p - y)
    Update rule:       w := w - alpha * dJ/dw
                       b := b - alpha * dJ/db

Mirrors the structure of `src/linear_regression.py` so the two implementations
read side-by-side. No Python loops over training examples.
"""

from __future__ import annotations

import numpy as np


_EPS = 1e-12  # avoid log(0) inside BCE


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip to avoid overflow in exp for very large negative z. The clipped
    # range (-500, 500) corresponds to sigmoid values 0 and 1 within float64
    # precision, so behavior at the asymptotes is preserved.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)


def compute_cost(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lam: float = 0.0
) -> float:
    m = X.shape[0]
    p = predict_proba(X, w, b)
    bce = -np.mean(y * np.log(p + _EPS) + (1 - y) * np.log(1 - p + _EPS))
    reg = (lam / (2 * m)) * float(w @ w)
    return float(bce + reg)


def compute_gradients(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lam: float = 0.0
) -> tuple[np.ndarray, float]:
    m = X.shape[0]
    err = predict_proba(X, w, b) - y          # shape (m,)
    dw = (X.T @ err) / m + (lam / m) * w       # L2 term on weights only
    db = float(err.mean())
    return dw, db


class LogisticRegressionGD:
    """Logistic regression trained by batch gradient descent with optional L2."""

    def __init__(
        self,
        alpha: float = 0.1,
        n_iters: int = 2000,
        lam: float = 0.0,
        verbose: bool = False,
    ):
        self.alpha = alpha
        self.n_iters = n_iters
        self.lam = lam
        self.verbose = verbose
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.cost_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        self.cost_history = []

        for i in range(self.n_iters):
            dw, db = compute_gradients(X, y, self.w, self.b, self.lam)
            self.w = self.w - self.alpha * dw
            self.b = self.b - self.alpha * db

            cost = compute_cost(X, y, self.w, self.b, self.lam)
            self.cost_history.append(cost)

            if self.verbose and (i % max(1, self.n_iters // 10) == 0):
                print(f"iter {i:>6d} | cost = {cost:.6f}")

            if not np.isfinite(cost):
                if self.verbose:
                    print(f"Cost diverged at iter {i}; stopping.")
                break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return predict_proba(X, self.w, self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str) -> None:
        np.savez(path, w=self.w, b=np.array(self.b))

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionGD":
        data = np.load(path)
        model = cls()
        model.w = data["w"]
        model.b = float(data["b"])
        return model
