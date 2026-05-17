"""Linear regression from scratch — vectorized batch gradient descent.

This is the centerpiece of the project. Every line corresponds directly to a
concept from Andrew Ng's ML Specialization (Course 1, Weeks 1-2):

    Model:        f(x) = w . x + b           (vectorized: X @ w + b)
    Cost (MSE):   J(w,b) = (1/2m) * sum((f(x) - y)^2)
    Gradients:    dJ/dw = (1/m) * X.T @ (f(x) - y)
                  dJ/db = (1/m) * sum(f(x) - y)
    Update rule:  w := w - alpha * dJ/dw
                  b := b - alpha * dJ/db

No Python loops over training examples — everything is NumPy matrix ops.
"""

from __future__ import annotations

import numpy as np


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b


def compute_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    m = X.shape[0]
    err = predict(X, w, b) - y
    return float((err @ err) / (2 * m))


def compute_gradients(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> tuple[np.ndarray, float]:
    m = X.shape[0]
    err = predict(X, w, b) - y          # shape (m,)
    dw = (X.T @ err) / m                # shape (n,)
    db = float(err.mean())
    return dw, db


class LinearRegressionGD:
    """Linear regression trained by batch gradient descent."""

    def __init__(self, alpha: float = 0.01, n_iters: int = 1000, verbose: bool = False):
        self.alpha = alpha
        self.n_iters = n_iters
        self.verbose = verbose
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.cost_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        self.cost_history = []

        for i in range(self.n_iters):
            dw, db = compute_gradients(X, y, self.w, self.b)
            self.w = self.w - self.alpha * dw
            self.b = self.b - self.alpha * db

            cost = compute_cost(X, y, self.w, self.b)
            self.cost_history.append(cost)

            if self.verbose and (i % max(1, self.n_iters // 10) == 0):
                print(f"iter {i:>6d} | cost = {cost:.6f}")

            # Stop early if cost has blown up (alpha too large).
            if not np.isfinite(cost):
                if self.verbose:
                    print(f"Cost diverged at iter {i}; stopping.")
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return predict(X, self.w, self.b)

    def save(self, path: str) -> None:
        np.savez(path, w=self.w, b=np.array(self.b))

    @classmethod
    def load(cls, path: str) -> "LinearRegressionGD":
        data = np.load(path)
        model = cls()
        model.w = data["w"]
        model.b = float(data["b"])
        return model
