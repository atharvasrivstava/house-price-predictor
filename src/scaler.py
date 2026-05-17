"""Z-score feature scaling, implemented from scratch.

Mirrors the formula from Andrew Ng's ML Specialization (Course 1, Week 2):
    x_scaled = (x - mu) / sigma

We persist mu and sigma so the Streamlit app can scale a user's input row
identically to how the training data was scaled.
"""

import numpy as np


class ZScoreScaler:
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, X: np.ndarray) -> "ZScoreScaler":
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        # Guard against zero-variance columns (e.g., a one-hot column that ended
        # up all zeros in this split). Leave those columns untouched.
        self.sigma = np.where(self.sigma == 0, 1.0, self.sigma)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sigma

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        np.savez(path, mu=self.mu, sigma=self.sigma)

    @classmethod
    def load(cls, path: str) -> "ZScoreScaler":
        data = np.load(path)
        s = cls()
        s.mu = data["mu"]
        s.sigma = data["sigma"]
        return s
