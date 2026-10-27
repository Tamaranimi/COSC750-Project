from __future__ import annotations

import numpy as np


class DiagonalGaussianBayes:
    """Two-class Gaussian Bayes model with diagonal covariance matrices.

    This is intentionally implemented with NumPy for the course classical baseline
    and the proposed minimum-risk decision rule.
    """

    def __init__(self, var_floor: float = 1e-6):
        if var_floor <= 0:
            raise ValueError("var_floor must be positive")
        self.var_floor = float(var_floor)
        self.means_: dict[int, np.ndarray] = {}
        self.vars_: dict[int, np.ndarray] = {}
        self.priors_: dict[int, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DiagonalGaussianBayes":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        classes = np.unique(y)
        if set(classes.tolist()) != {0, 1}:
            raise ValueError("Both binary classes 0 and 1 are required")
        for cls in (0, 1):
            Xc = X[y == cls]
            self.means_[cls] = Xc.mean(axis=0)
            self.vars_[cls] = np.maximum(Xc.var(axis=0), self.var_floor)
            self.priors_[cls] = float(len(Xc) / len(X))
        return self

    def _log_likelihood(self, X: np.ndarray, cls: int) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        mu = self.means_[cls]
        var = self.vars_[cls]
        return -0.5 * np.sum(np.log(2.0 * np.pi * var) + ((X - mu) ** 2) / var, axis=1)

    def predict_proba(self, X: np.ndarray, attack_prior: float | None = None) -> np.ndarray:
        if not self.means_:
            raise RuntimeError("Model has not been fitted")
        if attack_prior is None:
            p1 = self.priors_[1]
        else:
            if not 0.0 < attack_prior < 1.0:
                raise ValueError("attack_prior must be between 0 and 1")
            p1 = float(attack_prior)
        p0 = 1.0 - p1
        l0 = self._log_likelihood(X, 0) + np.log(max(p0, 1e-15))
        l1 = self._log_likelihood(X, 1) + np.log(max(p1, 1e-15))
        # numerically stable two-class softmax
        m = np.maximum(l0, l1)
        e0 = np.exp(l0 - m)
        e1 = np.exp(l1 - m)
        prob1 = e1 / (e0 + e1)
        return np.column_stack([1.0 - prob1, prob1])
