"""Slot error estimator: p_hat = P(the extracted value is wrong | Whisper features).

L2-regularised logistic regression over the numeric Whisper features plus a
one-hot slot type, then calibration on a SEPARATE split (temperature or Platt
scaling) so that "30%" really means about 30% of such values are wrong.

numpy only (fitted by Newton's method). Missing features (for example
``agreement`` when K = 0) are imputed with the training mean, so a row with no
ASR features at all gets the slot-type prior. Models are saved as JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import NUM_FEATURES, SLOT_TYPES


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _nll(p, y) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 100) -> np.ndarray:
    """Newton/IRLS for L2 logistic regression. Last weight is the (unpenalised) bias."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    R = np.eye(Xb.shape[1]) * l2
    R[-1, -1] = 0.0
    for _ in range(iters):
        p = _sigmoid(Xb @ w)
        g = Xb.T @ (p - y) + R @ w
        H = (Xb.T * (p * (1 - p))) @ Xb + R + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def _golden(f, lo: float, hi: float, tol: float = 1e-5) -> float:
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return (a + b) / 2


class SlotErrorEstimator:
    def __init__(self, features: list[str] | None = None, C: float = 1.0):
        self.features = list(features or NUM_FEATURES)
        self.C = C
        self.mean = self.std = self.w = None
        self.method, self.T, self.a, self.b = "none", 1.0, 1.0, 0.0

    @property
    def names(self) -> list[str]:
        return self.features + [f"slot_type={t}" for t in SLOT_TYPES]

    def _raw(self, rows: list[dict]) -> np.ndarray:
        num = np.array([[np.nan if r.get(f) is None else float(r[f]) for f in self.features] for r in rows],
                       dtype=float).reshape(len(rows), len(self.features))
        cat = np.array([[1.0 if r.get("slot_type") == t else 0.0 for t in SLOT_TYPES] for r in rows])
        return num, cat

    def _matrix(self, rows: list[dict]) -> np.ndarray:
        num, cat = self._raw(rows)
        z = (num - self.mean) / self.std
        z[np.isnan(z)] = 0.0                       # impute with the training mean
        return np.hstack([z, cat])

    # y = 1 means the slot hypothesis is WRONG
    def fit(self, rows: list[dict], y) -> "SlotErrorEstimator":
        num, _ = self._raw(rows)
        self.mean = np.nan_to_num(np.nanmean(num, axis=0)) if len(num) else np.zeros(len(self.features))
        std = np.nanstd(num, axis=0)
        self.std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
        self.w = fit_logreg(self._matrix(rows), np.asarray(y, float), l2=1.0 / self.C)
        return self

    def logit(self, rows: list[dict]) -> np.ndarray:
        X = self._matrix(rows)
        return X @ self.w[:-1] + self.w[-1]

    def calibrate(self, rows: list[dict], y, method: str = "temperature") -> "SlotErrorEstimator":
        """Fit on a calibration split that was NOT used in fit()."""
        z, y = self.logit(rows), np.asarray(y, float)
        self.method = method
        if method == "temperature":
            self.T = math.exp(_golden(lambda lt: _nll(_sigmoid(z / math.exp(lt)), y),
                                      math.log(0.05), math.log(20)))
        elif method == "platt":
            w = fit_logreg(z.reshape(-1, 1), y, l2=1e-6)
            self.a, self.b = float(w[0]), float(w[1])
        return self

    def predict(self, rows: list[dict]) -> np.ndarray:
        z = self.logit(rows)
        if self.method == "temperature":
            return _sigmoid(z / self.T)
        if self.method == "platt":
            return _sigmoid(self.a * z + self.b)
        return _sigmoid(z)

    def predict_one(self, features: dict | None, slot_type: str) -> float:
        return float(self.predict([{**(features or {}), "slot_type": slot_type}])[0])

    def coefficients(self) -> dict[str, float]:
        return dict(sorted(zip(self.names, self.w[:-1].tolist()), key=lambda kv: kv[1]))

    # --------------------------------------------------------------- saving
    def to_dict(self) -> dict:
        return {"features": self.features, "C": self.C, "mean": self.mean.tolist(),
                "std": self.std.tolist(), "w": self.w.tolist(), "slot_types": SLOT_TYPES,
                "method": self.method, "T": self.T, "a": self.a, "b": self.b}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SlotErrorEstimator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("slot_types") != SLOT_TYPES:
            raise ValueError("estimator was trained with a different slot type list")
        est = cls(d["features"], d["C"])
        est.mean, est.std, est.w = np.array(d["mean"]), np.array(d["std"]), np.array(d["w"])
        est.method, est.T, est.a, est.b = d["method"], d["T"], d["a"], d["b"]
        return est


# ---------------------------------------------------------------- metrics
def ece(y, p, n_bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum((bins == m).mean() * abs(y[bins == m].mean() - p[bins == m].mean())
                     for m in range(n_bins) if (bins == m).any()))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def auroc(y, p) -> float:
    """Mann-Whitney AUROC with average ranks for ties (nan if one class only)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
