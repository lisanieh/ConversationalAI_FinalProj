"""
estimator.py — slot error estimator p_hat = P(slot wrong | features), with calibration and held-out evaluation.

Procedure (matches the proposal's held-out protocol):
  1. Use only clean + noise_* conditions; split by dialogue_id into fit (70%) / calib (30%)
     -- must group by dialogue: slots within a dialogue are strongly correlated, so a random split leaks.
  2. Fit standardization + L2 logistic regression on fit.
  3. Calibrate on calib: temperature (1 parameter), platt (2 parameters), isotonic (non-parametric).
     Default is temperature: fewest parameters, least likely to overfit when moving to unseen conditions.
  4. Report calibration_metrics.full_report on held-out conditions (music, sidetalk, real speech).
Ablations: drop=["agree_k"] removes sample agreement; calibrator="none" removes the calibration step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from features import CATEGORICAL, FEATURES


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


class SlotErrorEstimator:
    def __init__(self, calibrator: str = "temperature", drop: tuple[str, ...] = (), C: float = 1.0):
        self.calibrator, self.C = calibrator, C
        self.num = [f for f in FEATURES if f not in drop]
        self.cat = [f for f in CATEGORICAL if f not in drop]

    def _base(self):
        pre = ColumnTransformer([("num", StandardScaler(), self.num),
                                 ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat)])
        return make_pipeline(pre, LogisticRegression(C=self.C, max_iter=2000))

    def fit(self, df: pd.DataFrame, group_col: str = "dialogue_id", seed: int = 0):
        gss = GroupShuffleSplit(n_splits=1, test_size=.3, random_state=seed)
        fi, ci = next(gss.split(df, groups=df[group_col]))
        fit, cal = df.iloc[fi], df.iloc[ci]
        self.model = self._base().fit(fit[self.num + self.cat], fit["y"])
        z = self.model.decision_function(cal[self.num + self.cat])
        y = cal["y"].to_numpy()
        if self.calibrator == "temperature":
            nll = lambda T: -np.mean(y * np.log(_sigmoid(z / T) + 1e-12) + (1 - y) * np.log(1 - _sigmoid(z / T) + 1e-12))
            self.T = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
        elif self.calibrator == "platt":
            self.platt = LogisticRegression(C=1e6).fit(z.reshape(-1, 1), y)
        elif self.calibrator == "isotonic":
            self.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(_sigmoid(z), y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        z = self.model.decision_function(df[self.num + self.cat])
        if self.calibrator == "temperature":
            return _sigmoid(z / self.T)
        if self.calibrator == "platt":
            return self.platt.predict_proba(z.reshape(-1, 1))[:, 1]
        if self.calibrator == "isotonic":
            return self.iso.predict(_sigmoid(z))
        return _sigmoid(z)

    def coefficients(self) -> pd.Series:
        names = self.model[0].get_feature_names_out()
        return pd.Series(self.model[-1].coef_[0], index=names).sort_values()
