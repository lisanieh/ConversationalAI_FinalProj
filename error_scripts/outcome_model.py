"""
outcome_model.py — estimates each action's r_a (fix rate), e_a (break rate) and c_a (user effort) from simulation results.

Input table repairs (one row per repair actually executed, produced by the end-to-end runs):
    slot_type, condition, action, was_wrong (v_hat wrong before repair), final_wrong (wrong after repair), effort
Estimates:
    r_a = P(final_wrong=0 | was_wrong=1)        e_a = P(final_wrong=1 | was_wrong=0)
Smoothed with the Beta(1,1) posterior mean (k+1)/(n+2), so small cells never give 0 or 1; 95% credible intervals included.
Cells that are too small (n < min_n) are flagged; merge them to action x condition.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import beta

from u2r import ActionParams


def _beta(k: int, n: int) -> tuple[float, float, float]:
    a, b = k + 1, n - k + 1
    return a / (a + b), beta.ppf(.025, a, b), beta.ppf(.975, a, b)


def estimate(repairs: pd.DataFrame, by=("slot_type", "condition", "action"), min_n: int = 30) -> pd.DataFrame:
    rows = []
    for key, g in repairs.groupby(list(by)):
        w, c = g[g.was_wrong == 1], g[g.was_wrong == 0]
        r = _beta(int((w.final_wrong == 0).sum()), len(w))
        e = _beta(int((c.final_wrong == 1).sum()), len(c))
        rows.append(dict(zip(by, key), n_wrong=len(w), n_right=len(c),
                         r=r[0], r_lo=r[1], r_hi=r[2], e=e[0], e_lo=e[1], e_hi=e[2],
                         c=g.effort.mean(), small=(len(w) < min_n or len(c) < min_n)))
    return pd.DataFrame(rows)


def to_params(table: pd.DataFrame, **where) -> dict[str, ActionParams]:
    t = table
    for k, v in where.items():
        t = t[t[k] == v]
    P = {row.action: ActionParams(row.c, row.r, row.e) for row in t.itertuples()}
    P.setdefault("ACCEPT", ActionParams(0.0, 0.0, 0.0))
    return P
