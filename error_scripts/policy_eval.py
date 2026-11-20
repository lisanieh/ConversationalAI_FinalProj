"""
policy_eval.py — evaluates p_hat and U2R together: policy comparison (RQ3 part 2) and the crossover test (RQ3 part 1).

Input outcomes table (factorial design: every slot run once under every action, same random seed):
    dialogue_id, slot_id, condition, slot_type, p_hat, raw_conf, y (wrong before repair),
    final_wrong_<A>, effort_<A>   for A in {ACCEPT, CONFIRM, SPELL, TYPE}
So any policy's outcome can be looked up instead of re-running the pipeline per policy (cheap, and paired comparisons are stronger).
Note: this assumes a slot's repair outcome does not affect later turns. Multi-turn effects (task success, turns until first error)
still need real end-to-end dialogue runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from u2r import ACTIONS, ActionParams, choose, spell_type_threshold


def realized(df: pd.DataFrame, actions: pd.Series, Cw: float) -> pd.DataFrame:
    fw = np.choose(actions.map(ACTIONS.index).to_numpy(), [df[f"final_wrong_{a}"] for a in ACTIONS])
    ef = np.choose(actions.map(ACTIONS.index).to_numpy(), [df[f"effort_{a}"] for a in ACTIONS])
    return pd.DataFrame({"dialogue_id": df.dialogue_id, "action": actions.values,
                         "wrong_commit": fw, "effort": ef, "cost": ef + Cw * fw})


def policies(df: pd.DataFrame, params_for, Cw: float, conf_thr: float = .5) -> dict[str, pd.Series]:
    """params_for(row) -> dict[action, ActionParams] (look up the outcome_model table by slot_type / estimated SNR)."""
    const = lambda a: pd.Series([a] * len(df), index=df.index)
    u2r = df.apply(lambda r: choose(r.p_hat, params_for(r), Cw,
                                    ACTIONS if r.get("can_type", 1) else ACTIONS[:3])[0], axis=1)
    return {
        "no_repair": const("ACCEPT"),
        "always_confirm": const("CONFIRM"),
        "always_spell": const("SPELL"),          # current industry practice
        "always_type": const("TYPE"),
        "raw_conf_threshold": pd.Series(np.where(df.raw_conf < conf_thr, "SPELL", "ACCEPT"), index=df.index),
        "U2R": u2r,
        # oracle: knows the true y; an upper bound that shows how far U2R is from ideal
        "oracle": pd.Series(np.where(df.y == 1, "TYPE", "ACCEPT"), index=df.index),
    }


def per_dialogue(res: pd.DataFrame) -> pd.DataFrame:
    return res.groupby("dialogue_id")[["wrong_commit", "effort", "cost"]].mean()


def observed_crossover(df: pd.DataFrame, Cw: float, n_bins: int = 10) -> float | None:
    """Empirical crossover: bin by p_hat, take the realized cost difference SPELL - TYPE, and linearly interpolate the zero crossing."""
    d = (df.effort_SPELL + Cw * df.final_wrong_SPELL) - (df.effort_TYPE + Cw * df.final_wrong_TYPE)
    q = pd.qcut(df.p_hat, n_bins, duplicates="drop")
    g = pd.DataFrame({"p": df.p_hat, "d": d}).groupby(q, observed=True).mean()
    p, dd = g.p.to_numpy(), g.d.to_numpy()
    for i in range(len(dd) - 1):
        if dd[i] < 0 <= dd[i + 1] or dd[i] >= 0 > dd[i + 1]:
            return float(p[i] - dd[i] * (p[i + 1] - p[i]) / (dd[i + 1] - dd[i]))
    return None  # no crossover in range: report the dominant action


def predicted_crossover(params: dict[str, ActionParams], Cw: float) -> float | None:
    return spell_type_threshold(params["SPELL"], params["TYPE"], Cw)


def success_effort_curve(df: pd.DataFrame, params_for, Cw_grid) -> pd.DataFrame:
    """Sweep Cw: U2R picks different actions at each Cw, giving one (mean effort, success = 1 - wrong_commit) point each."""
    pts = []
    for Cw in Cw_grid:
        acts = policies(df, params_for, Cw)["U2R"]
        r = realized(df, acts, Cw)
        pts.append({"Cw": Cw, "effort": r.effort.mean(), "success": 1 - r.wrong_commit.mean()})
    return pd.DataFrame(pts)


def success_effort_auc(curve: pd.DataFrame, effort_max: float) -> float:
    """One definition of the area under the success-effort curve: normalize effort to [0,1], trapezoid-integrate success;
    extend left to effort=0 (success of the lowest-effort point) and right to 1 (success of the highest-effort point).
    Higher is better = succeeds more often for the same effort. effort_max must be fixed team-wide (e.g. mean effort of always_type) and stated in the appendix."""
    c = curve.sort_values("effort")
    x = np.clip(np.r_[0, c.effort / effort_max, 1], 0, 1)
    y = np.r_[c.success.iloc[0], c.success, c.success.iloc[-1]]
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))
