"""
evaluate.py — single entry point of the measurement layer: reads the team's standard logs and writes every error table in the paper.

    python evaluate.py --turns runs/turns.jsonl --goals runs/goals.json \
                       --slots runs/slots_labeled.jsonl --out results/

  --turns  per-turn log (format in the header of episode_metrics.py)
  --goals  {dialogue_id: {slot: value}}
  --slots  one row per slot, with at least y, error_type, condition, dialogue_id, p_hat; p_llm is evaluated too when present (RQ4)
Outputs:
  episode_summary.csv      condition x policy: task success, JGA, wrong commits, effort (with 95% dialogue-level CI)
  first_error_km.csv       Kaplan-Meier curves per condition x policy
  policy_tests.csv         per condition: paired difference of U2R vs each baseline, permutation p, Holm correction
  estimate_error.csv       per condition: ECE/Brier/AUROC... of p_hat (and p_llm), with a CI for ECE
  estimate_by_type.csv     AUROC per error type
  mistake_types.csv        share of each error type per condition
  reliability_<cond>.png   reliability diagrams
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

import calibration_metrics as cm
import episode_metrics as em
from stats_tests import cluster_bootstrap, holm, paired_bootstrap, paired_permutation

METRICS = ["task_success", "jga", "wrong_commit_rate", "effort_s"]


def episode_tables(turns: pd.DataFrame, goals: dict, out: str, ref_policy: str = "U2R", B: int = 1000):
    dlg = em.per_dialogue(turns, goals)
    dlg.to_csv(f"{out}/per_dialogue.csv", index=False)
    rows = []
    for (c, p), g in dlg.groupby(["condition", "policy"]):
        r = {"condition": c, "policy": p, "n_dialogues": g.dialogue_id.nunique()}
        for m in METRICS:
            est, lo, hi = cluster_bootstrap(g, lambda d, m=m: d[m].mean(), B=B)
            r.update({m: est, f"{m}_lo": lo, f"{m}_hi": hi})
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{out}/episode_summary.csv", index=False)

    km = [em.km_survival(g.first_error_turn, g.n_turns).assign(condition=c, policy=p)
          for (c, p), g in dlg.groupby(["condition", "policy"])]
    if km:
        pd.concat(km).to_csv(f"{out}/first_error_km.csv", index=False)

    # paired comparison: align the same (dialogue, seed) across policies
    tests = []
    for c, g in dlg.groupby("condition"):
        piv = g.pivot_table(index=["dialogue_id", "seed"], columns="policy", values="wrong_commit_rate")
        if ref_policy not in piv:
            continue
        pv, ci = {}, {}
        for p in piv.columns.drop(ref_policy):
            pair = piv[[ref_policy, p]].dropna()
            ci[p] = paired_bootstrap(pair[ref_policy].to_numpy(), pair[p].to_numpy())
            pv[p] = paired_permutation(pair[ref_policy].to_numpy(), pair[p].to_numpy())
        for p, h in holm(pv).items():
            tests.append({"condition": c, "vs": p, "metric": "wrong_commit_rate", "diff": ci[p][0],
                          "lo": ci[p][1], "hi": ci[p][2], **h})
    pd.DataFrame(tests).to_csv(f"{out}/policy_tests.csv", index=False)
    return dlg


def estimate_tables(slots: pd.DataFrame, out: str, B: int = 1000):
    pd.crosstab(slots.condition, slots.error_type, normalize="index").to_csv(f"{out}/mistake_types.csv")
    scores = [s for s in ("p_hat", "p_llm") if s in slots]
    rows, by_type = [], []
    for c, g in slots.groupby("condition"):
        for s in scores:
            gg = g.dropna(subset=[s])
            rep = cm.full_report(gg.y, gg[s])
            _, lo, hi = cluster_bootstrap(gg, lambda d, s=s: cm.ece(d.y, d[s]), B=B)
            rows.append({"condition": c, "score": s, **rep, "ece_lo": lo, "ece_hi": hi})
            for t, v in cm.per_error_type_auroc(gg.y, gg[s], gg.error_type).items():
                by_type.append({"condition": c, "score": s, "error_type": t, **v})
            cm.plot_reliability(gg.y, gg[s], f"{out}/reliability_{c}_{s}.png", title=f"{c} · {s}")
    pd.DataFrame(rows).to_csv(f"{out}/estimate_error.csv", index=False)
    pd.DataFrame(by_type).to_csv(f"{out}/estimate_by_type.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns")
    ap.add_argument("--goals")
    ap.add_argument("--slots")
    ap.add_argument("--out", default="results")
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.turns:
        episode_tables(pd.read_json(a.turns, lines=True), json.load(open(a.goals)), a.out, B=a.B)
    if a.slots:
        estimate_tables(pd.read_json(a.slots, lines=True), a.out, B=a.B)
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
