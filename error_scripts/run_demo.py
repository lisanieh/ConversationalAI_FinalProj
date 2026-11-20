"""
run_demo.py — runs the whole measurement chain on synthetic data to check the scripts work and produce example figures.
The synthetic data only exercises the code; its numbers have no scientific meaning. With real data, replace make_synthetic() with a JSONL reader.

    python run_demo.py      # writes to demo_output/
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

import calibration_metrics as cm
from align import label_slot, wer_counts
from estimator import SlotErrorEstimator
from outcome_model import estimate, to_params
from policy_eval import (observed_crossover, per_dialogue, policies, predicted_crossover, realized,
                         success_effort_auc, success_effort_curve)
from stats_tests import cluster_bootstrap, holm, paired_bootstrap, paired_permutation
from u2r import ACTIONS, decision_regions

if hasattr(sys.stdout, "reconfigure"):          # Windows consoles default to cp1252 and cannot print → or Δ
    sys.stdout.reconfigure(encoding="utf-8")

OUT = "demo_output"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

SLOT_TYPES = {"name": 1.2, "number": 0.6, "time": 0.4, "category": -0.6, "code": 1.5}
# condition: (SNR, hallucination tendency, used for training)
CONDS = {"clean": (35, 0.0, True), "noise_20": (20, 0.0, True), "noise_10": (10, 0.0, True),
         "noise_5": (5, 0.0, True), "noise_0": (0, 0.0, True),
         "music": (8, 1.0, False), "sidetalk": (6, 0.6, False), "real": (15, 0.3, False)}


def make_synthetic(n_dial: int = 160, slots_per: int = 6) -> pd.DataFrame:
    rows = []
    for cond, (snr, hallu, _) in CONDS.items():
        for d in range(n_dial):
            did = f"{cond}-{d}"
            for s in range(slots_per):
                st = rng.choice(list(SLOT_TYPES))
                snr_est = snr + rng.normal(0, 3)
                logit = -2.6 + SLOT_TYPES[st] + 0.12 * (15 - snr) + 0.8 * hallu * rng.random()
                y = int(rng.random() < 1 / (1 + np.exp(-logit)))
                if y:
                    et = rng.choice(["substitution", "deletion", "insertion", "extraction"],
                                    p=np.array([.55, .2, .1 + .3 * hallu, .15]) / (1 + .3 * hallu))
                else:
                    et = "correct"
                # features: deletions leave almost no token evidence -> hard for the estimator to detect
                sig = {"substitution": 1.0, "insertion": 0.7, "extraction": 0.2, "deletion": 0.15}.get(et, 0)
                min_lp = rng.normal(-0.3 - 1.6 * sig - 0.03 * (30 - snr), 0.5)
                rows.append(dict(
                    dialogue_id=did, slot_id=f"{did}-{s}", condition=cond, slot_type=st, y=y, error_type=et,
                    min_logprob=min_lp, mean_logprob=min_lp * 0.6 + rng.normal(0, .2),
                    span_found=int(et != "deletion" or rng.random() < .3),
                    agree_k=float(np.clip(1 - 0.6 * sig + rng.normal(0, .15), 0, 1)),
                    no_speech_prob=float(np.clip(0.05 + 0.4 * hallu * (et == "insertion") + rng.normal(0, .05), 0, 1)),
                    compression_ratio=1.4 + 0.8 * hallu * (et == "insertion") + rng.normal(0, .15),
                    avg_logprob=rng.normal(-0.25 - 0.02 * (30 - snr), 0.15),
                    snr_est_db=snr_est, can_type=int(rng.random() > .1),
                    # 'raw_conf' baseline = min word probability in the slot, no learning or calibration
                    raw_conf=float(np.exp(min_lp)),
                ))
    df = pd.DataFrame(rows)
    # simulated repair outcome for every action (factorial design)
    for a, (c0, r0, e0) in {"ACCEPT": (0, 0, 0), "CONFIRM": (4, .75, .02),
                            "SPELL": (10, .92, .02), "TYPE": (15, .97, .03)}.items():
        snr = df.condition.map(lambda c: CONDS[c][0])
        r = r0 - (0.02 * (25 - snr)).clip(0) if a in ("SPELL", "CONFIRM") else pd.Series(r0, index=df.index)
        r = r.clip(0, 1)
        fixed = rng.random(len(df)) < r
        broke = rng.random(len(df)) < e0
        df[f"final_wrong_{a}"] = np.where(df.y == 1, ~fixed, broke).astype(int)
        df[f"effort_{a}"] = c0 + (rng.gamma(2, 1, len(df)) if c0 else 0)
    return df


def main():
    # ---------- 0. alignment / labelling examples (mistake error)
    cases = [
        ("book a table for four people at seven thirty", "book a table for for people at 7 30", "4", "4"),
        ("i want the cambridge lodge", "i want the came bridge lodge", "cambridge lodge", "came bridge lodge"),
        ("for four people please", "for people please", "4", ""),
        ("", "thank you for watching", None, "thank you"),
        ("in the center of town", "in the centre of town", "centre", "north"),
    ]
    print("== Slot labelling examples ==")
    for ref, hyp, rv, hv in cases:
        print(f"{hyp!r:45s} → {label_slot(ref, hyp, rv, hv)}  WER={wer_counts(ref, hyp)['wer']:.2f}")

    df = make_synthetic()
    train = df[df.condition.map(lambda c: CONDS[c][2])]
    test = df[~df.condition.map(lambda c: CONDS[c][2])]
    print(f"\nTraining conditions {sorted(train.condition.unique())}, held-out {sorted(test.condition.unique())}")

    # ---------- 1. RQ1: error type distribution
    tab = pd.crosstab(df.condition, df.error_type, normalize="index").round(3)
    tab.to_csv(f"{OUT}/rq1_error_types_by_condition.csv")
    print("\n== RQ1 error type shares ==\n", tab)

    # ---------- 2. RQ2: estimate error
    report = {}
    for name, kw in {"full": {}, "no_agree_k": {"drop": ("agree_k",)}, "no_calibration": {"calibrator": "none"},
                     "isotonic": {"calibrator": "isotonic"}}.items():
        est = SlotErrorEstimator(**kw).fit(train)
        test_p = est.predict_proba(test)
        report[name] = {c: cm.full_report(g.y, test_p[test.condition == c]) for c, g in test.groupby("condition")}
        if name == "full":
            est_full, p_test = est, test_p
    report["raw_conf_baseline"] = {c: cm.full_report(g.y, 1 - g.raw_conf) for c, g in test.groupby("condition")}
    flat = pd.DataFrame({(m, c): v for m, d in report.items() for c, v in d.items()}).T[
        ["n", "base_rate", "ece", "ece_quantile", "brier", "reliability", "resolution", "nll", "auroc", "auprc"]]
    flat.to_csv(f"{OUT}/rq2_calibration_report.csv")
    print("\n== RQ2 held-out estimate error ==\n", flat.round(3))

    test = test.assign(p_hat=p_test)
    print("\nAUROC per error type:", json.dumps(cm.per_error_type_auroc(test.y, test.p_hat, test.error_type), indent=1))
    lo_hi = cluster_bootstrap(test, lambda d: cm.ece(d.y, d.p_hat), B=500)
    print(f"ECE (all held-out) = {lo_hi[0]:.3f}  95% CI [{lo_hi[1]:.3f}, {lo_hi[2]:.3f}]  (dialogue bootstrap)")
    cm.plot_reliability(test.y, test.p_hat, f"{OUT}/reliability_heldout.png", title="Held-out reliability (synthetic)")
    print("\nCoefficients (standardized; positive = more likely wrong):\n", est_full.coefficients().round(2))

    # ---------- 3. outcome model r_a, e_a (estimated on training conditions only)
    rep = []
    for a in ACTIONS[1:]:
        rep.append(pd.DataFrame({"slot_type": train.slot_type,
                                 "condition": np.where(train.condition.isin(["clean", "noise_20"]), "quiet", "noisy"),
                                 "action": a, "was_wrong": train.y, "final_wrong": train[f"final_wrong_{a}"],
                                 "effort": train[f"effort_{a}"]}))
    outcome_tbl = estimate(pd.concat(rep), by=("condition", "action"))
    outcome_tbl.to_csv(f"{OUT}/outcome_model.csv", index=False)
    print("\n== r_a / e_a estimates ==\n", outcome_tbl.round(3))
    P = {k: to_params(outcome_tbl, condition=k) for k in ("quiet", "noisy")}
    params_for = lambda r: P["quiet" if r.snr_est_db >= 17 else "noisy"]

    # ---------- 4. U2R thresholds and decision regions
    Cw = 60
    for k in P:
        print(f"[{k}] SPELL/TYPE p* = {predicted_crossover(P[k], Cw):.3f}; regions =",
              [(a, round(s, 3), round(e, 3)) for a, s, e in decision_regions(P[k], Cw)])

    # ---------- 5. RQ3: predicted vs observed crossover, policy comparison
    rows = []
    for c, g in test.groupby("condition"):
        key = "quiet" if CONDS[c][0] >= 17 else "noisy"
        rows.append({"condition": c, "pred_p*": predicted_crossover(P[key], Cw),
                     "obs_p*": observed_crossover(g, Cw)})
    print("\n== RQ3 crossover ==\n", pd.DataFrame(rows).round(3))

    pol = policies(test, params_for, Cw)
    per = {k: per_dialogue(realized(test, v, Cw)) for k, v in pol.items()}
    summ = pd.DataFrame({k: v.mean() for k, v in per.items()}).T
    print("\n== Policy comparison (held-out, per-dialogue mean) ==\n", summ.round(3))
    pvals, cis = {}, {}
    for k in per:
        if k in ("U2R", "oracle"):
            continue
        a, b = per["U2R"].cost.to_numpy(), per[k].cost.to_numpy()
        cis[k] = paired_bootstrap(a, b)
        pvals[k] = paired_permutation(a, b, B=5000)
    h = holm(pvals)
    for k in pvals:
        m, lo, hi = cis[k]
        print(f"U2R − {k:20s} cost Δ={m:7.2f} [{lo:7.2f},{hi:7.2f}]  p_holm={h[k]['p_holm']:.4f}")
    summ.to_csv(f"{OUT}/rq3_policy_comparison.csv")

    curve = success_effort_curve(test, params_for, [5, 10, 20, 40, 60, 100, 200, 400])
    print("\nSuccess-effort AUC (U2R) =", round(success_effort_auc(curve, effort_max=summ.loc["always_type", "effort"]), 3))


if __name__ == "__main__":
    main()
