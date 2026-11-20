"""Train the slot error estimator and the repair outcome model from FIT-phase logs.

  python -m assistant_agent.train --attempts logs/fit/attempts.jsonl          --replays logs/fit/replays.jsonl --out models/
  # ablations
  python -m assistant_agent.train ... --drop agreement --out models/no_agreement/
  python -m assistant_agent.train ... --calibration none --out models/no_calib/

Split 60/20/20 (train / calibration / test) by GOAL: every attempt from the same
goal (all policies, conditions, seeds) lands in the same split, so nothing leaks.
r_a / e_a / c_a come from the save-point replays when --replays is given,
otherwise from forced-policy runs (always confirm / spell / type).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .config import NUM_FEATURES
from .estimator import SlotErrorEstimator, auroc, brier, ece
from .outcomes import RepairOutcomeModel


def load_jsonl(*paths) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def estimator_rows(attempts: list[dict]) -> list[dict]:
    """Spoken, non-spurious values with ASR features; label 1 = first hypothesis wrong."""
    out = []
    for a in attempts:
        if a.get("missed") or a.get("spurious") or a.get("mode") != "speech" or not a.get("features"):
            continue
        out.append({**a["features"], "slot_type": a["slot_type"], "dialogue_id": a["dialogue_id"],
                    "y": 0 if a["first_correct"] else 1})
    return out


def split_by_goal(rows: list[dict], seed: int = 0, frac=(0.6, 0.2)):
    goals = sorted({r["dialogue_id"] for r in rows})
    random.Random(seed).shuffle(goals)
    a, b = int(len(goals) * frac[0]), int(len(goals) * (frac[0] + frac[1]))
    part = {g: 0 if i < a else 1 if i < b else 2 for i, g in enumerate(goals)}
    return tuple([r for r in rows if part[r["dialogue_id"]] == k] for k in range(3))


def report(name: str, y, p) -> None:
    print(f"  {name:26s} ECE={ece(y, p):.3f}  Brier={brier(y, p):.3f}  AUROC={auroc(y, p):.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", nargs="+", required=True)
    ap.add_argument("--replays", nargs="*", default=[], help="save-point replays (preferred for r/e/c)")
    ap.add_argument("--calibration", default="temperature", choices=["temperature", "platt", "none"])
    ap.add_argument("--drop", nargs="*", default=[], help="features to remove (ablation)")
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength")
    ap.add_argument("--min-n", type=int, default=20, help="min cell size before pooling r/e")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    attempts = load_jsonl(*a.attempts)
    rows = estimator_rows(attempts)
    if not rows:
        raise SystemExit("no spoken attempts with ASR features: run the fit phase with an ASR channel")
    print(f"{len(rows)} spoken slot values, {sum(r['y'] for r in rows) / len(rows):.1%} wrong at first")

    train, cal, test = split_by_goal(rows, a.seed)
    print(f"split by goal: train={len(train)} calibration={len(cal)} test={len(test)}")
    feats = [f for f in NUM_FEATURES if f not in a.drop]
    est = SlotErrorEstimator(feats, C=a.C).fit(train, [r["y"] for r in train])
    y = [r["y"] for r in test]
    print("Test split (fit conditions):")
    report("uncalibrated", y, est.predict(test))
    if a.calibration != "none":
        est.calibrate(cal, [r["y"] for r in cal], a.calibration)
        report(f"calibrated ({a.calibration})", y, est.predict(test))
        if a.calibration == "temperature":
            print(f"  T = {est.T:.3f}")
    print("Coefficients (standardised; positive = more likely wrong):")
    for k, v in est.coefficients().items():
        print(f"  {k:24s} {v:+.3f}")
    est.save(out / "estimator.json")

    if a.replays:
        replays = load_jsonl(*a.replays)
        outcomes = RepairOutcomeModel(min_n=a.min_n).fit_replays(replays)
        print(f"Repair outcomes from {len(replays)} save-point replays")
    else:
        outcomes = RepairOutcomeModel(min_n=a.min_n).fit(attempts)
        print("Repair outcomes from forced-policy runs")
    if not outcomes.table:
        print("WARNING: no replays or forced confirm/spell/type runs found; outcomes.json is empty")
    outcomes.save(out / "outcomes.json")
    print("Repair outcomes (pooled over slot types):")
    for k, v in outcomes.table.items():
        if k.endswith("|*"):
            c = f"{v['c']:.1f}s" if v.get("c") is not None else "prior"
            print(f"  {k:10s} r={v['r']:.3f}  e={v['e']:.3f}  c={c:>6s}  "
                  f"(n_wrong={v['n_wrong']}, n_right={v['n_right']})")
    print(f"Saved {out / 'estimator.json'} and {out / 'outcomes.json'}")


if __name__ == "__main__":
    main()
