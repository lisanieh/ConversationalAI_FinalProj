"""make_fake_logs.py — generates fake standard logs to demo evaluate.py (numbers are meaningless).
    python make_fake_logs.py && python evaluate.py --turns fake/turns.jsonl --goals fake/goals.json --slots fake/slots.jsonl --out fake_results
"""
import json, os
import numpy as np
from run_demo import make_synthetic, CONDS
from estimator import SlotErrorEstimator

rng = np.random.default_rng(1)
os.makedirs("fake", exist_ok=True)
SLOTS = ["people", "day", "time", "name"]
POL = {"no_repair": .25, "always_spell": .12, "always_type": .04, "U2R": .06}
goals, turns = {}, []
for i in range(60):
    did = f"D{i:03d}"
    goals[did] = {s: f"v{rng.integers(9)}" for s in SLOTS}
    for cond, (snr, *_ ) in CONDS.items():
        for pol, base in POL.items():
            err = min(.9, base * (1 + (20 - snr) / 15))
            state, gold = {}, {}
            for t, s in enumerate(SLOTS):
                gold[s] = goals[did][s]
                v = goals[did][s] if rng.random() > err else "WRONG"
                state[s] = v
                act = {"no_repair": None, "always_spell": "SPELL", "always_type": "TYPE",
                       "U2R": rng.choice(["ACCEPT", "TYPE"], p=[.7, .3])}[pol]
                rep = [] if act in (None, "ACCEPT") else [{"slot": s, "action": act, "effort_s": {"SPELL": 9, "TYPE": 14}[act]}]
                turns.append(dict(dialogue_id=did, seed=0, condition=cond, policy=pol, turn_id=t,
                                  gold_state=dict(gold), belief_state=dict(state), commits={s: v},
                                  repairs=rep, final=t == len(SLOTS) - 1))
json.dump(goals, open("fake/goals.json", "w"))
with open("fake/turns.jsonl", "w") as f:
    for r in turns: f.write(json.dumps(r) + "\n")
df = make_synthetic(80)
tr = df[df.condition.map(lambda c: CONDS[c][2])]
df["p_hat"] = SlotErrorEstimator().fit(tr).predict_proba(df)
df["p_llm"] = np.clip(0.15 + 0.3 * df.y + rng.normal(0, .2, len(df)), 0, 1)  # fake LLM self-reported confidence
df[["dialogue_id", "condition", "slot_type", "y", "error_type", "p_hat", "p_llm"]].to_json("fake/slots.jsonl", orient="records", lines=True)
print("ok")
