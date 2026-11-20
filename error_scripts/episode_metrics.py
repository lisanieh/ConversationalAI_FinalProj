"""
episode_metrics.py — the multi-turn metrics from the proposal's "How We Measure Effectiveness / Metrics".

Shared log (one JSONL line per turn of each (dialogue, seed, condition, policy); every teammate's runs must write this format):
{
  "dialogue_id": "PMUL0698", "seed": 0, "condition": "noise_5", "policy": "U2R", "turn_id": 3,
  "gold_state":   {"restaurant-book-people": "4", ...},   # true values of the slots the user has said so far (cumulative)
  "belief_state": {"restaurant-book-people": "4", ...},   # agent's dialogue state at the end of this turn (cumulative)
  "repairs": [{"slot": "restaurant-book-people", "action": "SPELL", "p_hat": 0.41,
               "p_llm": 0.2, "effort_s": 9.5, "typed_chars": 0}],
  "commits": {"restaurant-book-people": "4"},             # slots written or overwritten this turn
  "final": false                                          # true on the last turn
}
Also needed: each dialogue's full goal (all slots to fill in the MultiWOZ user goal), in goals.json: {dialogue_id: {slot: value}}.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from align import normalize


def _eq(a, b) -> bool:
    return normalize(str(a)) == normalize(str(b))


def _state_match(pred: dict, gold: dict) -> bool:
    keys = set(pred) | set(gold)
    return all(k in pred and k in gold and _eq(pred[k], gold[k]) for k in keys)


def _has_wrong(pred: dict, gold_full: dict) -> bool:
    """Whether any slot value in the state disagrees with the full goal (wrong, not merely unfilled)."""
    return any(k in gold_full and not _eq(v, gold_full[k]) for k, v in pred.items()) or \
        any(k not in gold_full for k in pred)


def per_dialogue(turns: pd.DataFrame, goals: dict) -> pd.DataFrame:
    """turns: all turns of one run. Returns one row per (dialogue_id, seed, condition, policy)."""
    out = []
    for key, g in turns.sort_values("turn_id").groupby(["dialogue_id", "seed", "condition", "policy"]):
        did = key[0]
        goal = goals[did]
        jga = np.mean([_state_match(r.belief_state, r.gold_state) for r in g.itertuples()])
        n_commit = n_wrong = 0
        for r in g.itertuples():
            for s, v in (r.commits or {}).items():
                n_commit += 1
                n_wrong += int(s not in goal or not _eq(v, goal[s]))
        first_err = next((r.turn_id for r in g.itertuples() if _has_wrong(r.belief_state, goal)), None)
        last = g.iloc[-1]
        reps = [x for r in g.repairs for x in (r or [])]
        acts = defaultdict(int)
        for x in reps:
            acts[x["action"]] += 1
        out.append({
            "dialogue_id": did, "seed": key[1], "condition": key[2], "policy": key[3],
            "n_turns": len(g),
            "task_success": int(_state_match(last.belief_state, goal)),   # final state exactly matches the full goal
            "jga": jga,
            "wrong_commit_rate": n_wrong / n_commit if n_commit else 0.0,
            "n_commits": n_commit,
            "first_error_turn": first_err,                                # None = no error at all (right-censored)
            "effort_s": sum(x.get("effort_s", 0) for x in reps),
            "typed_chars": sum(x.get("typed_chars", 0) for x in reps),
            "n_repairs": len(reps), **{f"n_{a}": c for a, c in acts.items()},
        })
    return pd.DataFrame(out).fillna({f"n_{a}": 0 for a in ("CONFIRM", "SPELL", "TYPE", "ACCEPT")})


def km_survival(first_error, n_turns) -> pd.DataFrame:
    """Kaplan-Meier estimate of 'turns until first error'. Dialogues without an error are right-censored at their length;
    they must not be dropped or treated as 'length+1', which would bias the comparison. Returns S(t) = P(no error by turn t)."""
    t = np.array([e if e is not None and not pd.isna(e) else n for e, n in zip(first_error, n_turns)], float)
    event = np.array([e is not None and not pd.isna(e) for e in first_error])
    S, rows = 1.0, []
    for u in np.unique(t[event]):
        at_risk = (t >= u).sum()
        d = ((t == u) & event).sum()
        S *= 1 - d / at_risk
        rows.append({"turn": int(u), "at_risk": int(at_risk), "errors": int(d), "survival": S})
    return pd.DataFrame(rows)


def summarize(dlg: pd.DataFrame) -> pd.DataFrame:
    cols = ["task_success", "jga", "wrong_commit_rate", "effort_s", "typed_chars", "n_repairs"]
    return dlg.groupby(["condition", "policy"])[cols].mean()
