"""Score one finished dialogue against the goal (the agent never sees the goal).

Produces the two log files every analysis reads:
  attempts.jsonl   one row per value the agent heard (+ one per missed goal slot)
  dialogues.jsonl  one row per dialogue

Joint goal accuracy here means the END-of-dialogue state equals the goal on
every goal slot (not the per-turn MultiWOZ JGA); say so in the paper.
"""

from __future__ import annotations

from .config import slot_type
from .values import canonical, values_match


def _align_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Word-level edit distance, returning (substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    d = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]   # (cost, S, D, I)
    for i in range(1, n + 1):
        d[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        d[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c, s, dd, ii = d[i - 1][j - 1]
            sub = (c, s, dd, ii) if ref[i - 1] == hyp[j - 1] else (c + 1, s + 1, dd, ii)
            c, s, dd, ii = d[i - 1][j]
            dele = (c + 1, s, dd + 1, ii)
            c, s, dd, ii = d[i][j - 1]
            ins = (c + 1, s, dd, ii + 1)
            d[i][j] = min(sub, dele, ins)
    _, S, D, I = d[n][m]
    return S, D, I


def classify_error(slot: str, ref: str | None, hyp: str | None) -> str:
    """correct | substitution | deletion | insertion (on canonical values).
    Emails and phones are compared character by character, other slots word by word."""
    r, h = canonical(slot, ref), canonical(slot, hyp)
    if not h:
        return "deletion" if r else "correct"
    if r == h:
        return "correct"
    split = list if slot in ("email", "phone") else str.split
    S, D, I = _align_counts(split(r.replace(" ", "")) if slot == "phone" else split(r),
                            split(h.replace(" ", "")) if slot == "phone" else split(h))
    if I and not S and not D:
        return "insertion"
    if D and not S and not I:
        return "deletion"
    return "substitution"


def score_dialogue(agent, customer, meta: dict) -> tuple[list[dict], dict]:
    """Returns (attempt rows, dialogue row). ``meta`` holds run_id, policy,
    condition, snr_db, ... and is copied into every row."""
    goal = customer.goal
    rows = []
    for a in agent.attempts:
        true = goal.slots.get(a["slot"])
        row = {**meta, "dialogue_id": goal.dialogue_id, "seed": goal.seed, "voice": goal.voice,
               "can_type": goal.can_type, **a, "true_value": true, "spurious": true is None,
               "missed": False}
        row["first_correct"] = values_match(a["slot"], a["hypothesis"], true) if true else False
        row["committed_correct"] = (values_match(a["slot"], a["committed_value"], true)
                                    if true and a["committed_value"] is not None else False)
        row["error_type"] = classify_error(a["slot"], true, a["hypothesis"]) if true else "spurious"
        rows.append(row)

    final = agent.final_state()
    heard = {a["slot"] for a in agent.attempts}
    for slot, true in goal.slots.items():           # goal slots the agent never got a value for
        if slot not in heard and slot not in final:
            rows.append({**meta, "dialogue_id": goal.dialogue_id, "seed": goal.seed,
                         "voice": goal.voice, "can_type": goal.can_type, "slot": slot,
                         "slot_type": slot_type(slot), "true_value": true, "hypothesis": None,
                         "missed": True, "spurious": False, "first_correct": False,
                         "committed_correct": False, "error_type": "deletion", "mode": None,
                         "action": None, "effort": 0.0, "features": None})

    per_slot = {s: values_match(s, final.get(s), v) for s, v in goal.slots.items()}
    real = [r for r in rows if not r["missed"] and not r["spurious"]]
    wrong = [r for r in real if r.get("committed_value") is not None and not r["committed_correct"]]
    joint = all(per_slot.values())
    pk = customer.packages
    dlg = {**meta, "dialogue_id": goal.dialogue_id, "seed": goal.seed, "voice": goal.voice,
           "can_type": goal.can_type, "domain": goal.domain, "agent_domain": agent.domain,
           "n_goal_slots": len(goal.slots), "n_turns": len(pk), "ended": customer.done,
           "joint_goal_correct": joint, "task_success": joint and customer.done,
           "slot_accuracy": sum(per_slot.values()) / len(per_slot),
           "wrong_slots": [s for s, ok in per_slot.items() if not ok],
           "n_wrong_commits": len(wrong),
           "turns_to_first_error": min((r["turn"] for r in wrong), default=None),
           "total_effort": sum(r.get("effort") or 0.0 for r in rows),
           "n_confirm": sum(r.get("action") == "confirm" for r in rows),
           "n_spell": sum(p.path == "spell" for p in pk),
           "n_type_requests": sum(p.assistant_action == "type" for p in pk),
           "n_cannot_type": sum(p.path == "cannot_type" for p in pk),
           "typed_chars": sum(p.typed_chars or 0 for p in pk),
           "n_spurious": sum(r["spurious"] for r in rows), "n_missed": sum(r["missed"] for r in rows)}
    return rows, dlg
