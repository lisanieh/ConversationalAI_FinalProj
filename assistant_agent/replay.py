"""Save-point replays: try all four repairs on the very same heard value.

During a dialogue, just before the agent hears a customer turn, the whole world
(customer + LLM brain, agent, recognizer channel with its random state) is
saved. If the agent would take in new spoken values on that turn, each value is
replayed from the save point four times, once per action (accept / confirm /
spell / type), with every other new value accepted. Each branch runs only until
that value is committed, and records whether it ended wrong and how many
seconds of the customer's time the repair cost. The main dialogue then goes on
unchanged.

Because the branches share the save point and random state, the four outcomes
are paired value by value: this gives r_a / e_a / c_a for U2R (outcomes.py)
and the "factorial" table that error_scripts/policy_eval.py reads
(``final_wrong_<A>``, ``effort_<A>`` for A in ACCEPT, CONFIRM, SPELL, TYPE).

Cost: each saved turn with new values costs one probe plus 3 branches per value,
each a few turns (and LLM calls when the agent extracts with an LLM).
"""

from __future__ import annotations

import copy
import math

from .config import ACTIONS
from .effort import EffortModel
from .policies import ForcedFor, Fixed
from .scoring import repair_seconds
from .values import values_match

MAX_BRANCH_TURNS = 6


def _branch(world, slot: str, action: str, pkg, max_turns: int = MAX_BRANCH_TURNS):
    """Replay one turn from the save point with ``action`` forced for ``slot``.
    Returns (customer, agent, attempt) after the attempt is committed (or the cap)."""
    customer, agent, channel = copy.deepcopy(world)
    agent.policy = ForcedFor(slot, action) if action != "probe" else Fixed("accept")
    n_before = len(agent.attempts)
    out = agent.step(channel(pkg))
    new = [a for a in agent.attempts[n_before:] if a["mode"] == "speech"]
    if action == "probe":
        return customer, agent, new
    att = next((a for a in new if a["slot"] == slot), None)
    for _ in range(max_turns):
        if att is None or att["committed_value"] is not None:
            break
        nxt = customer.respond(out)
        out = agent.step(channel(nxt))
    return customer, agent, att


def replay_turn(customer, agent, channel, pkg, effort: EffortModel, meta: dict) -> list[dict]:
    """Rows for every new spoken value heard on this turn (empty if none)."""
    p = agent.pending
    if p is not None and p.action not in ("normal", "ask_type", "domain"):
        return []                                    # a repair answer: no new values here
    world = (customer, agent, channel)
    _, _, probe = _branch(world, "", "probe", pkg)
    rows = []
    goal = customer.goal
    for first in probe:
        slot = first["slot"]
        true = goal.slots.get(slot)
        if true is None:
            continue                                 # spurious slot: nothing to repair towards
        feats = first["features"] or {}
        row = {**meta, "dialogue_id": goal.dialogue_id, "goal_seed": goal.seed,
               "slot_id": f"{goal.dialogue_id}|{meta.get('seed', 0)}|{pkg.turn}|{slot}",
               "turn": pkg.turn, "slot": slot, "slot_type": first["slot_type"], "can_type": goal.can_type,
               "true_value": true, "hypothesis": first["hypothesis"], "transcript": first["transcript"],
               "ref_text": pkg.true_text, "p_hat": first["p_hat"], "llm_p_wrong": first["llm_p_wrong"],
               "raw_conf": math.exp(feats["mean_logprob"]) if feats.get("mean_logprob") is not None else None,
               "features": first["features"],
               "y": int(not values_match(slot, first["hypothesis"], true))}
        for action in ACTIONS:
            A = action.upper()
            if action == "accept":
                row[f"final_wrong_{A}"], row[f"effort_{A}"] = row["y"], 0.0
                row[f"value_{A}"] = first["hypothesis"]
                continue
            bc, ba, att = _branch(world, slot, action, pkg)
            committed = att["committed_value"] if att else None
            row[f"final_wrong_{A}"] = int(not values_match(slot, committed, true))
            row[f"effort_{A}"] = repair_seconds(att, ba, bc, effort)[0] if att else 0.0
            row[f"value_{A}"] = committed
            row[f"channel_{A}"] = att["repair_channel"] if att else None
        rows.append(row)
    return rows


def drive(customer, agent, channel, max_turns: int = 40, on_turn=None) -> dict:
    """Same loop as robot_customer.run_dialogue, with a hook called with each
    customer package BEFORE the agent hears it (the save point)."""
    pkg = None
    for _ in range(max_turns):
        if pkg is not None and on_turn is not None:
            on_turn(pkg)
        pkg = customer.respond(agent.step(channel(pkg) if pkg is not None else None))
        if customer.done:
            break
    return {"done": customer.done, "turns": len(customer.packages), "hit_turn_limit": not customer.done}
