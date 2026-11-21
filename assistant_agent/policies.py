"""U2R and the baseline repair policies.

Eq. (1):  E[C(a)] = c_a + p (1 - r_a) C_w + (1 - p) e_a C_w

Setting E[C(spell)] = E[C(type)] gives the closed-form threshold

          (c_T - c_S)/C_w + (e_T - e_S)
   p* = ---------------------------------      typing is cheaper when p_hat > p*
          (r_T - r_S) + (e_T - e_S)

Policy specs (``run_agent.py --policies``):
  accept | confirm | spell | type    fixed action for every heard value
  typed                              ask every slot by typing from the start (typed baseline)
  threshold[:<tau>]                  low raw confidence -> type (Bell et al., 2000 style)
  threshold-confirm[:<tau>]          low raw confidence -> confirm (the classic ASR-confidence rule)
  phat-confirm[:<t>]                 calibrated p_hat > t -> confirm. Same estimator as U2R but no
                                     cost model: the ablation that shows what Eq. (1) adds.
  llm                                the LLM picks the repair (CHOOSE prompt)
  u2r | u2r:<C_w> | u2r-notype       the method; C_w sweep (seconds); ablation without TYPE

c_a and C_w are both in seconds of the customer's time (see config.py).
"""

from __future__ import annotations

from .config import ACTIONS, C_W, EFFORT
from .outcomes import RepairOutcomeModel


class Policy:
    name = "base"
    ask_action = "normal"          # how each slot is first asked
    needs_p_hat = False

    def choose(self, ctx: dict) -> str:
        raise NotImplementedError


class Fixed(Policy):
    def __init__(self, action: str, name: str | None = None, ask_action: str = "normal"):
        self.action, self.name, self.ask_action = action, name or action, ask_action

    def choose(self, ctx):
        return self.action


class ConfidenceThreshold(Policy):
    """Raw Whisper confidence (mean token logprob of the value) below ``tau`` -> ``action``."""

    def __init__(self, tau: float = -0.5, action: str = "type"):
        self.tau, self.action = tau, action
        self.name = f"threshold:{tau}" if action == "type" else f"threshold-{action}:{tau}"

    def choose(self, ctx):
        f = ctx.get("features") or {}
        return self.action if f.get("mean_logprob") is not None and f["mean_logprob"] < self.tau else "accept"


class PHatThreshold(Policy):
    """Calibrated p_hat above ``t`` -> ``action``, else accept."""
    needs_p_hat = True

    def __init__(self, t: float = 0.2, action: str = "confirm"):
        self.t, self.action, self.name = t, action, f"phat-{action}:{t}"

    def choose(self, ctx):
        if ctx.get("p_hat") is None:
            raise ValueError("phat policies need p_hat: pass --estimator")
        return self.action if ctx["p_hat"] > self.t else "accept"


class LLMChoice(Policy):
    name = "llm"

    def choose(self, ctx):
        return ctx["agent"].llm_choose(ctx["transcript"], ctx["slot"], ctx["hyp"])


class U2R(Policy):
    needs_p_hat = True

    def __init__(self, outcomes: RepairOutcomeModel, C_w: float = C_W, effort: dict | None = None,
                 actions=ACTIONS, name: str = "u2r"):
        self.outcomes, self.C_w, self.effort = outcomes, C_w, effort or EFFORT
        self.actions, self.name = tuple(actions), name

    def c(self, action: str, slot_type: str) -> float:
        """c_a in seconds: measured in the fit phase if available, else the prior."""
        measured = self.outcomes.cost(action, slot_type)
        return self.effort[action] if measured is None else measured

    def expected_costs(self, p: float, slot_type: str) -> dict[str, float]:
        costs = {}
        for a in self.actions:
            r, e = self.outcomes.get(a, slot_type)
            costs[a] = self.c(a, slot_type) + p * (1 - r) * self.C_w + (1 - p) * e * self.C_w
        return costs

    def choose(self, ctx):
        if ctx.get("p_hat") is None:
            raise ValueError("U2R needs p_hat: pass --estimator")
        actions = tuple(a for a in self.actions if a in ctx.get("allowed", self.actions)
                        and (a != "type" or ctx.get("can_type", True)))
        costs = {a: c for a, c in self.expected_costs(ctx["p_hat"], ctx["slot_type"]).items() if a in actions}
        ctx["expected_costs"] = costs
        return min(costs, key=costs.get)

    def p_star(self, slot_type: str) -> float | None:
        rS, eS = self.outcomes.get("spell", slot_type)
        rT, eT = self.outcomes.get("type", slot_type)
        denom = (rT - rS) + (eT - eS)
        if abs(denom) < 1e-9:
            return None
        p = ((self.c("type", slot_type) - self.c("spell", slot_type)) / self.C_w + (eT - eS)) / denom
        return p if 0 <= p <= 1 else None          # None: one action wins everywhere


class ForcedFor(Policy):
    """Save-point replays: force ``action`` for one slot, accept everything else."""

    def __init__(self, slot: str, action: str):
        self.slot, self.action, self.name = slot, action, f"replay:{action}"

    def choose(self, ctx):
        return self.action if ctx["slot"] == self.slot else "accept"


def make_policy(spec: str, outcomes: RepairOutcomeModel | None = None) -> Policy:
    if spec in ACTIONS:
        return Fixed(spec)
    if spec == "typed":
        return Fixed("accept", "typed", ask_action="type")
    if spec.startswith("threshold-confirm"):
        return ConfidenceThreshold(float(spec.split(":")[1]) if ":" in spec else -0.5, "confirm")
    if spec.startswith("threshold"):
        return ConfidenceThreshold(float(spec.split(":")[1]) if ":" in spec else -0.5)
    if spec.startswith("phat-confirm"):
        return PHatThreshold(float(spec.split(":")[1]) if ":" in spec else 0.2, "confirm")
    if spec == "llm":
        return LLMChoice()
    if spec.startswith("u2r"):
        if outcomes is None:
            raise ValueError("U2R needs --outcomes (from the fit phase)")
        if spec == "u2r-notype":
            return U2R(outcomes, actions=("accept", "confirm", "spell"), name=spec)
        return U2R(outcomes, C_w=float(spec.split(":")[1]) if ":" in spec else C_W, name=spec)
    raise ValueError(f"unknown policy {spec!r}")
