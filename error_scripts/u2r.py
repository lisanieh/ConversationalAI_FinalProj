"""
u2r.py — the U2R decision rule, closed-form thresholds p*, and decision regions.

Proposal equation (1):
    E[C(a)] = c_a + p(1 − r_a)Cw + (1 − p) e_a Cw
Written as an affine function of p:
    E[C(a)] = α_a + β_a · p,   α_a = c_a + e_a·Cw,   β_a = Cw·(1 − r_a − e_a)
Therefore:
  * each action is a straight line, and the policy is the lower envelope of these lines;
  * the crossover of any two actions a, b is p*_{ab} = (alpha_b - alpha_a) / (beta_a - beta_b);
  * ACCEPT: c=0, r=0, e=0 → E = Cw·p。

Units: c_a and Cw must share a unit (user-seconds or turns recommended); r_a / e_a are probabilities.
r_a, e_a, c_a may vary with (slot_type, condition) -> p* is a per-slot threshold; that is what "per slot" means.
"""
from __future__ import annotations

from dataclasses import dataclass

ACTIONS = ("ACCEPT", "CONFIRM", "SPELL", "TYPE")


@dataclass(frozen=True)
class ActionParams:
    c: float  # user cost (effort)
    r: float  # P(fixed | original value wrong)
    e: float  # P(broken | original value right), e.g. a typo, or spelled letters misheard by Whisper


def line(a: ActionParams, Cw: float) -> tuple[float, float]:
    return a.c + a.e * Cw, Cw * (1 - a.r - a.e)


def expected_cost(p: float, a: ActionParams, Cw: float) -> float:
    return a.c + p * (1 - a.r) * Cw + (1 - p) * a.e * Cw


def choose(p: float, params: dict[str, ActionParams], Cw: float,
           available: tuple[str, ...] = ACTIONS) -> tuple[str, dict]:
    """Return the argmin action and the expected cost of every action. `available` expresses "this user cannot type"."""
    costs = {k: expected_cost(p, params[k], Cw) for k in available}
    return min(costs, key=costs.get), costs


def crossover(a: ActionParams, b: ActionParams, Cw: float) -> float | None:
    """p* where the two actions have equal expected cost; None for parallel lines.
    For p > p*, the action with the smaller slope is better."""
    (aa, ba), (ab, bb) = line(a, Cw), line(b, Cw)
    if abs(ba - bb) < 1e-12:
        return None
    return (ab - aa) / (ba - bb)


def spell_type_threshold(spell: ActionParams, typ: ActionParams, Cw: float) -> float | None:
    """The proposal's testable prediction: the SPELL / TYPE crossover.
    Expanded: p* = [(c_T - c_S) + Cw(e_T - e_S)] / [Cw((r_T - r_S) + (e_T - e_S))]
    Intuition: if typing costs more than spelling (c_T > c_S), p* rises; when noise lowers r_S the denominator grows and p* falls -> the noisier, the earlier we switch to typing."""
    return crossover(spell, typ, Cw)


def accept_threshold(a: ActionParams, Cw: float) -> float:
    """ACCEPT vs action a: repairing pays off when p >= (c_a + e_a*Cw) / ((r_a + e_a)*Cw)."""
    return (a.c + a.e * Cw) / ((a.r + a.e) * Cw)


def decision_regions(params: dict[str, ActionParams], Cw: float,
                     available: tuple[str, ...] = ACTIONS, grid: int = 10001) -> list[tuple[str, float, float]]:
    """Sweep p in [0,1] and return [(action, p_from, p_to), ...]. Used for decision plots and the threshold table in the paper."""
    regions, cur, start = [], None, 0.0
    for i in range(grid):
        p = i / (grid - 1)
        a, _ = choose(p, params, Cw, available)
        if a != cur:
            if cur is not None:
                regions.append((cur, start, p))
            cur, start = a, p
    regions.append((cur, start, 1.0))
    return regions


if __name__ == "__main__":
    # example parameters (placeholders; replace with estimates from outcome_model.py)
    clean = {"ACCEPT": ActionParams(0, 0, 0), "CONFIRM": ActionParams(4, .70, .02),
             "SPELL": ActionParams(10, .85, .03), "TYPE": ActionParams(14, .97, .02)}
    noisy = dict(clean, CONFIRM=ActionParams(4, .55, .05), SPELL=ActionParams(12, .45, .10))
    for name, P in [("clean", clean), ("noise 0 dB", noisy)]:
        print(name, "SPELL/TYPE p* =", round(spell_type_threshold(P["SPELL"], P["TYPE"], 60), 3),
              "regions:", [(a, round(s, 3), round(e, 3)) for a, s, e in decision_regions(P, 60)])


def plot_cost_lines(params: dict[str, ActionParams], Cw: float, path: str, title: str = "") -> None:
    """Plot the four expected-cost lines and the lower envelope: the clearest single figure for the paper or a meeting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    p = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for k, col in zip(ACTIONS, ["#999999", "#e69f00", "#2a6fdb", "#009e73"]):
        ax.plot(p, [expected_cost(x, params[k], Cw) for x in p], label=k, c=col, lw=1.5)
    env = [min(expected_cost(x, params[k], Cw) for k in ACTIONS) for x in p]
    ax.plot(p, env, c="k", lw=3, alpha=.25, label="U2R (lower envelope)")
    for a, s, e in decision_regions(params, Cw)[1:]:
        ax.axvline(s, ls=":", c="grey", lw=1)
    ax.set_xlabel("p_hat = P(slot value is wrong)")
    ax.set_ylabel("expected cost (user-seconds)")
    ax.set_title(title or f"U2R expected cost, Cw={Cw}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
