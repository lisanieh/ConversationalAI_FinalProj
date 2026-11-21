"""Shared constants for the assistant agent: actions, costs, slots and slot types.

The slot schema is taken from ``robot_customer.goal_builder`` so the agent asks
for exactly the slots the customer's goal can contain.
"""

from __future__ import annotations

from robot_customer.goal_builder import CONTACT_SLOTS, DOMAIN_SLOTS

DOMAINS = tuple(DOMAIN_SLOTS)                       # ("hotel", "restaurant")
ALL_SLOTS = sorted({s for v in DOMAIN_SLOTS.values() for s in v} | set(CONTACT_SLOTS))


def schema(domain: str) -> list[str]:
    """Slots the agent collects for a domain, in asking order."""
    return DOMAIN_SLOTS[domain] + CONTACT_SLOTS


# Repair actions U2R chooses between. They are also the labels the robot
# customer's router reads (lowercase, see robot_customer/router.py).
ACTIONS = ("accept", "confirm", "spell", "type")

# c_a in Eq. (1): user effort per action, and C_w: cost of committing a wrong
# value, both in SECONDS of the customer's time. These are only priors: once the
# fit phase has run, U2R uses the measured mean effort per action (outcomes.json,
# from effort.EffortModel). C_w = 60 s means "a wrong booking costs the customer
# about a minute to notice and fix"; it is swept in the robustness analysis.
# "reask" is the plain voice re-ask used after a "no" to a confirmation.
EFFORT = {"accept": 0.0, "confirm": 3.0, "spell": 10.0, "type": 12.0, "reask": 4.0,
          "readback": 3.0, "retype": 12.0}     # readback / retype: the checks after a typed value
C_W = 60.0

SLOT_TYPE = {
    "name": "name", "email": "email", "phone": "phone",
    "people": "number", "stay": "number", "time": "time", "day": "day",
    "area": "category", "pricerange": "category", "type": "category", "food": "category",
}
SLOT_TYPES = ["name", "email", "phone", "number", "time", "day", "category", "other"]
ENTITY_SLOTS = {"name", "email", "phone"}


def allowed_actions(slot: str) -> tuple[str, ...]:
    """Spelling or typing only makes sense for open-vocabulary entities.
    Spelling "8" or "18:00" letter by letter is not a real repair (it decoded
    "18:00" as "1 800"), so other slots get accept or confirm only."""
    return ACTIONS if slot in ENTITY_SLOTS else ("accept", "confirm")


def slot_type(slot: str) -> str:
    return SLOT_TYPE.get(slot, "other")


# Whisper features of the slot error estimator (the audio part computes them).
NUM_FEATURES = ["min_logprob", "mean_logprob", "agreement",
                "no_speech_prob", "compression_ratio", "snr_est"]
