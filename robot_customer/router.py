"""Step 3: Router.

Reads the label the assistant sends with each turn (the action U2R picked and
the slot it targets) and decides how the customer's answer leaves:

  label                    path    who writes the reply
  normal / accept/ confirm  speak   the LLM (natural reply)
  spell                     spell   no LLM: exact goal value, letter by letter
  type, user can type       type    no LLM: exact goal value (+ typos)
  type, user can't type     speak   fixed line "Sorry, I'm driving, I can't type"

The router never guesses from the assistant's wording.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .goal_builder import Goal


class Path(str, Enum):
    SPEAK = "speak"
    SPELL = "spell"
    TYPE = "type"
    CANNOT_TYPE = "cannot_type"   # speak path with the fixed refusal line


VALID_ACTIONS = {"normal", "accept", "confirm", "spell", "type"}


@dataclass
class AssistantTurn:
    """What the assistant sends each turn. ``action`` is U2R's choice."""
    text: str
    action: str = "normal"          # normal | accept | confirm | spell | type
    slot: str | None = None         # which slot the question targets

    @classmethod
    def from_dict(cls, d: dict) -> "AssistantTurn":
        return cls(text=d["text"], action=(d.get("action") or "normal").lower(), slot=d.get("slot"))


def route(turn: AssistantTurn, goal: Goal) -> Path:
    action = turn.action.lower()
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown assistant action {turn.action!r}; expected one of {sorted(VALID_ACTIONS)}")
    if action in ("spell", "type"):
        if turn.slot not in goal.slots:
            # Asked to spell/type something the customer has no value for:
            # let the LLM answer naturally ("I don't have a preference").
            return Path.SPEAK
        if action == "spell":
            return Path.SPELL
        return Path.TYPE if goal.can_type else Path.CANNOT_TYPE
    return Path.SPEAK
