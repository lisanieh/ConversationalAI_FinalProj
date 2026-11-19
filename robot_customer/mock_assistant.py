"""A stand-in assistant for testing the robot customer before the real
Whisper + Llama agent is connected.

It walks through the goal's slots in order, asking each with the action that a
fixed ``policy`` gives for that slot (e.g. always spell emails), reads back the
contact details with a confirm, and then says the booking is confirmed.

It "understands" replies by cheating: it takes ``typed_text`` for typed turns,
the spelled value for spell turns, and the package's ``true_value`` for spoken
answers. That is fine for checking the customer's wiring and behaviour; it is
not a model of the real assistant.
"""

from __future__ import annotations

from .goal_builder import CONTACT_SLOTS, DOMAIN_SLOTS
from .package import TurnPackage

QUESTIONS = {
    "area": "Which area would you like?",
    "pricerange": "What price range are you looking for?",
    "type": "Would you prefer a hotel or a guesthouse?",
    "food": "What kind of food would you like?",
    "day": "What day is the booking for?",
    "people": "How many people is it for?",
    "stay": "How many nights will you stay?",
    "time": "What time would you like the table?",
    "name": "Can I have your name, please?",
    "email": "What is your email address?",
    "phone": "And a phone number we can reach you on?",
}

FIXED_POLICIES = {
    "always_normal": {},
    "always_spell": {s: "spell" for s in CONTACT_SLOTS},
    "always_type": {s: "type" for s in CONTACT_SLOTS},
    "mixed": {"name": "spell", "email": "type", "phone": "normal"},
}


class MockAssistant:
    def __init__(self, domain: str, policy: str | dict = "mixed", confirm_contacts: bool = True):
        self.slots = DOMAIN_SLOTS[domain] + CONTACT_SLOTS
        self.policy = FIXED_POLICIES[policy] if isinstance(policy, str) else policy
        self.confirm_contacts = confirm_contacts
        self.state: dict[str, str] = {}
        self.queue: list[tuple[str, str]] = []   # pending (action, slot)
        self.pending: tuple[str, str] | None = None
        self.finished = False

    def step(self, pkg: TurnPackage | None) -> dict:
        if pkg is None:
            return {"text": "Hello, how can I help you today?", "action": "normal", "slot": None}

        # 1) Update state from the customer's last turn.
        if self.pending:
            action, slot = self.pending
            if pkg.path == "cannot_type":
                self.queue.insert(0, ("spell", slot))         # fall back to spelling
            elif action == "confirm":
                if pkg.true_text.lower().startswith(("no", "that's wrong", "not")):
                    self.queue.insert(0, ("normal", slot))
            elif pkg.mode == "typed":
                self.state[slot] = pkg.typed_text
            elif pkg.true_value is not None:
                self.state[slot] = pkg.true_value
        self.pending = None

        # 2) Choose the next turn.
        if self.queue:
            action, slot = self.queue.pop(0)
            return self._ask(action, slot)
        for slot in self.slots:
            if slot not in self.state:
                return self._ask(self.policy.get(slot, "normal"), slot)
        if self.confirm_contacts:
            for slot in ("email", "phone"):
                key = f"_confirmed_{slot}"
                if key not in self.state:
                    self.state[key] = "1"
                    return self._ask("confirm", slot)
        self.finished = True
        return {"text": "Great, your booking is confirmed. Is there anything else?",
                "action": "normal", "slot": None}

    def _ask(self, action: str, slot: str) -> dict:
        self.pending = (action, slot)
        if action == "spell":
            text = f"Could you spell your {slot} for me, letter by letter?"
        elif action == "type":
            text = f"Could you type your {slot} into the chat box, please?"
        elif action == "confirm":
            text = f"Just to confirm, your {slot} is {self.state.get(slot, '')}, is that right?"
        else:
            text = QUESTIONS.get(slot, f"What is your {slot}?")
        return {"text": text, "action": action, "slot": slot}

    def final_state(self) -> dict:
        return {k: v for k, v in self.state.items() if not k.startswith("_")}
