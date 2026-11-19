"""Step 2: Customer brain.

``LLMBrain`` talks to any OpenAI-compatible chat endpoint, which covers both
back ends in the plan:
  * vLLM on PACE:  ``vllm serve Qwen/Qwen2.5-7B-Instruct``  -> http://<node>:8000/v1
  * Ollama locally: ``ollama pull qwen2.5:7b-instruct``     -> http://localhost:11434/v1

Temperature and seed are fixed so a dialogue can be rerun exactly (vLLM honours
the seed per request; Ollama honours it via the OpenAI-compatible API too).

``ScriptedBrain`` is a rule-based stand-in with no model, for unit tests and for
checking the pipeline wiring on a laptop.

Message roles: from the brain's point of view the *assistant agent* is the
"user" and the customer's own past replies are "assistant" messages.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Protocol

from .goal_builder import Goal
from .prompts import CUSTOMER_SYSTEM_PROMPT, DONE_TOKEN


class Brain(Protocol):
    def reply(self, goal: Goal, history: list[dict]) -> str: ...


def build_messages(goal: Goal, history: list[dict]) -> list[dict]:
    """history items: {"speaker": "assistant"|"customer", "text": str}."""
    system = CUSTOMER_SYSTEM_PROMPT.format(domain=goal.domain, goal=goal.describe())
    msgs = [{"role": "system", "content": system}]
    for h in history:
        role = "user" if h["speaker"] == "assistant" else "assistant"
        msgs.append({"role": role, "content": h["text"]})
    return msgs


_PREFIX = re.compile(r"^\s*(customer|user|me|caller)\s*:\s*", re.I)


def clean_llm_output(text: str) -> str:
    """Strip artefacts small LLMs add: role prefixes, wrapping quotes, stage
    directions in *asterisks* or (parentheses), and extra lines."""
    text = text.strip()
    text = _PREFIX.sub("", text)
    # Cut anything after the customer's own turn FIRST: LLMs sometimes go on to
    # write the assistant's reply (and a [DONE]) themselves. Only the customer's
    # first paragraph counts, so a [DONE] written later must not end the call.
    text = re.split(r"\n\s*\n|\n\s*(?:assistant|agent|receptionist|system)\s*:|"
                    r"\s(?:assistant|agent|receptionist)\s*:", text, flags=re.I)[0]
    text = re.sub(r"\*[^*]*\*", "", text)          # *smiles*
    text = re.sub(r"\([^()\d]*\)", "", text)       # (pauses), but keep "(404)"
    # Small models often drop the brackets ("DONE."). Accept any upper-case
    # DONE as the end token; lower-case "done" stays ordinary speech.
    done_re = re.compile(r"\[?\bDONE\b\]?\.?")
    done = bool(done_re.search(text))
    text = done_re.sub("", text)
    text = " ".join(text.split()).strip().strip('"').strip()
    return f"{text} {DONE_TOKEN}".strip() if done else text


class LLMBrain:
    def __init__(self, base_url: str = "http://localhost:11434/v1",
                 model: str = "qwen2.5:7b-instruct",
                 temperature: float = 0.7, seed: int = 42,
                 max_tokens: int = 80, api_key: str = "EMPTY", timeout: float = 120):
        self.base_url = base_url.rstrip("/")
        self.model, self.temperature, self.seed = model, temperature, seed
        self.max_tokens, self.api_key, self.timeout = max_tokens, api_key, timeout
        self.last_raw: str | None = None   # unprocessed LLM text of the last call

    def reply(self, goal: Goal, history: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": build_messages(goal, history),
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())
        self.last_raw = body["choices"][0]["message"]["content"]
        return clean_llm_output(self.last_raw)


class ScriptedBrain:
    """Deterministic, model-free customer for tests. Answers the slot it is
    asked about (``asked_slot`` from the router), corrects wrong read-backs, and
    finishes when the assistant says the booking is done."""

    OPENERS = {
        "hotel": "Hi, I'm looking for a {type} in the {area}.",
        "restaurant": "Hi, I'd like to book a table at a {food} place in the {area}.",
    }
    TEMPLATES = {
        "area": "In the {v}, please.", "pricerange": "Something {v}.",
        "type": "A {v}.", "food": "{v} food.", "day": "On {v}.",
        "people": "For {v} people.", "stay": "{v} nights.", "time": "At {v}.",
        "name": "It's {v}.", "email": "My email is {v}.", "phone": "My number is {v}.",
    }

    def __init__(self):
        self.asked_slot: str | None = None
        self.asked_action: str = "normal"

    def reply(self, goal: Goal, history: list[dict]) -> str:
        last = history[-1]["text"].lower() if history else ""
        customer_turns = sum(h["speaker"] == "customer" for h in history)
        if customer_turns == 0:
            return self.OPENERS[goal.domain].format(**goal.slots)
        if "booked" in last or "booking is confirmed" in last:
            return f"Great, thank you so much! {DONE_TOKEN}"
        slot = self.asked_slot
        if slot in goal.slots:
            value = goal.slots[slot]
            if self.asked_action == "confirm":
                return "Yes, that's right." if value.lower() in last else f"No, it's {value}."
            return self.TEMPLATES.get(slot, "{v}.").format(v=value)
        return "I don't have a preference."
