"""Step 1: Goal builder.

Builds the customer's secret booking form (the answer key) before each dialogue.

Booking slots come from MultiWOZ (real user goals), contact details come from
Faker, and two per-user properties are sampled here: whether the user can type,
and which TTS voice speaks for them.

Two sources for the booking part:
  * A local MultiWOZ 2.1-style ``data.json`` (each dialogue has a ``goal`` field
    with ``{"hotel": {"info": {...}, "book": {...}}, ...}``). Pass its path.
  * A built-in sampler over the MultiWOZ ontology values. It is used when no
    file is given, so everything runs offline.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    from faker import Faker
except ImportError:  # Faker is optional; a small fallback list is used instead.
    Faker = None

# --- MultiWOZ ontology values (subset relevant to hotel / restaurant booking) ---
AREAS = ["north", "south", "east", "west", "centre"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
PRICE_RANGES = ["cheap", "moderate", "expensive"]
FOODS = ["italian", "chinese", "indian", "british", "european", "thai",
         "french", "mexican", "japanese", "korean", "turkish", "seafood"]
HOTEL_TYPES = ["hotel", "guesthouse"]
TIMES = ["11:30", "12:00", "12:15", "13:00", "17:45", "18:00", "18:30",
         "19:00", "19:15", "19:30", "20:00", "20:30"]

# Kokoro voice ids (American/British, male/female). Your TTS teammate can swap
# these for Piper voice names; the customer only stores the id.
DEFAULT_VOICES = ["af_heart", "af_bella", "am_michael", "am_adam",
                  "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]

# Slots the assistant must collect, in a natural order, per domain.
DOMAIN_SLOTS = {
    "hotel": ["area", "pricerange", "type", "day", "people", "stay"],
    "restaurant": ["area", "pricerange", "food", "day", "people", "time"],
}
CONTACT_SLOTS = ["name", "email", "phone"]


@dataclass
class Goal:
    dialogue_id: str
    domain: str                      # "hotel" | "restaurant"
    slots: dict[str, str]            # booking + contact slots, written form
    can_type: bool
    voice: str
    seed: int
    source: str = "ontology"         # "multiwoz:<id>" or "ontology"
    slot_order: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        """Human-readable goal text used inside the customer prompt."""
        lines = [f"- domain: {self.domain}"]
        lines += [f"- {k}: {v}" for k, v in self.slots.items()]
        return "\n".join(lines)


def _normalize(value: str) -> str:
    return str(value).strip().lower()


def load_multiwoz_goals(path: str | Path, domains=("hotel", "restaurant")) -> list[dict]:
    """Extract hotel/restaurant booking goals from a MultiWOZ 2.1-style data.json.

    Returns a list of ``{"source", "domain", "slots"}`` dicts, keeping only goals
    that contain a booking (``book`` part) for one of ``domains``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for dial_id, dial in data.items():
        goal = dial.get("goal", {})
        for dom in domains:
            g = goal.get(dom) or {}
            info, book = g.get("info") or {}, g.get("book") or {}
            if not book:
                continue
            slots = {}
            for k in DOMAIN_SLOTS[dom]:
                v = info.get(k, book.get(k))
                if v not in (None, "", "dontcare"):
                    slots[k] = _normalize(v)
            if {"day", "people"} <= slots.keys():
                out.append({"source": f"multiwoz:{dial_id}", "domain": dom, "slots": slots})
            break  # one domain per dialogue keeps goals single-task
    return out


# MultiWOZ 2.2 slot names ("hotel-bookday") -> our slot names ("day").
_MWZ22_SLOT = {"area": "area", "pricerange": "pricerange", "type": "type", "food": "food",
               "bookday": "day", "bookpeople": "people", "bookstay": "stay", "booktime": "time"}


def _goal_from_state(source: str, dom: str, slot_values: dict) -> dict | None:
    """slot_values: {"hotel-bookday": ["saturday"], ...} (2.2 keeps several surface forms; take the first)."""
    slots = {}
    for src, ours in _MWZ22_SLOT.items():
        vals = slot_values.get(f"{dom}-{src}")
        v = vals[0] if isinstance(vals, list) and vals else vals
        if ours in DOMAIN_SLOTS[dom] and v not in (None, "", "dontcare", "none"):
            slots[ours] = _normalize(str(v)).replace("guest house", "guesthouse")
    if not {"day", "people"} <= slots.keys():
        return None
    return {"source": source, "domain": dom, "slots": {k: slots[k] for k in DOMAIN_SLOTS[dom] if k in slots}}


def _user_states_original(dial: dict) -> tuple[str, list[str], list[tuple[str, dict]]]:
    """Original MultiWOZ 2.2 release (datasets/multiwoz/data/MultiWOZ_2.2/<split>/dialogues_*.json)."""
    states = [(fr["service"], fr["state"]["slot_values"])
              for t in dial["turns"] if t["speaker"] == "USER"
              for fr in t.get("frames", []) if "state" in fr]
    return dial["dialogue_id"], dial.get("services", []), states


def _user_states_hf(dial: dict) -> tuple[str, list[str], list[tuple[str, dict]]]:
    """Hugging Face export (download_datasets.py, pfb30/multi_woz_v22): turns are column lists,
    speaker 0 = USER, and each state stores slots_values as name / list columns."""
    turns, states = dial["turns"], []
    for spk, frames in zip(turns["speaker"], turns["frames"]):
        if spk not in (0, "USER"):
            continue
        for service, st in zip(frames.get("service", []), frames.get("state", [])):
            sv = st.get("slots_values") or {}
            states.append((service, dict(zip(sv.get("slots_values_name", []), sv.get("slots_values_list", [])))))
    return dial["dialogue_id"], dial.get("services", []), states


def _read_dialogues(path: Path):
    """Yield (parser, dialogue) from a 2.2 file in either layout (JSON list or JSON lines)."""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        rows = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    for d in rows:
        yield (_user_states_hf if isinstance(d.get("turns"), dict) else _user_states_original), d


def load_multiwoz22_goals(path: str | Path, domains=("hotel", "restaurant")) -> list[dict]:
    """Booking goals from MultiWOZ 2.2 = the user's FINAL dialogue state per domain.

    ``path`` is a split folder (``.../MultiWOZ_2.2/train``), the whole release
    folder (all splits), a single ``dialogues_*.json``, or the JSON-lines file
    written by download_datasets.py. One goal per dialogue (its first booked
    hotel/restaurant domain), keeping only goals with a day and a party size.
    """
    path = Path(path)
    files = sorted(path.rglob("dialogues_*.json")) if path.is_dir() else [path]
    if path.is_dir() and not files:
        files = sorted(path.rglob("*.json"))
    out = []
    for f in files:
        for parse, dial in _read_dialogues(f):
            did, services, states = parse(dial)
            final: dict[str, dict] = {}
            for service, sv in states:                    # later turns overwrite earlier ones
                if service in domains:
                    final[service] = sv
            for dom in [s for s in services if s in domains] or list(final):
                g = _goal_from_state(f"multiwoz22:{did}", dom, final.get(dom, {}))
                if g:
                    out.append(g)
                    break
    return out


def load_goals(path: str | Path) -> list[dict]:
    """Detect the MultiWOZ version: 2.1 (one data.json with a "goal" per dialogue) or 2.2."""
    path = Path(path)
    if path.is_file() and path.name == "data.json":
        return load_multiwoz_goals(path)
    if path.is_file():
        head = path.read_text(encoding="utf-8")[:4000].lstrip()
        if head.startswith("{") and '"goal"' in head:
            return load_multiwoz_goals(path)
    return load_multiwoz22_goals(path)


def _sample_ontology_goal(rng: random.Random) -> dict:
    domain = rng.choice(["hotel", "restaurant"])
    slots = {
        "area": rng.choice(AREAS),
        "pricerange": rng.choice(PRICE_RANGES),
        "day": rng.choice(DAYS),
        "people": str(rng.randint(1, 6)),
    }
    if domain == "hotel":
        slots["type"] = rng.choice(HOTEL_TYPES)
        slots["stay"] = str(rng.randint(1, 5))
    else:
        slots["food"] = rng.choice(FOODS)
        slots["time"] = rng.choice(TIMES)
    ordered = {k: slots[k] for k in DOMAIN_SLOTS[domain] if k in slots}
    return {"source": "ontology", "domain": domain, "slots": ordered}


_FALLBACK_FIRST = ["varun", "maya", "daniel", "priya", "lucas", "sofia", "omar", "chen"]
_FALLBACK_LAST = ["kamath", "nguyen", "garcia", "patel", "oconnor", "schmidt", "li", "haddad"]


def _contact_details(rng: random.Random, seed: int) -> dict[str, str]:
    if Faker is not None:
        fake = Faker("en_US")
        fake.seed_instance(seed)
        first, last = fake.first_name(), fake.last_name()
    else:
        first, last = rng.choice(_FALLBACK_FIRST).title(), rng.choice(_FALLBACK_LAST).title()

    # Realistic email built from the name ("varun.k@gmail.com" style) so the
    # spoken form is plausible; Faker's free_email uses unrelated names.
    f, l = first.lower(), "".join(c for c in last.lower() if c.isalpha())
    local = rng.choice([f"{f}.{l[0]}", f"{f}.{l}", f"{f[0]}{l}", f"{f}{l}{rng.randint(1, 99)}"])
    email = f"{local}@{rng.choice(EMAIL_DOMAINS)}"

    # US-style 10-digit phone in a fixed "404 555 1234" layout (area code and
    # exchange never start with 0/1, matching the NANP).
    phone = (f"{rng.randint(2, 9)}{rng.randint(0, 9)}{rng.randint(0, 9)} "
             f"{rng.randint(2, 9)}{rng.randint(0, 9)}{rng.randint(0, 9)} "
             f"{rng.randint(0, 9999):04d}")
    return {"name": f"{first} {last}", "email": email, "phone": phone}


class GoalBuilder:
    """Creates reproducible goals: the same seed always gives the same goal."""

    def __init__(self, multiwoz_path: str | Path | None = None,
                 cannot_type_rate: float = 0.20,
                 voices: list[str] | None = None):
        self.pool = load_goals(multiwoz_path) if multiwoz_path else None
        if self.pool is not None and not self.pool:
            raise ValueError(f"no hotel/restaurant booking goals found in {multiwoz_path}")
        self.cannot_type_rate = cannot_type_rate
        self.voices = voices or DEFAULT_VOICES

    def build(self, seed: int, dialogue_id: str | None = None) -> Goal:
        rng = random.Random(seed)
        base = rng.choice(self.pool) if self.pool else _sample_ontology_goal(rng)
        slots = dict(base["slots"])
        slots.update(_contact_details(rng, seed))
        return Goal(
            dialogue_id=dialogue_id or f"dlg_{seed:05d}",
            domain=base["domain"],
            slots=slots,
            can_type=rng.random() >= self.cannot_type_rate,
            voice=rng.choice(self.voices),
            seed=seed,
            source=base["source"],
            slot_order=list(slots.keys()),
        )
