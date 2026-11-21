"""Slot extraction (dialogue state tracking): customer text -> {slot: raw value}.

Both extractors return ``(domain or None, {slot: raw_value})``. Values are raw,
"as heard"; ``values.canonical`` turns them into the comparison form later.
Copying as heard matters: if the extractor silently fixed "nort" to "north",
the recognition error would vanish before U2R could see it.

  LLMExtractor   the real one (Llama, EXTRACT prompt)
  RuleExtractor  regex + MultiWOZ ontology. No model; used for tests and for
                 checking the pipeline. It also decodes confirmation answers.
"""

from __future__ import annotations

import json
import re

from robot_customer.goal_builder import AREAS, DAYS, FOODS, PRICE_RANGES

from .config import ALL_SLOTS, DOMAINS
from .llm import parse_json
from .prompts import EXTRACT_PROMPT
from .values import DIGIT_WORDS, DONTCARE, DONTCARE_RE, digits_from, spoken_email_to_written

_NUM = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
_EMAIL_RE = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}")
_SPOKEN_EMAIL_RE = re.compile(r"\b([a-z0-9]+(?:\s+(?:dot|dash|underscore|plus|[a-z0-9]+))*"
                              r"\s+at\s+[a-z0-9]+(?:\s+dot\s+[a-z]+)+)\b")
_EMAIL_LEAD = re.compile(r"^.*\b(?:e-?mail(?: address)?(?: is|'s)?|it's|it is|that's|that is|sure|yes|okay|ok)\b[\s,:]*")


def _email_answer(low: str) -> str | None:
    """An email in an answer to 'what is your email?', in any mix Whisper writes:
    'varun.k at gmail.com', 'megan chang 50 at gmail dot com', 'meganchang50@gmail.com'."""
    s = _EMAIL_LEAD.sub("", low.strip(), count=1)
    m = _EMAIL_RE.search(spoken_email_to_written(s))
    return m.group().strip(".") if m else None


_TIME_RE = re.compile(r"\b(\d{1,2}(?:[:.]\d{2})?\s*(?:[ap]\.?\s?m\b\.?)?|\d{1,2}[:.]\d{2})|"
                      r"\b((?:half|quarter) (?:past|to) [a-z]+|noon|midday)\b|"
                      r"\b((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
                      r"(?:[ -](?:oh [a-z]+|fifteen|thirty|forty[ -]five|forty|fifty|[a-z]+ty))?"
                      r"(?: ?[ap]\.? ?m\b\.?| o'?clock)?)", re.I)
_NAME_RE = re.compile(r"(?:my name is|my name's|name is|the name is|it's|it is|this is|i'm|i am)"
                      r"\s+([a-z][a-z'\-]*(?:\s+[a-z][a-z'\-]*){0,3})", re.I)
_NAME_STOP = {"please", "thanks", "thank", "and", "the", "my", "it", "is", "that"}
_NOT_NAME = {"yes", "no", "not", "that's", "wrong", "right", "correct", "sorry", "it", "it's", "is",
             "the", "i", "i'm", "can't", "what", "pardon", "again", "preference"}
# MultiWOZ restaurant foods (datasets/multiwoz/db/restaurant_db.json) plus the ontology sampler's.
MULTIWOZ_FOODS = ["african", "asian oriental", "british", "chinese", "european", "french", "gastropub",
                  "indian", "international", "italian", "japanese", "korean", "lebanese", "mediterranean",
                  "mexican", "modern european", "north american", "portuguese", "seafood", "spanish",
                  "thai", "turkish", "vietnamese"]
FOOD_TERMS = sorted(set(FOODS) | set(MULTIWOZ_FOODS), key=len, reverse=True)
_HOTEL_WORDS = r"\b(hotel|guesthouse|guest house|stay|nights?|room|accommodation)\b"
_REST_WORDS = r"\b(restaurant|table|food|eat|dinner|lunch|cuisine)\b"


_DIGITISH = set(DIGIT_WORDS) - {"o"} | {"double", "triple"}


def _digit_span(text: str) -> str | None:
    """Longest run of digit tokens ('404 555 1234', 'four oh four, five ...'), as written."""
    best, run = None, []
    for m in list(re.finditer(r"\d+|[a-z']+", text)) + [None]:
        if m is not None and (m.group().isdigit() or m.group() in _DIGITISH):
            run.append(m)
            continue
        if run and (best is None or len(run) > len(best)):
            best = run
        run = []
    return text[best[0].start():best[-1].end()] if best else None


def _first(text: str, terms) -> str | None:
    hits = [(m.start(), t) for t in terms for m in [re.search(rf"\b{re.escape(t)}\b", text)] if m]
    return min(hits)[1] if hits else None


class RuleExtractor:
    def extract(self, text: str, asked_slot: str | None = None,
                domain: str | None = None) -> tuple[str | None, dict]:
        low = " ".join(text.lower().split())
        out: dict[str, str] = {}
        dom = None
        if re.search(_HOTEL_WORDS, low):
            dom = "hotel"
        elif re.search(_REST_WORDS, low):
            dom = "restaurant"

        if asked_slot and DONTCARE_RE.search(low):
            return dom, {asked_slot: DONTCARE}

        # Ontology words are safe to pick up whether or not they were asked for.
        food = _first(low, FOOD_TERMS)
        no_food = low.replace(food, " ") if food else low      # "north american" is not area "north"
        area = _first(re.sub(r"\b(city )?(center|central)\b", "centre", no_food), AREAS)
        if area:
            out["area"] = area
        price = _first(re.sub(r"\binexpensive\b", "cheap", low), PRICE_RANGES + ["moderately"])
        if price:
            out["pricerange"] = price
        if re.search(r"\bguest ?house\b", low):
            out["type"] = re.search(r"\bguest ?house\b", low).group()
        elif re.search(r"\bhotel\b", low) and (asked_slot == "type"
                                              or re.search(r"\b(a|an)\s+(\w+\s+)?hotel\b", low)):
            out["type"] = "hotel"
        if food:
            out["food"] = food
        elif asked_slot == "food":
            m = re.search(r"([a-z ]+?)\s+(?:food|cuisine)\b", low)
            if m:
                out["food"] = m.group(1).split(" a ")[-1].replace("some ", "").strip()
        day = _first(low, DAYS)
        if day:
            out["day"] = day

        # Numbers only with a clear unit, or when that slot was asked.
        m = re.search(_NUM + r"\s+(?:people|persons|guests|of us|adults)", low)
        if m:
            out["people"] = m.group(1)
        m = re.search(_NUM + r"\s+nights?", low)
        if m:
            out["stay"] = m.group(1)
        if asked_slot in ("people", "stay") and asked_slot not in out:
            m = re.search(r"\b" + _NUM + r"\b", low)
            if m:
                out[asked_slot] = m.group(1)
        if asked_slot == "time" or re.search(r"\d[:.]\d{2}|\b[ap]\.?\s?m\b", low):
            m = _TIME_RE.search(low)
            if m:
                out["time"] = next(g for g in m.groups() if g).strip()

        # Contact details.
        m = _EMAIL_RE.search(low)
        if asked_slot == "email":
            e = _email_answer(low)
            if e:
                out["email"] = e
        elif m:
            out["email"] = m.group().strip(".")
        span = _digit_span(low)
        if span and len(digits_from(span)) >= 7 and (asked_slot == "phone" or re.search(r"\d{3}\D{0,2}\d{3}\D{0,2}\d{4}", low)):
            out["phone"] = span
            for k in ("people", "stay", "time"):
                out.pop(k, None)
        if asked_slot == "name":
            m = _NAME_RE.search(text)
            if m:
                words = [w for w in m.group(1).split() if w.lower() not in _NAME_STOP]
                if words:
                    out["name"] = " ".join(words)
            elif len(low.split()) <= 4 and not (set(re.findall(r"[a-z']+", low)) & _NOT_NAME):
                out["name"] = re.sub(r"[^a-z' \-]", "", low).strip()
        return dom, out


class LLMExtractor:
    """EXTRACT prompt on the agent's LLM. Falls back to rules if the reply is not JSON."""

    def __init__(self, llm, fallback: RuleExtractor | None = None):
        self.llm, self.fallback = llm, fallback or RuleExtractor()
        self.last_raw: str | None = None
        self.filled_from_rules: str | None = None     # the asked slot, if the rules supplied it

    def messages(self, text: str, asked_slot: str | None, domain: str | None) -> list[dict]:
        system = EXTRACT_PROMPT.format(slots=", ".join(ALL_SLOTS), domain=domain or "unknown",
                                       asked=asked_slot or "nothing in particular")
        return [{"role": "system", "content": system},
                {"role": "user", "content": f"Customer: {json.dumps(text)}"}]

    def extract(self, text: str, asked_slot: str | None = None,
                domain: str | None = None) -> tuple[str | None, dict]:
        self.filled_from_rules = None
        self.last_raw = self.llm.chat(self.messages(text, asked_slot, domain), max_tokens=150)
        out = parse_json(self.last_raw)
        if not isinstance(out.get("slots"), dict):          # not the JSON we asked for
            return self.fallback.extract(text, asked_slot, domain)
        dom = out.get("domain")
        dom = dom.lower() if isinstance(dom, str) and dom.lower() in DOMAINS else None
        slots = {k: str(v).strip() for k, v in (out.get("slots") or {}).items()
                 if k in ALL_SLOTS and v not in (None, "", "none", "null")}
        # Valid JSON can still drop the slot that was just asked (a bare "377 963 1537."
        # came back as {}). The rules are good at the asked slot, so fill it from them.
        if asked_slot and asked_slot not in slots:
            rule_dom, rule_vals = self.fallback.extract(text, asked_slot, domain)
            if rule_vals.get(asked_slot):
                slots[asked_slot] = rule_vals[asked_slot]
                self.filled_from_rules = asked_slot
            dom = dom or rule_dom
        return dom, slots
