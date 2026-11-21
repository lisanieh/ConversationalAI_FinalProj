"""Repair requests and decoding the customer's repair answers.

Requests are templates. The robot customer reads only the label
(``{"action", "slot"}``), so the wording never changes what it does.

Decoders are rules, written against what the robot customer produces:
  spell    ``spell_out``:  'V. A. R. U. N. dot. K. at. G. M. A. I. L. dot. com.'
                           'four. oh. four, five. five. five, one. two. three. four.'
           Whisper may also write it as 'V-A-R-U-N dot K at ...' or 'varun dot k ...'.
  type     ``typed_text`` (with typos) arrives as is.
  confirm  a natural "Yes, that's right." / "No, it's ...". The value of a
           correction is extracted from the answer.
  cannot   ``CANNOT_TYPE_LINE`` ("Sorry, I'm driving, I can't type right now.").
"""

from __future__ import annotations

import re

from .values import DIGIT_WORDS, canonical, display, format_phone

LABEL = {"area": "area", "pricerange": "price range", "type": "type of place",
         "food": "type of food", "day": "day", "people": "number of people",
         "stay": "number of nights", "time": "time", "name": "name",
         "email": "email address", "phone": "phone number"}

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

GREETING = "Hello, how can I help you today?"
ASK_DOMAIN = "Are you looking to book a hotel or a restaurant?"
BOOKED = "Great, your booking is confirmed. Is there anything else?"


def question(slot: str) -> str:
    return QUESTIONS.get(slot, f"What is the {LABEL.get(slot, slot)}?")


def request(action: str, slot: str, value: str = "") -> str:
    label = LABEL.get(slot, slot)
    if action == "confirm":
        return f"Just to confirm, the {label} is {display(slot, value)}, is that right?"
    if action == "spell":
        return f"Sorry, could you spell the {label} for me, letter by letter?"
    if action == "type":
        return f"Sorry, it's a bit noisy. Could you type the {label} into the chat box, please?"
    if action == "ask_type":                      # the "typed" baseline asks by typing from the start
        return f"Could you type the {label} into the chat box, please?"
    if action == "reask":
        return f"Sorry about that. Could you tell me the {label} again?"
    if action == "readback":                      # after a typed value, before committing it
        return f"Thanks. I have {display(slot, value)}, is that right?"
    if action == "recheck":                       # after "No, it's ...": read the correction back
        return f"Sorry about that. So the {label} is {display(slot, value)}, is that right?"
    if action == "retype":
        return f"Sorry, could you type the {label} again, please?"
    return question(slot)


# ------------------------------------------------------------------- decoding
_SYMBOLS = {"dot": ".", "period": ".", "point": ".", "at": "@", "dash": "-", "hyphen": "-",
            "underscore": "_", "plus": "+", "apostrophe": "'"}
_SPELL_DIGITS = {k: v for k, v in DIGIT_WORDS.items() if k not in ("o",)}
_FILLER = {"its", "it's", "is", "that", "the", "and", "my", "letter", "letters", "spelled",
           "capital", "sorry", "okay", "ok", "sure", "yes", "um", "uh"}
_EMAIL_RE = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}")
CANNOT_TYPE_RE = re.compile(r"\b(can'?t|cannot|can not|unable to|not able to)\s+(type|text)\b|\bdriving\b", re.I)


def is_cannot_type(text: str) -> bool:
    return bool(CANNOT_TYPE_RE.search(text or ""))


def decode_spelled(text: str, slot: str) -> str:
    """Rebuild a value from its spelled-out form. Returns "" if nothing usable.

    Commas separate words (``spell_out`` joins words with ', '); within a word,
    single letters, symbol words and digit words are glued together. Whole
    words (Whisper sometimes merges letters back, and TLDs are said as words)
    are kept as they are.
    """
    low = text.lower().replace("’", "'")
    if slot == "phone":
        d = "".join(_SPELL_DIGITS.get(t, t if t.isdigit() else "") for t in re.findall(r"\d|[a-z]+", low))
        return format_phone(d)
    if slot == "email" and _EMAIL_RE.search(low):   # recognizer wrote the address itself
        return canonical(slot, _EMAIL_RE.search(low).group())
    words = []
    if slot == "email":   # 'varun.kiatgmail.com': a dot inside a word is part of the address
        low = re.sub(r"(?<=[a-z0-9])\.(?=[a-z0-9])", " dot ", low)
    # Whisper joins spelled units with hyphens ('V-A-R-U-N', 'B-AT-O-U-T'); a real hyphen
    # is said as "dash", so every hyphen between letters/digits is just a separator.
    low = re.sub(r"(?<=[a-z0-9])-(?=[a-z0-9])", " ", low)
    for group in re.split(r",|;|\bspace\b", low):
        units = []
        for tok in re.findall(r"[a-z]+|\d|[@._+\-']", group):
            if tok in _FILLER and slot != "email":
                continue
            if tok in _SYMBOLS and not (slot == "name" and tok not in ("dash", "hyphen", "apostrophe")):
                units.append(_SYMBOLS[tok])
            elif tok in _SPELL_DIGITS and tok != "oh" and slot != "name":
                units.append(_SPELL_DIGITS[tok])
            elif tok in ".":
                continue                                   # the pause after each letter
            else:
                units.append(tok)
        if units:
            words.append("".join(units))
    sep = "" if slot == "email" else " "
    return canonical(slot, sep.join(words))


_YES_RE = re.compile(r"^\s*(yes|yeah|yep|yup|correct|right|that's right|that is right|exactly|"
                     r"perfect|sure|that's correct|it is|uh huh)\b", re.I)
_NO_RE = re.compile(r"^\s*(no|nope|nah|not quite|that's wrong|that is wrong|incorrect|"
                    r"that's not right|actually|wrong)\b", re.I)


def decode_confirm(text: str, slot: str, hypothesis: str, extractor) -> tuple[str, str | None]:
    """Returns (verdict, value): ('yes', hypothesis), ('no', correction) or ('no', None)."""
    _, vals = extractor.extract(text, asked_slot=slot)
    corr = canonical(slot, vals.get(slot)) if vals.get(slot) else None
    if _NO_RE.search(text):
        if corr and corr != hypothesis:
            return "no", corr
        return "no", None
    if _YES_RE.search(text):
        return "yes", hypothesis
    if corr:                                       # no yes/no word, but a value: take it
        return ("yes", hypothesis) if corr == hypothesis else ("no", corr)
    return "yes", hypothesis                       # unclear answer: keep the value
