"""Canonical slot values.

The goal stores values in written form ("19:30", "404 555 1234", "3",
"varun.k@gmail.com"), but the customer and Whisper may produce "seven thirty p m",
"four oh four ...", "three" or "varun dot k at gmail dot com". ``canonical``
maps both sides to one form, so "is this value right?" is a plain string
comparison. It only rewrites the FORMAT; a misheard value stays wrong
("nort" does not become "north").
"""

from __future__ import annotations

import re

from robot_customer.goal_builder import AREAS, DAYS, FOODS, HOTEL_TYPES, PRICE_RANGES

_UNITS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
          "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
DIGIT_WORDS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3",
               "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
DONTCARE_RE = re.compile(
    r"\b(no (particular |specific |real )?preference|(don'?t|do not) have (a|any) (particular |specific |real )?preference|"
    r"don'?t (mind|care)|doesn'?t matter|dontcare|whatever|anywhere|up to you|"
    r"any(thing|where)?( [a-z]+){0,3} (is fine|will do|works|is ok(ay)?|is good)|"      # "any good spot will do"
    r"(i'?m|i am) (open to|flexible|fine with|happy with) (any|anything|whatever)|"
    r"not (particular|fussy|picky)|"
    r"(no|(don'?t|do not) have (a|any)) (specific|particular) [a-z ]{0,25}in mind|"   # "no specific price range in mind"
    r"(what(ever)? you|you can|just) (recommend|suggest|choose|pick)|"
    r"any (kind|type|sort)( of)?|any (area|part|location|price|price range|cuisine|food|place|spot|day|time)|"
    r"^any( one)?$)\b")
DONTCARE = "dontcare"

_DAY_ABBR = {"mon": "monday", "tue": "tuesday", "tues": "tuesday", "wed": "wednesday",
             "thu": "thursday", "thur": "thursday", "thurs": "thursday", "fri": "friday",
             "sat": "saturday", "sun": "sunday"}


def normalize(s: str | None) -> str:
    if not s:
        return ""
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    s = re.sub(r"^\s*(the|a|an)\s+", "", s)
    return " ".join(s.split())


def words_to_int(text: str) -> int | None:
    """First number in the text: '3', 'three', 'twenty one', 'twenty-one'."""
    m = re.search(r"\d+", text)
    toks = re.findall(r"[a-z]+", text.lower().replace("-", " "))
    for i, t in enumerate(toks):
        if t in _TENS:
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            val = _TENS[t] + (_UNITS[nxt] if nxt in _UNITS and _UNITS[nxt] < 10 else 0)
            word_pos = text.lower().find(t)
            break
        if t in _UNITS and t != "oh":
            val, word_pos = _UNITS[t], text.lower().find(t)
            break
    else:
        return int(m.group()) if m else None
    if m and m.start() < word_pos:
        return int(m.group())
    return val


def _minutes_from_words(toks: list[str]) -> int:
    if not toks:
        return 0
    if toks[0] == "oh" and len(toks) > 1 and toks[1] in _UNITS:
        return _UNITS[toks[1]]
    if toks[0] in _TENS:
        return _TENS[toks[0]] + (_UNITS[toks[1]] if len(toks) > 1 and toks[1] in _UNITS
                                 and _UNITS[toks[1]] < 10 else 0)
    if toks[0] in _UNITS and _UNITS[toks[0]] >= 10:
        return _UNITS[toks[0]]
    return 0


def parse_time(text: str) -> str | None:
    """'19:30', '7:30 pm', 'seven thirty p m', 'half past seven' -> 'HH:MM'.

    A time with no am/pm and an hour of 1-10 is read as p.m.: booking times in
    MultiWOZ run from late morning to evening ("7:30" means 19:30).
    """
    s = text.lower().replace("p.m.", "pm").replace("a.m.", "am")
    s = re.sub(r"\b([ap])\s+m\b", r"\1m", s)
    hour = minute = None
    suffix = re.search(r"\b(am|pm)\b", s)
    suffix = suffix.group(1) if suffix else None

    m = (re.search(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)\b", s) or re.search(r"\b(\d{1,2})[:.](\d{2})\b", s)
         or re.fullmatch(r"\s*(\d{1,2})(\d{2})\s*", s))          # spelled digits: '1830'
    if m:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
    elif re.search(r"\b(noon|midday)\b", s):
        hour, minute, suffix = 12, 0, "pm"
    else:
        toks = re.findall(r"[a-z]+", s.replace("-", " "))
        rel = re.search(r"\b(half|quarter) (past|to) ([a-z]+)", s)
        if rel and rel.group(3) in _UNITS:
            h = _UNITS[rel.group(3)]
            if rel.group(2) == "past":
                hour, minute = h, 30 if rel.group(1) == "half" else 15
            else:
                hour, minute = h - 1, 45
        else:
            for i, t in enumerate(toks):
                if t in _UNITS and 1 <= _UNITS[t] <= 23 and t != "oh":
                    hour = _UNITS[t]
                    minute = _minutes_from_words(toks[i + 1:i + 3])
                    break
    if hour is None or not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    elif suffix is None and 1 <= hour <= 10:
        hour += 12
    return f"{hour:02d}:{minute:02d}"


def digits_from(text: str) -> str:
    """'four oh four, 555 12 34' -> '4045551234'. Handles 'double five'."""
    out, double = [], False
    for tok in re.findall(r"\d+|[a-z]+", text.lower()):
        if tok.isdigit():
            out.append(tok * (2 if double else 1) if len(tok) == 1 else tok)
        elif tok in DIGIT_WORDS:
            out.append(DIGIT_WORDS[tok] * (2 if double else 1))
        elif tok in ("double", "triple"):
            double = True
            continue
        double = False
    return "".join(out)


def format_phone(digits: str) -> str:
    if len(digits) == 11 and digits[0] == "1":
        digits = digits[1:]
    return f"{digits[:3]} {digits[3:6]} {digits[6:]}" if len(digits) == 10 else digits


_SPOKEN_SYMBOLS = {"dot": ".", "at": "@", "dash": "-", "hyphen": "-", "underscore": "_", "plus": "+"}


def spoken_email_to_written(text: str) -> str:
    """'varun dot k at gmail dot com' -> 'varun.k@gmail.com'."""
    toks = re.findall(r"[a-z0-9]+|[@._+\-]", text.lower())
    out = [_SPOKEN_SYMBOLS.get(t, DIGIT_WORDS.get(t, t) if t not in ("o", "oh") else t) for t in toks]
    return "".join(out)


def _find_term(text: str, terms) -> str | None:
    for t in terms:
        if re.search(rf"\b{re.escape(t)}\b", text):
            return t
    return None


def canonical(slot: str, value: str | None) -> str:
    if value is None:
        return ""
    low = str(value).strip().lower().strip(" .,!?;:")
    if not low:
        return ""
    if low == DONTCARE or DONTCARE_RE.search(low):
        return DONTCARE
    if slot == "email":
        if "@" not in low and re.search(r"\bat\b", low):
            low = spoken_email_to_written(low)
        low = re.sub(r"(?<=\d)-(?=\d)", "", low)          # Whisper writes "fifty" / "five oh" as "5-0"
        return re.sub(r"\s+", "", low).strip(".")
    if slot == "phone":
        d = digits_from(low)
        return format_phone(d) if d else normalize(low)
    if slot in ("people", "stay"):
        n = words_to_int(low)
        return str(n) if n is not None else normalize(low)
    if slot == "time":
        return parse_time(low) or normalize(low)
    if slot == "day":
        for tok in re.findall(r"[a-z]+", low):
            if tok in DAYS:
                return tok
            if tok in _DAY_ABBR:
                return _DAY_ABBR[tok]
        return normalize(low)
    if slot == "name":
        return " ".join(re.sub(r"[^a-z' \-]", " ", low).split())
    n = normalize(low)
    if slot == "area":
        n = re.sub(r"\b(city )?(center|central)\b", "centre", n)
        return _find_term(n, AREAS) or n
    if slot == "pricerange":
        n = re.sub(r"\binexpensive\b", "cheap", re.sub(r"\bmoderately\b", "moderate", n))
        return _find_term(n, PRICE_RANGES) or n
    if slot == "type":
        n = re.sub(r"\bguest house\b", "guesthouse", n)
        return _find_term(n, ["guesthouse"] + [t for t in HOTEL_TYPES if t != "guesthouse"]) or n
    if slot == "food":
        return re.sub(r"\s*\b(food|cuisine)\b", "", n).strip()
    return n


def values_match(slot: str, a: str | None, b: str | None) -> bool:
    ca, cb = canonical(slot, a), canonical(slot, b)
    if slot == "name" and ca and cb and DONTCARE not in (ca, cb):
        # Spacing and hyphens are format: Whisper writes spelled letters as 'VARUNKAMATH'.
        return re.sub(r"[^a-z]", "", ca) == re.sub(r"[^a-z]", "", cb) != ""
    return bool(ca) and ca == cb


def display(slot: str, value: str) -> str:
    """How the agent reads a value back in a confirmation."""
    if slot == "name":
        return " ".join(w.capitalize() for w in value.split())
    return value
