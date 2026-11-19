"""Step 4: Text cleanup for speech.

TTS reads text literally, so before synthesis we rewrite the written text the
way a person would say it:

    varun.k@gmail.com  ->  varun dot k at gmail dot com
    404 555 1234       ->  four oh four, five five five, one two three four
    3pm / 19:30        ->  three p m / seven thirty p m

``to_spoken`` handles natural sentences (speak path). ``spell_out`` produces the
letter-by-letter form for the spell path. The written original is never
modified; callers keep it as the answer key.
"""

from __future__ import annotations

import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
DIGIT_WORDS = ["oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
SPELL_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# Words people say whole inside emails instead of spelling them.
SAY_AS_WORD = {"gmail", "yahoo", "outlook", "hotmail", "icloud", "aol", "proton",
               "com", "org", "net", "edu", "gov", "io", "co", "uk"}
# Spoken top-level domains that are spelled when said ("e d u").
SPELLED_TLDS = {"edu": "e d u", "gov": "gov", "io": "i o", "uk": "u k"}

SYMBOL_WORDS = {".": "dot", "@": "at", "-": "dash", "_": "underscore", "+": "plus"}

ABBREVIATIONS = {
    r"\bmon\b": "monday", r"\btue(s)?\b": "tuesday", r"\bwed\b": "wednesday",
    r"\bthu(r|rs)?\b": "thursday", r"\bfri\b": "friday", r"\bsat\b": "saturday",
    r"\bsun\b": "sunday", r"\bppl\b": "people", r"\bpls\b": "please",
    r"\bapprox\.?\b": "approximately", r"\bno\.\s*(?=\d)": "number ",
}

_VOWELS = set("aeiouy")


def number_to_words(n: int) -> str:
    """0..9999 to English words (enough for party sizes, nights, years)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + (f"-{_ONES[o]}" if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return f"{_ONES[h]} hundred" + (f" {number_to_words(r)}" if r else "")
    th, r = divmod(n, 1000)
    return f"{number_to_words(th)} thousand" + (f" {number_to_words(r)}" if r else "")


def digits_spoken(digits: str) -> str:
    """'404' -> 'four oh four' (how people read phone digits)."""
    return " ".join(DIGIT_WORDS[int(d)] for d in digits)


def phone_spoken(phone: str) -> str:
    """Speak a phone number in its natural groups: 3-3-4 for 10 digits."""
    d = re.sub(r"\D", "", phone)
    if len(d) == 11 and d[0] == "1":
        return "one, " + phone_spoken(d[1:])
    groups = [d[:3], d[3:6], d[6:]] if len(d) == 10 else re.findall(r"\d{1,4}", d)
    return ", ".join(digits_spoken(g) for g in groups)


def time_spoken(hour: int, minute: int, suffix: str | None) -> str:
    """Speak a time. 24h times are converted to 12h with 'a m' / 'p m'."""
    if suffix is None:
        suffix = "pm" if hour >= 12 else "am"
        hour = hour % 12 or 12
    if minute == 0:
        mins = ""
    elif minute < 10:
        mins = f" oh {_ONES[minute]}"
    else:
        mins = f" {number_to_words(minute)}"
    return f"{number_to_words(hour)}{mins} {' '.join(suffix.lower().replace('.', ''))}"


def _speak_token(tok: str) -> str:
    """A piece of an email local part or domain, between symbols."""
    out = []
    for part in re.findall(r"[A-Za-z]+|\d+", tok):
        if part.isdigit():
            out.append(" ".join(SPELL_DIGITS[int(c)] for c in part))
        elif part.lower() in SPELLED_TLDS:
            out.append(SPELLED_TLDS[part.lower()])
        elif part.lower() in SAY_AS_WORD:
            out.append(part.lower())
        elif len(part) <= 2 or not (_VOWELS & set(part.lower())):
            out.append(" ".join(part.lower()))       # 'vk' -> 'v k'
        else:
            out.append(part.lower())
    return " ".join(out)


def email_spoken(email: str) -> str:
    """'varun.k@gmail.com' -> 'varun dot k at gmail dot com'."""
    pieces = re.split(r"([.@\-_+])", email.strip())
    return " ".join(SYMBOL_WORDS[p] if p in SYMBOL_WORDS else _speak_token(p)
                    for p in pieces if p)


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)")
TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)(?![a-z])", re.I)
TIME24_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.I)
NUM_RE = re.compile(r"\b\d+\b")

_ORDINAL_FIX = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def _ordinal(n: int) -> str:
    w = number_to_words(n)
    head, _, last = w.rpartition(" ") if " " in w else ("", "", w)
    hy_head, sep, last = last.rpartition("-") if "-" in last else ("", "", last)
    if last in _ORDINAL_FIX:
        last = _ORDINAL_FIX[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    last = f"{hy_head}{sep}{last}"
    return f"{head} {last}".strip()


def to_spoken(text: str) -> str:
    """Rewrite a natural sentence for TTS. Order matters: emails and phones
    first (they contain digits and dots), then times, ordinals, plain numbers."""
    s = EMAIL_RE.sub(lambda m: email_spoken(m.group()), text)
    s = PHONE_RE.sub(lambda m: phone_spoken(m.group()), s)
    s = TIME_RE.sub(lambda m: time_spoken(int(m.group(1)), int(m.group(2) or 0), m.group(3)), s)
    s = TIME24_RE.sub(lambda m: time_spoken(int(m.group(1)), int(m.group(2)), None), s)
    s = ORDINAL_RE.sub(lambda m: _ordinal(int(m.group(1))), s)
    s = NUM_RE.sub(lambda m: number_to_words(int(m.group())) if len(m.group()) <= 4
                   else digits_spoken(m.group()), s)
    for pat, rep in ABBREVIATIONS.items():
        s = re.sub(pat, rep, s, flags=re.I)
    s = s.replace("&", " and ")
    return " ".join(s.split())


def spell_out(value: str, slot: str | None = None) -> str:
    """Letter-by-letter form for the spell path, with a pause after each unit.

    'varun.k@gmail.com' -> 'V. A. R. U. N. dot. K. at. G. M. A. I. L. dot. com.'
    '404 555 1234'      -> 'four. oh. four, five. five. five, one. two. three. four.'
    Common TLDs ('com', 'org', ...) are said as words, as people do.
    """
    value = value.strip()
    if slot == "phone" or re.fullmatch(r"[\d\s().+\-]+", value):
        d = re.sub(r"\D", "", value)
        groups = [d[:3], d[3:6], d[6:]] if len(d) == 10 else [d]
        return ", ".join(". ".join(DIGIT_WORDS[int(c)] for c in g) for g in groups) + "."

    # Each whitespace-separated word ("Varun", "Kamath") is spelled on its own
    # and words are joined with a comma, giving a longer pause between them.
    tlds = {"com", "org", "net", "edu", "gov"}
    words = []
    for word in value.split():
        units: list[str] = []
        tokens = re.findall(r"[A-Za-z]+|\d|[.@\-_+']", word)
        for i, tok in enumerate(tokens):
            if tok in SYMBOL_WORDS:
                units.append(SYMBOL_WORDS[tok])
            elif tok == "'":
                units.append("apostrophe")
            elif tok.isdigit():
                units.append(SPELL_DIGITS[int(tok)])
            elif tok.lower() in tlds and i > 0 and tokens[i - 1] == "." and i == len(tokens) - 1:
                units.append(tok.lower())
            else:
                units.extend(c.upper() for c in tok)
        words.append(". ".join(units))
    return ", ".join(words) + "."
