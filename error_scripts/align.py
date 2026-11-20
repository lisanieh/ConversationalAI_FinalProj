"""
align.py — measures "mistake error": word-level alignment, WER, slot-level labels and error types.

Core output (one row per slot):
    y            : 1 = the slot value v_hat after Whisper + extraction is wrong, 0 = correct
    error_type   : correct / substitution / deletion / insertion / extraction
                   - substitution : the reference words of the slot were replaced ("wrong words")
                   - deletion     : all reference words of the slot were deleted ("missing words")
                   - insertion    : the slot is not in the reference, but the system filled a value ("made-up words" / hallucination)
                   - extraction   : ASR got the slot span right, but the LLM extracted the wrong value (not an ASR error)

Usage:
    python align.py slots.jsonl labeled.jsonl
Each input row needs at least: ref_text, transcript, slot_type, ref_value, hyp_value
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------- normalization
_NUM_WORDS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50,
}
_SPELLING_VARIANTS = {"center": "centre", "theater": "theatre", "colour": "color"}


def normalize(text: str | None) -> str:
    """Lower-case, strip punctuation, number words -> digits, unify US/UK spellings.
    Note: the whole team must use the same normalization, and the same function for ref and hyp, or errors are created artificially."""
    if text is None:
        return ""
    t = text.lower()
    t = re.sub(r"(\d):(\d)", r"\1 \2", t)          # 7:30 -> 7 30
    t = re.sub(r"[^\w\s']", " ", t)
    toks = []
    for w in t.split():
        w = _SPELLING_VARIANTS.get(w, w)
        toks.append(str(_NUM_WORDS[w]) if w in _NUM_WORDS else w)
    # merge "twenty one" -> "21"
    out: list[str] = []
    for w in toks:
        if out and w.isdigit() and out[-1].isdigit() and int(out[-1]) % 10 == 0 \
                and 20 <= int(out[-1]) <= 90 and int(w) < 10:
            out[-1] = str(int(out[-1]) + int(w))
        else:
            out.append(w)
    return " ".join(out)


# ---------------------------------------------------------------- alignment
@dataclass
class Op:
    kind: str          # "C" correct, "S", "D", "I"
    ref_i: int | None  # index into reference words
    hyp_j: int | None  # index into transcript words


def align(ref: list[str], hyp: list[str]) -> list[Op]:
    """Standard Levenshtein dynamic programming + backtrace. Costs S=D=I=1.
    Ties prefer C/S > D > I (the usual convention, consistent with jiwer/sclite)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
            d[i][j] = min(sub, d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(Op("C" if ref[i - 1] == hyp[j - 1] else "S", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(Op("D", i - 1, None))
            i -= 1
        else:
            ops.append(Op("I", None, j - 1))
            j -= 1
    return ops[::-1]


def wer_counts(ref_text: str, hyp_text: str) -> dict:
    ref, hyp = normalize(ref_text).split(), normalize(hyp_text).split()
    ops = align(ref, hyp)
    c = {k: sum(o.kind == k for o in ops) for k in "CSDI"}
    c["N"] = len(ref)
    c["wer"] = (c["S"] + c["D"] + c["I"]) / max(len(ref), 1)
    return c


def _find_span(words: list[str], value: list[str]) -> tuple[int, int] | None:
    """[start, end) of the first occurrence of value in words; None if not found."""
    if not value:
        return None
    for s in range(len(words) - len(value) + 1):
        if words[s:s + len(value)] == value:
            return s, s + len(value)
    return None


# ---------------------------------------------------------------- slot labels
def label_slot(ref_text: str, transcript: str, ref_value: str | None,
               hyp_value: str | None) -> dict:
    """Return y, error_type and the S/D/I counts inside the slot span."""
    rv, hv = normalize(ref_value), normalize(hyp_value)
    y = int(rv != hv)
    ref_w, hyp_w = normalize(ref_text).split(), normalize(transcript).split()
    ops = align(ref_w, hyp_w)

    if not rv and hv:                       # the user never said it, but the system filled a value
        return {"y": 1, "error_type": "insertion", "span_S": 0, "span_D": 0, "span_I": 0}
    span = _find_span(ref_w, rv.split())
    if span is None:                        # slot value not in this reference sentence (e.g. from context)
        return {"y": y, "error_type": "extraction" if y else "correct",
                "span_S": 0, "span_D": 0, "span_I": 0}
    s, e = span
    in_span = [o for o in ops if o.ref_i is not None and s <= o.ref_i < e]
    # insertions inside the span also count
    idx = {id(o): k for k, o in enumerate(ops)}
    if in_span:
        lo, hi = idx[id(in_span[0])], idx[id(in_span[-1])]
        ins = [o for o in ops[lo:hi + 1] if o.kind == "I"]
    else:
        ins = []
    nS = sum(o.kind == "S" for o in in_span)
    nD = sum(o.kind == "D" for o in in_span)
    nI = len(ins)

    if not y:
        etype = "correct"
    elif nS == 0 and nD == 0 and nI == 0:
        etype = "extraction"               # ASR was right, the LLM extraction was wrong
    elif nD == len(in_span):
        etype = "deletion"
    elif nS > 0:
        etype = "substitution"
    elif nI > 0:
        etype = "insertion"
    else:
        etype = "deletion"                  # partial deletion
    return {"y": y, "error_type": etype, "span_S": nS, "span_D": nD, "span_I": nI}


def main(inp: str, out: str) -> None:
    with open(inp) as f, open(out, "w") as g:
        for line in f:
            r = json.loads(line)
            r.update(label_slot(r["ref_text"], r["transcript"], r.get("ref_value"), r.get("hyp_value")))
            r.update({f"utt_{k}": v for k, v in wer_counts(r["ref_text"], r["transcript"]).items()})
            g.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
