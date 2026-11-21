"""Step 5: Typo script for the type path.

Adds realistic keyboard mistakes to an exact value, in the spirit of the HIVE
perturbations (Hu et al., 2026): a neighbouring key, two swapped letters, or a
missing letter. Each character is independently corrupted with probability
``rate`` (default 2%). A seeded RNG makes the output repeatable.

Self-review: people look at a short field before pressing send. With
``review`` > 0, each typo is independently noticed and fixed with that
probability (it then costs a backspace and a retype, counted in
``n_fixed``). ``review=0`` gives the old per-character model exactly, which
is far too harsh for long fields: at 2% per character a 22-character email
has a typo 36% of the time. ``review`` is a modelling assumption: sweep it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

_ROWS = ["1234567890-", "qwertyuiop", "asdfghjkl", "zxcvbnm,."]


def _build_neighbors() -> dict[str, str]:
    pos = {c: (r, i) for r, row in enumerate(_ROWS) for i, c in enumerate(row)}
    nbrs = {}
    for c, (r, i) in pos.items():
        near = []
        for dr in (-1, 0, 1):
            for di in (-1, 0, 1):
                if (dr, di) == (0, 0):
                    continue
                rr, ii = r + dr, i + di
                if 0 <= rr < len(_ROWS) and 0 <= ii < len(_ROWS[rr]):
                    near.append(_ROWS[rr][ii])
        nbrs[c] = "".join(near)
    return nbrs


QWERTY_NEIGHBORS = _build_neighbors()


@dataclass
class TypoResult:
    typed: str
    original: str
    edits: list[dict] = field(default_factory=list)  # [{"kind", "pos", "from", "to"}]
    n_fixed: int = 0                                 # typos caught in self-review before sending

    @property
    def has_typo(self) -> bool:
        return self.typed != self.original


def add_typos(text: str, rate: float = 0.02, rng: random.Random | None = None,
              kinds: tuple[str, ...] = ("neighbor", "swap", "drop"),
              weights: tuple[float, ...] = (0.5, 0.25, 0.25), review: float = 0.0) -> TypoResult:
    rng = rng or random.Random(0)
    chars = list(text)
    out: list[str] = []
    edits: list[dict] = []
    n_fixed = 0
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isspace() or rng.random() >= rate:
            out.append(c)
            i += 1
            continue
        if review > 0 and rng.random() < review:   # noticed and fixed before sending
            n_fixed += 1
            out.append(c)
            i += 1
            continue
        kind = rng.choices(kinds, weights)[0]
        if kind == "swap" and i + 1 < len(chars) and chars[i + 1] != c:
            out += [chars[i + 1], c]
            edits.append({"kind": "swap", "pos": i, "from": c + chars[i + 1], "to": chars[i + 1] + c})
            i += 2
            continue
        if kind == "drop":
            edits.append({"kind": "drop", "pos": i, "from": c, "to": ""})
            i += 1
            continue
        # neighbor (also the fallback when a swap is impossible)
        pool = QWERTY_NEIGHBORS.get(c.lower())
        if pool:
            n = rng.choice(pool)
            n = n.upper() if c.isupper() else n
            out.append(n)
            edits.append({"kind": "neighbor", "pos": i, "from": c, "to": n})
        else:
            out.append(c)  # symbols like '@' have no neighbour entry: leave as is
        i += 1
    return TypoResult(typed="".join(out), original=text, edits=edits, n_fixed=n_fixed)
