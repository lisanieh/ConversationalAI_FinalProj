"""Repair outcome model, per (action, slot type):

  r_a = P(the repair fixes the value | the first hypothesis was wrong)
  e_a = P(the repair breaks the value | the first hypothesis was right)
  c_a = mean customer seconds the repair costs (effort model)

Two sources:
  fit_replays()  save-point replays: every heard value tried under all four
                 actions from the same saved state (replay.py). Preferred:
                 the comparison is paired, value by value.
  fit()          forced-policy runs (always confirm / spell / type).

Beta(1,1) smoothing keeps small cells away from 0 and 1, and a cell with fewer
than ``min_n`` examples on either side falls back to the pooled rate. Customers
who cannot type are included, so TYPE's r/e/c already price in the fallback to
spelling.
"""

from __future__ import annotations

import json
from pathlib import Path

FORCED = ("confirm", "spell", "type")


class RepairOutcomeModel:
    def __init__(self, table: dict | None = None, min_n: int = 20):
        self.table = table or {}       # "action|slot_type" and "action|*" -> {r, e, c, n_wrong, n_right}
        self.min_n = min_n

    @staticmethod
    def _rates(cases: list[tuple[bool, bool, float | None]]) -> dict:
        """cases: (first_wrong, final_wrong, effort_s)."""
        wrong = [c for c in cases if c[0]]
        right = [c for c in cases if not c[0]]
        fixed = sum(not c[1] for c in wrong)
        broken = sum(c[1] for c in right)
        eff = [c[2] for c in cases if c[2] is not None]
        return {"r": (fixed + 1) / (len(wrong) + 2), "e": (broken + 1) / (len(right) + 2),
                "c": sum(eff) / len(eff) if eff else None,
                "n_wrong": len(wrong), "n_right": len(right)}

    def _fill(self, cases_by_action: dict[str, list[tuple[str, tuple]]]) -> "RepairOutcomeModel":
        for action, items in cases_by_action.items():
            if not items:
                continue
            self.table[f"{action}|*"] = self._rates([c for _, c in items])
            for st in sorted({st for st, _ in items}):
                self.table[f"{action}|{st}"] = self._rates([c for s, c in items if s == st])
        return self

    def fit(self, attempts: list[dict]) -> "RepairOutcomeModel":
        rows = [a for a in attempts if a.get("policy") in FORCED and a.get("mode") == "speech"
                and not a.get("missed") and a.get("true_value") is not None
                and a.get("committed_value") is not None]
        return self._fill({act: [(r["slot_type"], (not r["first_correct"], not r["committed_correct"],
                                                   r.get("effort_s")))
                                 for r in rows if r["policy"] == act] for act in FORCED})

    def fit_replays(self, replays: list[dict]) -> "RepairOutcomeModel":
        return self._fill({act: [(r["slot_type"], (bool(r["y"]), bool(r[f"final_wrong_{act.upper()}"]),
                                                   r[f"effort_{act.upper()}"]))
                                 for r in replays] for act in FORCED})

    def _cell(self, action: str, slot_type: str) -> dict:
        cell = self.table.get(f"{action}|{slot_type}")
        if not cell or min(cell["n_wrong"], cell["n_right"]) < self.min_n:
            cell = self.table.get(f"{action}|*")
        if not cell:
            raise KeyError(f"no outcome data for {action!r}; run the fit phase first")
        return cell

    def get(self, action: str, slot_type: str) -> tuple[float, float]:
        if action == "accept":
            return 0.0, 0.0
        cell = self._cell(action, slot_type)
        return cell["r"], cell["e"]

    def cost(self, action: str, slot_type: str) -> float | None:
        """Measured mean seconds for this action, or None (then the prior in config is used)."""
        if action == "accept":
            return 0.0
        return self._cell(action, slot_type).get("c")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.table, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, min_n: int = 20) -> "RepairOutcomeModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), min_n)
