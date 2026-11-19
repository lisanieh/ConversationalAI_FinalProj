"""Repair outcome model, per (action, slot type):

  r_a = P(the repair fixes the value | the first hypothesis was wrong)
  e_a = P(the repair breaks the value | the first hypothesis was right)

Estimated from FIT-phase runs with forced policies (always confirm / spell /
type). Beta(1,1) smoothing keeps small cells away from 0 and 1, and a cell with
fewer than ``min_n`` examples on either side falls back to the pooled rate.

A forced "type" run includes the customers who cannot type (the agent falls
back to spelling), so r/e of TYPE already price in that risk.
"""

from __future__ import annotations

import json
from pathlib import Path

FORCED = ("confirm", "spell", "type")


class RepairOutcomeModel:
    def __init__(self, table: dict | None = None, min_n: int = 20):
        self.table = table or {}       # "action|slot_type" and "action|*" -> {r, e, n_wrong, n_right}
        self.min_n = min_n

    @staticmethod
    def _rates(rows: list[dict]) -> dict:
        wrong = [r for r in rows if not r["first_correct"]]
        right = [r for r in rows if r["first_correct"]]
        fixed = sum(bool(r["committed_correct"]) for r in wrong)
        broken = sum(not r["committed_correct"] for r in right)
        return {"r": (fixed + 1) / (len(wrong) + 2), "e": (broken + 1) / (len(right) + 2),
                "n_wrong": len(wrong), "n_right": len(right)}

    def fit(self, attempts: list[dict]) -> "RepairOutcomeModel":
        rows = [a for a in attempts if a.get("policy") in FORCED and a.get("mode") == "speech"
                and not a.get("missed") and a.get("true_value") is not None
                and a.get("committed_value") is not None]
        for action in FORCED:
            d = [r for r in rows if r["policy"] == action]
            if not d:
                continue
            self.table[f"{action}|*"] = self._rates(d)
            for st in sorted({r["slot_type"] for r in d}):
                self.table[f"{action}|{st}"] = self._rates([r for r in d if r["slot_type"] == st])
        return self

    def get(self, action: str, slot_type: str) -> tuple[float, float]:
        if action == "accept":
            return 0.0, 0.0
        cell = self.table.get(f"{action}|{slot_type}")
        if not cell or min(cell["n_wrong"], cell["n_right"]) < self.min_n:
            cell = self.table.get(f"{action}|*")
        if not cell:
            raise KeyError(f"no outcome data for {action!r}; run the fit phase first")
        return cell["r"], cell["e"]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.table, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, min_n: int = 20) -> "RepairOutcomeModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), min_n)
