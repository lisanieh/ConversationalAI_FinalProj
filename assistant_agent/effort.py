"""Effort model: how many seconds of the customer's time one turn costs.

    turn time = listening to the agent's question
              + answering: speaking (TTS audio length, or words / speaking rate),
                           spelling (units / spelling rate), or
                           typing (switch overhead + characters / typing rate)
              + a fixed response latency

When the TTS step has filled ``audio_seconds`` that measured length is used;
until then the length is estimated from the text. All rates are parameters, so
the team's own timings can replace the defaults (see ``calibrate``):

  python -m assistant_agent.effort --timings team_timing.csv --out models/effort.json

team_timing.csv has columns  kind,units,seconds  with kind in
speech (units = words), spell (units = letters/digits/symbols said),
typing (units = characters typed), listen (units = words heard). Each kind is
fitted as seconds = overhead + units / rate (least squares).

Defaults: conversational speech about 150 words/min; mobile typing 36 words/min
(Palin et al., 2019) = 3 characters/s; spelled units at about 0.6 s each.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


def _words(text: str | None) -> int:
    return len(re.findall(r"[A-Za-z0-9@']+", text or ""))


def _spell_units(text: str | None) -> int:
    return len(re.findall(r"[^\s.,;]+", text or ""))


@dataclass
class EffortModel:
    speech_wps: float = 2.5          # words per second when speaking
    listen_wps: float = 2.7          # words per second of the agent's TTS
    spell_ups: float = 1.6           # spelled units per second (letters with pauses)
    typing_cps: float = 3.0          # characters per second on a phone
    type_switch_s: float = 4.0       # pick up the phone, open the chat box, send
    turn_latency_s: float = 0.5      # thinking / response latency per turn
    source: str = "defaults"

    def listen_seconds(self, agent_text: str | None) -> float:
        return _words(agent_text) / self.listen_wps

    def answer_seconds(self, pkg) -> float:
        if pkg.mode == "typed":
            return self.type_switch_s + (pkg.typed_chars or 0) / self.typing_cps
        if pkg.audio_seconds:
            return float(pkg.audio_seconds)
        if pkg.path == "spell":
            return _spell_units(pkg.spoken_text) / self.spell_ups
        return _words(pkg.spoken_text) / self.speech_wps

    def turn_seconds(self, pkg, agent_text: str | None) -> float:
        """One exchange: the customer hears the agent's turn, then answers it."""
        return self.listen_seconds(agent_text) + self.turn_latency_s + self.answer_seconds(pkg)

    # ------------------------------------------------------------ saving
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path | None) -> "EffortModel":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))) if path else cls()


def _fit_line(pts: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares seconds = a + b * units. Returns (a, b); a >= 0."""
    n = len(pts)
    mx, my = sum(u for u, _ in pts) / n, sum(s for _, s in pts) / n
    sxx = sum((u - mx) ** 2 for u, _ in pts)
    b = sum((u - mx) * (s - my) for u, s in pts) / sxx if sxx > 0 else my / max(mx, 1e-9)
    a = my - b * mx
    if a < 0:                         # no negative overhead: refit through the origin
        a, b = 0.0, sum(u * s for u, s in pts) / max(sum(u * u for u, _ in pts), 1e-9)
    return a, b


def calibrate(rows: list[dict], base: EffortModel | None = None) -> EffortModel:
    m = EffortModel(**asdict(base or EffortModel()))
    by: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        by.setdefault(r["kind"].strip().lower(), []).append((float(r["units"]), float(r["seconds"])))
    if by.get("typing"):
        a, b = _fit_line(by["typing"])
        m.typing_cps, m.type_switch_s = 1 / b, a
    for kind, attr in (("speech", "speech_wps"), ("spell", "spell_ups"), ("listen", "listen_wps")):
        if by.get(kind):
            a, b = _fit_line(by[kind])
            setattr(m, attr, 1 / b)
            if kind == "speech":
                m.turn_latency_s = a
    m.source = "team timings: " + ", ".join(f"{k} n={len(v)}" for k, v in sorted(by.items()))
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit the effort model from the team's own timings.")
    ap.add_argument("--timings", required=True, help="CSV with columns kind,units,seconds")
    ap.add_argument("--out", default="models/effort.json")
    a = ap.parse_args(argv)
    with open(a.timings, newline="", encoding="utf-8") as f:
        m = calibrate(list(csv.DictReader(f)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.save(a.out)
    print(json.dumps(asdict(m), indent=2))


if __name__ == "__main__":
    main()
