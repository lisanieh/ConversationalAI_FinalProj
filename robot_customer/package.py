"""The hand-off "package": one JSON record per customer turn.

This is the contract with the rest of the system:
  * TTS teammate      reads ``spoken_text`` + ``voice`` for speech turns, writes
                      ``audio_file`` and fills ``audio_seconds``.
  * Noise / Whisper   reads ``audio_file`` when ``mode == "speech"``.
  * Assistant         gets Whisper's transcript (speech) or ``typed_text`` (typed).
  * Estimator/scoring compares against ``true_text`` / ``true_value``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TurnPackage:
    dialogue_id: str
    turn: int                         # customer turn index, starting at 0
    mode: str                         # "speech" | "typed"
    path: str                         # speak | spell | type | cannot_type
    slot: str | None                  # slot this turn answers (from the assistant label)
    assistant_action: str             # label the assistant sent

    true_text: str                    # what the customer meant, written form (answer key)
    true_value: str | None            # exact goal value for the slot, if any

    spoken_text: str | None = None    # TTS input (speech turns only)
    typed_text: str | None = None     # what reaches the assistant (typed turns only)
    typo_edits: list[dict] = field(default_factory=list)

    voice: str | None = None
    seed: int | None = None           # goal seed
    run_seed: int = 0                 # repetition of the same goal
    audio_file: str | None = None     # suggested path; written by the TTS step
    audio_seconds: float | None = None  # filled by the TTS step
    typed_chars: int | None = None    # effort proxy for typed turns

    done: bool = False                # customer said [DONE] and it was accepted
    done_ignored: bool = False        # customer said [DONE] while still being asked something
    llm_raw: str | None = None        # unprocessed LLM output (speak path), for debugging
    prompt_version: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def write_jsonl(packages: list[TurnPackage], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in packages:
            f.write(p.to_json() + "\n")


def read_jsonl(path: str | Path) -> list[TurnPackage]:
    with Path(path).open(encoding="utf-8") as f:
        return [TurnPackage(**json.loads(line)) for line in f if line.strip()]
