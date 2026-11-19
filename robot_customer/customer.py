"""The robot customer: ties steps 1-5 together and emits one TurnPackage per turn.

    goal -> brain (LLM) -> router -> speak : text cleanup ------> spoken_text  (-> TTS)
                                  -> spell : letter-by-letter --> spoken_text  (-> TTS)
                                  -> type  : typo script -------> typed_text   (-> assistant)

TTS is out of scope here. If a ``tts`` callable is given it is called with
``(spoken_text, voice, out_path)`` and must return the audio length in seconds;
otherwise ``audio_file`` holds the suggested path and ``audio_seconds`` stays
None for the TTS step to fill.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

from . import prompts
from .brain import Brain
from .goal_builder import Goal
from .package import TurnPackage, write_jsonl
from .router import AssistantTurn, Path as RoutePath, route
from .speech_text import spell_out, to_spoken
from .typo import add_typos

TTSFn = Callable[[str, str, Path], float]


class RobotCustomer:
    def __init__(self, goal: Goal, brain: Brain, typo_rate: float = 0.02,
                 out_dir: str | Path | None = None, tts: TTSFn | None = None):
        self.goal, self.brain = goal, brain
        self.typo_rate, self.tts = typo_rate, tts
        self.out_dir = Path(out_dir) / goal.dialogue_id if out_dir else None
        self.history: list[dict] = []      # {"speaker", "text", ...} both sides
        self.packages: list[TurnPackage] = []
        self.done = False

    # ------------------------------------------------------------------ turns
    def respond(self, assistant: AssistantTurn | dict) -> TurnPackage:
        if isinstance(assistant, dict):
            assistant = AssistantTurn.from_dict(assistant)
        self.history.append({"speaker": "assistant", "text": assistant.text,
                             "action": assistant.action, "slot": assistant.slot})

        turn = len(self.packages)
        path = route(assistant, self.goal)
        slot = assistant.slot if assistant.slot in self.goal.slots else None
        value = self.goal.slots.get(slot) if slot else None

        pkg = TurnPackage(
            dialogue_id=self.goal.dialogue_id, turn=turn, mode="speech",
            path=path.value, slot=slot, assistant_action=assistant.action,
            true_text="", true_value=value, voice=self.goal.voice,
            seed=self.goal.seed, prompt_version=prompts.PROMPT_VERSION,
        )

        if path is RoutePath.SPEAK:
            self._tell_brain(assistant)
            raw = self.brain.reply(self.goal, self._brain_history())
            pkg.llm_raw = getattr(self.brain, "last_raw", None)
            said_done = prompts.DONE_TOKEN in raw
            # [DONE] only counts once the assistant has stopped asking: not on
            # the opening turn, and not while its label targets a slot or asks
            # to confirm/spell/type. LLMs sometimes say [DONE] far too early.
            asking = (turn == 0 or assistant.slot is not None
                      or assistant.action not in ("normal", "accept"))
            pkg.done = said_done and not asking
            pkg.done_ignored = said_done and asking
            text = raw.replace(prompts.DONE_TOKEN, "").strip()
            pkg.true_text = text
            pkg.spoken_text = to_spoken(text)
        elif path is RoutePath.CANNOT_TYPE:
            pkg.true_text = prompts.CANNOT_TYPE_LINE
            pkg.spoken_text = to_spoken(prompts.CANNOT_TYPE_LINE)
        elif path is RoutePath.SPELL:
            pkg.true_text = value
            pkg.spoken_text = spell_out(value, slot)
        elif path is RoutePath.TYPE:
            # Separate RNG per (dialogue seed, turn) so reruns give identical typos.
            rng = random.Random(self.goal.seed * 1000 + turn)
            res = add_typos(value, self.typo_rate, rng)
            pkg.mode = "typed"
            pkg.true_text = value
            pkg.typed_text = res.typed
            pkg.typo_edits = res.edits
            pkg.typed_chars = len(res.typed)
            pkg.voice = None

        if pkg.mode == "speech":
            pkg.audio_file = self._audio_path(turn)
            if self.tts is not None:
                pkg.audio_seconds = float(self.tts(pkg.spoken_text, self.goal.voice, Path(pkg.audio_file)))

        self.history.append({"speaker": "customer", "text": pkg.true_text, "path": pkg.path,
                             "mode": pkg.mode, "typed_text": pkg.typed_text})
        self.packages.append(pkg)
        self.done = pkg.done
        return pkg

    # ---------------------------------------------------------------- helpers
    def _tell_brain(self, assistant: AssistantTurn) -> None:
        """Model-free brains (ScriptedBrain) read the router label directly."""
        if hasattr(self.brain, "asked_slot"):
            self.brain.asked_slot = assistant.slot
            self.brain.asked_action = assistant.action

    def _brain_history(self) -> list[dict]:
        return [{"speaker": h["speaker"], "text": h["text"]} for h in self.history]

    def _audio_path(self, turn: int) -> str:
        base = self.out_dir or Path(self.goal.dialogue_id)
        return str(base / "audio" / f"turn_{turn:02d}.wav")

    # ------------------------------------------------------------------ saving
    def save(self, final_state: dict | None = None) -> Path | None:
        if not self.out_dir:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "goal.json").write_text(json.dumps(self.goal.to_dict(), indent=2), encoding="utf-8")
        write_jsonl(self.packages, self.out_dir / "turns.jsonl")
        (self.out_dir / "transcript.json").write_text(json.dumps(self.history, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
        if final_state is not None:
            (self.out_dir / "final_state.json").write_text(json.dumps(final_state, indent=2), encoding="utf-8")
        self.save_prompts(self.out_dir)
        return self.out_dir

    def save_prompts(self, folder: Path) -> None:
        """Write every prompt word for word (required for the paper)."""
        system = prompts.CUSTOMER_SYSTEM_PROMPT.format(domain=self.goal.domain, goal=self.goal.describe())
        text = (f"PROMPT_VERSION: {prompts.PROMPT_VERSION}\n\n"
                f"=== CUSTOMER_SYSTEM_PROMPT (template) ===\n{prompts.CUSTOMER_SYSTEM_PROMPT}\n"
                f"=== CUSTOMER_SYSTEM_PROMPT (filled for this dialogue) ===\n{system}\n"
                f"=== CANNOT_TYPE_LINE ===\n{prompts.CANNOT_TYPE_LINE}\n")
        (folder / "prompts.txt").write_text(text, encoding="utf-8")


def run_dialogue(customer: RobotCustomer, assistant_step: Callable[[TurnPackage | None], dict],
                 max_turns: int = 20) -> dict:
    """Drive a dialogue until the customer says [DONE] or ``max_turns`` is hit.

    ``assistant_step(prev_package)`` returns the assistant's next turn as
    ``{"text", "action", "slot"}``; it gets ``None`` on the first call. In the
    full system it wraps noise -> Whisper -> assistant; in text-only tests it
    reads ``true_text`` / ``typed_text`` directly.
    """
    pkg = None
    for _ in range(max_turns):
        pkg = customer.respond(assistant_step(pkg))
        if customer.done:
            break
    return {"done": customer.done, "turns": len(customer.packages),
            "hit_turn_limit": not customer.done}
