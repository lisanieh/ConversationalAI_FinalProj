"""Robot customer (simulated user) for the U2R modality-switching testbed.

Pipeline (this package covers steps 1-5; TTS is the next stage):
  1 goal_builder  -> secret goal (answer key), can_type flag, voice
  2 brain         -> natural reply from a separate LLM (Qwen 2.5 7B via vLLM/Ollama)
  3 router        -> speak / spell / type, from the assistant's action label
  4 speech_text   -> text cleanup for TTS, letter-by-letter spelling
  5 typo          -> realistic typos for typed answers
  package         -> one JSON record per turn handed to TTS / Whisper / assistant
"""

from .brain import LLMBrain, ScriptedBrain
from .customer import RobotCustomer, run_dialogue
from .goal_builder import Goal, GoalBuilder
from .package import TurnPackage, read_jsonl, write_jsonl
from .router import AssistantTurn, Path, route
from .speech_text import spell_out, to_spoken
from .typo import add_typos

__all__ = ["Goal", "GoalBuilder", "LLMBrain", "ScriptedBrain", "RobotCustomer", "run_dialogue",
           "TurnPackage", "read_jsonl", "write_jsonl", "AssistantTurn", "Path", "route",
           "spell_out", "to_spoken", "add_typos"]
