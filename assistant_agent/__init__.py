"""Assistant agent for the U2R modality-switching testbed (Marina's part).

  extract    LLM (Llama) dialogue state tracking, values copied as heard
  estimator  p_hat = P(heard value is wrong) from Whisper features, calibrated
  outcomes   r_a / e_a per (repair action, slot type), from forced-policy runs
  policies   U2R (Eq. 1, closed-form p*) and the baselines
  agent      U2RAgent.step(heard) -> {"text", "action", "slot"} for the robot customer
  channel    what the agent hears: text-only, mock ASR, or the real Whisper step
  scoring    attempts / dialogue rows against the goal
"""

from .agent import U2RAgent
from .channel import ASRChannel, Heard, MockASRChannel, TextChannel, slot_features
from .estimator import SlotErrorEstimator
from .extract import LLMExtractor, RuleExtractor
from .llm import ChatLLM
from .outcomes import RepairOutcomeModel
from .policies import U2R, make_policy
from .scoring import score_dialogue

__all__ = ["U2RAgent", "ASRChannel", "Heard", "MockASRChannel", "TextChannel", "slot_features",
           "SlotErrorEstimator", "LLMExtractor", "RuleExtractor", "ChatLLM",
           "RepairOutcomeModel", "U2R", "make_policy", "score_dialogue"]
