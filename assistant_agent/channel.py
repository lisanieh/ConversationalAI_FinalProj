"""What the agent hears from one customer turn, and the Whisper features of a slot.

A *channel* is any callable ``channel(pkg: TurnPackage) -> Heard``. The agent
only ever sees a ``Heard``, never the package's answer-key fields.

  TextChannel     perfect recognition, no ASR features. For reading dialogues.
  MockASRChannel  fake recognizer: more errors at low SNR, word confidences
                  loosely tied to errors. For testing fit -> train -> eval on a
                  laptop. NEVER report its numbers.
  ASRChannel      wraps the real audio step (TTS file -> noise -> Whisper). Give
                  it a ``transcribe(pkg) -> dict`` with the keys below.

ASR dict / Heard fields (the audio part fills these):
  text               1-best transcript (temperature 0)
  words              [(word, probability), ...] for the 1-best transcript
  samples            K transcripts sampled at temperature > 0 (for agreement)
  no_speech_prob, compression_ratio   Whisper's own segment statistics
  snr_est            blind SNR estimate (never the true SNR)
"""

from __future__ import annotations

import math
import random
import re
import string
from dataclasses import dataclass, field
from typing import Callable

from .values import normalize


@dataclass
class Heard:
    text: str
    mode: str = "speech"                  # "speech" | "typed"
    turn: int = 0                          # customer turn index (TurnPackage.turn)
    words: list[tuple[str, float]] | None = None
    samples: list[str] = field(default_factory=list)
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    snr_est: float | None = None

    @property
    def has_asr(self) -> bool:
        return self.mode == "speech" and bool(self.words)


def _speech_source(pkg) -> str:
    """What the customer said aloud, in the form Whisper would roughly write it.
    Spelled turns keep the letter-by-letter form; other turns use the written text."""
    return pkg.spoken_text if pkg.path == "spell" else pkg.true_text


def _typed(pkg) -> Heard:
    return Heard(text=pkg.typed_text or "", mode="typed", turn=pkg.turn)


class TextChannel:
    def __call__(self, pkg) -> Heard:
        if pkg.mode == "typed":
            return _typed(pkg)
        return Heard(text=_speech_source(pkg), turn=pkg.turn)


class ASRChannel:
    def __init__(self, transcribe: Callable[[object], dict]):
        self.transcribe = transcribe

    def __call__(self, pkg) -> Heard:
        if pkg.mode == "typed":
            return _typed(pkg)
        a = self.transcribe(pkg)
        return Heard(text=a["text"], turn=pkg.turn, words=[tuple(w) for w in a.get("words") or []],
                     samples=list(a.get("samples") or []), no_speech_prob=a.get("no_speech_prob"),
                     compression_ratio=a.get("compression_ratio"), snr_est=a.get("snr_est"))


# ---------------------------------------------------------------- mock ASR
_KEEP = {"the", "is", "it's", "no", "yes", "that's", "right", "sorry", "i", "i'm", "can't",
         "type", "now", "driving", "thanks", "thank", "you", "hi", "a", "my", "and", "please",
         "dot", "at", "for", "in", "on", "people", "nights", "food"}


class MockASRChannel:
    """Plumbing test only. Conditions: clean, noise, music, sidetalk (with an SNR)."""

    def __init__(self, condition: str = "clean", snr_db: float | None = None, seed: int = 0, K: int = 5):
        self.condition, self.snr_db, self.K = condition, snr_db, K
        self.rng = random.Random(f"{seed}|{condition}|{snr_db}")

    def _rates(self) -> tuple[float, float]:
        if self.condition == "clean" or self.snr_db is None:
            return 0.02, 0.0
        p = 0.02 + 0.6 / (1 + math.exp((self.snr_db - 2) / 3))
        extra = {"noise": (0.0, 0.0), "music": (0.02, 0.05), "sidetalk": (0.05, 0.08)}.get(self.condition, (0, 0))
        return min(0.9, p + extra[0]), extra[1]

    def _corrupt(self, text: str, p_err: float, p_ins: float):
        rng, out, words, n_ins = self.rng, [], [], 0
        for tok in text.split():
            core = re.sub(r"[^a-z0-9'@.]", "", tok.lower()).strip(".")
            if not core or core in _KEEP:
                out.append(tok)
                words.append((tok, 1 - rng.betavariate(1, 30)))
                continue
            pw = min(0.9, p_err * (0.7 if core.isdigit() else 0.5 + min(len(core), 12) / 6))
            r = rng.random()
            if r < pw * 0.7:                                        # substitution
                if len(core) > 3 and rng.random() < 0.4:
                    k = rng.randrange(1, len(core))
                    new = core[:k] + " " + core[k:]
                else:
                    k = rng.randrange(len(core))
                    new = core[:k] + rng.choice(string.ascii_lowercase) + core[k + 1:]
                out.append(new)
                words += [(w, rng.uniform(0.15, 0.85)) for w in new.split()]
            elif r < pw:                                            # deletion
                continue
            else:
                out.append(tok)
                conf = 1 - rng.betavariate(1.2, 12) - 0.3 * p_err * rng.random()
                words.append((tok, min(1.0, max(0.01, conf))))
            if rng.random() < p_ins:                                # insertion
                w = rng.choice(["yeah", "so", "la", "oh", "and"])
                out.append(w)
                words.append((w, rng.uniform(0.2, 0.8)))
                n_ins += 1
        return " ".join(out), words, n_ins

    def __call__(self, pkg) -> Heard:
        if pkg.mode == "typed":
            return _typed(pkg)
        text = _speech_source(pkg)
        p_err, p_ins = self._rates()
        hyp, words, n_ins = self._corrupt(text, p_err, p_ins)
        samples = [self._corrupt(text, p_err, p_ins)[0] for _ in range(self.K)]
        snr = self.snr_db if self.snr_db is not None else 35.0
        return Heard(text=hyp, turn=pkg.turn, words=words, samples=samples,
                     no_speech_prob=self.rng.uniform(0, 0.05) + 0.1 * n_ins,
                     compression_ratio=1.0 + self.rng.uniform(0, 0.4) + 0.3 * n_ins,
                     snr_est=snr + self.rng.gauss(0, 2))


# ------------------------------------------------------------ slot features
def _find_span(tokens: list[str], target: list[str]) -> tuple[int, int] | None:
    n = len(target)
    for i in range(len(tokens) - n + 1):
        if n and tokens[i:i + n] == target:
            return i, i + n
    return None


def slot_features(heard: Heard, raw_value: str) -> dict | None:
    """Whisper features for one extracted value (None for typed or text-only turns).

    Confidences are taken over the words of the value when they can be found in
    the transcript, otherwise over the whole utterance. ``agreement`` is the
    share of the K sampled transcripts that also contain the value.
    """
    if not heard.has_asr:
        return None
    tok_lists = [normalize(w).split() for w, _ in heard.words]
    flat, probs = [], []
    for toks, (_, p) in zip(tok_lists, heard.words):
        for t in toks:
            flat.append(t)
            probs.append(p)
    target = normalize(raw_value).split()
    span = _find_span(flat, target)
    sel = probs[span[0]:span[1]] if span else (probs or [p for _, p in heard.words])
    logp = [math.log(min(1.0, max(1e-6, p))) for p in sel]
    hyp_n = normalize(raw_value)
    agreement = (sum(hyp_n in normalize(s) for s in heard.samples) / len(heard.samples)
                 if heard.samples and hyp_n else None)
    return {"min_logprob": min(logp), "mean_logprob": sum(logp) / len(logp),
            "agreement": agreement, "no_speech_prob": heard.no_speech_prob,
            "compression_ratio": heard.compression_ratio, "snr_est": heard.snr_est,
            "value_found": span is not None}
