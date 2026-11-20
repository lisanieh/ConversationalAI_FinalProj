"""Audio connector: robot customer turn -> Kokoro TTS -> MUSAN noise -> Whisper -> agent.

This links the live dialogue to the audio scripts in script/ (Lisa's part):

  KokoroTTS     the robot customer's ``tts`` hook. Speaks ``spoken_text`` with the
                goal's voice, resamples 24 kHz -> 16 kHz (as generate_tts.py),
                returns the length in seconds (fills ``audio_seconds``, which the
                effort model then uses). The pipeline language follows the voice
                (af_/am_ -> American 'a', bf_/bm_ -> British 'b').
  NoiseMixer    MUSAN pools and SNR mixing from script/mix_musan.py
                (``collect_musan_files``: vocal music only, ``mix_with_snr``:
                active-speech RMS, ``estimate_blind_snr``), reused as is.
  WhisperASR    greedy transcript with per-word probabilities
                (``transcribe(word_timestamps=True)``), segment statistics, and K
                transcripts sampled at temperature 0.7 for agreement.
  AudioChannel  the channel the agent hears through: ``channel(pkg) -> Heard``.

Conditions: clean | noise | music | sidetalk (= MUSAN speech, "side_talk" in
mix_musan.py), each with an SNR. Typed turns never touch audio.

Save-point replays deep-copy the channel and the customer: the models and the
noise cache are shared (never copied), the random state is copied so every
branch hears exactly the same noise.

Requires (GPU node): torch, kokoro, soundfile, librosa, openai-whisper; MUSAN
in datasets/musan/ (noise/, music/, speech/).
"""

from __future__ import annotations

import copy
import importlib.util
import random
import sys
from pathlib import Path

from .channel import Heard

ROOT = Path(__file__).resolve().parents[1]
CONDITION_POOL = {"noise": "noise", "music": "music", "sidetalk": "side_talk", "side_talk": "side_talk"}
SAMPLE_RATE = 16000


def _load_script(name: str):
    """Import script/<name>.py by path (script/ is not a package)."""
    key = f"script_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "script" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def _device(device: str | None) -> str:
    if device:
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class _Shared:
    """Mixin: deepcopy returns the same object (models must not be copied)."""

    def __deepcopy__(self, memo):
        return self


# ------------------------------------------------------------------- TTS
class KokoroTTS(_Shared):
    def __init__(self, device: str | None = None, speed: float = 1.0, save_dir: str | Path | None = None):
        self.device, self.speed = _device(device), speed
        self.save_dir = Path(save_dir) if save_dir else None
        self._pipes: dict[str, object] = {}
        self.cache: dict[str, object] = {}         # audio_file path -> 16 kHz float32 array

    def _pipe(self, voice: str):
        lang = "b" if voice[:1] == "b" else "a"
        if lang not in self._pipes:
            from kokoro import KPipeline
            self._pipes[lang] = KPipeline(lang_code=lang, device=self.device)
        return self._pipes[lang]

    def synth(self, text: str, voice: str):
        import librosa
        import numpy as np
        chunks = []
        for _, _, audio in self._pipe(voice)(text, voice=voice, speed=self.speed):
            if audio is not None:
                chunks.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
        if not chunks:
            raise ValueError(f"Kokoro produced no audio for {text!r}")
        return librosa.resample(np.concatenate(chunks).astype("float32"), orig_sr=24000, target_sr=SAMPLE_RATE)

    def __call__(self, text: str, voice: str, out_path: Path) -> float:
        """robot_customer hook: (spoken_text, voice, audio_file) -> seconds."""
        audio = self.synth(text, voice)
        if len(self.cache) >= 256:                    # keep recent turns only (replays reread them)
            self.cache.pop(next(iter(self.cache)))
        self.cache[str(out_path)] = audio
        if self.save_dir is not None:
            import soundfile as sf
            p = self.save_dir / Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, audio, SAMPLE_RATE, subtype="PCM_16")
        return len(audio) / SAMPLE_RATE


# ----------------------------------------------------------------- noise
class NoiseMixer(_Shared):
    def __init__(self, musan_root: str | Path = ROOT / "datasets" / "musan", max_cached: int = 64):
        self.mm = _load_script("mix_musan")
        self.root = Path(musan_root)
        if not self.root.exists():
            raise FileNotFoundError(f"MUSAN not found at {self.root} (noise/, music/, speech/); "
                                    "on PACE-ICE download https://www.openslr.org/resources/17/musan.tar.gz")
        self.pools = self.mm.collect_musan_files(str(self.root))
        self._cache: dict[str, object] = {}
        self.max_cached = max_cached

    def _noise(self, path: str):
        if path not in self._cache:
            if len(self._cache) >= self.max_cached:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = self.mm.load_audio_16k(path)
        return self._cache[path]

    def mix(self, speech, condition: str, snr_db: float | None, rng: random.Random):
        """Returns (mixed audio, MUSAN source or None)."""
        if condition == "clean" or snr_db is None:
            return speech, None
        pool = self.pools[CONDITION_POOL[condition]]
        if not pool:
            raise ValueError(f"no MUSAN files for condition {condition!r} under {self.root}")
        src = rng.choice(pool)
        return self.mm.mix_with_snr(speech, self._noise(src), float(snr_db), rng), src

    def blind_snr(self, audio) -> float:
        return self.mm.estimate_blind_snr(audio)


# --------------------------------------------------------------- Whisper
class WhisperASR(_Shared):
    def __init__(self, model: str = "small.en", device: str | None = None, K: int = 5,
                 sample_temperature: float = 0.7):
        import whisper
        self.whisper, self.device = whisper, _device(device)
        self.model = whisper.load_model(model, device=self.device)
        self.K, self.T = K, sample_temperature

    def recognize(self, audio, seed: int = 0) -> dict:
        import numpy as np
        import torch
        w, fp16 = self.whisper, self.device == "cuda"
        audio = np.asarray(audio, dtype=np.float32)
        res = self.model.transcribe(audio, language="en", temperature=0.0, word_timestamps=True,
                                    condition_on_previous_text=False, fp16=fp16)
        segs = res.get("segments") or []
        words = [(wd["word"].strip(), float(wd["probability"])) for s in segs for wd in s.get("words", [])
                 if wd["word"].strip()]
        torch.manual_seed(seed)                     # reproducible sampled transcripts
        mel = w.log_mel_spectrogram(w.pad_or_trim(audio), n_mels=self.model.dims.n_mels).to(self.model.device)
        opts = w.DecodingOptions(language="en", temperature=self.T, without_timestamps=True, fp16=fp16)
        samples = [w.decode(self.model, mel, opts).text.strip() for _ in range(self.K)]
        return {"text": res["text"].strip(), "words": words, "samples": samples,
                "no_speech_prob": max((s["no_speech_prob"] for s in segs), default=0.0),
                "compression_ratio": max((s["compression_ratio"] for s in segs), default=1.0),
                "avg_logprob": min((s["avg_logprob"] for s in segs), default=0.0)}


# --------------------------------------------------------------- channel
class AudioChannel:
    """channel(pkg) -> Heard, for speech turns: cached TTS audio -> noise -> Whisper."""

    def __init__(self, tts: KokoroTTS, asr: WhisperASR, mixer: NoiseMixer | None,
                 condition: str = "clean", snr_db: float | None = None, seed: int = 0,
                 save_dir: str | Path | None = None):
        if condition != "clean" and mixer is None:
            raise ValueError(f"condition {condition!r} needs a NoiseMixer (MUSAN)")
        self.tts, self.asr, self.mixer = tts, asr, mixer
        self.condition, self.snr_db = condition, snr_db
        self.rng = random.Random(f"{seed}|{condition}|{snr_db}")
        self.seed = seed
        self.save_dir = Path(save_dir) if save_dir else None
        self.last: dict = {}                         # details of the last turn, for logging

    def __deepcopy__(self, memo):                     # share models, copy the random state
        new = copy.copy(self)
        new.rng = copy.deepcopy(self.rng, memo)
        new.last = dict(self.last)
        return new

    def __call__(self, pkg) -> Heard:
        if pkg.mode == "typed":
            return Heard(text=pkg.typed_text or "", mode="typed", turn=pkg.turn)
        speech = self.tts.cache.get(str(pkg.audio_file))
        if speech is None:                            # TTS hook not attached: synthesize here
            speech = self.tts.synth(pkg.spoken_text, pkg.voice)
            pkg.audio_seconds = len(speech) / SAMPLE_RATE
        mixed, src = self.mixer.mix(speech, self.condition, self.snr_db, self.rng) if self.mixer \
            else (speech, None)
        if self.save_dir is not None:
            import soundfile as sf
            p = self.save_dir / f"{self.condition}_{self.snr_db:g}" if self.snr_db is not None \
                else self.save_dir / self.condition
            p = p / Path(pkg.audio_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, mixed, SAMPLE_RATE, subtype="PCM_16")
        snr_est = self.mixer.blind_snr(mixed) if self.mixer else _load_script("mix_musan").estimate_blind_snr(mixed)
        a = self.asr.recognize(mixed, seed=self.rng.randrange(2**31))
        self.last = {"musan_source": src, "snr_est": snr_est, "avg_logprob": a["avg_logprob"]}
        return Heard(text=a["text"], turn=pkg.turn, words=a["words"], samples=a["samples"],
                     no_speech_prob=a["no_speech_prob"], compression_ratio=a["compression_ratio"],
                     snr_est=snr_est)
