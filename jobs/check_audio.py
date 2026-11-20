"""Audio smoke test: Kokoro -> (MUSAN) -> Whisper on a few customer-style turns.

  python jobs/check_audio.py [--musan datasets/musan] [--whisper-model small.en] [--out audio_check]

Prints what Whisper heard, the features the estimator would get, and what the
agent extracts / decodes. Writes the WAVs to --out so you can listen to them.
Exit code 1 if any step fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant_agent.audio import KokoroTTS, NoiseMixer, WhisperASR, AudioChannel  # noqa: E402
from assistant_agent.channel import slot_features  # noqa: E402
from assistant_agent.extract import RuleExtractor  # noqa: E402
from assistant_agent.repair import decode_spelled  # noqa: E402
from assistant_agent.values import values_match  # noqa: E402
from robot_customer.package import TurnPackage  # noqa: E402
from robot_customer.speech_text import spell_out, to_spoken  # noqa: E402

CASES = [  # (slot, written text the customer says, goal value, path, voice)
    ("name", "It's Varun Kamath.", "Varun Kamath", "speak", "af_heart"),
    ("name", None, "Varun Kamath", "spell", "bf_emma"),
    ("email", "My email is varun.k@gmail.com.", "varun.k@gmail.com", "speak", "am_michael"),
    ("email", None, "varun.k@gmail.com", "spell", "bm_george"),
    ("phone", "My number is 404 555 1234.", "404 555 1234", "speak", "af_bella"),
    ("time", "At 19:30, please.", "19:30", "speak", "bf_isabella"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musan", default=None, help="MUSAN root; also tests noise:10 and music:5")
    ap.add_argument("--whisper-model", default="small.en")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--out", default="audio_check")
    a = ap.parse_args()

    t0 = time.time()
    tts = KokoroTTS(save_dir=a.out)
    asr = WhisperASR(a.whisper_model, K=a.K)
    mixer = NoiseMixer(a.musan) if a.musan else None
    print(f"[load] Kokoro + Whisper {a.whisper_model} on {tts.device}"
          f"{' + MUSAN ' + str({k: len(v) for k, v in mixer.pools.items()}) if mixer else ''}"
          f" in {time.time() - t0:.0f}s\n")
    conditions = [("clean", None)] + ([("noise", 10.0), ("music", 5.0)] if mixer else [])
    rules, failures = RuleExtractor(), 0

    for i, (slot, text, value, path, voice) in enumerate(CASES):
        spoken = spell_out(value, slot) if path == "spell" else to_spoken(text)
        pkg = TurnPackage(dialogue_id="check", turn=i, mode="speech", path=path, slot=slot,
                          assistant_action="spell" if path == "spell" else "normal",
                          true_text=value if path == "spell" else text, true_value=value,
                          spoken_text=spoken, voice=voice, audio_file=f"check/turn_{i:02d}.wav")
        t = time.time()
        pkg.audio_seconds = tts(spoken, voice, Path(pkg.audio_file))
        print(f"--- {slot} / {path} / {voice}: {pkg.audio_seconds:.1f}s audio, TTS {time.time() - t:.1f}s")
        print(f"    TTS input : {spoken}")
        for cond, snr in conditions:
            ch = AudioChannel(tts, asr, mixer, cond, snr, seed=i, save_dir=a.out)
            t = time.time()
            h = ch(pkg)
            if path == "spell":
                got = decode_spelled(h.text, slot)
            else:
                got = rules.extract(h.text, asked_slot=slot)[1].get(slot)
            ok = values_match(slot, got, value)
            feats = slot_features(h, got or "") or {}
            tag = f"{cond}" + (f":{snr:g}" if snr is not None else "")
            print(f"    [{tag:9s}] heard: {h.text!r}  ({time.time() - t:.1f}s)")
            print(f"               value: {got!r} -> {'OK' if ok else 'WRONG'} | "
                  f"min_logprob={feats.get('min_logprob', float('nan')):.2f} "
                  f"agreement={feats.get('agreement')} snr_est={h.snr_est:.1f} "
                  f"no_speech={h.no_speech_prob:.2f}")
            if cond == "clean" and not ok:
                failures += 1
        print()

    print(f"WAVs in {Path(a.out).resolve()}  (listen to a few: names, spelled letters, noise levels)")
    if failures:
        print(f"CHECK: {failures} clean-speech case(s) were not recovered. Read the transcripts above: "
              "if names are missing from the TTS, Kokoro's espeak fallback for unknown words is not "
              "installed (pip install 'misaki[en]' espeakng-loader).")
    print(f"total {time.time() - t0:.0f}s")
    sys.exit(1 if failures > 2 else 0)


if __name__ == "__main__":
    main()
