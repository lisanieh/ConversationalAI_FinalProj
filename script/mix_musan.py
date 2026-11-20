import os
import glob
import json
import random
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa


def load_audio_16k(path: str) -> np.ndarray:
    """read audio file and convert to 16kHz mono float32"""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


def compute_active_rms(audio: np.ndarray, top_db: float = 35.0) -> float:
    """exclude silent segments at the beginning and end, and compute RMS energy of the active speech region"""
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return float(np.sqrt(np.mean(audio ** 2)) + 1e-8)
    active_samples = np.concatenate([audio[start:end] for start, end in intervals])
    return float(np.sqrt(np.mean(active_samples ** 2)) + 1e-8)


def estimate_blind_snr(
    mixed_audio: np.ndarray, frame_length: int = 400, hop_length: int = 160
) -> float:
    """
    Estimated SNR：
    calculate the 85th percentile and 15th percentile ratio for Slot Error Estimator
    """
    frames = librosa.util.frame(
        mixed_audio, frame_length=frame_length, hop_length=hop_length
    )
    frame_energy = np.mean(frames ** 2, axis=0) + 1e-10
    noise_power = np.percentile(frame_energy, 15)
    signal_plus_noise_power = np.percentile(frame_energy, 85)

    snr_linear = max((signal_plus_noise_power - noise_power) / noise_power, 1e-3)
    estimated_snr = 10.0 * np.log10(snr_linear)
    return float(np.clip(estimated_snr, -10.0, 40.0))


def collect_musan_files(musan_root: str):
    """scan MUSAN directory and filter vocal music files based on ANNOTATIONS"""
    noise_files = glob.glob(
        os.path.join(musan_root, "noise", "**", "*.wav"), recursive=True
    )
    speech_files = glob.glob(
        os.path.join(musan_root, "speech", "**", "*.wav"), recursive=True
    )

    vocal_music_files = []
    music_dir = os.path.join(musan_root, "music")

    for ann_file in glob.glob(
        os.path.join(music_dir, "**", "ANNOTATIONS"), recursive=True
    ):
        subdir = os.path.dirname(ann_file)
        with open(ann_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[2].upper() == "Y":
                    wav_candidate = os.path.join(subdir, f"{parts[0]}.wav")
                    if os.path.exists(wav_candidate):
                        vocal_music_files.append(wav_candidate)

    if not vocal_music_files:
        print("[warning] can not find vocals in ANNOTATIONS, using all music/*.wav")
        vocal_music_files = glob.glob(
            os.path.join(music_dir, "**", "*.wav"), recursive=True
        )

    return {
        "noise": noise_files,
        "music": vocal_music_files,
        "side_talk": speech_files,
    }


def mix_with_snr(
    speech: np.ndarray, noise: np.ndarray, target_snr_db: float, rng: random.Random
) -> np.ndarray:
    """mix speech and background noise according to the specified target_snr_db, and prevent clipping"""
    speech_len = len(speech)
    noise_len = len(noise)

    if noise_len < speech_len:
        repeats = (speech_len // noise_len) + 1
        noise_aligned = np.tile(noise, repeats)[:speech_len]
    else:
        start_idx = rng.randint(0, noise_len - speech_len)
        noise_aligned = noise[start_idx : start_idx + speech_len]

    rms_speech = compute_active_rms(speech)
    rms_noise = float(np.sqrt(np.mean(noise_aligned ** 2)) + 1e-8)

    alpha = rms_speech / (rms_noise * (10.0 ** (target_snr_db / 20.0)))
    mixed = speech + alpha * noise_aligned

    max_amp = np.max(np.abs(mixed))
    if max_amp > 0.99:
        mixed = mixed * (0.99 / max_amp)

    return mixed.astype(np.float32)


def run_mixing_from_turns(
    turns_jsonl_path: str,
    musan_root: str,
    output_base_dir: str,
    output_manifest_path: str,
    seed: int = 42,
):
    rng = random.Random(seed)
    musan_pools = collect_musan_files(musan_root)
    print(
        f"MUSAN loading completed -> Noise: {len(musan_pools['noise'])} files, "
        f"Vocal Music: {len(musan_pools['music'])} files, Side Talk: {len(musan_pools['side_talk'])} files"
    )

    # align Milestone Table 1 and Held-out protocol
    conditions = [
        ("train_calib", "clean", [None]),
        ("train_calib", "noise", [20.0, 10.0, 5.0, 0.0]),
        ("heldout_test", "music", [10.0, 5.0]),
        ("heldout_test", "side_talk", [10.0, 5.0]),
    ]

    turns_path = Path(turns_jsonl_path)
    dialogue_id = turns_path.parent.name

    with open(turns_path, "r", encoding="utf-8") as f:
        turn_records = [json.loads(line) for line in f if line.strip()]

    # only process turns in "speech" mode that have generated audio files
    speech_turns = [
        r for r in turn_records if r.get("mode") == "speech" and r.get("audio_file")
    ]

    out_records = []
    for idx, rec in enumerate(speech_turns):
        clean_wav_path = rec["audio_file"]
        if not os.path.exists(clean_wav_path):
            print(f"[warning] clean audio file not found: {clean_wav_path}")
            continue

        speech_audio = load_audio_16k(clean_wav_path)
        turn_tag = rec.get("turn_id", f"turn_{idx:02d}")
        utt_id = f"{dialogue_id}_{turn_tag}"

        for split, cond, snr_list in conditions:
            out_dir = Path(output_base_dir) / dialogue_id / split
            out_dir.mkdir(parents=True, exist_ok=True)

            for snr in snr_list:
                if cond == "clean":
                    mixed_audio = speech_audio
                    musan_src = None
                    snr_tag = "clean"
                else:
                    musan_src = rng.choice(musan_pools[cond])
                    noise_audio = load_audio_16k(musan_src)
                    mixed_audio = mix_with_snr(speech_audio, noise_audio, snr, rng)
                    snr_tag = f"{cond}_{int(snr)}dB"

                est_snr = estimate_blind_snr(mixed_audio)
                mixed_utt_id = f"{utt_id}_{snr_tag}_seed{seed}"
                out_wav_path = out_dir / f"{mixed_utt_id}.wav"

                sf.write(
                    str(out_wav_path), mixed_audio, samplerate=16000, subtype="PCM_16"
                )

                out_records.append(
                    {
                        "utt_id": mixed_utt_id,
                        "dialogue_id": dialogue_id,
                        "turn_id": turn_tag,
                        "clean_audio_path": str(clean_wav_path).replace("\\", "/"),
                        "audio_path": str(out_wav_path).replace("\\", "/"),
                        "split": split,
                        "condition": cond,
                        "target_snr_db": snr,
                        "estimated_snr_db": round(est_snr, 2),
                        "path": rec.get("path"),
                        "slot": rec.get("slot"),
                        "assistant_action": rec.get("assistant_action"),
                        "true_text": rec.get("true_text"),
                        "true_value": rec.get("true_value"),
                        "spoken_text": rec.get("spoken_text"),
                        "voice": rec.get("voice"),
                        "audio_seconds": rec.get("audio_seconds"),
                        "musan_source": (
                            os.path.relpath(musan_src, musan_root).replace("\\", "/")
                            if musan_src
                            else None
                        ),
                        "seed": seed,
                    }
                )

    os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"[completed] Generated {len(out_records)} mixed audio files, manifest saved to {output_manifest_path}"
    )


if __name__ == "__main__":
    run_mixing_from_turns(
        turns_jsonl_path="runs/dlg_00001/turns.jsonl",
        musan_root="datasets/musan",
        output_base_dir="mixed_audio",
        output_manifest_path="manifests/mixed_manifest.jsonl",
        seed=42,
    )