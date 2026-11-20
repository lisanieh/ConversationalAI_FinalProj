import os
import json
from pathlib import Path
import numpy as np
import torch
import librosa
import soundfile as sf
from kokoro import KPipeline

# initialize Kokoro Pipeline（'a' = Amercian,  'b' = British）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipelines = {
    'a': KPipeline(lang_code='a', device=device),
    'b': KPipeline(lang_code='b', device=device),
}

def kokoro_tts(text: str, voice: str, out_path: Path) -> float:
    """
    [Option A] get the clean spoken_text、voice ID and output path,
    write 16 kHz mono WAV send the duration (float)。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # based on Kokoro voice naming rules automatically select American ('af_', 'am_') or British ('bf_', 'bm_')
    lang_code = 'b' if voice.startswith(('bf_', 'bm_')) else 'a'
    pipeline = pipelines[lang_code]

    # directly synthesize preprocessed spoken_text
    generator = pipeline(text, voice=voice, speed=1.0)
    audio_chunks = [audio for _, _, audio in generator if audio is not None]

    if not audio_chunks:
        raise ValueError(f"TTS generation failed: text='{text}', voice='{voice}'")

    full_audio_24k = np.concatenate(audio_chunks)

    # resample to Whisper's required 16,000 Hz mono
    audio_16k = librosa.resample(full_audio_24k, orig_sr=24000, target_sr=16000)
    sf.write(str(out_path), audio_16k, samplerate=16000, subtype='PCM_16')

    # return audio file duration (float)
    duration_seconds = float(len(audio_16k) / 16000.0)
    return round(duration_seconds, 4)


def process_turns_jsonl(turns_jsonl_path: str):
    """
    [Option B] read the turns.jsonl in batch：
    for mode == "speech" generate 16kHz WAV and send audio_seconds
    """
    turns_path = Path(turns_jsonl_path)
    updated_lines = []

    with open(turns_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)

            # only TTS in speech mode, skip in typed mode
            if record.get("mode") == "speech":
                spoken_text = record["spoken_text"]
                voice = record["voice"]
                audio_file = Path(record["audio_file"])

                duration = kokoro_tts(spoken_text, voice, audio_file)
                record["audio_seconds"] = duration
                print(f"[TTS finish] {audio_file} ({duration}s) | text: {spoken_text}")

            updated_lines.append(json.dumps(record, ensure_ascii=False))

    # write updated audio_seconds back to turns.jsonl
    with open(turns_path, "w", encoding="utf-8") as f:
        for line in updated_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    # test Option B：read turns.jsonl
    sample_turns_file = "runs/dlg_00001/turns.jsonl"
    if os.path.exists(sample_turns_file):
        process_turns_jsonl(sample_turns_file)
    else:
        # test in single utterance mode (Option A)
        sec = kokoro_tts(
            text="V. A. R. U. N. dot. K. at. G. M. A. I. L. dot. C. O. M.",
            voice="bf_emma",
            out_path=Path("runs/test_audio/turn_01.wav")
        )
        print(f"Single utterance test completed, audio duration: {sec} seconds")