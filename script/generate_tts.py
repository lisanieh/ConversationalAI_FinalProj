import os
import json
import torch
import librosa
import soundfile as sf
from kokoro import KPipeline

# 1. initialize Kokoro Pipeline（'a' = American English；'b' = British English）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipeline = KPipeline(lang_code='a', device=device)

def format_spelling_text(word: str) -> str:
    """turn a word into a spelled-out format, e.g., "Cambridge" -> "C, A, M, B, R, I, D, G, E" """
    cleaned = word.strip().upper()
    return ", ".join(list(cleaned))

def generate_utterance(
    utt_id: str,
    text: str,
    output_dir: str,
    voice: str = 'af_heart',
    speed: float = 1.0,
    is_spelled: bool = False
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    
    # turn the text into spelled-out format if is_spelled is True
    tts_input = format_spelling_text(text) if is_spelled else text
    
    # Kokoro TTS in streaming mode
    generator = pipeline(tts_input, voice=voice, speed=speed)
    audio_chunks = []
    for _, _, audio in generator:
        if audio is not None:
            audio_chunks.append(audio)
            
    if not audio_chunks:
        raise ValueError(f"TTS failed, no audio: {utt_id}")
        
    import numpy as np
    full_audio_24k = np.concatenate(audio_chunks)
    
    # 2. resample：change Kokoro 24,000 Hz into Whisper 16,000 Hz
    audio_16k = librosa.resample(full_audio_24k, orig_sr=24000, target_sr=16000)
    
    # 3. store into 16-bit PCM WAV
    out_path = os.path.join(output_dir, f"{utt_id}.wav")
    sf.write(out_path, audio_16k, samplerate=16000, subtype='PCM_16')
    return out_path

if __name__ == "__main__":
    # testing：normal dialogue / spell format / diff tones
    sample_tasks = [
        {
            "utt_id": "MUL0001_turn1_normal",
            "text": "I would like to book a train from Cambridge to London Kings Cross.",
            "voice": "af_heart",      # american female voice
            "is_spelled": False,
            "slots": [{"domain": "train", "slot_name": "departure", "slot_value": "cambridge"}]
        },
        {
            "utt_id": "MUL0001_turn2_spell",
            "text": "Cambridge",
            "voice": "af_heart",
            "is_spelled": True,       # if system asked for spelling, you can set is_spelled=True to generate spelled-out format
            "slots": [{"domain": "train", "slot_name": "departure", "slot_value": "cambridge"}]
        },
        {
            "utt_id": "MUL0002_turn1_male",
            "text": "My reference number is X 7 B 9 2 K.",
            "voice": "am_michael",    # american male voice (can be used to test different speakers)
            "is_spelled": False,
            "slots": [{"domain": "hotel", "slot_name": "ref", "slot_value": "x7b92k"}]
        }
    ]

    output_dir = "raw_tts"
    manifest_path = "manifests/raw_tts_manifest.jsonl"
    os.makedirs("manifests", exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        for task in sample_tasks:
            wav_path = generate_utterance(
                utt_id=task["utt_id"],
                text=task["text"],
                output_dir=output_dir,
                voice=task["voice"],
                is_spelled=task["is_spelled"]
            )
            record = {
                "utt_id": task["utt_id"],
                "raw_audio_path": wav_path.replace("\\", "/"),
                "ground_truth_text": task["text"],
                "voice": task["voice"],
                "is_spelled": task["is_spelled"],
                "slots": task["slots"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[finish] {task['utt_id']} -> {wav_path}")