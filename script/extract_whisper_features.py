import os
import json
import torch
import whisper

def run_whisper_extraction(
    mixed_manifest_path: str,
    output_json_path: str,
    model_size: str = "base.en",
    k_samples: int = 5,
    sample_temp: float = 0.7
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"downloading Whisper model ({model_size}) to {device}...")
    model = whisper.load_model(model_size, device=device)
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=model.is_multilingual)

    with open(mixed_manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    results = []
    for idx, rec in enumerate(records):
        audio_path = rec["audio_path"]
        
        # 1. input audio and turn into Log-Mel Spectrogram
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

        # 2. main transcription (Greedy Decoding, temperature=0.0) to get stable text and token-level logprobs
        greedy_options = whisper.DecodingOptions(
            language="en",
            temperature=0.0,
            without_timestamps=True,
            fp16=(device == "cuda")
        )
        greedy_res = whisper.decode(model, mel, greedy_options)

        # retrieve token's text and log probabilities
        token_details = []
        for tid, lprob in zip(greedy_res.tokens, greedy_res.token_logprobs):
            # filter out Whisper special control tokens (eot, sot, etc.)
            if tid < tokenizer.eot:
                token_str = tokenizer.decode([tid])
                token_details.append({
                    "token_id": int(tid),
                    "token": token_str,
                    "logprob": round(float(lprob), 4)
                })

        # 3. K Temperature Sampling
        k_transcripts = []
        sample_options = whisper.DecodingOptions(
            language="en",
            temperature=sample_temp,
            without_timestamps=True,
            fp16=(device == "cuda")
        )
        for _ in range(k_samples):
            s_res = whisper.decode(model, mel, sample_options)
            k_transcripts.append(s_res.text.strip())

        # 4. create downstream Llama and Mistake Estimator needed JSON structure
        output_record = {
            "utt_id": rec["utt_id"],
            "base_utt_id": rec["base_utt_id"],
            "split": rec["split"],
            "condition": rec["condition"],
            "target_snr_db": rec["target_snr_db"],
            "is_spelled": rec["is_spelled"],
            "ground_truth_text": rec["ground_truth_text"],
            "ground_truth_slots": rec["slots"],
            # Whisper output and feature
            "transcript": greedy_res.text.strip(),
            "utterance_features": {
                "no_speech_prob": round(float(greedy_res.no_speech_prob), 4),
                "compression_ratio": round(float(greedy_res.compression_ratio), 4),
                "estimated_snr_db": rec["estimated_snr_db"]
            },
            "tokens": token_details,
            "k_sampled_transcripts": k_transcripts
        }
        results.append(output_record)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(records):
            print(f"Progress: {idx + 1}/{len(records)} | {rec['utt_id']} -> \"{output_record['transcript']}\"")

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[finish] Whisper features saved to {output_json_path}")

if __name__ == "__main__":
    run_whisper_extraction(
        mixed_manifest_path="manifests/mixed_manifest.jsonl",
        output_json_path="manifests/whisper_features.jsonl",
        model_size="base.en",
        k_samples=5,
        sample_temp=0.7
    )