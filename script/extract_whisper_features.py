import os
import re
import json
from collections import defaultdict
import torch
import whisper
import jiwer


def normalize_text(text: str) -> str:
    """normalize English text (convert to lowercase, remove punctuation, merge extra whitespace), ensuring fair WER and Slot matching"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_slot_match(true_value: str, transcript: str, is_spell: bool = False) -> bool:
    """
    Determine if the ground_truth true_value is correctly present in the Whisper transcript:
    - Regular utterances: Compare normalized strings
    - Spelled utterances (spell): Remove whitespace and compare contiguous letters (e.g., 'c a m b r i d g e' -> 'cambridge')
    """
    if not true_value:
        return True
    norm_val = normalize_text(str(true_value))
    norm_trans = normalize_text(transcript)

    if is_spell:
        compact_val = norm_val.replace(" ", "")
        compact_trans = norm_trans.replace(" ", "")
        return compact_val in compact_trans
    return norm_val in norm_trans


def run_whisper_and_eval(
    mixed_manifest_path: str,
    output_json_path: str,
    model_size: str = "base.en",
    k_samples: int = 5,
    sample_temp: float = 0.7,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model ({model_size}) to {device}...")
    model = whisper.load_model(model_size, device=device)
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=model.is_multilingual)

    with open(mixed_manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    results = []
    # used for Milestone Table 1
    stats = defaultdict(
        lambda: {
            "refs": [],
            "hyps": [],
            "total_slots": 0,
            "slot_errors": 0,
            "subs": 0,
            "dels": 0,
            "ins": 0,
        }
    )

    for idx, rec in enumerate(records):
        audio_path = rec["audio_path"]
        cond = rec["condition"]
        snr = rec.get("target_snr_db")
        cond_label = "Clean" if cond == "clean" else f"{cond} {int(snr)} dB"

        # 1. turn audio into Log-Mel
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

        # 2. Main transfer (Greedy Decoding, temperature=0.0)
        greedy_options = whisper.DecodingOptions(
            language="en",
            temperature=0.0,
            without_timestamps=True,
            fp16=(device == "cuda"),
        )
        greedy_res = whisper.decode(model, mel, greedy_options)
        transcript = greedy_res.text.strip()

        # extract Token-level logprobs
        token_details = []
        for tid, lprob in zip(greedy_res.tokens, greedy_res.token_logprobs):
            if tid < tokenizer.eot:
                token_details.append(
                    {
                        "token_id": int(tid),
                        "token": tokenizer.decode([tid]),
                        "logprob": round(float(lprob), 4),
                    }
                )

        # 3. K temperature sampling
        k_transcripts = []
        sample_options = whisper.DecodingOptions(
            language="en",
            temperature=sample_temp,
            without_timestamps=True,
            fp16=(device == "cuda"),
        )
        for _ in range(k_samples):
            s_res = whisper.decode(model, mel, sample_options)
            k_transcripts.append(s_res.text.strip())

        # 4. Calculate error metrics to fill Milestone Table 1
        ref_norm = normalize_text(rec.get("true_text", ""))
        hyp_norm = normalize_text(transcript)
        if ref_norm:
            word_measures = jiwer.process_words(ref_norm, hyp_norm)
            stats[cond_label]["refs"].append(ref_norm)
            stats[cond_label]["hyps"].append(hyp_norm)
            stats[cond_label]["subs"] += word_measures.substitutions
            stats[cond_label]["dels"] += word_measures.deletions
            stats[cond_label]["ins"] += word_measures.insertions

        true_val = rec.get("true_value")
        is_spell = rec.get("path") == "spell"
        slot_correct = evaluate_slot_match(true_val, transcript, is_spell=is_spell)
        if true_val:
            stats[cond_label]["total_slots"] += 1
            if not slot_correct:
                stats[cond_label]["slot_errors"] += 1

        # 5. Output standard JSON structure for Yifan (Estimator/U2R) and Llama Agent
        output_record = {
            "utt_id": rec["utt_id"],
            "dialogue_id": rec["dialogue_id"],
            "turn_id": rec["turn_id"],
            "split": rec["split"],
            "condition": cond,
            "target_snr_db": snr,
            "path": rec.get("path"),
            "slot": rec.get("slot"),
            "assistant_action": rec.get("assistant_action"),
            "true_text": rec.get("true_text"),
            "true_value": true_val,
            "slot_correct": slot_correct,
            "transcript": transcript,
            "utterance_features": {
                "no_speech_prob": round(float(greedy_res.no_speech_prob), 4),
                "compression_ratio": round(float(greedy_res.compression_ratio), 4),
                "estimated_snr_db": rec["estimated_snr_db"],
            },
            "tokens": token_details,
            "k_sampled_transcripts": k_transcripts,
        }
        results.append(output_record)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(records):
            print(f"Progress: {idx + 1}/{len(records)} | {rec['utt_id']} -> \"{transcript}\"")

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 6. Directly print the statistics in Overleaf Table 1 format
    print("\n" + "=" * 68)
    print("Milestone Table 1 Data Summary (can be directly inserted into Overleaf main.tex):")
    print(f"{'Condition':<18} | {'WER (%)':<8} | {'Slot err (%)':<12} | {'Sub (%)':<7} | {'Del (%)':<7} | {'Ins (%)':<7}")
    print("-" * 68)
    for cond_label, d in stats.items():
        wer_pct = jiwer.wer(d["refs"], d["hyps"]) * 100 if d["refs"] else 0.0
        slot_err_pct = (d["slot_errors"] / d["total_slots"] * 100) if d["total_slots"] > 0 else 0.0
        total_err_ops = d["subs"] + d["dels"] + d["ins"]
        sub_pct = (d["subs"] / total_err_ops * 100) if total_err_ops > 0 else 0.0
        del_pct = (d["dels"] / total_err_ops * 100) if total_err_ops > 0 else 0.0
        ins_pct = (d["ins"] / total_err_ops * 100) if total_err_ops > 0 else 0.0
        print(
            f"{cond_label:<18} | {wer_pct:<8.1f} | {slot_err_pct:<12.1f} | "
            f"{sub_pct:<7.1f} | {del_pct:<7.1f} | {ins_pct:<7.1f}"
        )
    print("=" * 68)


if __name__ == "__main__":
    run_whisper_and_eval(
        mixed_manifest_path="manifests/mixed_manifest.jsonl",
        output_json_path="manifests/whisper_features.jsonl",
        model_size="base.en",
        k_samples=5,
        sample_temp=0.7,
    )