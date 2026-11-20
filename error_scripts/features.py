"""
features.py — extracts the input features for p_hat for each slot from Whisper output.

Expected Whisper fields (agreed with the speech pipeline; openai-whisper with transcribe(..., word_timestamps=True)):
    words    : [{"word": str, "probability": float}, ...]     # main decode (temperature=0)
    segments : [{"avg_logprob": float, "no_speech_prob": float, "compression_ratio": float}, ...]
    samples  : [str, ...]   # K sampled transcripts at temperature>0 (e.g. K=5, T=0.7)
    snr_est_db : float      # estimated SNR (use the estimate for both training and testing, never the true value in training)

Output features (one-to-one with the proposal's Components (i)):
    min_logprob, mean_logprob : min / mean of log word probability within the slot span
    span_found                : whether the slot value can be located in the transcript (if 0, logprob falls back to the segment mean)
    agree_k                   : share of the K samples that agree with the main decode on the slot value
    no_speech_prob, compression_ratio, avg_logprob : segment level (worst segment)
    snr_est_db, slot_type     : condition and type
"""
from __future__ import annotations

import math
from align import normalize

FEATURES = ["min_logprob", "mean_logprob", "span_found", "agree_k",
            "no_speech_prob", "compression_ratio", "avg_logprob", "snr_est_db"]
CATEGORICAL = ["slot_type"]


def _word_span(words: list[dict], value: str) -> list[dict] | None:
    toks = [normalize(w["word"]) for w in words]
    target = normalize(value).split()
    if not target:
        return None
    # allow one Whisper word to become several tokens after normalization (e.g. "7:30")
    flat, owner = [], []
    for k, t in enumerate(toks):
        for piece in t.split():
            flat.append(piece)
            owner.append(k)
    for s in range(len(flat) - len(target) + 1):
        if flat[s:s + len(target)] == target:
            ks = sorted(set(owner[s:s + len(target)]))
            return [words[k] for k in ks]
    return None


def slot_value_in(text: str, value: str) -> bool:
    """Cheap agreement proxy: does the sampled transcript contain the slot value?
    Stricter alternative: run LLM slot extraction on every sample and compare values. The paper must say which one is used."""
    v = normalize(value)
    return bool(v) and f" {v} " in f" {normalize(text)} "


def extract(row: dict) -> dict:
    segs = row.get("segments") or [{}]
    worst = {
        "avg_logprob": min(s.get("avg_logprob", 0.0) for s in segs),
        "no_speech_prob": max(s.get("no_speech_prob", 0.0) for s in segs),
        "compression_ratio": max(s.get("compression_ratio", 1.0) for s in segs),
    }
    hv = row.get("hyp_value") or ""
    span = _word_span(row.get("words", []), hv) if hv else None
    if span:
        lps = [math.log(max(w["probability"], 1e-6)) for w in span]
        min_lp, mean_lp, found = min(lps), sum(lps) / len(lps), 1
    else:
        min_lp = mean_lp = worst["avg_logprob"]
        found = 0
    samples = row.get("samples") or []
    agree = (sum(slot_value_in(s, hv) for s in samples) / len(samples)) if samples and hv else 0.0
    return {"min_logprob": min_lp, "mean_logprob": mean_lp, "span_found": found,
            "agree_k": agree, **worst, "snr_est_db": row.get("snr_est_db", 30.0),
            "slot_type": row.get("slot_type", "other")}
