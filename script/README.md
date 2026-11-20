# Speech Channel Pipeline (`TTS` → `MUSAN Mixing` → `Whisper Feature Extraction`)

This module implements the simulated speech channel for the **When Is Switching Modality Worth It?** project. It takes simulated user turns from `robot_customer`, synthesizes 16 kHz speech via **Kokoro-82M**, mixes background audio from **MUSAN** across controlled acoustic conditions, and extracts ASR transcripts and uncertainty features via **Whisper**.

---

## 1. Prerequisites & Setup

1. **System Dependency (for Kokoro G2P)**:
* Install `espeak-ng` (on Windows, install `espeak-ng-X64.msi`; on Linux/PACE-ICE, load or install `espeak-ng`).


2. **Python Packages**:
* Install required dependencies: `pip install torch librosa soundfile kokoro openai-whisper jiwer`


3. **Dataset Placement**:
* Download and extract [MUSAN](https://www.openslr.org/17/) to `datasets/musan/` (containing `noise/`, `music/`, and `speech/`).

---

## 2. Pipeline Scripts & Execution Order

### Step 1: Text-to-Speech (`generate_tts.py`)

Synthesizes `mode == "speech"` turns into **16 kHz, mono, 16-bit PCM WAV** files using `Kokoro-82M`. Supports both American (`af_*`, `am_*`) and British (`bf_*`, `bm_*`) voices and preserves pre-formatted spelling pauses from `spoken_text`.

* **Option A (Inline Function for `RobotCustomer`)**:
* Import and call: `from script.generate_tts import kokoro_tts`
* Returns audio duration in seconds (`float`): `duration_sec = kokoro_tts(text, voice, out_path)`


* **Option B (Batch from `turns.jsonl`)**:
* Run: `python script/generate_tts.py`
* Reads `runs/<dialogue_id>/turns.jsonl`, generates WAV files for `mode == "speech"`, and populates `audio_seconds`.

---

### Step 2: Acoustic Noise Mixing (`mix_musan.py`)

Mixes clean TTS audio with MUSAN background audio using active-speech RMS scaling (excluding silence) and computes a blind SNR estimate (`estimated_snr_db`).

* **Splits & Conditions (Held-out Protocol)**:
* `train_calib`: `clean`, `noise` at `20 dB`, `10 dB`, `5 dB`, `0 dB`
* `heldout_test`: `music` (filtered for vocals via `ANNOTATIONS`) and `side_talk` at `10 dB`, `5 dB`


* **Run**: `python script/mix_musan.py`
* **Output**: Mixed `.wav` files under `mixed_audio/` and metadata manifest at `manifests/mixed_manifest.jsonl`.

---

### Step 3: Whisper ASR & Feature Extraction (`extract_whisper_features.py`)

Runs greedy decoding (`temperature=0.0`) for the primary transcript and token log-probabilities, plus $K=5$ temperature-sampled passes (`temperature=0.7`) for slot agreement estimation. Also prints Table 1 summary metrics (`WER`, `Slot err.`, `Sub.`, `Del.`, `Ins.`).

* **Run**: `python script/extract_whisper_features.py`
* **Output**: `manifests/whisper_features.jsonl`

---

## 3. Output JSON Schema (`whisper_features.jsonl`)

Each line in `manifests/whisper_features.jsonl` is a JSON object consumed by the downstream LLM agent and slot error estimator:

```
{
  "utt_id": "dlg_00001_turn_01_noise_10dB_seed42",
  "dialogue_id": "dlg_00001",
  "turn_id": "turn_01",
  "split": "train_calib",
  "condition": "noise",
  "target_snr_db": 10.0,
  "path": "normal",
  "slot": "hotel-area",
  "assistant_action": "ASK",
  "true_text": "I need a hotel in the centre of town.",
  "true_value": "centre",
  "slot_correct": true,
  "transcript": "I need a hotel in the center of town.",
  "utterance_features": {
    "no_speech_prob": 0.012,
    "compression_ratio": 1.35,
    "estimated_snr_db": 9.42
  },
  "tokens": [
    {"token_id": 40, "token": "I", "logprob": -0.05}
  ],
  "k_sampled_transcripts": [
    "I need a hotel in the center of town.",
    "I need a hotel in the sender of town."
  ]
}

```