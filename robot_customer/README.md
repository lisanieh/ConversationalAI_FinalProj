# Robot customer (simulated user)

Steps 1 to 5 of the simulated user, ending at the TTS input. TTS, the noise mixer, Whisper and the assistant are other modules.

```
goal_builder ─► brain (LLM) ─► router ─┬─ speak ─► to_spoken()  ─► spoken_text ─► [TTS]
                                       ├─ spell ─► spell_out()  ─► spoken_text ─► [TTS]
                                       └─ type  ─► add_typos()  ─► typed_text  ─► assistant
```

| File | Step | What it does |
|---|---|---|
| `goal_builder.py` | 1 | Secret goal (MultiWOZ booking + Faker name/email/phone), `can_type` (80/20), TTS voice |
| `prompts.py` | 2 | Every prompt, word for word (`PROMPT_VERSION`) |
| `brain.py` | 2 | `LLMBrain` (OpenAI-compatible: vLLM or Ollama, fixed temperature and seed), `ScriptedBrain` (no model, for tests) |
| `router.py` | 3 | The assistant's label (`normal/accept/confirm/spell/type` + slot) picks speak / spell / type / cannot_type |
| `speech_text.py` | 4 | Text cleanup for TTS, letter-by-letter spelling |
| `typo.py` | 5 | Neighbouring key / swap / dropped letter, 2% per character, seeded |
| `package.py` | — | `TurnPackage`, one JSON line per turn (the hand-off) |
| `customer.py` | — | `RobotCustomer.respond()` and `run_dialogue()` |
| `mock_assistant.py` | — | A stand-in assistant for testing before the real agent exists |

## Run

```bash
pip install -r requirements.txt
python -m unittest discover tests                       # 21 tests
python run_text_only.py --brain scripted --n 5          # no model needed
python run_text_only.py --brain llm --n 20              # Ollama with qwen2.5:7b-instruct
python run_text_only.py --brain llm --base-url http://<node>:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
```

Each dialogue writes `runs/<dialogue_id>/`: `goal.json` (answer key), `turns.jsonl` (packages), `transcript.json`, `final_state.json`, `prompts.txt`.

To use real MultiWOZ goals, download MultiWOZ 2.1 and pass `--multiwoz path/to/data.json`. Without it, goals are sampled from the MultiWOZ ontology values.

## Hand-off to TTS (for the TTS teammate)

You have two options.

**A. Plug in a function.** It receives the cleaned text, the voice id and the output path. It writes a 16 kHz mono WAV and returns the length in seconds:

```python
def kokoro_tts(text: str, voice: str, out_path: Path) -> float: ...
customer = RobotCustomer(goal, brain, out_dir="runs", tts=kokoro_tts)
```

**B. Batch from files.** Read `turns.jsonl`. For each record with `mode == "speech"`, synthesize `spoken_text` with `voice` to `audio_file`, then fill in `audio_seconds`.

Voice ids are Kokoro names (`af_heart`, `am_michael`, `bf_emma`, …) in `goal_builder.DEFAULT_VOICES`. Pass `GoalBuilder(voices=[...])` to use Piper voices.

### Package fields (one per customer turn)

| Field | Example | Notes |
|---|---|---|
| `mode` | `speech` / `typed` | Whisper only for `speech` |
| `path` | `speak`, `spell`, `type`, `cannot_type` | How the reply was produced |
| `slot`, `assistant_action` | `email`, `spell` | Taken from the assistant's label |
| `true_text` | `varun.k@gmail.com` | **Answer key**, written form |
| `true_value` | `varun.k@gmail.com` | Exact goal value of the slot (null if none) |
| `spoken_text` | `V. A. R. U. N. dot. K. at. …` | **TTS input** (speech only) |
| `typed_text`, `typo_edits`, `typed_chars` | `varun.k@gmial.com` | Typed turns only |
| `voice`, `seed`, `prompt_version` | `bf_emma`, `42`, `v1` | For reproducibility |
| `audio_file`, `audio_seconds` | `runs/dlg_00042/audio/turn_07.wav`, null | Path suggested here. **TTS writes the file and fills in the seconds** |
| `done` | `true` | The customer said [DONE] |

### What the assistant must send each turn

```json
{"text": "Could you spell your email?", "action": "spell", "slot": "email"}
```

`action` is one of `normal | accept | confirm | spell | type`. Slot names follow MultiWOZ: `area, pricerange, type, food, day, people, stay, time`, plus `name, email, phone`.
