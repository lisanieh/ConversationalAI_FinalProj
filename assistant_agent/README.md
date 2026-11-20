# Assistant agent (U2R)

This is the agent the robot customer talks to. For every value it hears, it estimates how likely the value is wrong (p̂). It then chooses the cheapest way to make sure the value is right: **accept**, **confirm**, **spell** or **type**.

```
TurnPackage ─► channel ─► Heard ─► extract (Llama) ─► features ─► p̂ ─► U2R ─► {"text","action","slot"} ─► robot customer
     │      (text / mock / audio)                                                 │
     │       audio = Kokoro TTS → MUSAN noise → Whisper   decode the repair answer ◄┘  (spell / type / confirm)
     └─ goals: MultiWOZ 2.2 (or 2.1) final user states + Faker contacts
```

| File | What it does |
|---|---|
| `config.py` | Actions, effort priors `c_a` and `C_w` (seconds), slot types. Slots come from `robot_customer.goal_builder` |
| `values.py` | Canonical values. `"seven thirty p m"` → `19:30`, `"four oh four …"` → `404 555 1234`. Changes the format only, never the value |
| `prompts.py` | Every LLM prompt word for word (`PROMPT_VERSION`): EXTRACT, MISHEAR (RQ4), CHOOSE (LLM baseline) |
| `llm.py` | OpenAI-compatible client (vLLM or Ollama). Temperature 0, fixed seed |
| `extract.py` | `LLMExtractor` (copies values as heard) and `RuleExtractor` (no model, for tests; knows the MultiWOZ foods) |
| `repair.py` | Request templates. Decoders for the customer's `spell_out` output, typed text, yes/no answers and the "can't type" line |
| `channel.py` | `Heard`, `TextChannel`, `MockASRChannel`, `ASRChannel` (generic hook), `slot_features` |
| `audio.py` | **Audio connector**: `KokoroTTS` (the customer's TTS hook), `NoiseMixer` (reuses `script/mix_musan.py`), `WhisperASR`, `AudioChannel` |
| `effort.py` | Customer seconds per turn (TTS length, speaking, spelling, typing), fittable to the team's own timings |
| `estimator.py` | Logistic regression plus temperature or Platt calibration, in numpy only. ECE, Brier, AUROC |
| `outcomes.py` | r_a, e_a and c_a per (action, slot type), with Beta(1,1) smoothing and a pooled fallback |
| `replay.py` | Save-point replays: every heard value tried under all 4 actions from the same saved state |
| `policies.py` | U2R (Eq. 1) with the closed-form p\*, plus the baselines |
| `agent.py` | `U2RAgent.step(heard) -> dict`, the dialogue policy |
| `scoring.py` | Adds the answer key afterwards and writes `attempts.jsonl` and `dialogues.jsonl` rows |
| `export.py` | Writes `for_eval/` in the format `error_scripts/` reads, and merges several runs |
| `train.py` | Splits fit logs 60/20/20 by goal, trains, calibrates, reports, and saves `estimator.json` and `outcomes.json` |

## Run

```bash
python -m unittest discover tests                                    # 48 tests, no model needed

# Read a few dialogues (perfect recognition, real MultiWOZ 2.2 goals)
python run_agent.py --policies confirm --channel text --n 3 --show \
       --multiwoz datasets/multiwoz/data/MultiWOZ_2.2/test

# The whole loop offline with the mock recognizer (never report these numbers)
python run_agent.py --phase fit  --channel mock --n 40 --seeds 0 1 2 --logdir logs/fit
python -m assistant_agent.train --attempts logs/fit/attempts.jsonl --replays logs/fit/replays.jsonl --out models/
python run_agent.py --phase eval --channel mock --n 40 --seeds 0 1 2 --logdir logs/eval \
       --estimator models/estimator.json --outcomes models/outcomes.json
cd error_scripts && python evaluate.py --turns ../logs/eval/for_eval/eval_turns.jsonl \
       --goals ../logs/eval/for_eval/goals.json --slots ../logs/eval/for_eval/eval_slots.jsonl --out ../results
```

### Real audio (GPU node)

Kokoro speaks each customer turn with the goal's voice. MUSAN noise is mixed in at the condition's SNR, using Lisa's functions from `script/mix_musan.py`. Whisper then transcribes the result, giving word probabilities, K sampled transcripts and segment statistics. Typed turns skip all of this. The TTS length fills `audio_seconds`, so effort uses real speech durations.

```bash
pip install -r requirements.txt            # torch first, matching CUDA
# MUSAN in datasets/musan/{noise,music,speech}
python run_agent.py --channel whisper --whisper-model small.en --K 5 --musan datasets/musan \
       --multiwoz datasets/multiwoz/data/MultiWOZ_2.2/train --phase fit --n 100 --seeds 0 1 2 --logdir logs/fit
#  --save-audio audio_out/   keeps the clean and mixed WAVs for listening checks
```

Real models: the customer runs on Qwen and the agent on Llama, so the agent never "understands itself".

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct         --port 8000 --gpu-memory-utilization 0.3
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001 --gpu-memory-utilization 0.3
python run_agent.py --brain llm --customer-url http://localhost:8000/v1 --customer-model Qwen/Qwen2.5-7B-Instruct \
       --agent-extractor llm --agent-url http://localhost:8001/v1 --ask-llm ...
```

- **Goals:** `--multiwoz` takes MultiWOZ 2.2 or 2.1, as described below. Use **train** goals for the fit phase and **test** goals for eval.
  - MultiWOZ 2.2 can be a split folder, the release folder, or the JSON from `download_datasets.py`. A goal is the user's final hotel or restaurant state.
  - MultiWOZ 2.1 is a `data.json` file.
  - Without `--multiwoz`, goals are sampled from the ontology.
- **Fit phase:** clean speech plus noise at 20, 15, 10 and 5 dB (Lisa's levels). The main dialogue uses `accept`, with save-point replays of all 4 repairs. The runs train the estimator and give r_a, e_a and c_a.
- **Eval phase:** music and side talk only (held out), run with every policy. Nothing from this phase is used for fitting.
- **Policies:** `typed`, `accept`, `confirm`, `spell`, `type`, `threshold:-0.5`, `llm`, `u2r`, `u2r-notype`, `u2r:<C_w seconds>`.
- **Paired comparisons:** `--seeds` repeats each goal. The goal, voice and can-type trait stay the same under every policy and seed.
- **Logs:** files are overwritten on each run, so give each run its own `--logdir`. `python -m assistant_agent.export dirA dirB --out merged/` merges the `for_eval/` folders of several jobs.

## The rules the agent follows

- Typed input is accepted as is.
- The agent asks one question per turn. Several values heard in one turn queue their repairs.
- A "no" to a confirmation without a new value leads to spelling for name, email and phone, and to a plain re-ask for other slots.
- After the customer says they can't type, the agent never asks TYPE again. A queued or chosen TYPE becomes `spell_fallback`.
- Spelling is decoded from the customer's `spell_out` form. Whisper styles such as `V-A-R-U-N` and `varun dot k at …` also work. The test round-trips every goal value for 300 seeds.
- A slot that gets no value after 2 questions is left empty and logged as `missed`.

## Log fields (one row per heard value, `attempts.jsonl`)

`slot, slot_type, turn, mode, raw, hypothesis, transcript, features, p_hat, llm_p_wrong, expected_costs, action, repair_channel, repair_turns, repair_received, committed_value, effort_s`, plus the answer key: `true_value, first_correct, committed_correct, error_type, missed, spurious`. `dialogues.jsonl` has `joint_goal_correct` (end-of-dialogue state, not the per-turn MultiWOZ JGA), `task_success`, `n_wrong_commits`, `turns_to_first_error`, `total_effort_s`, `total_user_time_s`, `n_type_requests`, `typed_chars`, and more.

## Known limits

- `RuleExtractor` only knows the ontology words. It is a test stand-in; real runs use `--agent-extractor llm`.
- Under the mock recognizer, SPELL does badly because every letter is a token that can be dropped. That is an artifact of the mock.
- The customer's typo rate (2% per character) makes about 1 in 8 typed values wrong, so TYPE has a high e_a. Matching it to HIVE is the customer's TODO.
- Replays multiply the cost of a run. Each saved turn with new values costs a probe, plus 3 branches per value, each a few turns. With audio, every branch turn is another TTS and Whisper call, so replays only belong in the fit phase.
