# Assistant agent (U2R)

This is the agent the robot customer talks to. For every value it hears, it estimates how likely the value is wrong (p̂). It then chooses the cheapest way to make sure the value is right: **accept**, **confirm**, **spell** or **type**.

```
TurnPackage ─► channel ─► Heard ─► extract (Llama) ─► features ─► p̂ ─► U2R ─► {"text","action","slot"} ─► robot customer
                (text / mock / Whisper)                                         │
                                         decode the repair answer ◄─────────────┘  (spell / type / confirm)
```

| File | What it does |
|---|---|
| `config.py` | Actions, effort `c_a`, `C_w`, slot types. Slots come from `robot_customer.goal_builder` |
| `values.py` | Canonical values. `"seven thirty p m"` → `19:30`, `"four oh four …"` → `404 555 1234`. Changes the format only, never the value |
| `prompts.py` | Every LLM prompt word for word (`PROMPT_VERSION`): EXTRACT, MISHEAR (RQ4), CHOOSE (LLM baseline) |
| `llm.py` | OpenAI-compatible client (vLLM or Ollama). Temperature 0, fixed seed |
| `extract.py` | `LLMExtractor` (copies values as heard) and `RuleExtractor` (no model, for tests) |
| `repair.py` | Request templates. Decoders for the customer's `spell_out` output, typed text, yes/no answers and the "can't type" line |
| `channel.py` | `Heard`, `TextChannel`, `MockASRChannel`, `ASRChannel` (the Whisper hook), `slot_features` |
| `estimator.py` | Logistic regression plus temperature or Platt calibration, in numpy only. ECE, Brier, AUROC |
| `outcomes.py` | r_a and e_a per (action, slot type), with Beta(1,1) smoothing and a pooled fallback |
| `policies.py` | U2R (Eq. 1) with the closed-form p\*, plus the baselines |
| `agent.py` | `U2RAgent.step(heard) -> dict`, the dialogue policy |
| `scoring.py` | Adds the answer key afterwards and writes `attempts.jsonl` and `dialogues.jsonl` rows |
| `train.py` | Splits fit logs 60/20/20 by goal, trains, calibrates, reports, and saves `estimator.json` and `outcomes.json` |

## Run

```bash
python -m unittest discover tests                                    # 45 tests, no model needed

# Read a few dialogues (perfect recognition)
python run_agent.py --policies confirm --channel text --n 3 --show

# The whole loop offline with the mock recognizer (about 30 s; never report these numbers)
python run_agent.py --phase fit  --channel mock --n 40 --logdir logs/fit
python -m assistant_agent.train --attempts logs/fit/attempts.jsonl --out models/
python run_agent.py --phase eval --channel mock --n 40 --logdir logs/eval \
       --estimator models/estimator.json --outcomes models/outcomes.json
```

Real models: the customer runs on Qwen and the agent on Llama, so the agent never "understands itself".

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct         --port 8000 --gpu-memory-utilization 0.3
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001 --gpu-memory-utilization 0.3
python run_agent.py --brain llm --customer-url http://localhost:8000/v1 --customer-model Qwen/Qwen2.5-7B-Instruct \
       --agent-extractor llm --agent-url http://localhost:8001/v1 --ask-llm ...
```

- **Fit phase:** clean speech and noise at 4 SNRs, run with the forced policies `accept`, `confirm`, `spell` and `type`. These runs are the training data for the estimator and for r_a / e_a.
- **Eval phase:** music and side talk only (held out), run with every policy. Nothing from this phase is used for fitting.
- **Policies:** `typed`, `accept`, `confirm`, `spell`, `type`, `threshold:-0.5`, `llm`, `u2r`, `u2r-notype`, `u2r:<C_w>`.
- **Paired comparisons:** the same seed gives the same goal, voice and can-type trait under every policy.
- **Logs:** files are overwritten on each run, so give each run its own `--logdir`. The log folder also gets `agent_prompts.txt` and `config.json`.

## The rules the agent follows

- Typed input is accepted as is.
- The agent asks one question per turn. Several values heard in one turn queue their repairs.
- A "no" to a confirmation without a new value leads to spelling for name, email and phone, and to a plain re-ask for other slots.
- After the customer says they can't type, the agent never asks TYPE again. A queued or chosen TYPE becomes `spell_fallback`.
- Spelling is decoded from the customer's `spell_out` form. Whisper styles such as `V-A-R-U-N` and `varun dot k at …` also work. The test round-trips every goal value for 300 seeds.
- A slot that gets no value after 2 questions is left empty and logged as `missed`.

## Plugging in Whisper (for the audio teammate)

Wrap your step (TTS file → noise → Whisper) in a function that receives a speech `TurnPackage` and returns:

```python
{"text": "...", "words": [("varun", 0.91), ...], "samples": ["...", ...],   # K transcripts at T > 0
 "no_speech_prob": 0.01, "compression_ratio": 1.2, "snr_est": 9.5}          # blind SNR, not the true one
channel = ASRChannel(my_transcribe)
```

Then add it to `make_channel` in `run_agent.py`. Typed turns never go through Whisper.

## Log fields (one row per heard value, `attempts.jsonl`)

`slot, slot_type, turn, mode, raw, hypothesis, transcript, features, p_hat, llm_p_wrong, expected_costs, action, repair_channel, repair_received, committed_value, effort`, plus the answer key: `true_value, first_correct, committed_correct, error_type, missed, spurious`. `dialogues.jsonl` has `joint_goal_correct` (end-of-dialogue state, not the per-turn MultiWOZ JGA), `task_success`, `n_wrong_commits`, `turns_to_first_error`, `total_effort`, `n_type_requests`, `typed_chars`, and more.

## Known limits

- `RuleExtractor` only knows the ontology words. It is a test stand-in; real runs use `--agent-extractor llm`.
- Under the mock recognizer, SPELL does badly because every letter is a token that can be dropped. That is an artifact of the mock.
- The customer's typo rate (2% per character) makes about 1 in 8 typed values wrong, so TYPE has a high e_a. Matching it to HIVE is the customer's TODO.
- U2R's cost for TYPE does not add the expected cost of falling back to spelling for users who can't type. The fit phase captures that cost in r/e instead.
