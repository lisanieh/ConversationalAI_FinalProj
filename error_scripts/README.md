# Measurement scripts: estimate error and mistake error

Measurement layer for *When Is Switching Modality Worth It?*: defines every error we report and computes it from one shared log format.
`estimator.py` and `u2r.py` are reference implementations, used to cross-check the agent's own estimator and U2R and to run ablations.

Requirements: numpy, pandas, scipy, scikit-learn, matplotlib (Python 3.10+).

## Quick start

```bash
python test_metrics.py        # 9 self-tests with hand-checkable answers
python run_demo.py            # whole chain on synthetic data -> demo_output/
python make_fake_logs.py && python evaluate.py --turns fake/turns.jsonl \
    --goals fake/goals.json --slots fake/slots.jsonl --out fake_results
```

The synthetic and fake data only exercise the code; their numbers mean nothing.

## Files

| File | What it measures / does | RQ |
|---|---|---|
| align.py | Mistake error: normalization, Levenshtein alignment, WER, slot label y and error type | RQ1 |
| features.py | Inputs for p_hat from Whisper output | RQ2 |
| estimator.py | Logistic regression + temperature / Platt / isotonic calibration, dialogue-grouped split | RQ2 |
| calibration_metrics.py | Estimate error: ECE, MCE, Brier decomposition, NLL, AUROC, AUPRC, reliability plots, AUROC per error type | RQ2, RQ4 |
| outcome_model.py | Repair fix rate r, break rate e and effort c per action (Beta smoothing, 95% intervals) | U2R |
| u2r.py | Expected cost, argmin decision, closed-form p*, decision regions, cost-line plot | U2R, RQ3 |
| policy_eval.py | Baseline policies vs U2R, observed vs predicted crossover, success-effort AUC | RQ3 |
| episode_metrics.py | Task success, JGA, wrong-commit rate, turns to first error (Kaplan-Meier), effort; defines the log format | RQ3 |
| stats_tests.py | Dialogue-level bootstrap, paired permutation test, Holm correction | all |
| audio_checks.py | Quality check: measured SNR of mixed audio vs nominal SNR | condition validity |
| evaluate.py | Single entry point: team logs in, every table and figure out | all |
| test_metrics.py | Self-tests for the metrics | - |
| run_demo.py, make_fake_logs.py | Synthetic demos | - |

## Log format

- `goals.json`: `{dialogue_id: {slot: value}}`
- `slots.jsonl`: one row per slot (Whisper fields, `hyp_value`, `p_hat`, optional `p_llm`); `align.py` adds `y` and `error_type`
- `turns.jsonl`: one row per turn of each (dialogue, seed, condition, policy); schema in the header of `episode_metrics.py`

## RQ4

Ask the LLM for the probability that each slot was misheard (verbal 0-100, or the token probability of "yes"), store it as `p_llm`, and evaluate it with `calibration_metrics.full_report(y, p_llm)` next to `p_hat`.
