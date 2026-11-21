# Running on PACE-ICE

The plan is to set up once, run the smoke test, then run the fit phase, train, run the eval phase, and evaluate. Every command runs from the repo root on the login node.

## 0. Upload and set up (once)

```bash
# from your laptop
scp -r ConversationalAI_FinalProj <gt-user>@login-ice.pace.gatech.edu:~/scratch/
ssh <gt-user>@login-ice.pace.gatech.edu
cd ~/scratch/ConversationalAI_FinalProj
mkdir -p logs                          # Slurm needs it before the first job
bash jobs/setup_env.sh                 # about 15 min: small env with Kokoro, Whisper and error_scripts
```

The jobs follow the course template (`CAI_A4 run_agent.sbatch`):
- vLLM runs from the course's shared env `/storage/ice1/0/4/lheck6/CS6604_ECE6256/envs/CAI_A4`, with models from the shared offline HF cache.
- Ports come from the job id, so two jobs on one node can't reach each other's server.
- `PYTHONNOUSERSITE=1` stops packages in `~/.local` from being used.

The setup script reports whether Llama-3.1-8B and Qwen2.5-7B are in the shared cache. Any model that's missing is downloaded to `~/scratch/hf`. For Llama, that needs the license accepted on huggingface.co and an HF login.

**MUSAN noise data** (11 GB, only if it isn't there yet):

```bash
cd ~/scratch && wget https://www.openslr.org/resources/17/musan.tar.gz && tar xzf musan.tar.gz
ln -sfn ~/scratch/musan ~/scratch/ConversationalAI_FinalProj/datasets/musan
```

**GPU line.** The jobs use `--gres=gpu:H100:1`, the same line as the course template. An H100 has 80 GB, enough for Qwen and Llama at 38% of the GPU each, plus Whisper and Kokoro. On a smaller card, `common.sh` serves Llama alone for both customer and agent and logs a warning. You can force that with `SAME_MODEL=1`.

## 1. Smoke test (about 30–60 min)

```bash
sbatch jobs/smoke_test.sbatch
squeue -u $USER
tail -f logs/smoke-<jobid>.out
```

It runs six steps, smallest first, so a failure tells you which piece broke:

| Step | What it checks | If it fails |
|---|---|---|
| 1 | Unit tests (49) | Code or package problem |
| 2 | `jobs/check_audio.py`: Kokoro → MUSAN → Whisper on names, spelled email, phone and time, with clean, noise:10 and music:5 audio | TTS, Whisper or MUSAN. Listen to `logs/smoke_<id>/audio_check/*.wav` |
| 3 | One audio dialogue without LLMs | Audio connector inside the dialogue loop |
| 4 | Starts vLLM: Qwen on :8000 (customer), Llama on :8001 (agent) | `logs/smoke_<id>/vllm_*.log` |
| 5 | Full stack, 2 goals, with replays, then the forced confirm, spell and type policies | LLM prompts and extraction. Read the printed dialogues |
| 6 | Mock fit → train → eval → `error_scripts/evaluate.py` | Measurement packages |

**What to send back:**
- The `real` times printed in step 5.
- The step 2 transcripts.
- A couple of the dialogues printed in step 5.

The step 5 times give the seconds per dialogue, which decides `N` below.

## 2. Fit phase → train

The fit phase uses train goals, clean audio plus noise at 20/15/10/5 dB, the `accept` main line, and save-point replays. Split it into one job per seed:

```bash
for s in 0 1 2; do
  sbatch --export=ALL,PHASE=fit,N=50,SEEDS=$s,TAG=fit_s$s jobs/run_phase.sbatch
done
```

When all three finish, train on the login node. This takes about a minute on CPU:

```bash
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh && conda activate ~/scratch/envs/u2r
python -m assistant_agent.train --attempts logs/fit_s*/attempts.jsonl \
       --replays logs/fit_s*/replays.jsonl --out models/fit
```

Check the printout. Calibration (ECE) should be small, AUROC well above 0.5, and each r/e/c cell should hold enough cases.

## 3. Eval phase (held-out music / side talk, every policy)

```bash
for s in 0 1 2; do
  sbatch --export=ALL,PHASE=eval,N=50,SEEDS=$s,MODELS=models/fit,TAG=eval_s$s jobs/run_phase.sbatch
done
```

The eval phase is the biggest run: 12 policies × 4 conditions × N goals per seed. If one job hits the time limit, split it further with `EXTRA="--conditions music:10 music:5"` or `EXTRA="--policies u2r accept spell type"`.

## 4. Merge and evaluate

```bash
python -m assistant_agent.export logs/eval_s0 logs/eval_s1 logs/eval_s2 --out logs/eval_all
cd error_scripts
python evaluate.py --turns ../logs/eval_all/eval_turns.jsonl --goals ../logs/eval_all/goals.json \
       --slots ../logs/eval_all/eval_slots.jsonl --out ../results
```

`evaluate.py` writes everything into `results/`:
- Episode summary with confidence intervals, and the U2R vs. baseline tests with Holm correction.
- Estimate error (ECE, Brier, AUROC) for p̂, and for `p_llm` (RQ4).
- Mistake types (RQ1) and reliability plots.

For the tipping point (observed vs. predicted p\*), the replay table is `logs/fit_s*/for_eval/replay_outcomes.jsonl`. Feed it to `error_scripts/policy_eval.py`.

## Knobs

| Variable | Default | Meaning |
|---|---|---|
| `N` | 20 | Goals per job |
| `SEEDS` | 0 | Run seeds, for example `"0 1"` |
| `FIRST_GOAL` | 0 | Offset, to split goals across jobs |
| `WHISPER` | small.en | Whisper model |
| `K` | 5 | Sampled transcripts per turn (agreement feature) |
| `EXTRA` | | Any other `run_agent.py` flags |
| `SAME_MODEL=1` | | Use Llama for the customer too |
| `HF_HUB_OFFLINE=0` | | Allow model downloads inside the job |
