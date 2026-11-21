#!/bin/bash
# Smoke test: checks every stage on the GPU, smallest sizes first, and times one
# real dialogue so you can size the full runs.     sbatch jobs/smoke_test.sbatch
#
# GPU: two 8B LLMs + Whisper want an 80 GB card (H100 / A100-80GB). On a smaller
# card common.sh serves one model for both sides. Edit the #SBATCH lines to
# your course allocation (`sinfo`, PACE-ICE docs): some courses also need -q / -A.
#SBATCH -J u2r-smoke
#SBATCH -N 1
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o logs/smoke-%j.out

set -uo pipefail
source jobs/common.sh
OUT=logs/smoke_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
MUSAN_ARG=""; [ -d datasets/musan/noise ] && MUSAN_ARG="--musan datasets/musan"
NOISE_COND=""; [ -n "$MUSAN_ARG" ] && NOISE_COND="noise:10"
MWZ=datasets/multiwoz/data/MultiWOZ_2.2

step() { echo; echo "################ $*"; }

step "1/6 unit tests (no models)"
python -m unittest discover tests 2>&1 | tail -3

step "2/6 audio: Kokoro -> MUSAN -> Whisper on 6 sample turns"
python jobs/check_audio.py $MUSAN_ARG --whisper-model small.en --out "$OUT/audio_check" \
  || echo "!! audio check reported problems (see above)"

step "3/6 one audio dialogue, no LLMs (scripted customer, rule extractor)"
time python run_agent.py --channel whisper $MUSAN_ARG --conditions clean $NOISE_COND \
  --policies confirm --n 1 --multiwoz $MWZ/test --logdir "$OUT/audio_scripted" \
  --out "$OUT/runs" --show

step "4/6 start the LLM servers"
start_llms "$OUT"

step "5/6 full stack: Qwen customer + Llama agent + audio, 2 goals, replays on (times it)"
time python run_agent.py --channel whisper $MUSAN_ARG $LLM_ARGS --conditions clean $NOISE_COND \
  --policies accept --replay --ask-llm --n 2 --multiwoz $MWZ/train \
  --logdir "$OUT/full_fit" --out "$OUT/runs" --show
time python run_agent.py --channel whisper $MUSAN_ARG $LLM_ARGS --conditions clean \
  --policies confirm spell type --n 2 --multiwoz $MWZ/test \
  --logdir "$OUT/full_repairs" --out "$OUT/runs" --show

step "6/6 measurement chain on mock data (checks error_scripts' packages on PACE)"
python run_agent.py --phase fit --channel mock --n 15 --seeds 0 1 --logdir "$OUT/mock_fit" --no-save-dialogues | tail -4
python -m assistant_agent.train --attempts "$OUT/mock_fit/attempts.jsonl" \
  --replays "$OUT/mock_fit/replays.jsonl" --out "$OUT/mock_models" --min-n 5 | tail -5
python run_agent.py --policies accept confirm spell type threshold-confirm:-0.5 phat-confirm:0.2 u2r \
  --conditions music:5 --channel mock --n 15 \
  --seeds 0 1 --estimator "$OUT/mock_models/estimator.json" --outcomes "$OUT/mock_models/outcomes.json" \
  --logdir "$OUT/mock_eval" --no-save-dialogues | tail -6
(cd error_scripts && python evaluate.py --turns "../$OUT/mock_eval/for_eval/eval_turns.jsonl" \
  --goals "../$OUT/mock_eval/for_eval/goals.json" --slots "../$OUT/mock_eval/for_eval/eval_slots.jsonl" \
  --out "../$OUT/mock_results" --B 200)

echo; echo "== smoke test done. Outputs in $OUT/ ; read logs/smoke-${SLURM_JOB_ID:-local}.out"
echo "   seconds per dialogue: see the 'real' times of step 5 (2 goals x conditions)"
