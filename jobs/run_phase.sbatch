#!/bin/bash
# One experiment job (fit or eval) with real audio and both LLMs. Split big runs
# into several jobs by seed or condition, then merge (see jobs/README.md).
#
#   sbatch --export=ALL,PHASE=fit,N=50,SEEDS="0",TAG=fit_s0 jobs/run_phase.sbatch
#   sbatch --export=ALL,PHASE=eval,N=50,SEEDS="0",MODELS=models/fit,TAG=eval_s0 jobs/run_phase.sbatch
#   extra run_agent.py flags:  EXTRA="--conditions music:5 sidetalk:5"
#
# Edit the #SBATCH lines to your allocation (see smoke_test.sbatch).
#SBATCH -J u2r-run
#SBATCH -N 1
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/run-%j.out

set -euo pipefail
source jobs/common.sh

PHASE=${PHASE:-fit}
N=${N:-20}
SEEDS=${SEEDS:-0}
FIRST_GOAL=${FIRST_GOAL:-0}
TAG=${TAG:-${PHASE}_${SLURM_JOB_ID:-local}}
LOGDIR=logs/$TAG
MODELS=${MODELS:-models/fit}
MWZ=datasets/multiwoz/data/MultiWOZ_2.2
EXTRA=${EXTRA:-}
mkdir -p "$LOGDIR"

start_llms "$LOGDIR"

COMMON="--channel whisper --musan datasets/musan --whisper-model ${WHISPER:-small.en} --K ${K:-5}
        --n $N --first-goal $FIRST_GOAL --seeds $SEEDS --logdir $LOGDIR --out $LOGDIR/runs $LLM_ARGS --ask-llm"

if [ "$PHASE" = fit ]; then
  # train goals; clean + noise 20/15/10/5 dB; accept main line + save-point replays
  time python run_agent.py --phase fit --multiwoz $MWZ/train $COMMON $EXTRA
elif [ "$PHASE" = eval ]; then
  # test goals; held-out music / side talk; every policy
  [ -f "$MODELS/estimator.json" ] || { echo "!! no $MODELS/estimator.json: run fit + train first"; exit 1; }
  time python run_agent.py --phase eval --multiwoz $MWZ/test $COMMON \
       --estimator "$MODELS/estimator.json" --outcomes "$MODELS/outcomes.json" $EXTRA
else
  echo "PHASE must be fit or eval"; exit 1
fi
echo "== done: $LOGDIR"
