#!/bin/bash
# Sourced by the sbatch files. Follows the course template (CAI_A4 run_agent.sbatch):
#   * vLLM runs from the course's shared env, models from the shared offline HF cache;
#   * our own env (jobs/setup_env.sh) runs the pipeline: Kokoro, Whisper, run_agent.py;
#   * each job takes its ports from its job id, so jobs sharing a node never talk to
#     somebody else's server.
# Customer = Qwen, agent = Llama: one vLLM server each on the same GPU.

SCR=${SCR:-$HOME/scratch}
ENV=${ENV:-$SCR/envs/u2r}
SHARED=${SHARED:-/storage/ice1/0/4/lheck6/CS6604_ECE6256}
SHARED_HF=${SHARED_HF:-$SHARED/hf_cache}
VLLM_ENV=${VLLM_ENV:-$SHARED/envs/CAI_A4}
CONDA_SH=${CONDA_SH:-/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh}

module load anaconda3 2>/dev/null || true
source "$CONDA_SH"
conda activate "$ENV"
export PYTHONNOUSERSITE=1              # ~/.local packages must not shadow the env's
export HF_HOME=$SCR/hf XDG_CACHE_HOME=$SCR/cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "== $(date)  node=$(hostname)  job=${SLURM_JOB_ID:-local}  env=$ENV"
nvidia-smi -L || true

CUSTOMER_MODEL=${CUSTOMER_MODEL:-Qwen/Qwen2.5-7B-Instruct}
AGENT_MODEL=${AGENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
JOB=${SLURM_JOB_ID:-$$}
AGENT_PORT=$((20000 + JOB % 20000))
CUSTOMER_PORT=$((AGENT_PORT + 20000))
AGENT_URL=http://127.0.0.1:$AGENT_PORT/v1
CUSTOMER_URL=http://127.0.0.1:$CUSTOMER_PORT/v1
SERVER_PIDS=()
trap 'kill "${SERVER_PIDS[@]}" 2>/dev/null || true' EXIT

hf_home_for() {   # the shared cache if it has the model, else our own (filled by setup_env.sh)
  local dir="models--${1//\//--}"
  if [ -d "$SHARED_HF/hub/$dir" ] || [ -d "$SHARED_HF/$dir" ]; then echo "$SHARED_HF"; else echo "$SCR/hf"; fi
}

serve() {         # model port gpu_fraction logfile
  local model=$1 port=$2 frac=$3 log=$4 hf
  hf=$(hf_home_for "$model")
  echo "== vLLM $model on :$port (gpu util $frac, HF_HOME=$hf)"
  bash -c "source '$CONDA_SH' && conda activate '$VLLM_ENV' && export PYTHONNOUSERSITE=1 \
           HF_HOME='$hf' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && \
           exec python -m vllm.entrypoints.openai.api_server --model '$model' \
             --host 127.0.0.1 --port $port --dtype bfloat16 --max-model-len 4096 \
             --gpu-memory-utilization $frac --seed 42" > "$log" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  for _ in $(seq 1 240); do                     # loading weights takes a few minutes
    curl -s "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "FATAL: vLLM for $model stopped. End of $log:"; tail -n 40 "$log"; exit 1
    fi
    sleep 5
  done
  if ! curl -s "http://127.0.0.1:$port/v1/models" | grep -q "\"$model\""; then
    echo "FATAL: the server on :$port does not serve $model"; curl -s "http://127.0.0.1:$port/v1/models"; exit 1
  fi
  echo "   ready"
}

start_llms() {    # $1 = folder for the server logs
  local L=${1:-logs} mem frac=0.38
  mkdir -p "$L"
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  # Two 8B models need ~16 GB of weights each, plus Whisper/Kokoro (~3 GB). Below 70 GB
  # only the agent model fits: it then plays the customer too (report as a limitation).
  if [ "${mem:-0}" -lt 70000 ] || [ "${SAME_MODEL:-0}" = 1 ]; then
    echo "!! one model for both customer and agent (GPU ${mem} MiB, SAME_MODEL=${SAME_MODEL:-0})"
    SAME_MODEL=1; frac=0.70
  fi
  # Sequential start: vLLM checks the free GPU memory when it starts.
  serve "$AGENT_MODEL" "$AGENT_PORT" $frac "$L/vllm_agent.log"
  if [ "${SAME_MODEL:-0}" = 1 ]; then
    CUSTOMER_MODEL=$AGENT_MODEL; CUSTOMER_URL=$AGENT_URL
  else
    serve "$CUSTOMER_MODEL" "$CUSTOMER_PORT" $frac "$L/vllm_customer.log"
  fi
  LLM_ARGS="--brain llm --customer-url $CUSTOMER_URL --customer-model $CUSTOMER_MODEL
            --agent-extractor llm --agent-url $AGENT_URL --agent-model $AGENT_MODEL"
}
