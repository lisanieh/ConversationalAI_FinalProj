#!/bin/bash
# Sourced by the sbatch files: activate the env, point caches to scratch, and
# optionally start the two LLM servers (customer = Qwen on :8000, agent = Llama on :8001).

SCR=${SCR:-$HOME/scratch}
ENV=${ENV:-$SCR/envs/u2r}
export HF_HOME=$SCR/hf XDG_CACHE_HOME=$SCR/cache HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "== $(date)  host=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CUSTOMER_MODEL=${CUSTOMER_MODEL:-Qwen/Qwen2.5-7B-Instruct}
AGENT_MODEL=${AGENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
CUSTOMER_URL=http://localhost:8000/v1
AGENT_URL=http://localhost:8001/v1

_wait_for() {   # url name logfile
  for _ in $(seq 1 90); do
    curl -s "$1/models" >/dev/null && { echo "   $2 ready"; return 0; }
    sleep 10
  done
  echo "!! $2 did not start; last lines of $3:"; tail -30 "$3"; exit 1
}

start_llms() {  # $1 = folder for the server logs
  local L=${1:-logs}; mkdir -p "$L"
  local mem; mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  # Two 8B models need ~16 GB of weights each, plus Whisper/Kokoro (~3 GB).
  # On a 40-48 GB card only the agent model fits: the customer then uses it too
  # (one model plays both sides; report that as a limitation).
  if [ "${mem:-0}" -lt 70000 ]; then
    echo "!! GPU has ${mem} MiB (< 70 GB): serving one model for both customer and agent"
    SAME_MODEL=1
  fi
  local frac=0.38; [ "${SAME_MODEL:-0}" = 1 ] && frac=0.70
  echo "== starting agent LLM $AGENT_MODEL (util $frac)"
  vllm serve "$AGENT_MODEL" --port 8001 --dtype bfloat16 --max-model-len 4096 \
       --gpu-memory-utilization $frac --seed 0 > "$L/vllm_agent.log" 2>&1 &
  _wait_for $AGENT_URL agent "$L/vllm_agent.log"       # sequential: vLLM checks free memory at start
  if [ "${SAME_MODEL:-0}" = 1 ]; then
    CUSTOMER_MODEL=$AGENT_MODEL; CUSTOMER_URL=$AGENT_URL
  else
    echo "== starting customer LLM $CUSTOMER_MODEL (util $frac)"
    vllm serve "$CUSTOMER_MODEL" --port 8000 --dtype bfloat16 --max-model-len 4096 \
         --gpu-memory-utilization $frac --seed 0 > "$L/vllm_customer.log" 2>&1 &
    _wait_for $CUSTOMER_URL customer "$L/vllm_customer.log"
  fi
  trap 'kill $(jobs -p) 2>/dev/null' EXIT
  LLM_ARGS="--brain llm --customer-url $CUSTOMER_URL --customer-model $CUSTOMER_MODEL
            --agent-extractor llm --agent-url $AGENT_URL --agent-model $AGENT_MODEL"
}
