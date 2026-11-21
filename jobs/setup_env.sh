#!/bin/bash
# One-time setup on the PACE-ICE LOGIN node (no GPU needed). Run from the repo root:
#     bash jobs/setup_env.sh
# Everything big (conda env, model weights, caches) goes to ~/scratch: the home quota is small.
set -euo pipefail

SCR=${SCR:-$HOME/scratch}
ENV=${ENV:-$SCR/envs/u2r}
SHARED_HF_CACHE="/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache"

mkdir -p "$SCR"/{envs,hf,cache}
export HF_HOME=$SCR/hf
export XDG_CACHE_HOME=$SCR/cache
export PIP_CACHE_DIR=$SCR/cache/pip

module load anaconda3                       # `module avail anaconda` if the name differs
if [ ! -d "$ENV" ]; then
    conda create -y -p "$ENV" python=3.11     # Kokoro needs Python < 3.13
fi
eval "$(conda shell.bash hook)"
conda activate "$ENV"

# vLLM first: it pins the CUDA build of torch that everything else then reuses.
pip install vllm
pip install -r requirements.txt
pip install "misaki[en]" espeakng-loader    # Kokoro's fallback for words it does not know (names!)
pip install -U "huggingface_hub[cli]"       # Ensure `hf` CLI is available
python -m spacy download en_core_web_sm     # misaki's English tokenizer

# Pre-download models so the GPU jobs do not depend on the network.
python - <<'EOF'
import whisper
whisper.load_model("small.en", device="cpu")
from kokoro import KPipeline
for lang in ("a", "b"):
    KPipeline(lang_code=lang, device="cpu")
for v in ["af_heart", "af_bella", "am_michael", "am_adam", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]:
    list(KPipeline(lang_code=v[0], device="cpu")("hello", voice=v))
print("Whisper + Kokoro + 8 voices cached")
EOF

# LLM weights:
# 1) Reuse TA's shared Llama-3.1-8B-Instruct cache on PACE-ICE if available (avoids gated approval & saves quota)
if [ -d "$SHARED_HF_CACHE/hub/models--meta-llama--Llama-3.1-8B-Instruct" ]; then
    mkdir -p "$HF_HOME/hub"
    ln -sfn "$SHARED_HF_CACHE/hub/models--meta-llama--Llama-3.1-8B-Instruct" \
            "$HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct"
    echo "Linked shared Llama-3.1-8B-Instruct from TA cache."
else
    hf auth whoami >/dev/null 2>&1 || hf auth login
    hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"
fi

# 2) Download Qwen2.5-7B-Instruct (ungated) into $SCR/hf
hf download Qwen/Qwen2.5-7B-Instruct

echo
echo "Done. Env: $ENV"
echo "MUSAN (11 GB) if not there yet (use -c to support resume on unstable network):"
echo "  cd $SCR && wget -c https://www.openslr.org/resources/17/musan.tar.gz && tar xzf musan.tar.gz"
echo "  ln -sfn $SCR/musan $(pwd)/datasets/musan"