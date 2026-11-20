#!/bin/bash
# One-time setup on the PACE-ICE LOGIN node (no GPU needed). Run from the repo root:
#     bash jobs/setup_env.sh
# Everything big (conda env, model weights, caches) goes to ~/scratch: the home quota is small.
set -euo pipefail

SCR=${SCR:-$HOME/scratch}
ENV=${ENV:-$SCR/envs/u2r}
mkdir -p "$SCR"/{envs,hf,cache}
export HF_HOME=$SCR/hf XDG_CACHE_HOME=$SCR/cache PIP_CACHE_DIR=$SCR/cache/pip

module load anaconda3                       # `module avail anaconda` if the name differs
if [ ! -d "$ENV" ]; then
  conda create -y -p "$ENV" python=3.11     # Kokoro needs Python < 3.13
fi
source activate "$ENV" 2>/dev/null || conda activate "$ENV"

# vLLM first: it pins the CUDA build of torch that everything else then reuses.
pip install vllm
pip install -r requirements.txt
pip install "misaki[en]" espeakng-loader    # Kokoro's fallback for words it does not know (names!)
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

# LLM weights. Llama is gated: accept the license on huggingface.co first, then log in once.
huggingface-cli whoami >/dev/null 2>&1 || huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

echo
echo "Done. Env: $ENV"
echo "MUSAN (11 GB) if not there yet:"
echo "  cd $SCR && wget https://www.openslr.org/resources/17/musan.tar.gz && tar xzf musan.tar.gz"
echo "  ln -sfn $SCR/musan $(pwd)/datasets/musan"
