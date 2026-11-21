#!/bin/bash
# One-time setup on the PACE-ICE LOGIN node (no GPU needed). Run from the repo root:
#     bash jobs/setup_env.sh
#
# vLLM and the Llama weights come from the course (shared env + shared HF cache, as in
# the CAI_A4 template), so this only builds a small env for the pipeline itself
# (Kokoro, Whisper, error_scripts) and caches what the shared cache does not have.
# Everything goes to ~/scratch: the home quota is small.
set -euo pipefail

SCR=${SCR:-$HOME/scratch}
ENV=${ENV:-$SCR/envs/u2r}
SHARED=${SHARED:-/storage/ice1/0/4/lheck6/CS6604_ECE6256}
SHARED_HF=$SHARED/hf_cache
CONDA_SH=${CONDA_SH:-/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh}
mkdir -p "$SCR"/{envs,hf,cache}
export HF_HOME=$SCR/hf XDG_CACHE_HOME=$SCR/cache PIP_CACHE_DIR=$SCR/cache/pip PYTHONNOUSERSITE=1

module load anaconda3 2>/dev/null || true
source "$CONDA_SH"
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.11      # Kokoro needs Python < 3.13
conda activate "$ENV"

pip install torch                                           # CUDA build (Linux default wheel)
pip install -r requirements.txt
pip install "misaki[en]" espeakng-loader                    # Kokoro's fallback for unknown words (names!)
python -m spacy download en_core_web_sm                     # misaki's English tokenizer

# Pre-download Whisper, Kokoro and the 8 customer voices, so jobs run offline.
python - <<'EOF'
import whisper
whisper.load_model("small.en", device="cpu")
from kokoro import KPipeline
pipes = {lang: KPipeline(lang_code=lang, device="cpu") for lang in ("a", "b")}
for v in ["af_heart", "af_bella", "am_michael", "am_adam", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]:
    list(pipes[v[0]]("hello", voice=v))
print("Whisper small.en + Kokoro + 8 voices cached")
EOF

# LLM weights: use the shared cache when it has them, otherwise download to ~/scratch/hf.
have() { [ -d "$SHARED_HF/hub/models--${1//\//--}" ] || [ -d "$SHARED_HF/models--${1//\//--}" ]; }
for m in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  if have "$m"; then
    echo "$m: in the shared course cache"
  else
    echo "$m: not in the shared cache, downloading to $HF_HOME"
    # Python API, not the CLI: huggingface-cli was renamed to `hf` in newer huggingface_hub.
    # Llama is gated: accept the license on huggingface.co and set HF_TOKEN first.
    python -c "from huggingface_hub import snapshot_download as d; print(d('$m', ignore_patterns=['original/*']))"
  fi
done

echo
echo "Done. Pipeline env: $ENV   (vLLM: shared course env)"
echo "MUSAN (11 GB) if not there yet:"
echo "  cd $SCR && wget https://www.openslr.org/resources/17/musan.tar.gz && tar xzf musan.tar.gz"
echo "  ln -sfn $SCR/musan $(pwd)/datasets/musan"
