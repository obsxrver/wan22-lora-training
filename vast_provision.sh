#!/bin/bash
# Provisioning script for Vast.ai to setup musubi-tuner with parallel installs/downloads
set -euo pipefail

# ---------- helpers ----------
pids=()
wait_all() {
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ $status -ne 0 ]]; then
    echo "One or more parallel tasks failed." >&2
    exit 1
  fi
}

# ---------- clone first (sequential) ----------
cd /workspace
if [[ ! -d musubi-tuner ]]; then
  git clone --recursive https://github.com/kohya-ss/musubi-tuner.git
fi
cd musubi-tuner
git fetch --all --tags --prune
git checkout d0a193061a23a51c90664282205d753605a641c1

# ---------- create dirs needed by downloads (sequential, cheap) ----------
mkdir -p models/text_encoders models/vae models/diffusion_models
mkdir -p /workspace/musubi-tuner/dataset

# ---------- venv + minimal tools (sequential to enable parallel later) ----------
PYTHON_BIN="$(command -v python3 || command -v python)"
"$PYTHON_BIN" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install -U pip wheel setuptools
# Ensure huggingface-cli exists before kicking off parallel HF downloads
python -m pip install -U huggingface_hub

# ---------- kick off parallel tasks ----------
# 1) apt packages (runs independent of Python/HF tasks)
(
  sudo apt-get update && \
  sudo apt-get install -y \
    libcudnn8=8.9.7.29-1+cuda12.2 \
    libcudnn8-dev=8.9.7.29-1+cuda12.2 \
    --allow-change-held-packages
) & pids+=($!)

# 2) Python deps (split to leverage parallelism)
(
  # editable install of musubi-tuner
  pip install -e .
) & pids+=($!)

(
  # light deps
  pip install protobuf six accelerate
) & pids+=($!)

(
  # big GPU deps
  pip install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    xformers==0.0.30 \
    --index-url https://download.pytorch.org/whl/cu128
) & pids+=($!)

# 3) Hugging Face model downloads (each in parallel)
(
  huggingface-cli download \
    Wan-AI/Wan2.1-I2V-14B-720P \
    models_t5_umt5-xxl-enc-bf16.pth \
    --local-dir models/text_encoders \
    --local-dir-use-symlinks False
) & pids+=($!)

(
  huggingface-cli download \
    Comfy-Org/Wan_2.1_ComfyUI_repackaged \
    split_files/vae/wan_2.1_vae.safetensors \
    --local-dir models/vae \
    --local-dir-use-symlinks False
) & pids+=($!)

(
  huggingface-cli download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models \
    --local-dir-use-symlinks False
) & pids+=($!)

(
  huggingface-cli download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models \
    --local-dir-use-symlinks False
) & pids+=($!)

# 4) Config/helper file fetches (parallel)
(
  cd /workspace/musubi-tuner/dataset && \
  curl -fsSL "https://raw.githubusercontent.com/obsxrver/wan22-lora-training/refs/heads/main/dataset.toml" -o dataset.toml
) & pids+=($!)

(
  cd /workspace && \
  curl -fsSL "https://raw.githubusercontent.com/AI-Characters/wan22-lora-training/main/train_helper.py" -o train_helper.py && \
  chmod +x train_helper.py
) & pids+=($!)

# ---------- wait for all parallel tasks ----------
wait_all

echo "✅ Setup complete."
