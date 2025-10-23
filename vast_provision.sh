#!/bin/bash
# Provisioning script for Vast.ai to setup musubi-tuner with correct install order and parallel model downloads
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

# ---------- prep (sequential) ----------
cd /workspace
if [[ ! -d musubi-tuner ]]; then
  git clone --recursive https://github.com/kohya-ss/musubi-tuner.git
fi
cd musubi-tuner
git fetch --all --tags --prune
#git checkout d0a193061a23a51c90664282205d753605a641c1

# directories for datasets and models
mkdir -p models/text_encoders models/vae models/diffusion_models
mkdir -p /workspace/musubi-tuner/dataset

# fetch dataset config and training helper ahead of parallel tasks
curl -fsSL "https://raw.githubusercontent.com/obsxrver/wan22-lora-training/main/dataset.toml" -o /workspace/musubi-tuner/dataset/dataset.toml
curl -fsSL "https://raw.githubusercontent.com/obsxrver/wan22-lora-training/main/run_wan_training.sh" -o /workspace/run_wan_training.sh
curl -fsSL "https://raw.githubusercontent.com/obsxrver/wan22-lora-training/main/analyze_training_logs.py" -o /workspace/analyze_training_logs.py
chmod +x /workspace/run_wan_training.sh
chmod +x /workspace/analyze_training_logs.py

# ensure huggingface-cli exists for downloads (system Python, outside venv)
python3 -m pip install -U huggingface_hub

# install vastai CLI for instance management and cloud storage
# Handle system package conflicts gracefully
python3 -m pip install -U vastai --break-system-packages || {
  echo "Warning: vastai installation had conflicts, trying alternative approach..."
  python3 -m pip install vastai --user --break-system-packages
}

# Set up vastai API key - prefer VASTAI_KEY, fallback to CONTAINER_API_KEY
if [[ -n "${VASTAI_KEY:-}" ]]; then
  echo "Setting up Vast.ai API key from VASTAI_KEY..."
  vastai set api-key "$VASTAI_KEY" || echo "Warning: Failed to set vastai API key"
elif [[ -n "${CONTAINER_API_KEY:-}" ]]; then
  echo "Setting up Vast.ai API key from CONTAINER_API_KEY..."
  vastai set api-key "$CONTAINER_API_KEY" || echo "Warning: Failed to set vastai API key"
else
  echo "No VASTAI_KEY or CONTAINER_API_KEY found. You'll need to set the API key manually:"
  echo "  vastai set api-key YOUR_API_KEY"
fi

# ---------- parallel tasks ----------
# Task 1: Install dependencies in the exact order from README
(
  set -euo pipefail
  cd /workspace/musubi-tuner

  sudo apt-get update
  #sudo apt-get install -y \
  ##  libcudnn8=8.9.7.29-1+cuda12.2 \
    #libcudnn8-dev=8.9.7.29-1+cuda12.2 \
    #--allow-change-held-packages
  #above should already be installecd on the image. 

  python3 -m venv venv
  # shellcheck disable=SC1091
  source venv/bin/activate

  pip install -e .
  pip install protobuf
  pip install six
  pip install matplotlib
  pip install torch==2.8.0 torchvision xformers --index-url https://download.pytorch.org/whl/cu128
) & pids+=($!)

# Task 2: Download all four models concurrently
(
  set -euo pipefail
  cd /workspace/musubi-tuner

  mkdir -p models/text_encoders models/vae models/diffusion_models

  huggingface-cli download \
    Wan-AI/Wan2.1-I2V-14B-720P \
    models_t5_umt5-xxl-enc-bf16.pth \
    --local-dir models/text_encoders \
    --local-dir-use-symlinks False &

  huggingface-cli download \
    Comfy-Org/Wan_2.1_ComfyUI_repackaged \
    split_files/vae/wan_2.1_vae.safetensors \
    --local-dir models/vae \
    --local-dir-use-symlinks False &

  huggingface-cli download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models \
    --local-dir-use-symlinks False &

  huggingface-cli download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models \
    --local-dir-use-symlinks False &

  wait
) & pids+=($!)

# ---------- wait for both tasks ----------
wait_all

echo "✅ Setup complete."
