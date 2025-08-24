#!/bin/bash
# Provisioning script for Vast.ai to setup musubi-tuner environment and download models
set -e

# clone musubi-tuner repository if not already present
cd /workspace
if [ ! -d musubi-tuner ]; then
  git clone --recursive https://github.com/kohya-ss/musubi-tuner.git
fi
cd musubi-tuner
# checkout the required branch and commit
if git rev-parse --verify feature-wan-2-2 >/dev/null 2>&1; then
  git checkout feature-wan-2-2
fi
git checkout d0a193061a23a51c90664282205d753605a641c1

# install system dependencies
apt-get update
apt-get install -y libcudnn8=8.9.7.29-1+cuda12.2 libcudnn8-dev=8.9.7.29-1+cuda12.2 --allow-change-held-packages

# create and activate virtual environment using available python
PYTHON_BIN=$(command -v python3 || command -v python)
"$PYTHON_BIN" -m venv venv
source venv/bin/activate

# install python dependencies
pip install -e .
pip install protobuf six accelerate huggingface_hub
pip install torch==2.7.0 torchvision==0.22.0 xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128

# download required models
mkdir -p models/text_encoders models/vae models/diffusion_models
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P models_t5_umt5-xxl-enc-bf16.pth --local-dir models/text_encoders
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/vae/wan_2.1_vae.safetensors --local-dir models/vae
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors --local-dir models/diffusion_models
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors --local-dir models/diffusion_models

# fetch the helper script to /workspace
cd /workspace
HELPER_URL="https://raw.githubusercontent.com/AI-Characters/wan22-lora-training/main/train_helper.py"
curl -L "$HELPER_URL" -o train_helper.py
chmod +x train_helper.py

