#!/bin/bash
# Provisioning script for Vast.ai to setup musubi-tuner and the training webui.
# Verified on  vastai/pytorch:cuda-12.9.1-auto
# For use with vastai/pytorch:latest docker image
set -euo pipefail
source /venv/main/bin/activate
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

cd /workspace
if [[ ! -d wan22-lora-training ]]; then
  git clone https://github.com/obsxrver/wan22-lora-training.git
fi
if [[ -n "${DEV:-}" ]]; then
  git checkout dev
fi
if [[ ! -d musubi-tuner ]]; then
  git clone --recursive https://github.com/kohya-ss/musubi-tuner.git
fi
cd musubi-tuner
git fetch --all --tags --prune

mkdir -p models/text_encoders models/vae models/diffusion_models
mkdir -p /workspace/musubi-tuner/dataset

curl -fsSL "https://raw.githubusercontent.com/obsxrver/wan22-lora-training/main/dataset.toml" -o /workspace/musubi-tuner/dataset/dataset.toml
curl -fsSL "https://raw.githubusercontent.com/obsxrver/wan22-lora-training/main/run_wan_training.sh" -o /workspace/run_wan_training.sh
curl -fsSL "https://raw.githubusercontent.com/obsxrver/wan22-lora-training/main/analyze_training_logs.py" -o /workspace/analyze_training_logs.py
chmod +x /workspace/run_wan_training.sh
chmod +x /workspace/analyze_training_logs.py

pip install -U "huggingface_hub>=0.20.0" --break-system-packages || \
pip install -U huggingface_hub --break-system-packages || \
pip install -U huggingface_hub

#fix bug vastai introduced in latest image
#TODO check if bug is patched and remove
/usr/bin/python3 -m pip install rich

if ! command -v vastai >/dev/null 2>&1; then
  pip install vastai
fi


if [[ -n "${VASTAI_KEY:-}" ]]; then
  echo "Setting up Vast.ai API key from VASTAI_KEY..."
  vastai set api-key "$VASTAI_KEY" || echo "Warning: Failed to set vastai API key"
fi

(
  set -euo pipefail
  cd /workspace/musubi-tuner

  sudo apt-get update -y
  
  pip install -e .
  pip install protobuf six matplotlib fastapi "uvicorn[standard]" python-multipart tomli torch torchvision
) & pids+=($!)


(
  set -euo pipefail
  cd /workspace/musubi-tuner

  mkdir -p models/text_encoders models/vae models/diffusion_models

  hf download \
    Wan-AI/Wan2.1-I2V-14B-720P \
    models_t5_umt5-xxl-enc-bf16.pth \
    --local-dir models/text_encoders &

  hf download \
    Comfy-Org/Wan_2.1_ComfyUI_repackaged \
    split_files/vae/wan_2.1_vae.safetensors \
    --local-dir models/vae &

  hf download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models &

  hf download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models &

  hf download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models &

  hf download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models &

  wait
) & pids+=($!)

# ---------- wait for both tasks ----------
wait_all

WEBUI_PORT=7865
cat <<'EOF' >/workspace/wan22-lora-training/start_wan_webui.sh
#!/bin/bash
set -euo pipefail
WEBUI_PORT="${WEBUI_PORT:-7865}"
cd /workspace/wan22-lora-training
source /venv/main/bin/activate
exec uvicorn webui.server:app --host 0.0.0.0 --port "${WEBUI_PORT}"
EOF
chmod +x /workspace/wan22-lora-training/start_wan_webui.sh

if command -v supervisorctl >/dev/null 2>&1; then
  sudo tee /etc/supervisor/conf.d/wan-training-webui.conf >/dev/null <<'EOF'
[program:wan-training-webui]
command=/bin/bash /workspace/wan22-lora-training/start_wan_webui.sh
directory=/workspace/wan22-lora-training
autostart=true
autorestart=true
stdout_logfile=/workspace/wan-training-webui.out.log
stderr_logfile=/workspace/wan-training-webui.err.log
stopasgroup=true
killasgroup=true
environment=PYTHONUNBUFFERED=1
EOF
  sudo supervisorctl reread || true
  sudo supervisorctl update || true
fi

PORTAL_ENTRY="0.0.0.0:${WEBUI_PORT}:${WEBUI_PORT}:/:WAN Training UI"
if [[ -n "${PORTAL_CONFIG:-}" ]]; then
  case "${PORTAL_CONFIG}" in
    *"${PORTAL_ENTRY}"*) ;;
    *) PORTAL_CONFIG="${PORTAL_CONFIG}|${PORTAL_ENTRY}" ;;
  esac
else
  PORTAL_CONFIG="${PORTAL_ENTRY}"
fi
export PORTAL_CONFIG

sudo tee /etc/profile.d/wan_portal.sh >/dev/null <<EOF
export PORTAL_CONFIG="${PORTAL_CONFIG}"
EOF

if command -v supervisorctl >/dev/null 2>&1; then
  sudo supervisorctl restart instance_portal || true
fi

echo "✅ Setup complete."
