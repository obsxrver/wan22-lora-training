#!/usr/bin/env bash
set -euo pipefail

# Simple WAN2.2 LoRA training runner
# - Prompts for title suffix, author, and dataset path (with sensible defaults)
# - Caches latents and text encoder outputs
# - Trains HIGH noise and LOW noise models
# - If 2+ GPUs are free, runs them concurrently; otherwise waits for a free GPU

MUSUBI_DIR="/workspace/musubi-tuner"
PYTHON="$MUSUBI_DIR/venv/bin/python"
ACCELERATE="$MUSUBI_DIR/venv/bin/accelerate"

VAE="$MUSUBI_DIR/models/vae/split_files/vae/wan_2.1_vae.safetensors"
T5="$MUSUBI_DIR/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth"
HIGH_DIT="$MUSUBI_DIR/models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
LOW_DIT="$MUSUBI_DIR/models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
DEFAULT_DATASET="$MUSUBI_DIR/dataset/dataset.toml"

require() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

ensure_accelerate_default() {
  local cfg="$HOME/.cache/huggingface/accelerate/default_config.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "No accelerate default config found; creating one..."
    "$ACCELERATE" config default
  fi
}

is_gpu_free() {
  local idx="$1"
  # If no processes are listed for this GPU, consider it free
  local procs
  procs=$(nvidia-smi -i "$idx" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E "[0-9]" || true)
  if [[ -z "$procs" ]]; then
    return 0
  else
    return 1
  fi
}

wait_for_free_gpu() {
  local excluded="${1:-}"
  while true; do
    local all_idxs
    all_idxs=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
    if [[ -z "$all_idxs" ]]; then
      echo "No NVIDIA GPUs detected (nvidia-smi returned nothing)." >&2
      exit 1
    fi
    for idx in $all_idxs; do
      # skip excluded ids (comma- or space-separated)
      if [[ -n "$excluded" ]] && [[ ",$excluded," == *",$idx,"* ]]; then
        continue
      fi
      if is_gpu_free "$idx"; then
        echo "$idx"
        return 0
      fi
    done
    sleep 10
  done
}

get_free_port() {
  python3 - "$@" <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

main() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required but not found in PATH." >&2
    exit 1
  fi

  # Prompt inputs with defaults
  echo "WAN2.2 LoRA simple runner"
  read -r -p "Title suffix (default: mylora): " TITLE_SUFFIX || true
  TITLE_SUFFIX=${TITLE_SUFFIX:-mylora}

  read -r -p "Author (default: authorName): " AUTHOR || true
  AUTHOR=${AUTHOR:-authorName}

  read -r -p "Dataset path (default: $DEFAULT_DATASET): " DATASET || true
  DATASET=${DATASET:-$DEFAULT_DATASET}

  HIGH_TITLE="WAN2.2-HighNoise_${TITLE_SUFFIX}"
  LOW_TITLE="WAN2.2-LowNoise_${TITLE_SUFFIX}"

  echo "Using:"
  echo "  Dataset: $DATASET"
  echo "  High title: $HIGH_TITLE"
  echo "  Low title:  $LOW_TITLE"
  echo "  Author:     $AUTHOR"

  # Validate required files
  require "$PYTHON"
  require "$ACCELERATE"
  require "$VAE"
  require "$T5"
  require "$HIGH_DIT"
  require "$LOW_DIT"
  require "$DATASET"

  cd "$MUSUBI_DIR"

  ensure_accelerate_default

  echo "Caching latents..."
  "$PYTHON" src/musubi_tuner/wan_cache_latents.py \
    --dataset_config "$DATASET" \
    --vae "$VAE"

  echo "Caching text encoder outputs..."
  "$PYTHON" src/musubi_tuner/wan_cache_text_encoder_outputs.py \
    --dataset_config "$DATASET" \
    --t5 "$T5"

  # Allocate distinct rendezvous ports to prevent EADDRINUSE
  HIGH_PORT=$(get_free_port)
  LOW_PORT=$(get_free_port)
  if [[ "$LOW_PORT" == "$HIGH_PORT" ]]; then
    LOW_PORT=$(get_free_port)
  fi

  echo "Waiting for a free GPU for HIGH noise training..."
  HIGH_GPU=$(wait_for_free_gpu)
  echo "Starting HIGH on GPU $HIGH_GPU (port $HIGH_PORT) -> run_high.log"
  MASTER_ADDR=127.0.0.1 MASTER_PORT="$HIGH_PORT" CUDA_VISIBLE_DEVICES="$HIGH_GPU" \
  "$ACCELERATE" launch --num_cpu_threads_per_process 1 --num_processes 1 --main_process_port "$HIGH_PORT" src/musubi_tuner/wan_train_network.py \
    --task t2v-A14B \
    --dit "$HIGH_DIT" \
    --vae "$VAE" \
    --t5 "$T5" \
    --dataset_config "$DATASET" \
    --xformers \
    --mixed_precision fp16 \
    --fp8_base \
    --optimizer_type adamw \
    --learning_rate 3e-4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --max_data_loader_n_workers 2 \
    --network_module networks.lora_wan \
    --network_dim 16 \
    --network_alpha 16 \
    --timestep_sampling shift \
    --discrete_flow_shift 1.0 \
    --max_train_epochs 100 \
    --save_every_n_epochs 100 \
    --seed 5 \
    --optimizer_args weight_decay=0.1 \
    --max_grad_norm 0 \
    --lr_scheduler polynomial \
    --lr_scheduler_power 8 \
    --lr_scheduler_min_lr_ratio=5e-5 \
    --output_dir "$MUSUBI_DIR/output" \
    --output_name "$HIGH_TITLE" \
    --metadata_title "$HIGH_TITLE" \
    --metadata_author "$AUTHOR" \
    --preserve_distribution_shape \
    --min_timestep 875 \
    --max_timestep 1000 \
    > "$PWD/run_high.log" 2>&1 &
  HIGH_PID=$!

  # Determine GPU count to decide if LOW can reuse the same GPU (single-GPU sequential case)
  GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
  echo "Waiting for a free GPU for LOW noise training..."
  if [[ "$GPU_COUNT" -gt 1 ]]; then
    LOW_GPU=$(wait_for_free_gpu "$HIGH_GPU")
  else
    # Single GPU: allow reuse of the same GPU after it becomes free (sequential)
    LOW_GPU=$(wait_for_free_gpu)
  fi
  echo "Starting LOW on GPU $LOW_GPU (port $LOW_PORT) -> run_low.log"
  MASTER_ADDR=127.0.0.1 MASTER_PORT="$LOW_PORT" CUDA_VISIBLE_DEVICES="$LOW_GPU" \
  "$ACCELERATE" launch --num_cpu_threads_per_process 1 --num_processes 1 --main_process_port "$LOW_PORT" src/musubi_tuner/wan_train_network.py \
    --task t2v-A14B \
    --dit "$LOW_DIT" \
    --vae "$VAE" \
    --t5 "$T5" \
    --dataset_config "$DATASET" \
    --xformers \
    --mixed_precision fp16 \
    --fp8_base \
    --optimizer_type adamw \
    --learning_rate 3e-4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --max_data_loader_n_workers 2 \
    --network_module networks.lora_wan \
    --network_dim 16 \
    --network_alpha 16 \
    --timestep_sampling shift \
    --discrete_flow_shift 1.0 \
    --max_train_epochs 100 \
    --save_every_n_epochs 100 \
    --seed 5 \
    --optimizer_args weight_decay=0.1 \
    --max_grad_norm 0 \
    --lr_scheduler polynomial \
    --lr_scheduler_power 8 \
    --lr_scheduler_min_lr_ratio=5e-5 \
    --output_dir "$MUSUBI_DIR/output" \
    --output_name "$LOW_TITLE" \
    --metadata_title "$LOW_TITLE" \
    --metadata_author "$AUTHOR" \
    --preserve_distribution_shape \
    --min_timestep 0 \
    --max_timestep 875 \
    > "$PWD/run_low.log" 2>&1 &
  LOW_PID=$!

  echo "HIGH PID: $HIGH_PID (GPU $HIGH_GPU), log: $PWD/run_high.log"
  echo "LOW  PID: $LOW_PID (GPU $LOW_GPU), log: $PWD/run_low.log"

  echo "Waiting for both trainings to finish..."
  wait "$HIGH_PID"
  wait "$LOW_PID"
  echo "✅ All done."
}

main "$@" 