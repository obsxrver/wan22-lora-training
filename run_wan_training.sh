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

check_low_vram() {
  # Get VRAM in MB for first GPU (assuming all GPUs are identical)
  local vram_mb
  vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [[ -z "$vram_mb" || ! "$vram_mb" =~ ^[0-9]+$ ]]; then
    echo "Warning: Could not detect GPU VRAM, defaulting to xformers" >&2
    return 1
  fi
  
  local vram_gb=$((vram_mb / 1024))
  echo "Detected GPU VRAM: ${vram_gb}GB" >&2
  
  if [[ "$vram_gb" -lt 33 ]]; then
    return 0  # Low VRAM
  else
    return 1  # High VRAM
  fi
}
# block swap on anything 32GB or less VRAM
determine_attention_flags() {
  if check_low_vram; then
    echo "--sdpa --blocks_to_swap 10"
  else
    echo "--sdpa"
  fi
}

get_cpu_threads() {
  local threads
  threads=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "8")
  echo "$threads"
}

prompt_for_valid_api_key() {
  echo ""
  echo "Your Vast.ai API key is invalid or expired."
  echo "To get a valid API key with full permissions:"
  echo "  1. Go to https://cloud.vast.ai/manage-keys/"
  echo "  2. Create a new API key"
  echo "  3. Enter it below"
  echo ""
  
  while true; do
    read -r -p "Enter your Vast.ai API key (or 'skip' to continue without cloud features): " api_key
    
    if [[ "$api_key" == "skip" ]]; then
      echo "Skipping API key setup. Cloud features will be disabled."
      return 1
    fi
    
    if [[ -z "$api_key" ]]; then
      echo "Please enter a valid API key or 'skip'."
      continue
    fi
    
    # Set the API key
    if vastai set api-key "$api_key"; then
      # Test if it works
      local output
      output=$(vastai show connections 2>&1)
      
      if echo "$output" | grep -q "failed with error 401: Authentication required"; then
        echo "API key is still invalid. Please try again."
        continue
      else
        echo "✅ API key set successfully!"
        return 0
      fi
    else
      echo "Failed to set API key. Please try again."
    fi
  done
}

check_cloud_configured() {
  # Check if Vast.ai cloud connections are configured
  if ! command -v vastai >/dev/null 2>&1; then
    echo "vastai CLI not found. Try: pip install vastai --user --break-system-packages" >&2
    return 1
  fi
  
  # Check if API key is valid by testing vastai show connections
  local output
  output=$(vastai show connections 2>&1)
  
  # Check for authentication error (401)
  if echo "$output" | grep -q "failed with error 401: Authentication required"; then
    echo "Current API key is invalid or expired." >&2
    prompt_for_valid_api_key
    return $?
  fi
  
  # Check if there are any cloud connections (skip header and URL lines)
  local connections
  connections=$(echo "$output" | grep -v "^ID" | grep -v "^https://" | head -1)
  if [[ -n "$connections" ]]; then
    return 0
  fi
  return 1
}

setup_vast_api_key() {
  # Set up Vast.ai API key for instance management
  if [[ -z "${CONTAINER_ID:-}" ]]; then
    echo "Warning: CONTAINER_ID not found. Cannot set up instance shutdown." >&2
    return 1
  fi
  
  # Check if API key is valid
  local output
  output=$(vastai show connections 2>&1)
  
  if echo "$output" | grep -q "failed with error 401: Authentication required"; then
    echo "Current API key is invalid for instance management."
    prompt_for_valid_api_key
    return $?
  else
    echo "Vast.ai API key is valid for instance management."
    return 0
  fi
}

upload_to_cloud() {
  local lora_path="$1"
  local lora_name="$2"
  
  if ! check_cloud_configured; then
    echo "No cloud connections configured in Vast.ai. Skipping upload." >&2
    return 1
  fi
  
  # Get the first available cloud connection
  local connection_id
  connection_id=$(vastai show connections 2>/dev/null | grep -v "^ID" | grep -v "^https://" | head -1 | awk '{print $1}')
  
  if [[ -z "$connection_id" ]]; then
    echo "No cloud connection ID found. Skipping upload." >&2
    return 1
  fi
  
  if [[ -z "${CONTAINER_ID:-}" ]]; then
    echo "Warning: CONTAINER_ID not found. Cannot upload to cloud." >&2
    return 1
  fi
  
  echo "Uploading $lora_name to cloud storage (connection: $connection_id)..."
  
  # Use vastai cloud copy to upload to cloud storage
  # Format: vastai cloud copy --src <src> --dst <dst> --instance <instance_id> --connection <connection_id> --transfer "Instance to Cloud"
  if vastai cloud copy --src "$lora_path" --dst "/loras/WAN/$lora_name" --instance "$CONTAINER_ID" --connection "$connection_id" --transfer "Instance to Cloud"; then
    echo "✅ Successfully uploaded $lora_name to cloud storage"
    return 0
  else
    echo "❌ Failed to upload $lora_name to cloud storage" >&2
    return 1
  fi
}

shutdown_instance() {
  if [[ -z "${CONTAINER_ID:-}" ]]; then
    echo "Warning: CONTAINER_ID not found. Cannot shutdown instance." >&2
    return 1
  fi
  
  if ! command -v vastai >/dev/null 2>&1; then
    echo "Warning: vastai CLI not found. Cannot shutdown instance." >&2
    return 1
  fi
  
  echo "Shutting down Vast.ai instance $CONTAINER_ID..."
  if vastai stop instance "$CONTAINER_ID"; then
    echo "✅ Instance shutdown initiated"
    return 0
  else
    echo "❌ Failed to shutdown instance" >&2
    return 1
  fi
}

calculate_cpu_params() {
  local threads
  threads=$(get_cpu_threads)
  local cpu_threads_per_process=$((threads / 4))
  local max_data_loader_workers=$((threads / 8))
  
  # Ensure minimum values
  if [[ "$cpu_threads_per_process" -lt 1 ]]; then
    cpu_threads_per_process=1
  fi
  if [[ "$max_data_loader_workers" -lt 1 ]]; then
    max_data_loader_workers=1
  fi
  
  echo "Detected $threads CPU threads" >&2
  echo "Setting --num_cpu_threads_per_process=$cpu_threads_per_process" >&2
  echo "Setting --max_data_loader_n_workers=$max_data_loader_workers" >&2
  
  echo "$cpu_threads_per_process $max_data_loader_workers"
}

prompt_or_env() {
  local var_name="$1"
  local prompt_text="$2"
  local default_value="$3"
  local env_key="$4"

  local env_value="${!env_key:-}"
  if [[ -n "$env_value" ]]; then
    printf -v "$var_name" '%s' "$env_value"
    echo "$prompt_text$env_value (from $env_key)"
    return
  fi

  if [[ -n "${WAN_NON_INTERACTIVE:-}" ]]; then
    printf -v "$var_name" '%s' "$default_value"
    echo "$prompt_text$default_value (default via WAN_NON_INTERACTIVE)"
    return
  fi

  read -r -p "$prompt_text" user_input || true
  if [[ -z "$user_input" ]]; then
    user_input="$default_value"
  fi
  printf -v "$var_name" '%s' "$user_input"
}

confirm_or_env() {
  local var_name="$1"
  local prompt_text="$2"
  local default_value="$3"
  local env_key="$4"

  local env_value="${!env_key:-}"
  if [[ -n "$env_value" ]]; then
    printf -v "$var_name" '%s' "$env_value"
    echo "$prompt_text$env_value (from $env_key)"
    return
  fi

  if [[ -n "${WAN_NON_INTERACTIVE:-}" ]]; then
    printf -v "$var_name" '%s' "$default_value"
    echo "$prompt_text$default_value (default via WAN_NON_INTERACTIVE)"
    return
  fi

  read -r -p "$prompt_text" user_input || true
  if [[ -z "$user_input" ]]; then
    user_input="$default_value"
  fi
  printf -v "$var_name" '%s' "$user_input"
}

main() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required but not found in PATH." >&2
    exit 1
  fi

  # Prompt inputs with defaults
  echo "WAN2.2 LoRA simple runner"
  prompt_or_env TITLE_SUFFIX "Title suffix (default: mylora): " "mylora" "WAN_TITLE_SUFFIX"

  prompt_or_env AUTHOR "Author (default: authorName): " "authorName" "WAN_AUTHOR"

  prompt_or_env DATASET "Dataset path (default: $DEFAULT_DATASET): " "$DEFAULT_DATASET" "WAN_DATASET"

  if [[ ! -f "$DATASET" ]]; then
    echo "Dataset config not found at $DATASET; downloading..."
    mkdir -p "$(dirname "$DATASET")"
    curl -fsSL "https://raw.githubusercontent.com/obsxrver/wan22-lora-training/main/dataset.toml" -o "$DATASET" || echo "Failed to download dataset.toml" >&2
  fi

  prompt_or_env SAVE_EVERY "Save every N epochs (default: 100): " "100" "WAN_SAVE_EVERY"

  CPU_PARAMS=($(calculate_cpu_params))
  DEFAULT_CPU_THREADS_PER_PROCESS=${CPU_PARAMS[0]}
  DEFAULT_MAX_DATA_LOADER_WORKERS=${CPU_PARAMS[1]}

  echo ""
  prompt_or_env CPU_THREADS_PER_PROCESS "CPU threads per process (default: $DEFAULT_CPU_THREADS_PER_PROCESS): " "$DEFAULT_CPU_THREADS_PER_PROCESS" "WAN_CPU_THREADS_PER_PROCESS"

  prompt_or_env MAX_DATA_LOADER_WORKERS "Max data loader workers (default: $DEFAULT_MAX_DATA_LOADER_WORKERS): " "$DEFAULT_MAX_DATA_LOADER_WORKERS" "WAN_MAX_DATA_LOADER_WORKERS"

  HIGH_TITLE="WAN2.2-HighNoise_${TITLE_SUFFIX}"
  LOW_TITLE="WAN2.2-LowNoise_${TITLE_SUFFIX}"

  echo ""
  echo "=== Post-Training Options ==="

  # Check for cloud storage upload option
  UPLOAD_CLOUD="Y"
  if check_cloud_configured; then
    echo "Cloud storage is configured in Vast.ai."
    confirm_or_env UPLOAD_CLOUD "Upload LoRAs to cloud storage after training? [Y/n]: " "Y" "WAN_UPLOAD_CLOUD"
  else
    echo "No cloud connections configured. To set up:"
    echo "  1. Install vastai CLI if missing: pip install vastai --user --break-system-packages"
    echo "  2. Go to Vast.ai Console > Cloud Connections"
    echo "  3. Add a connection to Google Drive, AWS S3, or other cloud provider"
    echo "  4. Follow the authentication steps"
    confirm_or_env UPLOAD_CLOUD "Upload LoRAs to cloud storage after training? [Y/n]: " "Y" "WAN_UPLOAD_CLOUD"
  fi

  # Check for instance shutdown option
  SHUTDOWN_INSTANCE="Y"
  if [[ -n "${CONTAINER_ID:-}" ]] && command -v vastai >/dev/null 2>&1; then
    echo "Vast.ai instance management available."
    confirm_or_env SHUTDOWN_INSTANCE "Shut down this instance after training to save costs? [Y/n]: " "Y" "WAN_SHUTDOWN_INSTANCE"
  else
    echo "Vast.ai CLI not available or not running on Vast.ai instance."
    confirm_or_env SHUTDOWN_INSTANCE "Shut down this instance after training to save costs? [Y/n]: " "Y" "WAN_SHUTDOWN_INSTANCE"
  fi

  echo ""
  echo "=== Configuration Summary ==="
  echo "  Dataset: $DATASET"
  echo "  High title: $HIGH_TITLE"
  echo "  Low title:  $LOW_TITLE"
  echo "  Author:     $AUTHOR"
  echo "  Save every: $SAVE_EVERY epochs"
  echo "  Upload to cloud: $UPLOAD_CLOUD"
  echo "  Auto-shutdown: $SHUTDOWN_INSTANCE"
  echo ""
  confirm_or_env PROCEED "Proceed with training? [Y/n]: " "Y" "WAN_PROCEED"
  if [[ ! "$PROCEED" =~ ^[Yy]?$ ]]; then
    echo "Training cancelled."
    exit 0
  fi

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

  ATTN_FLAGS=$(determine_attention_flags)
  echo "Using attention flags: $ATTN_FLAGS"

  echo "Using CPU parameters:"
  echo "  --num_cpu_threads_per_process: $CPU_THREADS_PER_PROCESS"
  echo "  --max_data_loader_n_workers: $MAX_DATA_LOADER_WORKERS"

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
  "$ACCELERATE" launch --num_cpu_threads_per_process "$CPU_THREADS_PER_PROCESS" --num_processes 1 --main_process_port "$HIGH_PORT" src/musubi_tuner/wan_train_network.py \
    --task t2v-A14B \
    --dit "$HIGH_DIT" \
    --vae "$VAE" \
    --t5 "$T5" \
    --dataset_config "$DATASET" \
    $ATTN_FLAGS \
    --mixed_precision fp16 \
    --fp8_base \
    --optimizer_type adamw \
    --learning_rate 3e-4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --max_data_loader_n_workers "$MAX_DATA_LOADER_WORKERS" \
    --network_module networks.lora_wan \
    --network_dim 16 \
    --network_alpha 16 \
    --timestep_sampling shift \
    --discrete_flow_shift 1.0 \
    --max_train_epochs 100 \
    --save_every_n_epochs "$SAVE_EVERY" \
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
  "$ACCELERATE" launch --num_cpu_threads_per_process "$CPU_THREADS_PER_PROCESS" --num_processes 1 --main_process_port "$LOW_PORT" src/musubi_tuner/wan_train_network.py \
    --task t2v-A14B \
    --dit "$LOW_DIT" \
    --vae "$VAE" \
    --t5 "$T5" \
    --dataset_config "$DATASET" \
    $ATTN_FLAGS \
    --mixed_precision fp16 \
    --fp8_base \
    --optimizer_type adamw \
    --learning_rate 3e-4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --max_data_loader_n_workers "$MAX_DATA_LOADER_WORKERS" \
    --network_module networks.lora_wan \
    --network_dim 16 \
    --network_alpha 16 \
    --timestep_sampling shift \
    --discrete_flow_shift 1.0 \
    --max_train_epochs 100 \
    --save_every_n_epochs "$SAVE_EVERY" \
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
  echo "✅ Training completed!"

  OUTPUT_DIR="$MUSUBI_DIR/output"
  RENAMED_OUTPUT="$MUSUBI_DIR/output-${TITLE_SUFFIX}"
  if [[ -d "$OUTPUT_DIR" ]]; then
    mv "$OUTPUT_DIR" "$RENAMED_OUTPUT"
  fi
  
  # Analyze training logs and generate plots
  echo ""
  echo "=== Analyzing Training Logs ==="
  if [[ -f "$PWD/run_high.log" || -f "$PWD/run_low.log" ]]; then
    "$PYTHON" /workspace/analyze_training_logs.py "$PWD" || echo "Warning: Log analysis failed"
    if [[ -d "$PWD/training_analysis" ]]; then
      mv "$PWD/training_analysis" "$RENAMED_OUTPUT/training_analysis"
    fi

    [[ -f "$PWD/run_high.log" ]] && cp "$PWD/run_high.log" "$RENAMED_OUTPUT/"
    [[ -f "$PWD/run_low.log" ]] && cp "$PWD/run_low.log" "$RENAMED_OUTPUT/"
  else
    echo "Warning: No log files found to analyze"
  fi
  
  # Execute pre-configured post-training actions
  if [[ "$UPLOAD_CLOUD" =~ ^[Yy]$ ]]; then
    echo ""
    echo "=== Uploading to Cloud Storage ==="
    upload_to_cloud "$RENAMED_OUTPUT" "${TITLE_SUFFIX}" || echo "Failed to upload output directory"
  fi
  
  if [[ "$SHUTDOWN_INSTANCE" =~ ^[Yy]$ ]]; then
    echo ""
    echo "=== Shutting Down Instance ==="
    if setup_vast_api_key; then
      echo "Instance will shut down in 10 seconds. Press Ctrl+C to cancel."
      sleep 10
      shutdown_instance
    else
      echo "Could not set up instance shutdown. Skipping auto-shutdown."
    fi
  fi
  
  echo "✅ All done."
}

main "$@" 