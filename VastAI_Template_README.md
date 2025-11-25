# Wan 2.2 LoRA Training Quickstart (Vast.ai Template)

Use this template to train Wan 2.2 LoRA with the simplest possible flow.

## 1) Launch an instance

- Open the template and click Rent:  
  [Wan 2.2 LoRA Training Quickstart](https://cloud.vast.ai/?ref_id=208628&creator_id=208628&name=Wan%202.2%20LoRA%20Training%20Quickstart)
- Pick a machine with at least one H100 (recommended). More GPUs allow concurrent high/low runs.

## 2) Add your dataset

- Put your captioned images and `.txt` files into:
  - `/workspace/musubi-tuner/dataset/`
- A default `dataset.toml` is already present at:
- `/workspace/wan22-lora-training/dataset-configs/dataset.toml`
  - Edit if needed, or keep defaults.

## 3) Run the simple runner

In a terminal on the instance:

```bash
cd /workspace/musubi-tuner
source venv/bin/activate
cd /workspace
bash run_wan_training.sh
```

The script will prompt you for:
- Title suffix (default: `mylora`) → final names:
  - `WAN2.2-HighNoise_<suffix>` and `WAN2.2-LowNoise_<suffix>`
- Author (default: `authorName`)
- Dataset path (default: `/workspace/wan22-lora-training/dataset-configs/dataset.toml`)

What it does:
- Caches latents and text encodings
- Trains HIGH-noise and LOW-noise LoRA
- Picks any free GPU; if 2+ are free, runs both trainings concurrently
- Writes logs to `run_high.log` and `run_low.log`
- Saves outputs to `/workspace/musubi-tuner/output/`
- **Analyzes training logs** and generates:
  - CSV files with step/loss data (`training_analysis/`)
  - Matplotlib plot of loss curves (`training_analysis/training_loss_plot.png`)
  - Summary statistics (`training_analysis/training_summary.txt`)
- Optionally uploads LoRAs + analysis to cloud storage (Google Drive, AWS S3, etc.)
- Optionally shuts down the instance to save costs

Notes:
- No manual `accelerate config` required; defaults are created automatically if missing.
- Port conflicts are avoided automatically by using unique rendezvous ports per run.

## Troubleshooting

- If you want to launch manually with explicit ports:
  ```bash
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 accelerate launch --main_process_port 29600 ...
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29601 accelerate launch --main_process_port 29601 ...
  ```
- If no GPUs appear, ensure your selected machine has NVIDIA GPUs and `nvidia-smi` works in the container.

## Post-Training Options

After training completes, the script will prompt you for:

### Cloud Storage Upload
- **Requirements**: Cloud connection configured in Vast.ai Console
- **Setup** (one-time):
  1. Go to [Vast.ai Console > Cloud Connections](https://cloud.vast.ai/connections)
  2. Add a connection to Google Drive, AWS S3, or other cloud provider
  3. Follow the authentication steps in the console
- **Result**: LoRAs uploaded to `loras/WAN/[lora_name]/` in your connected cloud storage

### Auto-Shutdown
- **Requirements**: Running on Vast.ai instance
- **Result**: Instance stops automatically to save costs
- **Safety**: 10-second countdown with option to cancel

## Where results go

- Trained LoRA files and metadata are under `/workspace/musubi-tuner/output/`.
- Training analysis (loss plots, CSVs, summary) are under `/workspace/training_analysis/`.
- If uploaded to cloud storage: `loras/WAN/[lora_name]/` contains:
  - LoRA `.safetensors` files
  - Training logs (`run_high.log`, `run_low.log`)
  - Analysis directory with plots and CSVs

## Analyzing existing logs

If you have training logs from a previous run:

```bash
python3 /workspace/analyze_training_logs.py /path/to/log/directory
```

This will create a `training_analysis/` directory with:
- `high_noise_loss.csv` - Step and loss data for high noise
- `low_noise_loss.csv` - Step and loss data for low noise  
- `training_loss_plot.png` - Visual comparison of both training runs
- `training_summary.txt` - Statistics (min, max, mean loss, etc.)

updated 20250824