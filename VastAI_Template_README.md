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
  - `/workspace/musubi-tuner/dataset/dataset.toml`
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
- Dataset path (default: `/workspace/musubi-tuner/dataset/dataset.toml`)

What it does:
- Caches latents and text encodings
- Trains HIGH-noise and LOW-noise LoRA
- Picks any free GPU; if 2+ are free, runs both trainings concurrently
- Writes logs to `run_high.log` and `run_low.log`
- Saves outputs to `/workspace/musubi-tuner/output/`

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

## Where results go

- Trained LoRA files and metadata are under `/workspace/musubi-tuner/output/`.

updated 20250824