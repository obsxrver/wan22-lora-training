#!/usr/bin/env python3
"""Helper script to cache latents and caption embeddings and launch WAN2.2 training.

This script assumes the musubi-tuner repository has been provisioned at
`/workspace/musubi-tuner` using `vast_provision.sh`.

It will automatically re-execute itself using the musubi-tuner virtual
environment so the user does not need to manually activate it.

The script caches latents and caption embeddings, writes a default
`accelerate` config, then launches both high- and low-noise training runs.
Use `--output-name` to choose the base name for the resulting LoRAs
(defaults to `my_wan_lora`). Each run appends `_high` or `_low` and uses
the same value as the metadata title. The `--author` flag controls the
metadata author field and defaults to `AI_Characters`.
"""

import argparse
import os
import pathlib
import subprocess
import sys

import torch


MUSUBI_DIR = pathlib.Path("/workspace/musubi-tuner")
VENV_PYTHON = MUSUBI_DIR / "venv" / "bin" / "python"
ACCELERATE = MUSUBI_DIR / "venv" / "bin" / "accelerate"
VAE_PATH = MUSUBI_DIR / "models/vae/split_files/vae/wan_2.1_vae.safetensors"
T5_PATH = MUSUBI_DIR / "models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth"
HIGH_MODEL_PATH = MUSUBI_DIR / "models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
LOW_MODEL_PATH = MUSUBI_DIR / "models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors"

# Re-exec inside the musubi-tuner virtual environment if not already using it.
if sys.executable != str(VENV_PYTHON) and VENV_PYTHON.exists():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])


def run(cmd, **kwargs):
    """Run a subprocess and stream output."""
    print("Running:", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def cache_latents_and_captions(dataset_config: pathlib.Path) -> None:
    run(
        [
            str(VENV_PYTHON),
            "src/musubi_tuner/wan_cache_latents.py",
            "--dataset_config",
            str(dataset_config),
            "--vae",
            str(VAE_PATH),
        ],
        cwd=MUSUBI_DIR,
    )
    run(
        [
            str(VENV_PYTHON),
            "src/musubi_tuner/wan_cache_text_encoder_outputs.py",
            "--dataset_config",
            str(dataset_config),
            "--t5",
            str(T5_PATH),
        ],
        cwd=MUSUBI_DIR,
    )


def ensure_accelerate_config() -> None:
    """Create a default accelerate config if missing."""
    config_dir = pathlib.Path.home() / ".cache" / "huggingface" / "accelerate"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "default_config.yaml"
    if not config_file.exists():
        run([str(ACCELERATE), "config", "default"], cwd=MUSUBI_DIR)


def build_train_cmd(
    model_path: pathlib.Path,
    dataset_config: pathlib.Path,
    save_every: int,
    output_name: str,
    author: str,
    min_timestep: int,
    max_timestep: int,
):
    return [
        str(ACCELERATE), "launch", "--num_cpu_threads_per_process", "1", "src/musubi_tuner/wan_train_network.py",
        "--task", "t2v-A14B",
        "--dit", str(model_path),
        "--vae", str(VAE_PATH),
        "--t5", str(T5_PATH),
        "--dataset_config", str(dataset_config),
        "--xformers",
        "--mixed_precision", "fp16",
        "--fp8_base",
        "--optimizer_type", "adamw",
        "--learning_rate", "3e-4",
        "--gradient_checkpointing",
        "--gradient_accumulation_steps", "1",
        "--max_data_loader_n_workers", "2",
        "--network_module", "networks.lora_wan",
        "--network_dim", "16",
        "--network_alpha", "16",
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", "1.0",
        "--max_train_epochs", "100",
        "--save_every_n_epochs", str(save_every),
        "--seed", "5",
        "--optimizer_args", "weight_decay=0.1",
        "--max_grad_norm", "0",
        "--lr_scheduler", "polynomial",
        "--lr_scheduler_power", "8",
        "--lr_scheduler_min_lr_ratio=5e-5",
        "--output_dir", "/workspace/musubi-tuner/output",
        "--output_name",
        output_name,
        "--metadata_title",
        output_name,
        "--metadata_author",
        author,
        "--preserve_distribution_shape",
        "--min_timestep", str(min_timestep),
        "--max_timestep", str(max_timestep),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache latents and start WAN2.2 training")
    parser.add_argument("--dataset-config", default="dataset/dataset.toml", dest="dataset_config")
    parser.add_argument("--num-samples", type=int, default=1, dest="num_samples")
    parser.add_argument("--output-name", default="my_wan_lora", dest="output_name")
    parser.add_argument("--author", default="AI_Characters", dest="author")
    args = parser.parse_args()

    dataset_config = pathlib.Path(args.dataset_config)
    if not dataset_config.is_absolute():
        dataset_config = MUSUBI_DIR / dataset_config

    save_every = max(1, 100 // max(1, args.num_samples))

    cache_latents_and_captions(dataset_config)
    ensure_accelerate_config()

    high_name = f"{args.output_name}_high"
    low_name = f"{args.output_name}_low"
    high_cmd = build_train_cmd(
        HIGH_MODEL_PATH,
        dataset_config,
        save_every,
        high_name,
        args.author,
        875,
        1000,
    )
    low_cmd = build_train_cmd(
        LOW_MODEL_PATH,
        dataset_config,
        save_every,
        low_name,
        args.author,
        0,
        875,
    )

    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2:
        env_high = os.environ.copy()
        env_high["CUDA_VISIBLE_DEVICES"] = "0"
        env_low = os.environ.copy()
        env_low["CUDA_VISIBLE_DEVICES"] = "1"
        procs = [
            subprocess.Popen(high_cmd, cwd=MUSUBI_DIR, env=env_high),
            subprocess.Popen(low_cmd, cwd=MUSUBI_DIR, env=env_low),
        ]
        for p in procs:
            p.wait()
    else:
        run(high_cmd, cwd=MUSUBI_DIR)
        run(low_cmd, cwd=MUSUBI_DIR)


if __name__ == "__main__":
    main()

