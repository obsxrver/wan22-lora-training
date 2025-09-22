# WAN2.2 LoRa Training Workflow and Quick Start

### Looking for help with captioning? Auto-caption in under a minute with [https://obsxrver.pro/SillyCaption](https://obsxrver.pro/SillyCaption)

## Quick Start

### Launch via Vast.ai Template

Use the pre-built template to skip manual setup. Click the template link, pick a machine (2 x RTX 5090 recommended for concurrent high + low model training), add your captioned images and .txt captions under `/workspace/musubi-tuner/dataset/`, then run the simple runner:

- Template: [Wan 2.2 LoRA Training Quickstart](https://cloud.vast.ai/?ref_id=208628&creator_id=208628&name=Wan%202.2%20LoRA%20Training%20Quickstart)

```bash
# on the instance
./run_wan_training.sh
```

## Automation Scripts

This repository includes helpers to streamline setup and training on Vast.ai.

- `vast_provision.sh` provisions a fresh instance, installs dependencies, downloads the required models, and pulls `train_helper.py` into `/workspace`.
- `run_wan_training.sh` Simple interactive training runner.

### Simple runner (recommended)

Use the new `run_wan_training.sh` for a minimal, interactive flow that:
- caches latents and text encodings
- trains HIGH noise and LOW noise
- waits for a free GPU for each job, running both concurrently if 2+ GPUs are free

```bash
# on the Vast instance
cd /workspace/musubi-tuner
source venv/bin/activate
cd /workspace
bash run_wan_training.sh
```

Prompts and defaults:
- Title suffix: defaults to `mylora`. Final names become `WAN2.2-HighNoise_<suffix>` and `WAN2.2-LowNoise_<suffix>`.
- Author: defaults to `authorName`.
- Dataset path: defaults to `/workspace/musubi-tuner/dataset/dataset.toml`.

## WAN2.2 LoRa Training Workflow - AI_Characters
*I did not make this guide. I am just archiving it from [(CivitAI)](https://civitai.com/articles/17740)* [(Original Author: AI_Characters)](https://civitai.com/user/AI_Characters)

## 1. VastAI

Launch an instance using the PyTorch (Vast) template on https://cloud.vast.ai (recommend using an H100)

Set up the environment:
```bash
sudo apt-get update
git clone --recursive https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
git checkout d0a193061a23a51c90664282205d753605a641c1
apt install -y libcudnn8=8.9.7.29-1+cuda12.2 libcudnn8-dev=8.9.7.29-1+cuda12.2 --allow-change-held-packages
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install protobuf
pip install six
pip install torch==2.7.0 torchvision==0.22.0 xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
```

Downloading the necessary models:

```bash
cd /workspace/musubi-tuner
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P models_t5_umt5-xxl-enc-bf16.pth --local-dir models/text_encoders
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/vae/wan_2.1_vae.safetensors --local-dir models/vae
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors --local-dir models/diffusion_models
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors --local-dir models/diffusion_models
```

## 2. Dataset and Captions

Put your images and captions into `/workspace/musubi-tuner/dataset/`.

Create the following `dataset.toml` and put it into `/workspace/musubi-tuner/dataset/`:

```toml
# resolution, caption_extension, batch_size, num_repeats, enable_bucket, bucket_no_upscale should be set in either general or datasets
# otherwise, the default values will be used for each item
# general configurations
[general]
resolution = [960 , 960]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/workspace/musubi-tuner/dataset"
cache_directory = "/workspace/musubi-tuner/dataset/cache"
num_repeats = 1 # optional, default is 1. Number of times to repeat the dataset. Useful to balance the multiple datasets with different sizes.
# other datasets can be added here. each dataset can have different configurations
```

## 3. Training

Use the following command whenever you open a new terminal window (in order to activate the venv and be in the correct folder):

```bash
cd /workspace/musubi-tuner
source venv/bin/activate
```

Run the following command to create the necessary latents for the training (needs to be rerun every time you change the dataset/captions):

```bash
python src/musubi_tuner/wan_cache_latents.py --dataset_config /workspace/musubi-tuner/dataset/dataset.toml --vae /workspace/musubi-tuner/models/vae/split_files/vae/wan_2.1_vae.safetensors
```

Run the following command to create the necessary text encoder cache for the training (needs to be rerun every time you change the dataset/captions):

```bash
python src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config /workspace/musubi-tuner/dataset/dataset.toml --t5 /workspace/musubi-tuner/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth
```

Run `accelerate config` once before training (select "no" for all options).

If training on an RTX 5090, you need to swap out `--xformers` for `--sdpa` (Blackwell has compat. issues with xformers), and add `--blocks_to_swap 10` (or you will OOM)

### High-noise training command:

```bash
accelerate launch --num_cpu_threads_per_process 1 src/musubi_tuner/wan_train_network.py \
--task t2v-A14B \
--dit /workspace/musubi-tuner/models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
--vae /workspace/musubi-tuner/models/vae/split_files/vae/wan_2.1_vae.safetensors \
--t5 /workspace/musubi-tuner/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth \
--dataset_config /workspace/musubi-tuner/dataset/dataset.toml \
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
--lr_scheduler_min_lr_ratio="5e-5" \
--output_dir /workspace/musubi-tuner/output \
--output_name WAN2.2-HighNoise_MyLoRA \
--metadata_title WAN2.2-HighNoise_MyLoRA \
--metadata_author MyName \
--preserve_distribution_shape \
--min_timestep 875 \
--max_timestep 1000
```

### Low-noise training command:

```bash
accelerate launch --num_cpu_threads_per_process 1 src/musubi_tuner/wan_train_network.py \
--task t2v-A14B \
--dit /workspace/musubi-tuner/models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors \
--vae /workspace/musubi-tuner/models/vae/split_files/vae/wan_2.1_vae.safetensors \
--t5 /workspace/musubi-tuner/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth \
--dataset_config /workspace/musubi-tuner/dataset/dataset.toml \
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
--lr_scheduler_min_lr_ratio="5e-5" \
--output_dir /workspace/musubi-tuner/output \
--output_name WAN2.2-LowNoise_MyLoRA \
--metadata_title WAN2.2-LowNoise_MyLoRA \
--metadata_author MyName \
--preserve_distribution_shape \
--min_timestep 0 \
--max_timestep 875
```
