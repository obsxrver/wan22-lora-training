#wan22-lora-training. 
This is a tool that started out as a simple training script for Wan 2.2 LoRA. It has since expanded to include a full FastAPI-Flask WebUI for training. 
The repository is designed to be used with Vast.AI, and is automatically porovisioned with a custom template. This is how it works: 
1. The instance template is created on vast.ai by customizing the existing PyTorch (Vast) template and setting vast_provision.sh set as the startup script. 
The docker environment variable PORTAL_CONFIG is set to enable the the training webui and the NOT YET IMPLEMENTED SillyCaption dataset management interface to be accessible from the vast.ai portal like so: 
PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard|localhost:7865:7865:/:wan-training-webui|localhost:7865:7865:/SillyCaption:SillyCaption Dataset Manager"
The user can optionally set the docker variable VASTAI_KEY to set the vastai key, the user can set this inside the webui as well.
2. vast_provision.sh runs on the first boot of the instance. It clones this repository, the musubi-tuner repository, and downloads dependencies, Wan 2.2 DITs, Text Encoders, and VAEs. 
It also configures the supervisorctl config for the webui and future apps 
3. The webui starts, and is accessed through the instance portal, using fastapi middleware to validate the token passed via uri/?token= (portal does this automatically) against the JUPYTER_TOKEN environment variable. 
From the webui, the user can upload their media dataset (video and image), and launch a dual-GPU LoRA training, and watch it in real time. 
NEXT STEPS: 
It is time for a major feature addition. We are going to create another FastAPI app by porting the standalone HTML app in the SillyCaption/ to flask. SillyCaption is a standalone repository I created. It is an advanced AI/API integrated captioning tool. 
The SillyCaption/ dirrectory currently contains the original source code. It is not integrated into the rest of the app in any way in its current form. 
Your task will be to
1. port SillyCaption to flask and integrate it into our existing app
2. add middleware authentication with JUPYTER_TOKEN
3. Delete the VLLM functionality, we will be using openrouter API only for simplicity. 
4. Add a feature to import the active dataset from /workspace/musubi-tuner/dataset/
5. Add a feature to export whatever dataset is loaded in SillyCaption to /workspace/musubi-tuner/dataset/
6. Add a feature to import/export the datasets in SillyCaption to/from a database library stored on the instance
7. Add intuitive navigation between the wan-training-webui and SillyCaption