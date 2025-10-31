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
We are going to integrate a dataset captioning interface, called sillycaption. 
The SillyCaption/ dirrectory currently contains the original source code for a standalone HTML app that captions image and video datasets using OpenRouter. It is not integrated into the rest of the app in any way in its current form. 
Your task will be to
1. port SillyCaption to flask and integrate it as a page /SillyCaption inside our existing app
2. Remove/Do Not Implement the VLLM functionality, we will be using openrouter API only for simplicity. Also remove/do not implement the hybrid TextToImage / ImageToImage functionality.  
3. Add a feature to import the active training dataset from /workspace/musubi-tuner/dataset/
4. Add a feature to export whatever dataset is loaded in SillyCaption to /workspace/musubi-tuner/dataset/
5. Add a feature to import/export the datasets in SillyCaption to/from a database library stored on the instance inside SillyCaption/Datasets/'Dataset Name'/[images and captions]
6. Add intuitive navigation between the wan-training-webui/ and the wan-training-webui/SillyCaption page
When finished, there should be a new page, wan-training-webui/SillyCaption, with all of the existing frontend and capability of the original except for the features I explicitally requested you leave out, and the added functionality I listed.
