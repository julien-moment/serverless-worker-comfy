# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1 libgl1-mesa-glx libglib2.0-0

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
RUN pip3 install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --upgrade -r requirements.txt

# Install runpod
RUN pip3 install runpod requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Stage 2: Download models
FROM base as downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE

# Change working directory to ComfyUI
WORKDIR /comfyui

# Missing folders
RUN echo "Creating missing folders..."  
RUN mkdir models/checkpoints/SDXL;
RUN mkdir models/checkpoints/SDXL/Lightning;
RUN mkdir models/ultralytics;
RUN mkdir models/ultralytics/bbox;
RUN mkdir models/clip_vision/SD1.5;
RUN mkdir models/ipadapter;
RUN mkdir models/controlnet/SDXL;
RUN mkdir models/loras/SDXL;
RUN mkdir models/insightface;
RUN mkdir models/facerestore_models;
RUN echo "Done"   

# Checkpoints
RUN echo "Downloading checkpoints..."  
RUN wget -O models/checkpoints/SDXL/Lightning/dreamshaperXL_lightningDPMSDE.safetensors https://huggingface.co/gingerlollipopdx/ModelsXL/resolve/57ff7db12ceda2efe09ac1048b6b25fb33406401/dreamshaperXL_lightningDPMSDE.safetensors
RUN wget -O models/ultralytics/bbox/face_yolov8m.pt https://huggingface.co/xingren23/comfyflow-models/resolve/976de8449674de379b02c144d0b3cfa2b61482f2/ultralytics/bbox/face_yolov8m.pt
#RUN wget -O custom_nodes/comfyui_controlnet_aux/ckpts/LiheYoung/Depth-Anything/checkpoints/depth_anything_vitb14.pth https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitb14.pth
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/clip_vision/SD1.5/clipvisionSD1.5.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors
RUN wget -O models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors
RUN wget -O models/controlnet/SDXL/diffusers_xl_depth_full.safetensors https://huggingface.co/lllyasviel/sd_control_collection/resolve/05cb13f62356f78d7c3a4ef10e460c5cda6bef8b/diffusers_xl_depth_full.safetensors
RUN wget -O models/loras/SDXL/tech_streetwear.safetensors https://civitai.com/api/download/models/166857?type=Model&format=SafeTensor
RUN wget -O models/insightface/inswapper_128.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx
RUN wget -O models/facerestore_models/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
RUN echo "Done"   

<<<<<<< HEAD
#instal custom nodes  
RUN echo "Installing custom nodes..." 
=======
COPY ./models/loras/SDXL /comfyui/models/loras/SDXL
>>>>>>> 535babb (save Dockerfile)

RUN cd custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack
RUN cd custom_nodes && git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus
RUN cd custom_nodes && git clone https://github.com/pikenrover/ComfyUI_PRNodes

# has to be separate
RUN cd custom_nodes && git clone https://github.com/Fannovel16/comfyui_controlnet_aux
RUN mkdir -p custom_nodes/comfyui_controlnet_aux/ckpts/LiheYoung/Depth-Anything/checkpoints
RUN wget -O custom_nodes/comfyui_controlnet_aux/ckpts/LiheYoung/Depth-Anything/checkpoints/depth_anything_vitb14.pth https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitb14.pth

RUN cd custom_nodes && git clone https://github.com/Gourieff/comfyui-reactor-node
RUN cd custom_nodes/comfyui-reactor-node && python3 install.py
RUN cd custom_nodes && git clone https://github.com/Acly/comfyui-tooling-nodes
RUN cd custom_nodes && git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts
RUN cd custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager
RUN cd custom_nodes && git clone https://github.com/Seedsa/Fooocus_Nodes

RUN cd custom_nodes && git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes

RUN cd custom_nodes && git clone https://github.com/evanspearman/ComfyMath
RUN cd custom_nodes && git clone https://github.com/rgthree/rgthree-comfy



# Stage 3: Final image
FROM base as final

ENV COMFYUI_PATH=/comfyui
ENV COMFYUI_MODEL_PATH=/comfyui/models

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models
COPY --from=downloader /comfyui/custom_nodes /comfyui/custom_nodes

RUN pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
RUN pip install opencv-python==4.7.0.72

#RUN cd custom_nodes/comfyui-reactor-node && pip install -r requirements.txt
RUN cd /comfyui/custom_nodes/comfyui-reactor-node && python3 install.py
RUN cd /comfyui/custom_nodes/Fooocus_Nodes && pip install -r requirements.txt

COPY ./input /comfyui/input

# Start the container
CMD /start.sh
