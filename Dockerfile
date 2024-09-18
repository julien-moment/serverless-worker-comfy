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

RUN mkdir -p models/checkpoints/SDXL/Lightning
RUN mkdir -p models/ultralytics/bbox
RUN mkdir -p models/clip_vision/SD1.5
RUN mkdir -p models/ipadapter
RUN mkdir -p models/controlnet/SDXL
RUN mkdir -p models/loras/SDXL
RUN mkdir -p models/insightface
RUN mkdir -p models/facerestore_models

# Download checkpoints/vae/LoRA to include in image based on model type

RUN wget -O models/checkpoints/SDXL/Lightning/dreamshaperXL_lightningDPMSDE.safetensors https://huggingface.co/gingerlollipopdx/ModelsXL/resolve/57ff7db12ceda2efe09ac1048b6b25fb33406401/dreamshaperXL_lightningDPMSDE.safetensors
RUN wget -O models/ultralytics/bbox/face_yolov8m.pt https://huggingface.co/xingren23/comfyflow-models/resolve/976de8449674de379b02c144d0b3cfa2b61482f2/ultralytics/bbox/face_yolov8m.pt
RUN wget -O models/clip_vision/SD1.5/clipvisionSD1.5.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors
RUN wget -O models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors
RUN wget -O models/controlnet/SDXL/diffusers_xl_depth_full.safetensors https://huggingface.co/lllyasviel/sd_control_collection/resolve/05cb13f62356f78d7c3a4ef10e460c5cda6bef8b/diffusers_xl_depth_full.safetensors
RUN wget -O models/loras/SDXL/tech_streetwear.safetensors https://civitai.com/api/download/models/166857?type=Model&format=SafeTensor
RUN wget -O models/insightface/inswapper_128.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx
RUN wget -O models/facerestore_models/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth


RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack custom_nodes/ComfyUI-Impact-Pack
#RUN git clone https://github.com/Seedsa/Fooocus_Nodes custom_nodes/Fooocus_Nodes
RUN git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus custom_nodes/ComfyUI_IPAdapter_plus

RUN git clone https://github.com/Gourieff/comfyui-reactor-node custom_nodes/comfyui-reactor-node
RUN cd custom_nodes/comfyui-reactor-node && python3 install.py

RUN git clone https://github.com/pikenrover/ComfyUI_PRNodes custom_nodes/ComfyUI_PRNodes

# has to be separate
RUN git clone https://github.com/Fannovel16/comfyui_controlnet_aux custom_nodes/comfyui_controlnet_aux
RUN mkdir -p custom_nodes/comfyui_controlnet_aux/ckpts/LiheYoung/Depth-Anything/checkpoints
RUN wget -O custom_nodes/comfyui_controlnet_aux/ckpts/LiheYoung/Depth-Anything/checkpoints/depth_anything_vitb14.pth https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitb14.pth

RUN git clone https://github.com/Acly/comfyui-tooling-nodes.git
RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git
# Stage 3: Final image
FROM base as final

ENV COMFYUI_PATH=/comfyui
ENV COMFYUI_MODEL_PATH=/comfyui/models

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models
COPY --from=downloader /comfyui/custom_nodes /comfyui/custom_nodes

RUN cd custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# Start the container
CMD /start.sh
