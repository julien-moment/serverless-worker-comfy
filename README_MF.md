
docker build -t docker.io/momentfactory/runpod-worker-comfy:photobooth .
# you can add --no-cache

docker push docker.io/momentfactory/runpod-worker-comfy:photobooth

# to start with API
docker run --rm  --runtime=nvidia -p 8000:8000 \
  -e SERVE_API_LOCALLY=true \
  docker.io/momentfactory/runpod-worker-comfy:photobooth 

# to test with ComfyUI - add this after   
# python3 /comfyui/main.py --listen 0.0.0.0 --port 6970


# to start with UI
docker run --rm  --runtime=nvidia -p 8001:6970 \
  -v $PWD/output:/comfyui/output \
  -e SERVE_API_LOCALLY=true \
  docker.io/momentfactory/runpod-worker-comfy:photobooth python3 /comfyui/main.py --listen 0.0.0.0 --port 6970


# format of the body

{
  "input": {
    "workflow": {}
  }
}