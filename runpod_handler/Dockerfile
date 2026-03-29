FROM python:3.11-slim

# System deps for torch + image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# torch with CUDA 12.1 for RunPod GPU instances
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    runpod \
    chromadb \
    numpy \
    Pillow

# Copy handler
COPY handler.py .

# DINOv2 downloads from torch hub on first load — pre-cache it
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')" || true

# RunPod serverless expects the handler to run as the entrypoint
CMD ["python", "-u", "handler.py"]
