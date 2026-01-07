FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No hardcoded entrypoint - command specified in Vertex AI job config
# Default to standard training if no command specified
CMD ["python", "-m", "temporal_vit.training.train"]
