FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "-m", "temporal_vit.training.train"]
