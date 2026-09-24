# Website + live demo (CPU). Build from the repository root:
#   docker build -f web/Dockerfile -t roadwatch-web .
#   docker run -p 7860:7860 roadwatch-web
# Hugging Face Spaces (Docker SDK) and Render run this image unchanged; see web/DEPLOY.md.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    YOLO_OFFLINE=true YOLO_CONFIG_DIR=/tmp/Ultralytics ROADWATCH_DEVICE=cpu \
    DEMO_JOBS_DIR=/tmp/roadwatch-jobs PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /tmp/req/requirements.txt
COPY web/requirements.txt /tmp/req/web-requirements.txt
# CPU-only torch keeps the image ~1.5 GB smaller than the CUDA wheels ("2.8.0+cpu" satisfies torch==2.8.0)
RUN pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/req/requirements.txt -r /tmp/req/web-requirements.txt
COPY . .
RUN useradd -m -u 1000 app && chown -R app /app
USER app
EXPOSE 7860
CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port ${PORT}"]
