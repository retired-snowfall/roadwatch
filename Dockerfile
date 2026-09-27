# Submission image: the same two commands as the pip route, inside a container.
#   docker build -t roadwatch .
#   docker run --gpus all -v /data/test:/data/test -v $PWD/out:/out roadwatch \
#       python run_submission.py --videos /data/test --out /out/predictions.json
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    YOLO_OFFLINE=true YOLO_CONFIG_DIR=/tmp/Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN bash weights/download.sh
CMD ["python", "run_submission.py", "--videos", "/data/test", "--out", "/out/predictions.json"]
