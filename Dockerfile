# Production image for the two-wheeler safety monitor.
#
# The trained YOLO weights are NOT baked into the image (they're large and
# gitignored). Mount them at run time and point MODEL_PATH at them:
#
#   docker build -t tws .
#   docker run --rm -p 5000:5000 \
#       -v "$PWD/weights:/weights:ro" \
#       -v tws-data:/data \
#       -e MODEL_PATH=/weights/best.pt \
#       -e TRAFFIC_DB_PATH=/data/traffic.db \
#       -e EVIDENCE_DIR=/data/evidence \
#       tws
#
# See docs/DEPLOYMENT.md for the full story (volumes, API key, limits).

FROM python:3.12-slim

# OpenCV needs these shared libraries even in headless use.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=5000 \
    FLASK_DEBUG=0

WORKDIR /app

# Install deps first for better layer caching. torch/ultralytics/easyocr are
# large, so this layer is the slow one and only rebuilds when deps change.
COPY requirements.txt requirements-prod.txt ./
RUN pip install --upgrade pip && pip install -r requirements-prod.txt

COPY . .

# Default persistent locations (override via env / volumes).
ENV TRAFFIC_DB_PATH=/data/traffic.db \
    EVIDENCE_DIR=/data/evidence \
    MODEL_PATH=/weights/best.pt
RUN mkdir -p /data

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health').status==200 else 1)"

# IMPORTANT: one worker, many threads. Cached models, the in-process video
# JobManager, and the rate limiter are all per-process state — multiple workers
# would each get their own copy and video-status polling could miss the worker
# that owns a job. Threads give concurrency without splitting that state.
# See docs/DECISIONS.md ("in-process jobs").
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "600", \
     "--bind", "0.0.0.0:5000", "app:app"]
