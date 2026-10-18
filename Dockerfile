# VISTA: one image serving the API and the built UI.
# Multi-arch: builds natively on linux/arm64 (AWS Graviton) and linux/amd64.
#
#   CPU (default):  docker build -t vista .
#   NVIDIA GPU:     docker build -t vista-gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130 .
#                   docker run --gpus all ...

FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ARG TORCH_INDEX=""
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/data/hf \
    VISTA_DATA_DIR=/data \
    VISTA_HOST=0.0.0.0 \
    VISTA_PORT=8000 \
    VISTA_FRONTEND_DIR=/app/frontend/dist \n    VISTA_DETECTOR_MODE=owlv2 \n    VISTA_DETECTION_CONFIDENCE_THRESHOLD=0.2 \n    VISTA_DETECTOR_LABELS="coin,key,wallet,pen,remote,phone,phone charger,charging cable,power adapter,earbuds,glasses,watch,mug,bottle,notebook,laptop,backpack,person"

# OpenCV runtime libs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
        --group owlv2 --group aws --group detection
# Optional: swap in a CUDA build of torch for GPU hosts.
RUN if [ -n "$TORCH_INDEX" ]; then \
        uv pip install --python .venv/bin/python --reinstall torch --index-url "$TORCH_INDEX"; \
    fi

COPY backend ./backend
COPY scripts ./scripts
COPY --from=ui /ui/dist ./frontend/dist

EXPOSE 8000
VOLUME ["/data"]
CMD [".venv/bin/python", "backend/run.py"]
