# FROM ubuntu:plucky AS builder_base
FROM python:3.11-slim AS base

ENV APP_DIR="/app"

FROM base AS builder

ENV PATH="$PATH:/root/.local/bin/"

WORKDIR ${APP_DIR}

RUN apt update && \
    apt install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    vim \
    curl && \
    apt autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Project dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --only-group inference \
    --frozen \
    --no-editable \
    --link-mode copy \
    --no-install-project

COPY . .

# Project source
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-default-groups --group inference --frozen --no-editable

FROM base AS dev

ENV PATH="$APP_DIR/.venv/bin:$PATH"

WORKDIR ${APP_DIR}

# Copy the environment only
COPY --from=builder --chown=app:app ${APP_DIR}/.venv .venv

# Add linker config to allow importing of the .venv dlls
RUN mkdir -p /etc/ld.so.conf.d/ && \
    echo "/app/.venv/lib/python3.11/site-packages/tensorrt_libs" > /etc/ld.so.conf.d/cuda-venv.conf && \
    echo "/app/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib" >> /etc/ld.so.conf.d/cuda-venv.conf && \
    echo "/app/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib" >> /etc/ld.so.conf.d/cuda-venv.conf && \
    echo "/app/.venv/lib/python3.11/site-packages/nvidia/curand/lib" >> /etc/ld.so.conf.d/cuda-venv.conf && \
    echo "/app/.venv/lib/python3.11/site-packages/nvidia/cublas/lib" >> /etc/ld.so.conf.d/cuda-venv.conf && \
    echo "/app/.venv/lib/python3.11/site-packages/nvidia/cufft/lib" >> /etc/ld.so.conf.d/cuda-venv.conf && \
    echo "/app/.venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib" >> /etc/ld.so.conf.d/cuda-venv.conf && \
    echo "/app/.venv/lib/python3.11/site-packages/nvidia/nvrtc/lib" >> /etc/ld.so.conf.d/cuda-venv.conf && \
    ldconfig

COPY main.py .

# Run the application
CMD ["fastapi", "dev", "main.py", "--port", "8000", "--host", "0.0.0.0"]
