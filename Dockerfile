# FROM ubuntu:plucky AS builder_base
FROM python:3.11-slim AS base

ENV APP_DIR="/app" \
    PATH="$APP_DIR/.venv/bin:$PATH"

# Add linker config to allow discovery of the .venv dlls
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

WORKDIR ${APP_DIR}

FROM base AS builder

ENV PATH="$PATH:/root/.local/bin/" \
    UV_CACHE_DIR="/root/.cache/uv" \
    UV_COMPILE_BYTECODE="true" \
    UV_CONCURRENT_DOWNLOADS="16" \
    UV_CONCURRENT_INSTALLS="16" \
    UV_CONCURRENT_BUILDS="16" \
    UV_FROZEN="true" \
    UV_LINK_MODE="copy" \
    UV_PYTHON_PREFERENCE="only-system" \
    REQUIRED_PACKAGE_GROUPS="inference"

RUN apt update && \
    apt install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl && \
    apt autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Project dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --only-group $REQUIRED_PACKAGE_GROUPS --no-editable --no-install-project
    # uv export --no-emit-project --no-editable --only-group $REQUIRED_PACKAGE_GROUPS > requirements.txt && \
    # uv venv && uv pip install -r requirements.txt

# Project source
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=src/,target=src/ \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --no-default-groups --group $REQUIRED_PACKAGE_GROUPS --no-editable

FROM base AS dev
ENV PATH="$APP_DIR/.venv/bin:$PATH"
COPY --from=builder --chown=app:app ${APP_DIR}/.venv .venv
COPY main.py .
CMD ["fastapi", "dev", "main.py", "--port", "8000", "--host", "0.0.0.0"]

FROM base AS prod
ENV PATH="$APP_DIR/.venv/bin:$PATH"
COPY --from=builder --chown=app:app ${APP_DIR}/.venv .venv
COPY main.py .
CMD ["fastapi", "run", "main.py", "--workers", "4", "--port", "8000", "--host", "0.0.0.0"]
