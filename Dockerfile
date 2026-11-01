# syntax=docker/dockerfile:1
# Lockfiles target Linux x86_64 / CPython 3.12. Build with --platform linux/amd64.
ARG PYTHON_IMAGE=python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337
ARG RUNTIME_PROFILE=cpu
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH HOME=/home/user \
    HF_HOME=/home/user/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/home/user/cache/matplotlib BACKEND_EMBEDDING_CACHE_DIR=/home/user/cache/backend \
    BACKEND_DATABASE_URL=sqlite:////home/user/data/backend.sqlite3 PORT=8000
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 user \
    && useradd --uid 1000 --gid 1000 --create-home user \
    && mkdir -p /home/user/cache /home/user/data \
    && chown -R 1000:1000 /home/user/cache /home/user/data
WORKDIR /app

FROM base AS core-dependencies
COPY requirements-runtime.lock /tmp/requirements.lock
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --require-hashes --only-binary=:all: -r /tmp/requirements.lock \
    && /opt/venv/bin/python -m pip check

FROM base AS cpu-dependencies
COPY requirements-runtime-embeddings.lock /tmp/requirements.lock
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --require-hashes --only-binary=:all: -r /tmp/requirements.lock \
    && /opt/venv/bin/python -m pip check \
    && /opt/venv/bin/python -c "import torch; assert torch.version.cuda is None, 'CUDA build is not allowed'"

FROM base AS source
# .dockerignore is an allowlist: no datasets, credentials, tests or frontend deps.
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY ml_training/ ./ml_training/
COPY config/ ./config/
COPY migrations/ ./migrations/
COPY alembic.ini ./
# Copy helper code separately so staged weights are not duplicated in two paths.
COPY deploy/*.py ./deploy/
# Empty by default. Stage only trusted, reviewed whole artifact trees explicitly.
COPY deploy/artifacts/ ./artifacts/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=6s --start-period=180s --retries=3 CMD ["python", "deploy/healthcheck.py"]
ENTRYPOINT ["python", "/app/deploy/entrypoint.py"]
USER 1000:1000

FROM source AS runtime-lite
COPY --from=core-dependencies /opt/venv /opt/venv
LABEL org.opencontainers.image.title="BanglaPSG API — rules and classical CPU models"

FROM runtime-lite AS lightweight

FROM source AS cpu
COPY --from=cpu-dependencies /opt/venv /opt/venv
LABEL org.opencontainers.image.title="BanglaPSG API — CPU-only neural runtime"

# Default full CPU runtime; hosts supporting build arguments can set
# RUNTIME_PROFILE=runtime-lite when intentionally serving only classical models.
FROM ${RUNTIME_PROFILE} AS runtime
