FROM python:3.12-slim

# Keep Python fast + logs visible in `docker logs`
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install `uv` without cache bloat
RUN pip install --no-cache-dir uv

# ── Layer caching: install deps BEFORE copying code ──
# Build context MUST be `pipeline/` (see build command at the bottom),
# so paths below are relative to `pipeline/`.
COPY src/api/requirements.txt ./requirements.api.txt
RUN uv pip install --system --no-cache -r requirements.api.txt

# ── Copy code, preserving the SAME import layout as training ──
# Training pickled the preprocessor as `src.features.feature_engineering.*`
# and `model_loader.py` does `from config.config import ...`,
# so the image must contain `src/` + `config/`, not just `api/`.
COPY src/api/ ./src/api/
COPY src/features/ ./src/features/
COPY src/__init__.py ./src/__init__.py
COPY config/ ./config/

EXPOSE 8000

# NOTE: double quotes are REQUIRED — Docker CMD exec form is strict JSON.
# `src.api.main:app` (not `main:app`) because main.py uses relative imports
# (`from .schemas import ...`), which only work as a package import.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Build CONTEXT must be `pipeline/` (it contains `src/` + `config/`):
# Run from the repo root:
#   docker build -f pipeline/src/api/Dockerfile -t trip-api ./pipeline
# Run (mount MLflow DB + artifacts, or point at a remote tracking server):
#   docker run -p 8000:8000 ^
#     -e MLFLOW_TRACKING_URI=sqlite:////app/mlflow_trip_duration.db ^
#     -e MODEL_NAME=trip_duration_model -e MODEL_ALIAS=champion ^
#     -v ${PWD}/pipeline/mlflow_trip_duration.db:/app/mlflow_trip_duration.db ^
#     -v ${PWD}/pipeline/mlruns:/app/mlruns trip-api
