# Multi-stage Dockerfile for the Python backend
# Supports x86_64 and ARM64

FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies needed by opencv, PyMuPDF, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ---------- dependency stage ----------
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- runtime stage ----------
FROM deps AS runtime

COPY . /app/backend
# Ensure the package is importable
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
