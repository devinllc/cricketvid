# ──────────────────────────────────────────────────────────
# Cricket Video Assessment System — Dockerfile
# Multi-stage build for lean production image
# ──────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System deps: FFmpeg + OpenCV headless requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────────────────────
WORKDIR /cricket

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── App code ────────────────────────────────────────────
COPY app/ ./app/

# Ensure required runtime dirs exist
RUN mkdir -p app/uploads app/processed app/static

# Download MediaPipe pose model at build time (avoids cold-start delay)
RUN python3 -c "\
import urllib.request, os; \
url='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'; \
path='app/ai/pose_landmarker_lite.task'; \
os.makedirs('app/ai', exist_ok=True); \
urllib.request.urlretrieve(url, path) if not os.path.exists(path) else None; \
print('Pose model ready:', path)"

# ── Runtime config ──────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV UVICORN_WORKERS=1
ENV ANALYSIS_INTERPOLATION_FACTOR=1

EXPOSE 8000

# Health check (Docker will restart container if unhealthy)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-1} --timeout-keep-alive 30"]
