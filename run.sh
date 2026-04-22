#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Cricket Video Assessment System — Quick Start Script
# Usage: bash run.sh
# ──────────────────────────────────────────────────────────────

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PORT="${PORT:-8000}"

echo ""
echo "🏏 Cricket Video Assessment System"
echo "──────────────────────────────────"

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 not found. Install from https://python.org"
  exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYTHON_VERSION detected"

# Check FFmpeg
if command -v ffmpeg &>/dev/null; then
  echo "✅ FFmpeg detected"
else
  echo "⚠️  FFmpeg not found — video enhancement will use OpenCV fallback"
  echo "   Install: brew install ffmpeg"
fi

# Create/activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
  echo "🔧 Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Install dependencies
echo "📦 Installing dependencies (this may take a minute on first run)..."
pip install --quiet --upgrade pip
pip install --quiet -r "$PROJECT_DIR/requirements.txt"
echo "✅ Dependencies installed"

# Launch server
echo ""
echo "🚀 Starting server on http://localhost:$PORT"
echo "   API Docs: http://localhost:$PORT/docs"
echo "   Upload UI: http://localhost:$PORT/"
echo ""

cd "$PROJECT_DIR"
uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload
