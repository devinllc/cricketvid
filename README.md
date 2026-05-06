# 🏏 Cricket Video Assessment System — POC

An AI-powered cricket practice video analysis system built with FastAPI, MediaPipe, OpenCV, and FFmpeg. Upload a batting or bowling video and receive a detailed biomechanical assessment report.

---

## 🚀 Quick Start (Recommended)

```bash
# 1. Clone / navigate to project
cd cricketVideo

# 2. Install FFmpeg (required for video enhancement)
brew install ffmpeg   # macOS
# apt install ffmpeg  # Ubuntu/Debian

# 3. Run the startup script (auto-creates venv & installs dependencies)
bash run.sh
```

Open your browser at: **http://localhost:8000**

> Tip: For the Ball Trajectory (Beta) video to play reliably in all browsers,
> install FFmpeg so the overlay is encoded as H.264:
>
> macOS: `brew install ffmpeg`  •  Ubuntu/Debian: `apt install ffmpeg`

> **Important:** For consistent local/server results, ensure `ANALYSIS_INTERPOLATION_FACTOR=1` is set in all environments.
> See [Environment Variables](#-environment-variables) below for details on avoiding local/server output mismatches.
>
> **Troubleshooting:** If production returns empty wagon wheel while local detects shots, see [TROUBLESHOOTING_LOCAL_SERVER_PARITY.md](TROUBLESHOOTING_LOCAL_SERVER_PARITY.md) for diagnosis steps.

---

## 🛠 Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## � Environment Variables

| Variable | Default | Purpose | Notes |
|---|---|---|---|
| `ANALYSIS_INTERPOLATION_FACTOR` | `1` | Frame interpolation multiplier (1 = no interpolation, 2+ = add synthetic frames) | **CRITICAL:** Must match between local/Docker/server to avoid different shot detections. Default is 1 for consistency. |
| `UVICORN_WORKERS` | `1` | Number of concurrent web workers | Set to 1 for small VPS (2+ for high-concurrency servers) |
| `PORT` | `8000` | Server port | |
| `PYTHONUNBUFFERED` | `1` | Unbuffered stdout for Docker logs | Already set in Dockerfile |

### Local/Server Consistency

If you see **different outputs** (empty wagon wheel on server but shots detected locally):
1. Check Docker `Dockerfile` and local shell have matching `ANALYSIS_INTERPOLATION_FACTOR` values
2. Verify FFmpeg is installed on server: `ffmpeg -version`
3. Check logs for frame count: `Extracted XXX frames ... | interpolation_factor=X`

Example setup for consistent multi-environment deployment:
```bash
# Local development
export ANALYSIS_INTERPOLATION_FACTOR=1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Docker build/run
docker build -t cricket-ai .
docker run -e ANALYSIS_INTERPOLATION_FACTOR=1 -p 8000:8000 cricket-ai
```

---

## �📁 Project Structure

```
cricketVideo/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── routes/
│   │   ├── video.py             # POST /upload-video, GET /status/{id}
│   │   └── report.py            # GET /report/{id}
│   ├── services/
│   │   ├── normalizer.py        # FFmpeg/OpenCV video enhancement
│   │   ├── frame_extractor.py   # OpenCV frame sampling
│   │   └── video_processor.py   # Pipeline orchestrator
│   ├── ai/
│   │   ├── pose_detector.py     # MediaPipe Pose detection
│   │   ├── metric_calculator.py # Biomechanical metric math
│   │   └── cricket_scorer.py    # Drill-specific scoring + issues
│   ├── reports/
│   │   └── report_builder.py    # JSON report assembler
│   ├── utils/
│   │   ├── job_store.py         # In-memory job tracking
│   │   └── logger.py            # Structured logging
│   ├── templates/
│   │   ├── index.html           # Upload UI (dark, premium)
│   │   └── report.html          # Report viewer (charts, metrics)
│   ├── static/
│   │   ├── styles.css           # Global CSS
│   │   └── app.js               # Upload page JS
│   ├── uploads/                 # Raw uploaded videos
│   └── processed/               # FFmpeg-normalized frames
├── requirements.txt
├── run.sh
└── README.md
```

---

## 🌐 API Endpoints

### `POST /upload-video`
Upload a cricket video for analysis.

**Form Data:**
| Field | Type | Values |
|---|---|---|
| `file` | File | `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` |
| `drill_type` | String | `straight_drive`, `cover_drive`, `bowling_action`, `spin_bowling` |

**Response (202):**
```json
{
  "job_id": "abc123-...",
  "status": "queued",
  "drill_type": "straight_drive",
  "poll_url": "/status/abc123-...",
  "report_url": "/report/abc123-..."
}
```

---

### `GET /status/{job_id}`
Poll job processing status.

**Statuses:** `queued` → `processing` → `complete` / `failed`

---

### `GET /report/{job_id}`
Get the full assessment report (JSON or HTML).

**Query params:** `?format=json` (default) | `?format=html` (browser viewer)

**Response:**
```json
{
  "job_id": "abc123",
  "player_score": 78.5,
  "performance_band": "Advanced",
  "drill_type": "Straight Drive",
  "drill_type_key": "straight_drive",
  "metrics": {
    "head_stability": 8.2,
    "footwork": 6.1,
    "shoulder_rotation": 7.5,
    "balance": 9.0,
    "bat_path": 7.8
  },
  "issues": [
    "Front foot not moving early enough into position"
  ],
  "recommendations": [
    "Step forward earlier — commit to the front foot drive as the ball pitches"
  ],
  "analysis_meta": {
    "video_filename": "batting_drill.mp4",
    "frame_count": 54,
    "detected_frame_count": 48,
    "pose_detection_rate": 0.889,
    "processing_time_sec": 12.4
  }
}
```

---

### Other Endpoints
| Endpoint | Description |
|---|---|
| `GET /` | Upload UI |
| `GET /health` | Health check |
| `GET /jobs` | List all jobs |
| `GET /docs` | Swagger UI |
| `GET /redoc` | ReDoc UI |

---

## 🎯 Supported Drills & Metrics

### Batting (Straight Drive / Cover Drive)
| Metric | Description |
|---|---|
| `head_stability` | Nose Y-position variance across frames |
| `footwork` | Front/back ankle displacement magnitude |
| `shoulder_rotation` | Shoulder line angle delta |
| `balance` | Hip center X deviation |
| `bat_path` | Wrist trajectory smoothness |

### Bowling (Bowling Action / Spin Bowling)
| Metric | Description |
|---|---|
| `runup_rhythm` | Hip height consistency in run-up phase |
| `arm_angle` | Elbow angle at delivery |
| `release_height` | Wrist Y relative to shoulder at release |
| `front_foot_landing` | Ankle displacement at delivery stride |
| `followthrough_balance` | Hip stability post-release |

---

## ⚙️ Requirements

| Tool | Version |
|---|---|
| Python | 3.11+ |
| FFmpeg | Any recent (via `brew install ffmpeg`) |
| RAM | 2GB minimum, 4GB+ recommended |
| Web workers | 1 for small VPS, 2 only on larger hosts |
| GPU | Not required — CPU-only |

---

## 🔧 Architecture

```
Upload Video
  → Save to /uploads
  → FFmpeg: denoise + normalize + scale to 720p
  → OpenCV: extract up to 60 evenly-spaced frames
  → MediaPipe Pose: 33 landmark keypoints per frame
  → NumPy: compute 5 biomechanical metrics
  → Rule-based scorer: generate score (0–100) + issues
  → Report builder: assemble JSON
  → Return via /report/{job_id}
```

---

## 🔧 Debugging Ball Detection Issues

If your production server detects zero shots while local works fine:

```bash
# Run the diagnostic tool
python3 diagnose_ball_detection.py cricket_video.mov
```

This tool will:
- ✅ Check if YOLO model is available
- ✅ Test OpenCV color+motion detection
- ✅ Verify frame extraction
- ✅ Suggest fixes for your specific issue

**Common fixes:**
```bash
# Disable YOLO, use CV-only fallback (faster, less accurate)
docker run -e BALL_TRACKER_MODEL="" -p 8000:8000 cricket-ai

# Use smaller YOLO model (nanomodel)
docker run -e BALL_TRACKER_MODEL="yolov8n.pt" -p 8000:8000 cricket-ai
```

See [TROUBLESHOOTING_LOCAL_SERVER_PARITY.md](TROUBLESHOOTING_LOCAL_SERVER_PARITY.md) for complete diagnosis.

---

## 🧪 Testing the API with curl

```bash
# Upload a video
curl -X POST http://localhost:8000/upload-video \
  -F "file=@/path/to/cricket.mp4" \
  -F "drill_type=straight_drive"

# Check status
curl http://localhost:8000/status/{job_id}

# Get report
curl http://localhost:8000/report/{job_id}
```

---

## 📝 Notes

- **Video quality**: Low-quality Android videos are enhanced via FFmpeg before processing
- **Processing time**: 10–45 seconds depending on video length and CPU
- **Storage**: All files stored locally. No cloud required for POC
- **Concurrency**: Multiple videos can be processed in parallel via background threads
- **Low-RAM deployment**: set `UVICORN_WORKERS=1` and `ANALYSIS_INTERPOLATION_FACTOR=1`
