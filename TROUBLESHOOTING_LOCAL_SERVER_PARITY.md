# Troubleshooting: Local vs. Production Server Parity

## Problem
Same video produces **different outputs** locally vs. on production/Docker:
- **Production:** Empty wagon wheel, "Ball could not be tracked..." message
- **Local:** Wagon wheel with shots detected

Processing times differ dramatically:
- **Production:** 111.3s (very slow)
- **Local:** 16.26s (normal)

---

## Root Causes

### 1. ⚠️ Ball Detection Failure (Most Common)
The production server returns **zero ball detections**, while local succeeds.

**Why this happens:**
- YOLO model attempts to load but times out or fails gracefully → only CV fallback runs
- CV detection thresholds too strict for video codec used on server
- Frame quality degraded due to codec differences between local recording and server processing

**Diagnostic signs:**
- Log: `"Extracted XXX frames..."` but `yolo_hits=0, cv_hits=0`
- Processing time > 90s (suggests YOLO timeout)

**Fix:**
```bash
# 1. Check if YOLO is loaded
curl http://your-server:8000/status/JOB_ID | grep -i "model_used"

# 2. Force CV-only fallback (disable YOLO on server)
docker run -e BALL_TRACKER_MODEL="" -p 8000:8000 cricket-ai

# 3. Check server logs for ball detection failures
docker logs CONTAINER_ID | grep -i "ball\|yolo\|detect"
```

---

### 2. ❌ ANALYSIS_INTERPOLATION_FACTOR Mismatch (Already Fixed)
Different frame counts between environments cause shot detection variations.

**Status:** ✅ Already configured correctly in current codebase
- Local: `export ANALYSIS_INTERPOLATION_FACTOR=1` (run.sh)
- Docker: `ENV ANALYSIS_INTERPOLATION_FACTOR=1` (Dockerfile)

**To verify:**
```bash
# Check local
grep "ANALYSIS_INTERPOLATION_FACTOR" run.sh

# Check Docker
grep "ANALYSIS_INTERPOLATION_FACTOR" Dockerfile

# Check server logs (should show interpolation_factor=1)
docker logs CONTAINER_ID | grep "Extracted.*frames"
```

---

### 3. 🔌 FFmpeg Codec Incompatibility
Server uses different video codec/encoding than local, causing frame degradation.

**Diagnostic signs:**
- Processing time > 100s (FFmpeg re-encoding delay)
- Frames on server appear different resolution/quality
- Log: "denoise + normalize + scale to 720p" taking too long

**Fix:**
```bash
# On server, pre-check FFmpeg is available
ffmpeg -version

# Check if video is being re-encoded (should take <5s for 110-frame video)
time ffmpeg -i video.mov -vf "yadif,scale=1280:720" -c:v libx264 output.mp4

# If > 5s, your server CPU is bottleneck
```

---

### 4. 📦 Memory/CPU Constraints
Server resources limited, causing timeouts or graceful failures.

**Diagnostic signs:**
- CPU usage spikes to 100% mid-processing
- Memory usage > 80%
- Process killed unexpectedly

**Fix:**
```bash
# Reduce workers
docker run -e UVICORN_WORKERS=1 -p 8000:8000 cricket-ai

# Reduce YOLO model size
docker run -e BALL_TRACKER_MODEL="yolov8n.pt" -p 8000:8000 cricket-ai
```

---

## Step-by-Step Diagnosis

### Step 1: Verify Frame Extraction
Check if frames are being extracted consistently:

```bash
# Local
export ANALYSIS_INTERPOLATION_FACTOR=1
python3 -c "
from app.services.frame_extractor import extract_frames
frames, fps = extract_frames('test_video.mov')
print(f'Frames: {len(frames)}, FPS: {fps}')
"

# Server (in Docker)
docker exec CONTAINER_ID python3 -c "
from app.services.frame_extractor import extract_frames
frames, fps = extract_frames('app/uploads/test_video.mov')
print(f'Frames: {len(frames)}, FPS: {fps}')
"
```

**Expected result:** Same frame count and FPS on both.

### Step 2: Test Ball Detection in Isolation
Check if ball detection is working:

```bash
python3 << 'EOF'
from app.services.ball_tracker import _detect_ball_centers
from app.services.frame_extractor import extract_frames

frames, fps = extract_frames('test_video.mov')
centers, model_used = _detect_ball_centers(frames)
detected = sum(1 for c in centers if c is not None)
print(f"Model: {model_used} | Detections: {detected}/{len(frames)} | Detection rate: {100*detected/len(frames):.1f}%")

if detected == 0:
    print("❌ CRITICAL: No ball detected. Check YOLO/CV pipeline.")
EOF
```

**Expected result:** Detection rate > 10%.

### Step 3: Compare Video Properties
Check if video codec/resolution differs:

```bash
# Local
ffprobe -v quiet -print_format json -show_format -show_streams test_video.mov

# Server
docker exec CONTAINER_ID ffprobe -v quiet -print_format json -show_format -show_streams app/uploads/test_video.mov
```

**Expected result:** Identical codec, resolution, fps.

---

## Quick Fixes (In Order of Likelihood)

### Fix 1: Disable YOLO on Server (Force CV Fallback)
```dockerfile
ENV BALL_TRACKER_MODEL=""
```

**Impact:** Faster processing (avoids YOLO timeout), but less accurate detection.

### Fix 2: Use Smaller YOLO Model
```dockerfile
ENV BALL_TRACKER_MODEL="yolov8n.pt"
```

**Impact:** Faster inference (nanomodel), slightly less accurate.

### Fix 3: Reduce Server Processing Workers
```dockerfile
ENV UVICORN_WORKERS=1
```

**Impact:** Prevents resource contention; slower concurrent processing.

### Fix 4: Pre-cache FFmpeg Codec
```bash
ffmpeg -codecs | grep h264  # Verify h264 available
```

If missing, install on server:
```bash
apt-get install -y ffmpeg
```

---

## Verify Fix
After applying a fix, re-upload the same video and check:

```bash
curl http://server:8000/report/JOB_ID | jq '.shot_summary.wagon_wheel'
```

**Expected:**
```json
{
  "dominant_side": "straight",
  "off_side_total": 0,
  "leg_side_total": 0,
  "straight_total": 1,
  "summary": "Off-side: 0 shots | Leg-side: 0 shots | Straight: 1 shots"
}
```

If still empty, proceed to step-by-step diagnosis above.

---

## Configuration Comparison

| Setting | Local | Docker | Status |
|---------|-------|--------|--------|
| `ANALYSIS_INTERPOLATION_FACTOR` | 1 | 1 | ✅ Matching |
| `BALL_TRACKER_MODEL` | auto | auto | ⚠️ May differ |
| `UVICORN_WORKERS` | 1 | 1 | ✅ Matching |
| FFmpeg | Installed | Installed | ✅ Both have |
| Python Version | 3.11+ | 3.11 | ✅ Matching |
| YOLO installed | Yes/No | Yes/No | ❓ Check logs |

---

## Long-Term Solution

1. **Add ball detection logging** to understand which frames fail
2. **Cache YOLO model** at build time (already done in Dockerfile)
3. **Add health check endpoint** that tests ball detection on sample frame
4. **Use cloud GPU** if YOLO too slow on server CPU

See [deployment_guide.md](deployment_guide.md) for production best practices.
