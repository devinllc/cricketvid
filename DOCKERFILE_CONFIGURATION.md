# Dockerfile Configuration Options

## Option 1: CV-Only (No YOLO)
**Best for:** Limited server resources, quick fix, acceptable accuracy loss

```dockerfile
# Add this line to Dockerfile
ENV BALL_TRACKER_MODEL=""
```

**Pros:**
- Fastest (avoids YOLO inference entirely)
- Reliable CPU-only solution
- Good for low-resource servers

**Cons:**
- Slightly lower detection accuracy
- May miss small balls or poor lighting

**Build & Run:**
```bash
docker build -t cricket-ai .
docker run -e BALL_TRACKER_MODEL="" -p 8000:8000 cricket-ai
```

---

## Option 2: Nano Model (YOLOv8n)
**Best for:** Balance of speed & accuracy

```dockerfile
# Add this line to Dockerfile
ENV BALL_TRACKER_MODEL="yolov8n.pt"
```

**Pros:**
- Faster than standard (v8s) model
- Better accuracy than CV-only
- Still accurate for most videos

**Cons:**
- Slower than CV-only
- Requires YOLO installed

**Build & Run:**
```bash
docker build -t cricket-ai .
docker run -e BALL_TRACKER_MODEL="yolov8n.pt" -p 8000:8000 cricket-ai
```

---

## Option 3: Standard Model (YOLOv8s) — Default
**Best for:** High-quality detection, server has resources

```dockerfile
# Default configuration — no change needed
ENV BALL_TRACKER_MODEL="yolov8s.pt"
```

**Pros:**
- Highest accuracy
- Most robust to poor video quality

**Cons:**
- Slowest option
- Requires more server resources

---

## Option 4: Custom — Test Different Models
```dockerfile
# Try other YOLO variants
ENV BALL_TRACKER_MODEL="yolov8m.pt"  # medium
ENV BALL_TRACKER_MODEL="yolov8l.pt"  # large
```

---

## Complete Updated Dockerfile Example

```dockerfile
# ... existing code ...

# ── Runtime config ──────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV UVICORN_WORKERS=1
ENV ANALYSIS_INTERPOLATION_FACTOR=1

# Choose one ball tracker configuration:
# Option 1: CV-only (fastest, less accurate)
# ENV BALL_TRACKER_MODEL=""

# Option 2: Nano model (balanced)
ENV BALL_TRACKER_MODEL="yolov8n.pt"

# Option 3: Standard model (most accurate, slower)
# ENV BALL_TRACKER_MODEL="yolov8s.pt"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## How to Apply

1. **Edit Dockerfile:**
   ```bash
   # Option A: Disable YOLO
   sed -i 's/ENV BALL_TRACKER_MODEL=.*/ENV BALL_TRACKER_MODEL=""/' Dockerfile
   
   # Option B: Use nano model
   sed -i 's/ENV BALL_TRACKER_MODEL=.*/ENV BALL_TRACKER_MODEL="yolov8n.pt"/' Dockerfile
   ```

2. **Rebuild & Redeploy:**
   ```bash
   docker build -t cricket-ai .
   docker run -p 8000:8000 cricket-ai
   ```

3. **Verify Fix:**
   ```bash
   curl http://localhost:8000/health  # Should return 200
   ```

---

## Performance Comparison

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| CV-Only | ⚡⚡ Fast | ⭐⭐⭐ Good | Low-resource servers |
| YOLOv8n | ⚡ Medium | ⭐⭐⭐⭐ Very Good | Balanced servers |
| YOLOv8s | 🐢 Slow | ⭐⭐⭐⭐⭐ Excellent | High-resource servers |

---

## Troubleshooting Each Option

### If using CV-Only and still no detections:
```bash
# Check if ball color ranges need tuning
python3 diagnose_ball_detection.py your_video.mov
```

### If using YOLOv8n and still slow (>60s):
```bash
# Check server resources
docker stats CONTAINER_ID

# May need to reduce workers
docker run -e UVICORN_WORKERS=1 -p 8000:8000 cricket-ai
```

### If using YOLOv8s and timeout:
```bash
# Definitely downgrade to nano or CV-only
docker run -e BALL_TRACKER_MODEL="yolov8n.pt" -p 8000:8000 cricket-ai
```

---

## Recommended Strategy

1. **Start with:** CV-Only (`BALL_TRACKER_MODEL=""`)
2. **Test:** Upload video, check if shots detected
3. **If working:** Done! 
4. **If not:** Run `diagnose_ball_detection.py` to see why
5. **Upgrade to:** Nano model (`BALL_TRACKER_MODEL="yolov8n.pt"`)
6. **If still failing:** Adjust CV detection thresholds (advanced)

---

**See also:**
- [TROUBLESHOOTING_LOCAL_SERVER_PARITY.md](TROUBLESHOOTING_LOCAL_SERVER_PARITY.md) — Full diagnosis
- [PRODUCTION_SETUP_SUMMARY.md](PRODUCTION_SETUP_SUMMARY.md) — Quick summary
