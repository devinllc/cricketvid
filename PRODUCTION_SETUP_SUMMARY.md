# 🏏 Ball Detection Parity Issue — Quick Summary

## Your Problem
Production server: **Empty wagon wheel, 111.3s processing**
Local: **Detected shots, 16.26s processing**

Both have matching frame counts (110) and `ANALYSIS_INTERPOLATION_FACTOR=1`, so frame extraction is consistent. The issue is **ball detection is failing on the server**.

---

## Why This Happens

The production server's ball detection pipeline (`_detect_ball_centers`) returns **zero detections**:

```
Production Flow:
├─ YOLO detection → ❌ (timeout or fails silently)
├─ CV fallback → ❌ (too strict for server video codec)
└─ Result: No valid ball tracked → "Ball could not be tracked" message

Local Flow:
├─ YOLO detection → ✅ Finds ball
├─ CV fallback → ✅ Works as backup
└─ Result: Shot detected successfully
```

**Root cause:** Server likely has:
1. YOLO timeout (causing 7x slowdown: 111.3s vs 16.26s)
2. Different video codec handling
3. Resource constraints preventing detection

---

## Quick Fixes (Try in Order)

### Fix 1: Disable YOLO (Force CV-Only)
**Fastest to implement**, slightly less accurate:

```bash
docker run -e BALL_TRACKER_MODEL="" -p 8000:8000 cricket-ai
```

**Why:** Avoids YOLO timeout; uses color+motion detection only.

### Fix 2: Use Nano Model
**Middle ground**, still accurate, faster:

```bash
docker run -e BALL_TRACKER_MODEL="yolov8n.pt" -p 8000:8000 cricket-ai
```

**Why:** Smaller model = faster inference on limited server resources.

### Fix 3: Check Server FFmpeg
**Verify infrastructure:**

```bash
docker exec CONTAINER_ID ffmpeg -version
# If missing:
apt-get install -y ffmpeg
```

---

## Verify Your Fix

After applying a fix, re-upload the same video and check:

```bash
curl http://server:8000/report/JOB_ID | jq '.analysis_meta.processing_time_sec'
```

Expected: Processing time drops significantly (from 111s → < 30s)

```bash
curl http://server:8000/report/JOB_ID | jq '.shot_summary.wagon_wheel.straight_total'
```

Expected: `1` (or > 0) instead of `0`

---

## Detailed Diagnosis

To understand exactly what's failing:

```bash
# Option A: Run on local first (as sanity check)
python3 diagnose_ball_detection.py test_video.mov

# Option B: Run inside Docker container
docker exec CONTAINER_ID python3 diagnose_ball_detection.py app/uploads/test_video.mov
```

This tool will tell you:
- ✅ Is YOLO available/working?
- ✅ Does OpenCV detect the ball?
- ✅ Are frames extracted correctly?
- ✅ What's the actual bottleneck?

---

## Files Created

| File | Purpose |
|------|---------|
| `TROUBLESHOOTING_LOCAL_SERVER_PARITY.md` | Comprehensive diagnosis & fixes |
| `diagnose_ball_detection.py` | Automated diagnostic tool |
| `PRODUCTION_SETUP_SUMMARY.md` | This file |

---

## Key Takeaway

**The issue is NOT frame extraction** (ANALYSIS_INTERPOLATION_FACTOR is correct).

**The issue IS ball detection failing on server** (likely YOLO timeout or CV detection too strict).

**Solution:** Disable YOLO or use smaller model to fall back to pure CV detection.

Try Fix 1 first → if it works, consider Fine-tuning later.

---

## Next Steps

1. **Immediate:** Apply Fix 1 or 2 above
2. **Verify:** Re-upload video, check processing time & shot_summary
3. **Diagnose:** Run `diagnose_ball_detection.py` if still failing
4. **Tune:** Adjust CV detection thresholds if needed (see TROUBLESHOOTING_LOCAL_SERVER_PARITY.md)
