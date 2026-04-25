# Cricket Vision Platform Design

## 1. Goal
Upgrade the current cricket video assessment application into a FullTrack AI-style broadcast-grade sports vision and AR analytics platform.

This document defines:
- Current state
- Target production architecture
- Recommended stack
- Vision pipeline upgrades
- Migration plan from MVP to platform

## 2. Current State (What Exists Today)

### 2.1 Strengths
- FastAPI API and report UI
- Background processing pipeline
- MediaPipe pose analysis
- OpenCV video processing
- Optional YOLO detection path
- Report generation with trajectory metadata

### 2.2 Current Bottlenecks
- In-memory job state and thread-per-job processing
- Limited production resilience and horizontal scaling
- Primarily 2D trajectory fitting with limited camera geometry
- No durable queue, no persistent analytics store, limited observability

## 3. Target Product Direction

Evolve from:
- Video analysis app

To:
- Sports vision plus AR analytics platform

Target user experience:
- Broadcast-like trajectory overlays
- Calibrated pitch and wicket alignment
- High-confidence event analytics (bounce, impact, shot type)
- Reliable processing at scale with job durability

## 4. Recommended Production Stack

### 4.1 Core Backend
- FastAPI
- Uvicorn
- Gunicorn
- Redis
- Celery

Reason:
- Replace thread-per-job with durable async job orchestration:
  API -> Redis Queue -> Worker Pool

### 4.2 Vision and AI Stack
- Ball detection:
  - YOLOv8 custom trained
  - RT-DETR (alternate high-accuracy detector)
  - SAHI slicing for tiny-object recall
- Pose and form:
  - Keep MediaPipe
  - Optional: MoveNet or OpenPose for specific scenarios
- Tracking:
  - Kalman Filter
  - ByteTrack
  - SORT or DeepSORT as fallback strategies

### 4.3 Geometry and Calibration Stack
- OpenCV homography
- Line detection for pitch and crease
- Perspective transform
- Vanishing point solver

Must detect:
- Crease lines
- Wicket line
- Pitch boundaries

### 4.4 Rendering Stack
- Option A (fast export): OpenCV + FFmpeg
- Option B (premium visuals): Blender scripted rendering
- Option C (interactive playback): Three.js web renderer
- Future live AR: Unity client pipeline

### 4.5 Data and Storage
- Object storage: Amazon Web Services S3
- Relational data: PostgreSQL
- Optional analytics/document store: MongoDB

### 4.6 Monitoring and Reliability
- Prometheus
- Grafana
- Sentry

## 5. Target System Architecture

1. User uploads video
2. API Gateway (FastAPI)
3. Auth and rate limiting middleware
4. Redis job queue
5. GPU worker cluster executes pipeline:
   - Video normalization
   - Camera calibration
   - Ball detect and track
   - Bounce detection
   - Trajectory physics
   - Pose analysis
   - Overlay rendering
6. Outputs stored in S3 and served via CDN
7. Report API plus dashboard consume metadata and assets

## 6. Internal Vision Pipeline (Target)

1. Frame decode
2. Pitch line and structure detection
3. Homography estimation
4. Ball detection (tiny-object optimized)
5. Kalman-smoothed tracking and ID consistency
6. Bounce detection
7. Bat impact detection
8. Physics-guided trajectory reconstruction
9. FullTrack-style overlay rendering
10. MP4 export and metadata packaging

## 7. Key Technical Upgrades Over Current System

### 7.1 Job Orchestration
Current:
- In-memory job store, restart loses jobs

Target:
- Redis + Celery + persistent DB state

### 7.2 Trajectory Modeling
Current:
- Basic polynomial fit

Target:
- Physics-informed model with:
  - Projectile motion
  - Bounce coefficient
  - Lateral movement component
  - Spline smoothing where appropriate

### 7.3 Calibration and AR Alignment
Current:
- Limited geometric alignment

Target:
- Full homography-driven world-to-image mapping for:
  - Blue pitch lane
  - Aligned wickets and crease
  - Consistent broadcast-grade trajectories

### 7.4 Confidence Engine
Every event output should include confidence signals, for example:

```json
{
  "tracking_confidence": 0.91,
  "bounce_confidence": 0.84,
  "impact_confidence": 0.87
}
```

## 8. Model Training Requirements

Custom labeled datasets should include:
- Ball bounding boxes
- Bounce point
- Pitch corner points
- Stump positions
- Bat impact frame

Recommended tooling:
- Roboflow
- CVAT

## 9. Infrastructure Recommendation

### 9.1 MVP-Plus
- Single GPU node for worker pool (for example Runpod or Lambda Labs)

### 9.2 Scale-Out
- Amazon Web Services ECS GPU autoscaling
- Queue-based workload balancing
- Multi-worker model version rollouts

## 10. Target Repository Structure

```text
app/
  api/
  workers/
  ai/
    ball_detector/
    tracker/
    pose/
    calibrator/
  engine/
    trajectory.py
    bounce.py
    physics.py
    renderer.py
  reports/
  storage/
```

## 11. Migration Plan

### Phase 1: Production Foundations
- Introduce Redis and Celery workers
- Replace in-memory jobs with persistent job metadata
- Add retry, timeout, dead-letter handling

### Phase 2: Vision Quality Core
- Integrate custom tiny-ball detector + SAHI
- Add Kalman + ByteTrack pipeline
- Add robust bounce and impact confidence models

### Phase 3: Calibration and Broadcast Overlay
- Implement line detection + homography service
- Calibrate pitch space and wicket anchors
- Upgrade overlay renderer with consistent AR alignment

### Phase 4: Platform and Productization
- S3 plus CDN output delivery
- Dashboard modernization (React)
- Monitoring stack and model observability

## 12. Product Roadmap

### V2
- Live mobile camera tracking
- Real-time speed estimates
- Bowling machine integration

### V3
- Spin axis prediction
- Seam tracking
- Hawk-eye style replay workflows

## 13. Recommended Core Stack (Final)
- FastAPI
- Redis
- Celery
- YOLOv8 custom detector
- MediaPipe
- OpenCV
- SciPy
- PostgreSQL
- Amazon Web Services S3
- FFmpeg
- React dashboard

## 14. Practical Truth
Improving only detector quality is not enough for FullTrack-level output.

The biggest visual quality gains come from the combined system:
- Calibration
- Tracking consistency
- Physics-aware smoothing
- AR-grade rendering

When these four are integrated, user-perceived quality jumps from analytics demo to broadcast product.
