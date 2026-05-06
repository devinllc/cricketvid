#!/usr/bin/env python3
"""
Diagnostic tool: Ball detection debugging for local/server parity issues.

Usage:
  python3 diagnose_ball_detection.py /path/to/video.mov
"""
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.services.frame_extractor import extract_frames
from app.utils.logger import get_logger

logger = get_logger(__name__)


def diagnose_yolo():
    """Check if YOLO model is available and working."""
    print("\n🔍 Checking YOLO availability...")
    try:
        from ultralytics import YOLO
        
        print("  ✅ ultralytics package available")
        
        # Try loading model
        try:
            model = YOLO("yolov8s.pt")
            print("  ✅ YOLOv8s model loaded successfully")
            return True
        except Exception as e:
            print(f"  ⚠️  YOLOv8s model load failed: {e}")
            # Try nano
            try:
                model = YOLO("yolov8n.pt")
                print("  ✅ YOLOv8n (nano) model loaded as fallback")
                return True
            except Exception as e2:
                print(f"  ❌ YOLOv8n model also failed: {e2}")
                return False
    except ImportError as e:
        print(f"  ❌ ultralytics not installed: {e}")
        return False


def diagnose_opencv_detection(frames: List[np.ndarray]) -> Tuple[int, int]:
    """Check if OpenCV color+motion detection works."""
    print("\n🔍 Checking OpenCV ball detection...")
    
    detected = 0
    failed = 0
    
    for i, frame in enumerate(frames[:10]):  # Test first 10 frames
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Cricket ball colors (red leather)
            lower_red1 = np.array([0, 80, 55])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 80, 55])
            upper_red2 = np.array([180, 255, 255])
            
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask_red1, mask_red2)
            
            if cv2.countNonZero(mask) > 0:
                detected += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Frame {i} processing error: {e}")
            failed += 1
    
    rate = 100 * detected / max(1, detected + failed)
    print(f"  ✅ OpenCV detection: {detected}/{detected+failed} frames with color match ({rate:.1f}%)")
    
    if rate < 50:
        print(f"  ⚠️  Low CV detection rate. Video codec or quality issue possible.")
    
    return detected, failed


def diagnose_frame_extraction(video_path: str) -> List[np.ndarray]:
    """Check if frames are extracted correctly."""
    print(f"\n🔍 Checking frame extraction from {Path(video_path).name}...")
    
    try:
        frames, fps = extract_frames(video_path)
        print(f"  ✅ Extracted {len(frames)} frames at {fps:.1f} FPS")
        
        if len(frames) == 0:
            print("  ❌ CRITICAL: No frames extracted!")
            return []
        
        # Check frame properties
        h, w = frames[0].shape[:2]
        print(f"  ✅ Frame size: {w}x{h}")
        
        # Check for degradation
        brightness_first = np.mean(frames[0])
        brightness_last = np.mean(frames[-1])
        brightness_diff = abs(brightness_first - brightness_last)
        print(f"  ℹ️  Brightness range: {brightness_first:.1f} → {brightness_last:.1f}")
        
        return frames
    except Exception as e:
        print(f"  ❌ Frame extraction failed: {e}")
        return []


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} VIDEO_PATH")
        print(f"\nExample: {sys.argv[0]} cricket_batting.mov")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("🏏 Cricket Video — Ball Detection Diagnosis")
    print("=" * 60)
    
    # Step 1: Frame extraction
    frames = diagnose_frame_extraction(video_path)
    if not frames:
        print("\n❌ Cannot proceed: frame extraction failed")
        sys.exit(1)
    
    # Step 2: YOLO availability
    yolo_available = diagnose_yolo()
    
    # Step 3: OpenCV fallback
    cv_detected, cv_failed = diagnose_opencv_detection(frames)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    
    if yolo_available and cv_detected > 0:
        print("✅ Both YOLO and OpenCV detection available — should work fine")
    elif yolo_available:
        print("⚠️  YOLO available but OpenCV detection low — try disabling YOLO:")
        print("    docker run -e BALL_TRACKER_MODEL='' -p 8000:8000 cricket-ai")
    elif cv_detected > 0:
        print("⚠️  YOLO unavailable, relying on OpenCV — detection rate: {:.1f}%".format(
            100 * cv_detected / (cv_detected + cv_failed)))
        print("    Consider installing ultralytics: pip install ultralytics")
    else:
        print("❌ CRITICAL: Neither YOLO nor OpenCV can detect ball!")
        print("   Possible causes:")
        print("   1. Video codec not supported")
        print("   2. Ball color outside detection ranges")
        print("   3. Video quality too degraded")
        print("   4. FFmpeg re-encoding failed")
    
    print("\nFor more help, see: TROUBLESHOOTING_LOCAL_SERVER_PARITY.md")


if __name__ == "__main__":
    main()
