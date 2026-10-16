"""
Batch Dataset Extractor for License Plate Detection
Extracts high-resolution vehicle crops from all input videos into a single unified folder
with frame skipping (every 5th frame) and smart front-focused padding.
"""

import os
import sys
import glob
import time
import csv
import cv2
import numpy as np
from ultralytics import YOLO

# Fix Windows console encoding for Unicode/emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

# -------------------------
# PATHS & CONFIGURATION
# -------------------------
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "plate_dataset_crops")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "dataset_manifest.csv")

# Locate YOLO object detection model
MODEL_PATH = os.path.join(PROJECT_ROOT, "Models", "object_detection.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "object_detection.pt")

TRACKER_CONFIG = "botsort.yaml"

# Vehicle classes: 1=car, 2=bus, 3=truck, 4=motorcycle, 5=rickshaw (0=person excluded)
VEHICLE_CLASSES = [1, 2, 3, 4, 5]
CLASS_NAMES = {
    1: "car",
    2: "bus",
    3: "truck",
    4: "motorcycle",
    5: "rickshaw"
}

# Performance & Extraction settings
FRAME_SKIP = 5           # Process every 5th frame for speed
RESIZE_RATIO = 0.4       # Resize factor for YOLO inference speed
CONF_THRES = 0.20        # Detection confidence threshold
MAX_CROPS_PER_VEHICLE = 5 # Up to 5 diverse angles per vehicle
ZONES_COUNT = 3          # Divide ROI into vertical approach zones
MAX_PER_ZONE = 2         # Max crops per approach zone
MIN_MOVEMENT_PX = 45     # Must move at least 45px before capturing another crop

# Smart padding for capturing plate area
PADDING_X_FRONT = 350
PADDING_X_BACK = 180
PADDING_Y = 350


def sanitize_bbox(x1, y1, x2, y2, max_w, max_h):
    return max(0, min(x1, max_w - 1)), max(0, min(y1, max_h - 1)), max(0, min(x2, max_w)), max(0, min(y2, max_h))


def get_direction_and_padding(x1, y1, x2, y2, track_id, history_dict, max_w, max_h):
    """Determine vehicle travel direction and apply front-focused padding."""
    center_x = (x1 + x2) // 2
    direction = "right"

    if track_id in history_dict and len(history_dict[track_id]) > 0:
        prev_x = history_dict[track_id][-1][0]
        dx = center_x - prev_x
        if dx < -2:
            direction = "left"
        elif dx > 2:
            direction = "right"
        else:
            direction = "right" if center_x > (max_w // 2) else "left"
    else:
        direction = "right" if center_x > (max_w // 2) else "left"

    if direction == "right":
        x1p = max(0, x1 - PADDING_X_BACK)
        x2p = min(max_w, x2 + PADDING_X_FRONT)
    else:
        x1p = max(0, x1 - PADDING_X_FRONT)
        x2p = min(max_w, x2 + PADDING_X_BACK)

    y1p = max(0, y1 - PADDING_Y)
    y2p = min(max_h, y2 + PADDING_Y)

    return x1p, y1p, x2p, y2p, direction


def process_video(video_path, model, manifest_writer, video_idx, total_videos):
    video_name = os.path.basename(video_path)
    video_stem = os.path.splitext(video_name)[0]
    # Clean stem for compact filenames (e.g. PXL_20260808_084128609 -> PXL_084128)
    if video_stem.startswith("PXL_"):
        parts = video_stem.split("_")
        clean_stem = f"{parts[0]}_{parts[-1][:6]}" if len(parts) >= 3 else video_stem
    else:
        clean_stem = video_stem

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {video_path}")
        return 0

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Standard ROI covering approach road area
    roi_top = int(0.26 * orig_h)
    roi_bottom = int(0.96 * orig_h)
    roi_height = max(1, roi_bottom - roi_top)
    zone_size = max(1, roi_height // ZONES_COUNT)

    small_w = int(orig_w * RESIZE_RATIO)
    small_h = int(orig_h * RESIZE_RATIO)
    scale_x = orig_w / small_w
    scale_y = orig_h / small_h

    print(f"\n{'='*70}")
    print(f"🎬 [{video_idx}/{total_videos}] Processing: {video_name}")
    print(f"   Resolution: {orig_w}x{orig_h} | FPS: {fps:.1f} | Frames: {total_frames}")
    print(f"   ROI: Y {roi_top} to {roi_bottom} | Frame Step: Every {FRAME_SKIP} frames")
    print(f"{'='*70}")

    # Tracking states
    saved_counts = {}   # track_id -> total crops
    zone_counts = {}    # track_id -> {zone_idx: count}
    last_centers = {}   # track_id -> (cx, cy)
    center_history = {} # track_id -> [(cx, cy), ...]

    frame_idx = 0
    crops_saved_this_video = 0
    t_start = time.time()

    while True:
        # Fast skipping with cap.grab()
        ret = cap.grab()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % FRAME_SKIP != 0:
            continue

        # Full decode only on the sampled frame
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            continue

        # Resize for high-speed tracking
        frame_small = cv2.resize(frame, (small_w, small_h))

        # Run BoT-SORT tracking
        try:
            results = model.track(
                source=frame_small,
                persist=True,
                tracker=TRACKER_CONFIG,
                classes=VEHICLE_CLASSES,
                conf=CONF_THRES,
                verbose=False
            )
        except Exception as e:
            continue

        if not results or len(results) == 0 or results[0].boxes is None or results[0].boxes.id is None:
            continue

        boxes = results[0].boxes
        track_ids = boxes.id.cpu().numpy().astype(int)
        coords = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        for box, track_id, cls_id, conf in zip(coords, track_ids, classes, confs):
            # Scale coordinates back to 4K resolution
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)

            x1, y1, x2, y2 = sanitize_bbox(x1, y1, x2, y2, orig_w, orig_h)
            box_w, box_h = x2 - x1, y2 - y1

            if box_w < 40 or box_h < 40:
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # ROI filter
            if center_y < roi_top or center_y > roi_bottom:
                continue

            zone_idx = min(ZONES_COUNT - 1, max(0, (center_y - roi_top) // zone_size))

            if track_id not in saved_counts:
                saved_counts[track_id] = 0
                zone_counts[track_id] = {z: 0 for z in range(ZONES_COUNT)}
                center_history[track_id] = []

            # Check limits
            if saved_counts[track_id] >= MAX_CROPS_PER_VEHICLE:
                continue
            if zone_counts[track_id][zone_idx] >= MAX_PER_ZONE:
                continue

            # Movement filter to ensure diverse angle
            if track_id in last_centers:
                prev_cx, prev_cy = last_centers[track_id]
                dist = np.hypot(center_x - prev_cx, center_y - prev_cy)
                if dist < MIN_MOVEMENT_PX:
                    continue

            # Smart padding for clear plate view
            x1p, y1p, x2p, y2p, direction = get_direction_and_padding(
                x1, y1, x2, y2, track_id, center_history, orig_w, orig_h
            )

            if x2p <= x1p or y2p <= y1p:
                continue

            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0 or crop.shape[0] < 50 or crop.shape[1] < 50:
                continue

            # Save in single unified output folder
            vehicle_type = CLASS_NAMES.get(cls_id, "vehicle")
            crop_filename = f"{clean_stem}_id{track_id:04d}_frame{frame_idx:06d}_{vehicle_type}.jpeg"
            crop_path = os.path.join(OUTPUT_DIR, crop_filename)

            cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Update counters
            saved_counts[track_id] += 1
            zone_counts[track_id][zone_idx] += 1
            last_centers[track_id] = (center_x, center_y)
            center_history[track_id].append((center_x, center_y))
            crops_saved_this_video += 1

            # Write manifest row
            manifest_writer.writerow([
                crop_filename,
                video_name,
                track_id,
                vehicle_type,
                frame_idx,
                round(float(conf), 2),
                box_w,
                box_h,
                crop.shape[1],
                crop.shape[0]
            ])

        # Progress reporting every 500 frames
        if frame_idx % 500 == 0:
            elapsed = time.time() - t_start
            pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            fps_proc = frame_idx / max(1e-3, elapsed)
            print(f"   ⏳ Frame {frame_idx}/{total_frames} ({pct:.1f}%) | "
                  f"Crops: {crops_saved_this_video} | Speed: {fps_proc:.1f} fps")

    cap.release()
    elapsed = time.time() - t_start
    print(f"✨ Finished {video_name}: {crops_saved_this_video} crops extracted in {elapsed:.1f}s")
    return crops_saved_this_video


def main():
    print("=" * 70)
    print("🚀 BATCH VEHICLE & PLATE DATASET EXTRACTOR")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.mp4")))

    if not video_files:
        print(f"❌ No .mp4 files found in {INPUT_DIR}")
        return

    print(f"📂 Output Directory: {OUTPUT_DIR}")
    print(f"📹 Found {len(video_files)} videos in {INPUT_DIR}")
    print(f"🧠 Loading YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # Initialize manifest CSV
    manifest_file = open(MANIFEST_PATH, "w", newline="", encoding="utf-8")
    manifest_writer = csv.writer(manifest_file)
    manifest_writer.writerow([
        "filename", "video_source", "track_id", "vehicle_type", "frame_idx",
        "confidence", "box_w", "box_h", "crop_w", "crop_h"
    ])

    total_crops = 0
    t_global_start = time.time()

    for idx, v_path in enumerate(video_files, 1):
        try:
            crops = process_video(v_path, model, manifest_writer, idx, len(video_files))
            total_crops += crops
            manifest_file.flush()
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(v_path)}: {e}")

    manifest_file.close()
    total_time = time.time() - t_global_start

    print("\n" + "=" * 70)
    print(f"🎉 BATCH EXTRACTION COMPLETE!")
    print(f"📸 Total Crops Saved: {total_crops}")
    print(f"📁 Destination Folder: {OUTPUT_DIR}")
    print(f"📝 Dataset Manifest: {MANIFEST_PATH}")
    print(f"⏱️ Total Time Elapsed: {total_time/60:.2f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
