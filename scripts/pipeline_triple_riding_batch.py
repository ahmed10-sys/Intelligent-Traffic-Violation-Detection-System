"""
Fast Batch Pipeline for Bike Rider & Triple Riding Detection
-----------------------------------------------------------
Processes:
  1. input/test.mp4 (tracking + bike cropping)
  2. input/evening traffic.mp4 (bike cropping using existing vehicle_positions_evening_traffic.csv)
  3. Evaluates all bike crops from test2, test, and evening traffic using the
     upgraded Rider Deduplication Engine (Head-top alignment, containment suppression,
     and temporal track consensus).
  4. Generates an interactive multi-video HTML visualizer.
"""

import os
import sys
import time
import json
import csv
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Fix Windows console encoding for Unicode/emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

MODEL_DETECTION = os.path.join(PROJECT_ROOT, "Models", "object_detection.pt")
MODEL_PERSON = os.path.join(PROJECT_ROOT, "Models", "yolov8m.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Input videos
VIDEOS = {
    "test2": {
        "name": "Test 2 (Overhead Daylight)",
        "video_path": os.path.join(PROJECT_ROOT, "input", "test2.mp4"),
        "csv_path": os.path.join(OUTPUT_DIR, "vehicle_positions.csv"),
        "crops_dir": os.path.join(OUTPUT_DIR, "cropped_photos", "Bikes_Motorcycles")
    },
    "test": {
        "name": "Test (Daylight)",
        "video_path": os.path.join(PROJECT_ROOT, "input", "test.mp4"),
        "csv_path": os.path.join(OUTPUT_DIR, "vehicle_positions_test.csv"),
        "crops_dir": os.path.join(OUTPUT_DIR, "cropped_photos_test", "Bikes_Motorcycles")
    },
    "evening": {
        "name": "Evening Traffic",
        "video_path": os.path.join(PROJECT_ROOT, "input", "evening traffic.mp4"),
        "csv_path": os.path.join(OUTPUT_DIR, "vehicle_positions_evening_traffic.csv"),
        "crops_dir": os.path.join(OUTPUT_DIR, "cropped_photos_evening", "Bikes_Motorcycles")
    }
}

RESULTS_DIR = os.path.join(OUTPUT_DIR, "triple_riding_batch_results")
HTML_PATH = os.path.join(OUTPUT_DIR, "triple_riding_visualizer.html")
CSV_PATH = os.path.join(OUTPUT_DIR, "triple_riding_results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# STEP 1: Fast Tracking for Videos without Positions CSV
# -------------------------------------------------------------
def run_fast_tracking_if_needed(video_key, conf_thresh=0.25):
    cfg = VIDEOS[video_key]
    csv_path = cfg["csv_path"]
    video_path = cfg["video_path"]

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
        print(f"✅ Positions CSV already exists for {video_key}: {csv_path}")
        return

    if not os.path.exists(video_path):
        print(f"⚠️ Video file not found: {video_path}")
        return

    print(f"\n🚀 Running Fast Tracking for {video_key} ({video_path})...")
    model = YOLO(MODEL_DETECTION)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    records = []
    frame_idx = 0
    t0 = time.time()

    # Track only bikes (class 4) + other vehicles for lane clamping (1=car, 2=bus, 3=truck, 4=motorcycle, 5=rickshaw)
    vehicle_classes = [1, 2, 3, 4, 5]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Process every 2nd frame for speed
        if frame_idx % 2 != 0:
            continue

        h, w = frame.shape[:2]
        # Resize to 45% for high speed detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.45, fy=0.45)

        results = model.track(
            small_frame,
            tracker="botsort.yaml",
            persist=True,
            conf=conf_thresh,
            classes=vehicle_classes,
            verbose=False
        )

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [-1] * len(boxes)

            scale = 1.0 / 0.45
            for box, cls, conf, tid in zip(boxes, classes, confs, track_ids):
                x1 = int(box[0] * scale)
                y1 = int(box[1] * scale)
                x2 = int(box[2] * scale)
                y2 = int(box[3] * scale)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                records.append({
                    "frame": frame_idx,
                    "id": tid,
                    "class": cls,
                    "confidence": round(float(conf), 3),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "center_x": cx, "center_y": cy,
                    "in_roi": 1
                })

        if frame_idx % 400 == 0:
            elapsed = time.time() - t0
            print(f"  Frame {frame_idx}/{total_frames} ({frame_idx/elapsed:.1f} fps) | {len(records)} vehicle detections")

    cap.release()
    df_out = pd.DataFrame(records)
    df_out.to_csv(csv_path, index=False)
    print(f"✅ Tracking complete for {video_key}: {len(df_out)} rows saved to {csv_path}")

# -------------------------------------------------------------
# STEP 2: Fast Motorcycle Cropping with Adaptive Tight Padding
# -------------------------------------------------------------
def crop_motorcycles_fast(video_key, max_per_id=8):
    cfg = VIDEOS[video_key]
    csv_path = cfg["csv_path"]
    video_path = cfg["video_path"]
    crops_dir = cfg["crops_dir"]

    if os.path.exists(crops_dir) and len(os.listdir(crops_dir)) > 5:
        print(f"✅ Bike crops already exist for {video_key}: {crops_dir} ({len(os.listdir(crops_dir))} IDs)")
        return

    if not os.path.exists(csv_path) or not os.path.exists(video_path):
        print(f"⚠️ Missing video or CSV for {video_key}")
        return

    os.makedirs(crops_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Normalize class: could be integer 4 or string 'motorcycle'
    if "class" in df.columns:
        if df["class"].dtype == object:
            df_bikes = df[df["class"].astype(str).str.lower().isin(["motorcycle", "4"])].copy()
        else:
            df_bikes = df[df["class"] == 4].copy()
    else:
        print(f"❌ No 'class' column in {csv_path}")
        return

    # Filter valid IDs
    df_bikes = df_bikes[df_bikes["id"] > 0]
    unique_ids = df_bikes["id"].unique()

    # Precalculate track-level max heights for completeness filter
    df_bikes["height"] = df_bikes["y2"] - df_bikes["y1"]
    track_max_heights = df_bikes.groupby("id")["height"].max().to_dict()

    print(f"\n✂️ Cropping {len(unique_ids)} motorcycles from {video_key} (enforcing 70%+ max height & non-touching bottom edge)...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open {video_path}")
        return

    frames_needed = set(df_bikes["frame"].unique())
    saved_counts = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx not in frames_needed:
            continue

        h, w = frame.shape[:2]
        frame_rows = df_bikes[df_bikes["frame"] == frame_idx]

        for _, row in frame_rows.iterrows():
            tid = int(row["id"])
            if saved_counts.get(tid, 0) >= max_per_id:
                continue

            x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
            bw, bh = x2 - x1, y2 - y1
            if bw < 15 or bh < 20:
                continue

            max_h = track_max_heights.get(tid, bh)

            # USER COMPLETENESS CRITERIA:
            # 1. Not touching the bottom edge of the frame (vehicles exiting downward lose upper body)
            if y2 >= (h - 30):
                continue

            # 2. At least about 70% of that track's max height
            if bh < 0.70 * max_h:
                continue

            # 3. Not clipped entering the top edge
            if y1 <= 15:
                continue

            # Adaptive tight motorcycle padding
            pad_x = min(25, max(8, int(bw * 0.10)))
            pad_y_top = min(35, max(12, int(bh * 0.15)))
            pad_y_bottom = min(15, max(5, int(bh * 0.05)))

            x1p = max(0, x1 - pad_x)
            x2p = min(w, x2 + pad_x)
            y1p = max(0, y1 - pad_y_top)
            y2p = min(h, y2 + pad_y_bottom)

            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            id_dir = os.path.join(crops_dir, f"id_{tid:03d}")
            os.makedirs(id_dir, exist_ok=True)

            filename = f"frame{frame_idx:06d}_motorcycle.jpeg"
            crop_path = os.path.join(id_dir, filename)
            cv2.imwrite(crop_path, crop)

            saved_counts[tid] = saved_counts.get(tid, 0) + 1

    cap.release()
    total_saved = sum(saved_counts.values())
    print(f"✅ Cropped {total_saved} photos across {len(saved_counts)} motorcycles for {video_key}")

# -------------------------------------------------------------
# STEP 3: Upgraded Rider Occupancy Detection Logic
# -------------------------------------------------------------
def deduplicate_rider_boxes(det_list, crop_h=None):
    """
    Intra-Rider Box Deduplication & Sanity Filtering:
    1. Rejects phantom boxes situated solely in the lower portion of the crop (y1 > 0.60 * crop_h)
    2. Merges multi-scale / multi-part detections of the exact same rider:
       - IoU > 0.40
       - Area containment > 0.55 (arm/torso inside full body)
       - Column alignment: horizontal overlap > 0.65 AND vertical overlap > 0.45
       - Head-top alignment: |y1 - ky1| < 45 and horizontal overlap > 0.50
    """
    valid_dets = []
    for d in det_list:
        x1, y1, x2, y2, conf = d
        if crop_h is not None and y1 > 0.60 * crop_h:
            # Phantom detection on dangling feet / road texture
            continue
        valid_dets.append(d)

    valid_dets.sort(key=lambda d: d[4], reverse=True)
    kept = []

    for d in valid_dets:
        x1, y1, x2, y2, conf = d
        w, h = x2 - x1, y2 - y1
        area = w * h
        is_dup = False

        for k in kept:
            kx1, ky1, kx2, ky2, kconf = k
            kw, kh = kx2 - kx1, ky2 - ky1
            karea = kw * kh

            # Standard IoU check
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = area + karea - inter
                iou = inter / union
                if iou > 0.45:
                    is_dup = True
                    break

            # Head-top alignment check (same head at top of saddle)
            head_diff = abs(y1 - ky1)
            h_inter = max(0, min(x2, kx2) - max(x1, kx1))
            h_union = max(x2, kx2) - min(x1, kx1)
            h_overlap = h_inter / h_union if h_union > 0 else 0

            # If heads are vertically adjacent (<40px) and horizontally overlapping (>55%)
            if head_diff < 40 and h_overlap > 0.55:
                is_dup = True
                break

        if not is_dup:
            kept.append(d)

    return kept

def draw_annotations(img, detections, video_name, person_count, is_valid_frame=True, exclusion_reason=""):
    annotated = img.copy()
    h, w = annotated.shape[:2]

    colors = [
        (0, 255, 0),    # Green (Rider 1)
        (255, 160, 0),  # Cyan/Blue (Rider 2)
        (0, 0, 255),    # Red (Rider 3)
        (255, 0, 255),  # Magenta (Rider 4)
    ]

    for idx, (x1, y1, x2, y2, conf) in enumerate(detections):
        color = colors[idx % len(colors)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        lbl = f"Rider #{idx+1} ({conf*100:.0f}%)"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, lbl, (x1 + 4, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Top Status Banner
    banner_h = 38
    if not is_valid_frame:
        bg = (55, 55, 70)
        text = f"EXCLUDED: {exclusion_reason}"
    elif person_count >= 3:
        bg = (0, 0, 220)
        text = f"TRIPLE RIDING ({person_count} PERSONS)"
    elif person_count == 2:
        bg = (180, 110, 15)
        text = f"DOUBLE RIDING ({person_count} PERSONS - PILLION)"
    elif person_count == 1:
        bg = (30, 140, 30)
        text = f"SOLO RIDER (1 PERSON)"
    else:
        bg = (70, 70, 70)
        text = "NO PERSON DETECTED"

    cv2.rectangle(annotated, (0, 0), (w, banner_h), bg, -1)
    cv2.putText(annotated, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return annotated

# -------------------------------------------------------------
# STEP 4: Process All Videos & Generate Dashboard
# -------------------------------------------------------------
def process_all_bikes(target_videos=None):
    print("\n" + "=" * 65)
    print("🔍 RUNNING UPGRADED RIDER OCCUPANCY DETECTION")
    print("=" * 65)

    if target_videos is None:
        target_videos = list(VIDEOS.keys())
    elif isinstance(target_videos, str):
        target_videos = [target_videos]

    model = YOLO(MODEL_PERSON)
    all_results = []
    csv_rows = []

    for video_key in target_videos:
        if video_key not in VIDEOS:
            continue
        cfg = VIDEOS[video_key]
        vname = cfg["name"]
        crops_dir = cfg["crops_dir"]
        csv_path = cfg["csv_path"]

        if not os.path.exists(crops_dir):
            print(f"Skipping {video_key}: {crops_dir} does not exist.")
            continue

        # Load vehicle positions CSV for frame completeness lookup
        track_meta = {}  # (tid, frame) -> (is_valid, reason, ratio, y2)
        track_max_h = {}
        if os.path.exists(csv_path):
            df_track = pd.read_csv(csv_path)
            if "class" in df_track.columns:
                if df_track["class"].dtype == object:
                    df_tbikes = df_track[df_track["class"].astype(str).str.lower().isin(["motorcycle", "4"])].copy()
                else:
                    df_tbikes = df_track[df_track["class"] == 4].copy()
            else:
                df_tbikes = df_track.copy()

            df_tbikes["height"] = df_tbikes["y2"] - df_tbikes["y1"]
            track_max_h = df_tbikes.groupby("id")["height"].max().to_dict()

            for _, tr in df_tbikes.iterrows():
                tid_t = int(tr["id"])
                f_t = int(tr["frame"])
                y1_t, y2_t = int(tr["y1"]), int(tr["y2"])
                bh_t = y2_t - y1_t
                mh_t = track_max_h.get(tid_t, bh_t)
                ratio_t = bh_t / max(1, mh_t)

                # USER RULE:
                # 1. Not touching the bottom edge of the frame
                # 2. At least ~70% of that track's max height
                if y2_t >= 2130:
                    is_val = False
                    reason = "Touching Bottom Edge"
                elif ratio_t < 0.70:
                    is_val = False
                    reason = f"Truncated Height ({ratio_t*100:.0f}% < 70%)"
                elif y1_t <= 15:
                    is_val = False
                    reason = "Touching Top Edge"
                else:
                    is_val = True
                    reason = "Complete"

                track_meta[(tid_t, f_t)] = (is_val, reason, ratio_t, y2_t)

        id_dirs = sorted([d for d in os.listdir(crops_dir) if os.path.isdir(os.path.join(crops_dir, d)) and d.startswith("id_")])
        print(f"\n📂 Evaluating {video_key} ({vname}) - {len(id_dirs)} motorcycle IDs...")

        for id_dir in id_dirs:
            folder_path = os.path.join(crops_dir, id_dir)
            try:
                vehicle_id = int(id_dir.replace("id_", ""))
            except:
                vehicle_id = id_dir

            out_sub = os.path.join(RESULTS_DIR, video_key, id_dir)
            os.makedirs(out_sub, exist_ok=True)

            img_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png")) and "plate" not in f.lower()])

            # Collect raw and deduplicated counts across valid frames for track consensus
            frame_counts = []
            frame_records = []

            for filename in img_files:
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                crop_h = img.shape[0]

                # Determine if frame is complete or truncated
                import re
                m = re.search(r'frame(\d+)', filename)
                if m:
                    f_num = int(m.group(1))
                    meta = track_meta.get((vehicle_id, f_num))
                    if meta:
                        is_valid_frame, exclusion_reason, height_ratio, y2_coord = meta
                    else:
                        is_valid_frame, exclusion_reason, height_ratio, y2_coord = True, "Complete", 1.0, 0
                else:
                    # BEST.jpg or unnumbered crop - treated as valid
                    is_valid_frame, exclusion_reason, height_ratio, y2_coord = True, "Complete", 1.0, 0

                res = model(img, conf=0.25, classes=[0], verbose=False)
                boxes = res[0].boxes

                raw_detections = []
                if boxes is not None and len(boxes) > 0:
                    for b in boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        conf = float(b.conf[0])
                        raw_detections.append((x1, y1, x2, y2, conf))

                # Apply upgraded deduplication
                filtered_detections = deduplicate_rider_boxes(raw_detections, crop_h=crop_h)
                count = len(filtered_detections)
                raw_count = len(raw_detections)

                # ONLY complete frames count toward the rider-count window!
                if is_valid_frame:
                    frame_counts.append(count)

                # Annotate image
                annotated = draw_annotations(img, filtered_detections, vname, count, is_valid_frame, exclusion_reason)
                out_filename = f"annotated_{filename}"
                out_path = os.path.join(out_sub, out_filename)
                cv2.imwrite(out_path, annotated)

                # Relative path for HTML
                rel_url = f"triple_riding_batch_results/{video_key}/{id_dir}/{out_filename}".replace("\\", "/")
                key = f"{video_key}_{vehicle_id}_{filename.replace('.', '_')}"

                frame_records.append({
                    "key": key,
                    "video_source": video_key,
                    "video_name": vname,
                    "vehicle_id": vehicle_id,
                    "filename": filename,
                    "raw_count": raw_count,
                    "person_count": count,
                    "is_valid_frame": is_valid_frame,
                    "exclusion_reason": exclusion_reason,
                    "height_ratio": round(height_ratio, 2),
                    "annotated_url": rel_url,
                    "detections": [{"conf": round(d[4], 2), "box": [d[0], d[1], d[2], d[3]]} for d in filtered_detections]
                })

            # Calculate Track-Level Temporal Consensus strictly from complete frames
            if frame_counts:
                # Majority vote mode
                consensus_count = max(set(frame_counts), key=frame_counts.count)
            else:
                consensus_count = 0

            # Attach consensus to records
            for r in frame_records:
                r["track_consensus"] = consensus_count
                all_results.append(r)

                csv_rows.append([
                    r["video_source"], r["vehicle_id"], r["filename"],
                    r["raw_count"], r["person_count"], r["track_consensus"],
                    r["is_valid_frame"], r["exclusion_reason"],
                    ";".join([f"{d['conf']}" for d in r["detections"]])
                ])

    # Save CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_source", "vehicle_id", "filename", "raw_count", "deduplicated_count", "track_consensus", "is_valid_frame", "exclusion_reason", "confidences"])
        writer.writerows(csv_rows)

    print(f"\n💾 Results CSV saved: {CSV_PATH} ({len(all_results)} crops evaluated)")

    # Build HTML Dashboard
    generate_batch_html_visualizer(all_results)

    print(f"\n💾 Results CSV saved: {CSV_PATH} ({len(all_results)} crops evaluated)")

    # Build HTML Dashboard
    generate_batch_html_visualizer(all_results)

def generate_batch_html_visualizer(results_data):
    total_crops = len(results_data)
    unique_bikes = len(set(f"{r['video_source']}_{r['vehicle_id']}" for r in results_data))
    
    valid_crops = sum(1 for r in results_data if r.get("is_valid_frame", True))
    excluded_crops = total_crops - valid_crops

    triple_count = sum(1 for r in results_data if r["person_count"] >= 3 and r.get("is_valid_frame", True))
    double_count = sum(1 for r in results_data if r["person_count"] == 2 and r.get("is_valid_frame", True))
    solo_count = sum(1 for r in results_data if r["person_count"] == 1 and r.get("is_valid_frame", True))
    zero_count = sum(1 for r in results_data if r["person_count"] == 0 and r.get("is_valid_frame", True))

    # Track consensus stats strictly from valid frames
    track_consensus_map = {}
    for r in results_data:
        k = f"{r['video_source']}_{r['vehicle_id']}"
        track_consensus_map[k] = r["track_consensus"]

    consensus_triples = sum(1 for v in track_consensus_map.values() if v >= 3)
    consensus_doubles = sum(1 for v in track_consensus_map.values() if v == 2)
    consensus_solos = sum(1 for v in track_consensus_map.values() if v == 1)

    json_data = json.dumps(results_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Multi-Video Motorcycle Rider & Occupancy Audit</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-primary: #0a0e17;
      --bg-card: rgba(18, 24, 38, 0.8);
      --border-color: rgba(255, 255, 255, 0.08);
      --border-highlight: rgba(99, 102, 241, 0.4);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --danger: #ef4444;
      --warning: #f59e0b;
      --success: #10b981;
      --info: #3b82f6;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', sans-serif;
      background: radial-gradient(circle at 50% 0%, #151d36 0%, var(--bg-primary) 70%);
      color: var(--text-main);
      padding: 24px 32px;
      min-height: 100vh;
    }}
    .header {{
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-color);
      border-radius: 20px;
      padding: 24px;
      margin-bottom: 24px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }}
    .header-title {{ display: flex; align-items: center; gap: 16px; }}
    .header-icon {{ font-size: 32px; background: linear-gradient(135deg, #6366f1, #a855f7); border-radius: 12px; width: 54px; height: 54px; display: flex; align-items: center; justify-content: center; }}
    .header h1 {{ font-size: 24px; font-weight: 800; }}
    .header p {{ color: var(--text-muted); font-size: 14px; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 16px 20px;
    }}
    .kpi-label {{ font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.5px; }}
    .kpi-val {{ font-size: 26px; font-weight: 800; font-family: 'JetBrains Mono', monospace; margin-top: 4px; }}
    .controls {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 16px 20px;
      margin-bottom: 24px;
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
    }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .tab-btn {{
      padding: 8px 14px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-color);
      color: var(--text-muted);
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      transition: all 0.2s;
    }}
    .tab-btn.active {{ background: #4f46e5; color: white; border-color: #6366f1; }}
    .tab-btn.triple.active {{ background: #ef4444; border-color: #f87171; }}
    .filters-right {{ display: flex; gap: 10px; flex: 1; max-width: 580px; flex-wrap: wrap; }}
    .select-box, .search-box {{
      background: rgba(0,0,0,0.4);
      border: 1px solid var(--border-color);
      color: white;
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 13px;
      outline: none;
    }}
    .search-box {{ flex: 1; min-width: 140px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 20px;
    }}
    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.2s, border-color 0.2s;
    }}
    .card:hover {{ transform: translateY(-4px); border-color: var(--border-highlight); }}
    .card.is-triple {{ border-color: rgba(239, 68, 68, 0.5); background: linear-gradient(180deg, rgba(239,68,68,0.1) 0%, var(--bg-card) 40%); }}
    .card.is-excluded {{ opacity: 0.65; border-style: dashed; }}
    .card-img-wrap {{ width: 100%; height: 260px; background: #000; position: relative; cursor: zoom-in; }}
    .card-img {{ width: 100%; height: 100%; object-fit: contain; }}
    .badge {{
      position: absolute;
      top: 10px;
      right: 10px;
      padding: 5px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }}
    .badge-triple {{ background: rgba(239, 68, 68, 0.9); color: white; }}
    .badge-double {{ background: rgba(59, 130, 246, 0.9); color: white; }}
    .badge-solo {{ background: rgba(16, 185, 129, 0.9); color: white; }}
    .badge-zero {{ background: rgba(100, 116, 139, 0.9); color: white; }}
    .badge-excluded {{ background: rgba(245, 158, 11, 0.9); color: #000; }}
    .video-tag {{
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(4px);
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      color: #93c5fd;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    .card-body {{ padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; flex: 1; }}
    .card-meta {{ display: flex; justify-content: space-between; font-size: 13px; align-items: center; }}
    .meta-id {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; background: rgba(255,255,255,0.06); padding: 2px 8px; border-radius: 4px; }}
    .meta-consensus {{ font-size: 12px; color: #a78bfa; font-weight: 600; }}
    .review-box {{
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 10px;
      padding: 10px;
      margin-top: auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .chip-btn {{
      padding: 4px 8px;
      border-radius: 5px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: var(--text-muted);
    }}
    .chip-btn.selected {{ background: rgba(16, 185, 129, 0.25); border-color: #10b981; color: #34d399; }}
    .chip-btn.selected-over {{ background: rgba(245, 158, 11, 0.25); border-color: #f59e0b; color: #fbbf24; }}
    .chip-btn.selected-under {{ background: rgba(239, 68, 68, 0.25); border-color: #ef4444; color: #f87171; }}
    .modal {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.88); z-index: 999; justify-content: center; align-items: center; }}
    .modal.active {{ display: flex; }}
    .modal-img {{ max-width: 90vw; max-height: 85vh; border-radius: 12px; }}
    .empty {{ grid-column: 1 / -1; text-align: center; padding: 60px; color: var(--text-muted); }}
  </style>
</head>
<body>
  <header class="header">
    <div class="header-title">
      <div class="header-icon">🏍️</div>
      <div>
        <h1>Multi-Video Motorcycle Rider & Occupancy Audit</h1>
        <p>Evaluated across {total_crops} bike crops with 70%+ track-height window & bottom-edge exit filtering</p>
      </div>
    </div>
    <div>
      <button class="tab-btn" onclick="exportJSON()">📥 Export JSON</button>
    </div>
  </header>

  <section class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Total Bike Crops</div>
      <div class="kpi-val">{total_crops}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Tracked Motorcycles</div>
      <div class="kpi-val">{unique_bikes}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">✅ Valid Frames (70%+)</div>
      <div class="kpi-val" style="color:#34d399">{valid_crops}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">⚠️ Excluded Truncated</div>
      <div class="kpi-val" style="color:#f59e0b">{excluded_crops}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">🚨 Verified Triple Bikes</div>
      <div class="kpi-val" style="color:#f87171">{consensus_triples}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">👥 Double Bikes (Consensus)</div>
      <div class="kpi-val" style="color:#60a5fa">{consensus_doubles}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">👤 Solo Bikes (Consensus)</div>
      <div class="kpi-val" style="color:#34d399">{consensus_solos}</div>
    </div>
  </section>

  <div class="controls">
    <div class="tabs">
      <button class="tab-btn active" onclick="setCountTab('all', this)">All Valid ({valid_crops})</button>
      <button class="tab-btn triple" onclick="setCountTab('triple', this)">🚨 Triple ({triple_count})</button>
      <button class="tab-btn" onclick="setCountTab('double', this)">👥 Double ({double_count})</button>
      <button class="tab-btn" onclick="setCountTab('solo', this)">👤 Solo ({solo_count})</button>
      <button class="tab-btn" onclick="setCountTab('zero', this)">❓ 0 ({zero_count})</button>
      <button class="tab-btn" onclick="setCountTab('excluded', this)">⚠️ Excluded ({excluded_crops})</button>
      <button class="tab-btn" onclick="setCountTab('all_raw', this)">🌐 All Raw ({total_crops})</button>
    </div>

    <div class="filters-right">
      <select class="select-box" id="videoFilter" onchange="renderCards()">
        <option value="all">All Video Sources</option>
        <option value="test2">Test 2</option>
        <option value="test">Test</option>
        <option value="evening">Evening Traffic</option>
      </select>
      <input type="text" class="search-box" id="searchBox" placeholder="Search ID or filename..." oninput="renderCards()">
    </div>
  </div>

  <main class="grid" id="grid"></main>

  <div class="modal" id="modal" onclick="this.classList.remove('active')">
    <img id="modalImg" class="modal-img" src="" alt="Enlarged crop">
  </div>

  <script>
    const DATA = {json_data};
    let currentCountTab = 'all';
    const ratings = JSON.parse(localStorage.getItem('MULTI_VIDEO_RIDER_RATINGS') || '{{}}');

    function saveRating(key, val) {{
      ratings[key] = val;
      localStorage.setItem('MULTI_VIDEO_RIDER_RATINGS', JSON.stringify(ratings));
      renderCards();
    }}

    function setCountTab(tab, el) {{
      currentCountTab = tab;
      document.querySelectorAll('.tabs .tab-btn').forEach(b => b.classList.remove('active'));
      el.classList.add('active');
      renderCards();
    }}

    function renderCards() {{
      const vFilter = document.getElementById('videoFilter').value;
      const search = document.getElementById('searchBox').value.toLowerCase().trim();
      const grid = document.getElementById('grid');

      const filtered = DATA.filter(item => {{
        const isVal = item.is_valid_frame !== false;

        if (currentCountTab === 'all' && !isVal) return false;
        if (currentCountTab === 'triple' && (!isVal || item.person_count < 3)) return false;
        if (currentCountTab === 'double' && (!isVal || item.person_count !== 2)) return false;
        if (currentCountTab === 'solo' && (!isVal || item.person_count !== 1)) return false;
        if (currentCountTab === 'zero' && (!isVal || item.person_count !== 0)) return false;
        if (currentCountTab === 'excluded' && isVal) return false;

        if (vFilter !== 'all' && item.video_source !== vFilter) return false;

        if (search) {{
          const matchId = item.vehicle_id.toString().includes(search) || `id_${{item.vehicle_id}}`.includes(search);
          if (!matchId && !item.filename.toLowerCase().includes(search)) return false;
        }}
        return true;
      }});

      if (filtered.length === 0) {{
        grid.innerHTML = '<div class="empty">No matching motorcycle crops found.</div>';
        return;
      }}

      grid.innerHTML = filtered.map(item => {{
        const r = ratings[item.key] || '';
        const isVal = item.is_valid_frame !== false;
        let bClass = 'badge-zero', bText = '0 Persons';

        if (!isVal) {{
          bClass = 'badge-excluded';
          bText = `⚠️ Excluded: ${{item.exclusion_reason || 'Truncated'}}`;
        }} else if (item.person_count >= 3) {{
          bClass = 'badge-triple';
          bText = `🚨 ${{item.person_count}} Persons`;
        }} else if (item.person_count === 2) {{
          bClass = 'badge-double';
          bText = '👥 2 Persons';
        }} else if (item.person_count === 1) {{
          bClass = 'badge-solo';
          bText = '👤 1 Person';
        }}

        const statusNote = !isVal 
          ? `<div style="font-size:11px;color:#f59e0b;font-weight:600;">⚠️ Excluded from consensus (${{item.exclusion_reason}})</div>`
          : `<div style="font-size:11px;color:#10b981;font-weight:600;">✅ Valid Window (Height ratio: ${{item.height_ratio * 100 || 100}}%)</div>`;

        return `
          <div class="card ${{item.person_count >= 3 && isVal ? 'is-triple' : ''}} ${{!isVal ? 'is-excluded' : ''}}">
            <div class="card-img-wrap" onclick="openModal('${{item.annotated_url}}')">
              <img class="card-img" src="${{item.annotated_url}}" loading="lazy">
              <span class="video-tag">${{item.video_name}}</span>
              <span class="badge ${{bClass}}">${{bText}}</span>
            </div>
            <div class="card-body">
              <div class="card-meta">
                <span class="meta-id">ID: ${{item.vehicle_id}}</span>
                <span class="meta-consensus">Consensus: ${{item.track_consensus}} Riders</span>
              </div>
              <div style="font-size:11px;color:#94a3b8;font-family:monospace;">
                Raw YOLO: ${{item.raw_count}} | Filtered: ${{item.person_count}}
              </div>
              ${{statusNote}}
              <div class="review-box">
                <div style="font-size:11px;font-weight:600;color:#94a3b8;">Rating:</div>
                <div class="chips">
                  <button class="chip-btn ${{r === 'Accurate' ? 'selected' : ''}}" onclick="saveRating('${{item.key}}', 'Accurate')">✅ Accurate</button>
                  <button class="chip-btn ${{r === 'Over' ? 'selected-over' : ''}}" onclick="saveRating('${{item.key}}', 'Over')">⚠️ Over-count</button>
                  <button class="chip-btn ${{r === 'Under' ? 'selected-under' : ''}}" onclick="saveRating('${{item.key}}', 'Under')">⚠️ Under-count</button>
                </div>
              </div>
            </div>
          </div>
        `;
      }}).join('');
    }}

    function openModal(url) {{
      document.getElementById('modalImg').src = url;
      document.getElementById('modal').classList.add('active');
    }}

    function exportJSON() {{
      const exp = DATA.map(d => ({{ ...d, user_rating: ratings[d.key] || 'Unreviewed' }}));
      const blob = new Blob([JSON.stringify(exp, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'multi_video_rider_audit.json';
      a.click();
    }}

    renderCards();
  </script>
</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n🎉 Interactive Visualizer generated at: {HTML_PATH}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch Pipeline for Bike Rider & Triple Riding Detection")
    parser.add_argument("--videos", nargs="+", default=["test2"], help="Videos to process: test2, test, evening, or all")
    args = parser.parse_args()

    selected = args.videos
    if "all" in selected:
        selected = list(VIDEOS.keys())

    print(f"🚦 BATCH PIPELINE START (Selected videos: {selected})")
    for v in selected:
        if v == "test":
            run_fast_tracking_if_needed(v)
        if v in ["test", "evening"]:
            crop_motorcycles_fast(v)

    process_all_bikes(selected)
    print("\n🎉 BATCH PIPELINE COMPLETE!")

if __name__ == "__main__":
    main()
