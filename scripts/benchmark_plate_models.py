"""
Benchmark License Plate Models: yolov8s_lpd.pt vs plate_detector.pt
====================================================================
Compares the existing license plate detection model with the newly trained model
on vehicle crop images from output/cropped_photos without altering the existing pipeline.

Evaluates:
1. Detection Recall (% of vehicles with license plates detected)
2. Average Detection Confidence
3. Crop Readability & OCR Accuracy (PaddleOCR + Pakistani plate logic)
4. Category breakdown (Bikes/Motorcycles, Cars, Rickshaws)
5. Generates a side-by-side CSV and an interactive HTML visual report
"""

import os
import sys
import re
import cv2
import csv
import math
import logging
import numpy as np
from datetime import datetime

# Windows terminal UTF-8 encoding support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
CROPPED_PHOTOS_DIR = os.path.join(PROJECT_ROOT, "output", "cropped_photos")
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")

MODEL_A_PATH = os.path.join(MODELS_DIR, "yolov8s_lpd.pt")
MODEL_B_PATH = os.path.join(MODELS_DIR, "plate_detector.pt")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "plate_model_comparison")
CROPS_A_DIR = os.path.join(OUTPUT_DIR, "crops", "yolov8s_lpd")
CROPS_B_DIR = os.path.join(OUTPUT_DIR, "crops", "plate_detector")
CSV_REPORT_PATH = os.path.join(OUTPUT_DIR, "plate_model_comparison_results.csv")
HTML_REPORT_PATH = os.path.join(OUTPUT_DIR, "plate_comparison_report.html")

CONFIDENCE_THRESHOLD = 0.40
CENTER_TOLERANCE = 0.40  # 40% distance from center

# Suppress PaddleOCR debug logs
logging.getLogger("ppocr").setLevel(logging.ERROR)


def is_centered(box, img_width, img_height, tolerance=CENTER_TOLERANCE):
    x1, y1, x2, y2 = box
    box_cx = (x1 + x2) / 2
    box_cy = (y1 + y2) / 2
    img_cx = img_width / 2
    img_cy = img_height / 2

    dist_x = abs(box_cx - img_cx) / img_width
    dist_y = abs(box_cy - img_cy) / img_height
    distance = math.sqrt(dist_x**2 + dist_y**2)
    return distance < tolerance


def apply_pakistan_correction(text):
    if not isinstance(text, str) or not text:
        return ""
    if len(text) > 6 and text[-1] in ['1', 'I', '|', 'l', ']', ')']:
        text = text[:-1]
    text = re.sub(r'[^A-Z0-9-]', '', text.upper())
    if "-" in text:
        parts = text.split('-')
        alpha_part = parts[0]
        num_part = parts[1] if len(parts) > 1 else ""
    else:
        match = re.search(r'\d', text)
        if match:
            idx = match.start()
            alpha_part = text[:idx]
            num_part = text[idx:]
        else:
            alpha_part, num_part = text, ""

    dict_to_letters = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B', '6': 'G'}
    dict_to_numbers = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'A': '4', 'Z': '7', 'G': '6', 'D': '0', 'Q': '0'}

    new_alpha = "".join([dict_to_letters.get(c, c) for c in alpha_part])
    new_num = "".join([dict_to_numbers.get(c, c) for c in num_part])
    final_text = new_alpha + new_num
    return final_text.replace('-', '')


def remove_red_emblem(img):
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 35, 35])
    upper_red1 = np.array([18, 255, 255])
    lower_red2 = np.array([160, 35, 35])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask[:, int(w * 0.35):] = 0
    mask[int(h * 0.55):, :] = 0
    if np.any(mask):
        bg_color = [int(x) for x in np.median(img[int(h * 0.6):, :], axis=(0, 1))]
        cleaned = img.copy()
        cleaned[mask > 0] = bg_color
        return cleaned
    return img


def preprocess_for_rec(img):
    if img is None:
        return None
    img = remove_red_emblem(img)
    scale = 200
    w = int(img.shape[1] * scale / 100)
    h = int(img.shape[0] * scale / 100)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)


def extract_best_plate(model, images_paths, padding=5):
    best_conf = -1.0
    best_crop = None
    best_box = None
    best_source = None

    for img_path in images_paths:
        frame_img = cv2.imread(img_path)
        if frame_img is None:
            continue
        h, w = frame_img.shape[:2]
        results = model.predict(frame_img, conf=CONFIDENCE_THRESHOLD, verbose=False)
        if not results:
            continue

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                if not is_centered(xyxy, w, h, CENTER_TOLERANCE):
                    continue
                if conf > best_conf:
                    best_conf = conf
                    best_source = os.path.basename(img_path)
                    x1, y1, x2, y2 = map(int, xyxy)
                    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                    x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
                    best_crop = frame_img[y1:y2, x1:x2].copy()
                    best_box = [x1, y1, x2, y2]

    return best_conf, best_crop, best_box, best_source


def run_ocr_on_crop(ocr_engine, crop_img):
    if crop_img is None or crop_img.size == 0:
        return "", "", 0.0, False

    processed = preprocess_for_rec(crop_img)
    try:
        result = ocr_engine.ocr(processed)
        raw_text = ""
        conf = 0.0
        if result:
            stack = [result]
            while stack:
                current = stack.pop(0)
                if isinstance(current, list):
                    stack.extend(current)
                elif isinstance(current, tuple) and len(current) == 2:
                    if isinstance(current[0], str):
                        raw_text += current[0]
                        conf = float(current[1])
        cleaned_text = apply_pakistan_correction(raw_text)
        is_readable = len(cleaned_text) >= 3
        return raw_text, cleaned_text, conf, is_readable
    except Exception as e:
        return "", "", 0.0, False


def build_html_report(results, summary):
    def escape(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows_html = []
    for r in results:
        # Paths relative to output dir for standalone viewing
        crop_a_rel = os.path.relpath(r['a_crop_path'], OUTPUT_DIR).replace("\\", "/") if r['a_crop_path'] else ""
        crop_b_rel = os.path.relpath(r['b_crop_path'], OUTPUT_DIR).replace("\\", "/") if r['b_crop_path'] else ""

        a_readable_badge = '<span class="badge readable">Readable</span>' if r['a_is_readable'] else '<span class="badge unreadable">Unreadable</span>' if r['a_detected'] else '<span class="badge missing">No Detection</span>'
        b_readable_badge = '<span class="badge readable">Readable</span>' if r['b_is_readable'] else '<span class="badge unreadable">Unreadable</span>' if r['b_detected'] else '<span class="badge missing">No Detection</span>'

        verdict_badge = {
            "New Model (plate_detector) Better": '<span class="badge win-b">New Model Superior</span>',
            "Previous Model (yolov8s_lpd) Better": '<span class="badge win-a">Previous Model Superior</span>',
            "Tie (Both Readable)": '<span class="badge tie">Tie (Both Readable)</span>',
            "Tie (Neither Detected)": '<span class="badge neither">Neither Detected</span>',
            "Tie (Neither Readable)": '<span class="badge neither">Both Detected, Unreadable</span>'
        }.get(r['verdict'], f'<span class="badge">{escape(r["verdict"])}</span>')

        img_a_tag = f'<img class="plate-img" src="{crop_a_rel}" alt="Model A Crop" loading="lazy" />' if crop_a_rel else '<div class="no-img">No Crop</div>'
        img_b_tag = f'<img class="plate-img" src="{crop_b_rel}" alt="Model B Crop" loading="lazy" />' if crop_b_rel else '<div class="no-img">No Crop</div>'

        row = f"""
        <tr>
            <td><strong>{escape(r['category'])}</strong><br><small class="text-muted">{escape(r['vehicle_id'])}</small></td>
            <td>
                {img_a_tag}
                <div class="crop-info">
                    <div>Conf: <strong>{r['a_det_conf']:.2f}</strong></div>
                    <div>Text: <code>{escape(r['a_cleaned_text']) if r['a_cleaned_text'] else '-'}</code></div>
                    <div>OCR Conf: {r['a_ocr_conf']:.2f}</div>
                    <div>{a_readable_badge}</div>
                </div>
            </td>
            <td>
                {img_b_tag}
                <div class="crop-info">
                    <div>Conf: <strong>{r['b_det_conf']:.2f}</strong></div>
                    <div>Text: <code>{escape(r['b_cleaned_text']) if r['b_cleaned_text'] else '-'}</code></div>
                    <div>OCR Conf: {r['b_ocr_conf']:.2f}</div>
                    <div>{b_readable_badge}</div>
                </div>
            </td>
            <td>{verdict_badge}</td>
        </tr>
        """
        rows_html.append(row)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>License Plate Model Benchmark: yolov8s_lpd vs plate_detector</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0b0f19;
            --surface: #141b2d;
            --surface-card: #1b253b;
            --border: #273553;
            --text-primary: #f0f4fc;
            --text-secondary: #94a3b8;
            --accent-a: #38bdf8;
            --accent-b: #10b981;
            --accent-gold: #f59e0b;
            --danger: #ef4444;
            --radius-md: 10px;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text-primary);
            font-family: 'Outfit', sans-serif;
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }}
        h1 {{ font-size: 26px; font-weight: 700; color: #fff; margin-bottom: 6px; }}
        .subtitle {{ color: var(--text-secondary); font-size: 14px; }}
        .badge-header {{
            display: inline-block;
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-a);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 6px;
            padding: 3px 10px;
            font-size: 12px;
            margin-left: 8px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .metric-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }}
        .metric-label {{
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-val {{
            font-size: 26px;
            font-weight: 700;
        }}
        .val-a {{ color: var(--accent-a); }}
        .val-b {{ color: var(--accent-b); }}
        .metric-sub {{ font-size: 12px; color: var(--text-secondary); margin-top: 4px; }}
        
        .winner-banner {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(56, 189, 248, 0.1));
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: var(--radius-md);
            padding: 18px 22px;
            margin-bottom: 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .winner-title {{ font-size: 18px; font-weight: 600; color: #fff; }}
        .winner-desc {{ font-size: 14px; color: var(--text-secondary); margin-top: 2px; }}

        .table-container {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 14px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }}
        th {{
            background: var(--surface-card);
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        .plate-img {{
            max-width: 170px;
            max-height: 70px;
            object-fit: contain;
            border-radius: 4px;
            border: 1px solid var(--border);
            background: #000;
            display: block;
            margin-bottom: 6px;
        }}
        .no-img {{
            width: 120px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255,255,255,0.03);
            border: 1px dashed var(--border);
            color: var(--text-secondary);
            font-size: 12px;
            border-radius: 4px;
            margin-bottom: 6px;
        }}
        code {{
            font-family: 'JetBrains Mono', monospace;
            background: rgba(0, 0, 0, 0.35);
            padding: 2px 6px;
            border-radius: 4px;
            color: #38bdf8;
            font-size: 13px;
        }}
        .crop-info {{ font-size: 12px; color: var(--text-secondary); }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            margin-top: 4px;
        }}
        .badge.readable {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge.unreadable {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge.missing {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .badge.win-b {{ background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }}
        .badge.win-a {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }}
        .badge.tie {{ background: rgba(255, 255, 255, 0.1); color: #e2e8f0; border: 1px solid rgba(255, 255, 255, 0.2); }}
        .badge.neither {{ background: rgba(148, 163, 184, 0.1); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.2); }}
        .text-muted {{ color: var(--text-secondary); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>License Plate Detection Benchmark <span class="badge-header">Isolated Evaluation</span></h1>
        <div class="subtitle">Comparing Previous Model (<code>yolov8s_lpd.pt</code>) vs Newly Trained Model (<code>plate_detector.pt</code>) &bull; Evaluated on {summary['total_vehicles']} Vehicles &bull; Generated {now_str}</div>
    </div>

    <div class="winner-banner">
        <div>
            <div class="winner-title">{summary['overall_winner_title']}</div>
            <div class="winner-desc">{summary['overall_winner_desc']}</div>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Vehicles Evaluated</div>
            <div class="metric-val">{summary['total_vehicles']}</div>
            <div class="metric-sub">Cars: {summary['cars_count']} | Bikes: {summary['bikes_count']} | Rickshaws: {summary['rickshaws_count']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Plates Detected</div>
            <div class="metric-val">
                <span class="val-a">{summary['a_detected_count']}</span> <span style="font-size:16px;color:var(--text-secondary)">vs</span> <span class="val-b">{summary['b_detected_count']}</span>
            </div>
            <div class="metric-sub">Old ({summary['a_detected_rate']:.1f}%) vs New ({summary['b_detected_rate']:.1f}%)</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Readable Plates (OCR)</div>
            <div class="metric-val">
                <span class="val-a">{summary['a_readable_count']}</span> <span style="font-size:16px;color:var(--text-secondary)">vs</span> <span class="val-b">{summary['b_readable_count']}</span>
            </div>
            <div class="metric-sub">Old ({summary['a_readable_rate']:.1f}%) vs New ({summary['b_readable_rate']:.1f}%)</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Detection Conf</div>
            <div class="metric-val">
                <span class="val-a">{summary['a_avg_det_conf']:.2f}</span> <span style="font-size:16px;color:var(--text-secondary)">vs</span> <span class="val-b">{summary['b_avg_det_conf']:.2f}</span>
            </div>
            <div class="metric-sub">Higher confidence indicates firmer detections</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg OCR Conf (Valid)</div>
            <div class="metric-val">
                <span class="val-a">{summary['a_avg_ocr_conf']:.2f}</span> <span style="font-size:16px;color:var(--text-secondary)">vs</span> <span class="val-b">{summary['b_avg_ocr_conf']:.2f}</span>
            </div>
            <div class="metric-sub">Crop clarity as measured by PaddleOCR</div>
        </div>
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th style="width: 140px;">Vehicle</th>
                    <th style="width: 320px;">Previous Model (<code>yolov8s_lpd.pt</code>)</th>
                    <th style="width: 320px;">New Model (<code>plate_detector.pt</code>)</th>
                    <th style="width: 160px;">Verdict</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with open(HTML_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    print("=" * 60)
    print("🔬 SMART LICENSE PLATE MODEL BENCHMARK")
    print("   Comparing yolov8s_lpd.pt vs plate_detector.pt")
    print("=" * 60)

    # 1. Verification
    if not os.path.exists(MODEL_A_PATH):
        print(f"❌ Previous model not found at: {MODEL_A_PATH}")
        return
    if not os.path.exists(MODEL_B_PATH):
        print(f"❌ New model not found at: {MODEL_B_PATH}")
        return
    if not os.path.exists(CROPPED_PHOTOS_DIR):
        print(f"❌ Cropped photos directory not found at: {CROPPED_PHOTOS_DIR}")
        return

    os.makedirs(CROPS_A_DIR, exist_ok=True)
    os.makedirs(CROPS_B_DIR, exist_ok=True)

    print(f"📂 Scanning dataset in: {CROPPED_PHOTOS_DIR}")
    print(f"   Model A (Previous): {os.path.basename(MODEL_A_PATH)}")
    print(f"   Model B (New):      {os.path.basename(MODEL_B_PATH)}")

    # 2. Load Models
    from ultralytics import YOLO
    from paddleocr import PaddleOCR

    print("\n⏳ Loading YOLO models into memory...")
    model_a = YOLO(MODEL_A_PATH)
    model_b = YOLO(MODEL_B_PATH)
    print("✅ Both YOLO models loaded.")

    print("⏳ Initializing PaddleOCR engine...")
    try:
        ocr = PaddleOCR(lang='en', det=False, use_angle_cls=False, show_log=False)
    except:
        ocr = PaddleOCR(lang='en', det=False, use_textline_orientation=False)
    print("✅ PaddleOCR initialized.")

    # 3. Collect vehicle folders
    vehicle_jobs = []
    categories = [d for d in os.listdir(CROPPED_PHOTOS_DIR) if os.path.isdir(os.path.join(CROPPED_PHOTOS_DIR, d))]

    for cat in sorted(categories):
        cat_path = os.path.join(CROPPED_PHOTOS_DIR, cat)
        for v_id in sorted(os.listdir(cat_path)):
            v_path = os.path.join(cat_path, v_id)
            if not os.path.isdir(v_path):
                continue
            images = [
                os.path.join(v_path, f)
                for f in os.listdir(v_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                and "plate" not in f.lower()
                and "best" not in f.lower()
            ]
            if images:
                vehicle_jobs.append({
                    "category": cat,
                    "vehicle_id": v_id,
                    "image_paths": images
                })

    total_vehicles = len(vehicle_jobs)
    print(f"\n🚘 Found {total_vehicles} vehicle folders with candidate images.")

    # 4. Run Benchmark
    results = []

    print("\n🚀 Running side-by-side inference & OCR...")
    for idx, job in enumerate(vehicle_jobs, 1):
        cat = job['category']
        v_id = job['vehicle_id']
        imgs = job['image_paths']

        # --- Model A Inference ---
        conf_a, crop_a, box_a, src_a = extract_best_plate(model_a, imgs)
        crop_a_path = ""
        if crop_a is not None:
            crop_a_filename = f"{cat}_{v_id}_crop_a.jpg"
            crop_a_path = os.path.join(CROPS_A_DIR, crop_a_filename)
            cv2.imwrite(crop_a_path, crop_a)

        raw_a, clean_a, ocr_conf_a, readable_a = run_ocr_on_crop(ocr, crop_a)

        # --- Model B Inference ---
        conf_b, crop_b, box_b, src_b = extract_best_plate(model_b, imgs)
        crop_b_path = ""
        if crop_b is not None:
            crop_b_filename = f"{cat}_{v_id}_crop_b.jpg"
            crop_b_path = os.path.join(CROPS_B_DIR, crop_b_filename)
            cv2.imwrite(crop_b_path, crop_b)

        raw_b, clean_b, ocr_conf_b, readable_b = run_ocr_on_crop(ocr, crop_b)

        # --- Decision Verdict ---
        # 1. Readability priority: if one is readable and the other isn't
        if readable_b and not readable_a:
            verdict = "New Model (plate_detector) Better"
        elif readable_a and not readable_b:
            verdict = "Previous Model (yolov8s_lpd) Better"
        elif readable_a and readable_b:
            # Both readable: compare OCR confidence first, then detection confidence
            if ocr_conf_b > ocr_conf_a + 0.05:
                verdict = "New Model (plate_detector) Better"
            elif ocr_conf_a > ocr_conf_b + 0.05:
                verdict = "Previous Model (yolov8s_lpd) Better"
            else:
                verdict = "Tie (Both Readable)"
        elif (crop_b is not None) and (crop_a is None):
            verdict = "New Model (plate_detector) Better"
        elif (crop_a is not None) and (crop_b is None):
            verdict = "Previous Model (yolov8s_lpd) Better"
        elif (crop_a is None) and (crop_b is None):
            verdict = "Tie (Neither Detected)"
        else:
            verdict = "Tie (Neither Readable)"

        results.append({
            "category": cat,
            "vehicle_id": v_id,
            "a_detected": crop_a is not None,
            "a_det_conf": conf_a if conf_a > 0 else 0.0,
            "a_crop_path": crop_a_path,
            "a_raw_text": raw_a,
            "a_cleaned_text": clean_a,
            "a_ocr_conf": ocr_conf_a,
            "a_is_readable": readable_a,
            "b_detected": crop_b is not None,
            "b_det_conf": conf_b if conf_b > 0 else 0.0,
            "b_crop_path": crop_b_path,
            "b_raw_text": raw_b,
            "b_cleaned_text": clean_b,
            "b_ocr_conf": ocr_conf_b,
            "b_is_readable": readable_b,
            "verdict": verdict
        })

        if idx % 10 == 0 or idx == total_vehicles:
            print(f"   [{idx:02d}/{total_vehicles:02d}] Processed {cat} / {v_id} -> Verdict: {verdict}")

    # 5. Aggregate Metrics
    a_detected_count = sum(1 for r in results if r['a_detected'])
    b_detected_count = sum(1 for r in results if r['b_detected'])

    a_readable_count = sum(1 for r in results if r['a_is_readable'])
    b_readable_count = sum(1 for r in results if r['b_is_readable'])

    a_det_confs = [r['a_det_conf'] for r in results if r['a_detected']]
    b_det_confs = [r['b_det_conf'] for r in results if r['b_detected']]
    a_avg_det_conf = float(np.mean(a_det_confs)) if a_det_confs else 0.0
    b_avg_det_conf = float(np.mean(b_det_confs)) if b_det_confs else 0.0

    a_ocr_confs = [r['a_ocr_conf'] for r in results if r['a_is_readable']]
    b_ocr_confs = [r['b_ocr_conf'] for r in results if r['b_is_readable']]
    a_avg_ocr_conf = float(np.mean(a_ocr_confs)) if a_ocr_confs else 0.0
    b_avg_ocr_conf = float(np.mean(b_ocr_confs)) if b_ocr_confs else 0.0

    cars_count = sum(1 for r in results if r['category'] == 'Cars')
    bikes_count = sum(1 for r in results if r['category'] == 'Bikes_Motorcycles')
    rickshaws_count = sum(1 for r in results if r['category'] == 'Rickshaws')

    # Winner verdict calculation
    score_b = (b_readable_count * 2) + b_detected_count + (b_avg_det_conf * 10)
    score_a = (a_readable_count * 2) + a_detected_count + (a_avg_det_conf * 10)

    if b_readable_count > a_readable_count:
        winner_title = "🏆 Winner: Newly Trained Model (plate_detector.pt)"
        winner_desc = f"Achieved {b_readable_count} readable plates ({b_readable_count/total_vehicles*100:.1f}%) vs {a_readable_count} ({a_readable_count/total_vehicles*100:.1f}%) for the previous model, with a detection count of {b_detected_count} vs {a_detected_count}."
    elif a_readable_count > b_readable_count:
        winner_title = "🏆 Winner: Previous Model (yolov8s_lpd.pt)"
        winner_desc = f"Achieved {a_readable_count} readable plates ({a_readable_count/total_vehicles*100:.1f}%) vs {b_readable_count} ({b_readable_count/total_vehicles*100:.1f}%) for the new model."
    else:
        if b_detected_count > a_detected_count:
            winner_title = "🏆 Winner: Newly Trained Model (plate_detector.pt)"
            winner_desc = f"Both tied on readable plates ({b_readable_count}), but the new model detected more plates ({b_detected_count} vs {a_detected_count}) with avg confidence {b_avg_det_conf:.2f} vs {a_avg_det_conf:.2f}."
        elif a_detected_count > b_detected_count:
            winner_title = "🏆 Winner: Previous Model (yolov8s_lpd.pt)"
            winner_desc = f"Both tied on readable plates ({a_readable_count}), but previous model detected more plates ({a_detected_count} vs {b_detected_count})."
        else:
            winner_title = "⚖️ Models Tied Overall"
            winner_desc = f"Both models achieved identical detection ({a_detected_count}) and readability counts ({a_readable_count})."

    summary = {
        "total_vehicles": total_vehicles,
        "cars_count": cars_count,
        "bikes_count": bikes_count,
        "rickshaws_count": rickshaws_count,
        "a_detected_count": a_detected_count,
        "b_detected_count": b_detected_count,
        "a_detected_rate": (a_detected_count / total_vehicles * 100) if total_vehicles else 0.0,
        "b_detected_rate": (b_detected_count / total_vehicles * 100) if total_vehicles else 0.0,
        "a_readable_count": a_readable_count,
        "b_readable_count": b_readable_count,
        "a_readable_rate": (a_readable_count / total_vehicles * 100) if total_vehicles else 0.0,
        "b_readable_rate": (b_readable_count / total_vehicles * 100) if total_vehicles else 0.0,
        "a_avg_det_conf": a_avg_det_conf,
        "b_avg_det_conf": b_avg_det_conf,
        "a_avg_ocr_conf": a_avg_ocr_conf,
        "b_avg_ocr_conf": b_avg_ocr_conf,
        "overall_winner_title": winner_title,
        "overall_winner_desc": winner_desc
    }

    # 6. Save CSV
    csv_headers = [
        "Category", "Vehicle_ID",
        "A_Detected", "A_Det_Confidence", "A_Plate_Text", "A_OCR_Confidence", "A_Readable",
        "B_Detected", "B_Det_Confidence", "B_Plate_Text", "B_OCR_Confidence", "B_Readable",
        "Verdict"
    ]
    with open(CSV_REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        for r in results:
            writer.writerow([
                r['category'], r['vehicle_id'],
                r['a_detected'], f"{r['a_det_conf']:.2f}", r['a_cleaned_text'], f"{r['a_ocr_conf']:.2f}", r['a_is_readable'],
                r['b_detected'], f"{r['b_det_conf']:.2f}", r['b_cleaned_text'], f"{r['b_ocr_conf']:.2f}", r['b_is_readable'],
                r['verdict']
            ])

    # 7. Save HTML Report
    build_html_report(results, summary)

    # 8. Terminal Output
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Vehicles Evaluated:      {total_vehicles}")
    print(f"  - Cars:                      {cars_count}")
    print(f"  - Bikes/Motorcycles:         {bikes_count}")
    print(f"  - Rickshaws:                 {rickshaws_count}")
    print("-" * 60)
    print(f"{'Metric':<30} | {'yolov8s_lpd.pt':<14} | {'plate_detector.pt':<14}")
    print("-" * 60)
    print(f"{'Plates Detected':<30} | {a_detected_count:<14} | {b_detected_count:<14}")
    print(f"{'Detection Rate':<30} | {summary['a_detected_rate']:.1f}%{'':<9} | {summary['b_detected_rate']:.1f}%{'':<9}")
    print(f"{'Avg Detection Conf':<30} | {a_avg_det_conf:.2f}{'':<10} | {b_avg_det_conf:.2f}{'':<10}")
    print(f"{'Readable Plates (OCR >= 3)':<30} | {a_readable_count:<14} | {b_readable_count:<14}")
    print(f"{'Readable Plate Rate':<30} | {summary['a_readable_rate']:.1f}%{'':<9} | {summary['b_readable_rate']:.1f}%{'':<9}")
    print(f"{'Avg OCR Conf (Valid)':<30} | {a_avg_ocr_conf:.2f}{'':<10} | {b_avg_ocr_conf:.2f}{'':<10}")
    print("=" * 60)
    print(f"{winner_title}")
    print(f"{winner_desc}")
    print("=" * 60)
    print(f"📄 Detailed CSV Report: {CSV_REPORT_PATH}")
    print(f"🌐 Interactive HTML Report: {HTML_REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
