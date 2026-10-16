"""
Model Detection Audit & Visualizer Generator
Generates an interactive, presentation-ready HTML dashboard to visually audit
the actual detections and bounding boxes produced by:
  - Models/seatbelt.pt (output/seatbelt_results)
  - Models/helmet.pt   (output/helmet_results)
"""

import os
import sys
import csv
import json
import webbrowser

# Fix Windows console encoding for Unicode/emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../output"))
SEATBELT_DIR = os.path.join(OUTPUT_DIR, "seatbelt_results")
HELMET_DIR = os.path.join(OUTPUT_DIR, "helmet_results")
SEATBELT_CSV = os.path.join(OUTPUT_DIR, "seatbelt_results.csv")
HELMET_CSV = os.path.join(OUTPUT_DIR, "helmet_results.csv")
MASTER_CSV = os.path.join(OUTPUT_DIR, "master_violation_log.csv")
HTML_OUTPUT = os.path.join(OUTPUT_DIR, "model_audit_visualizer.html")


def load_plate_map():
    """Load vehicle license plate info from master_violation_log.csv."""
    plate_map = {}
    if not os.path.exists(MASTER_CSV):
        return plate_map

    try:
        with open(MASTER_CSV, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = str(row.get("Vehicle_ID", "")).strip()
                cat = str(row.get("Category", "")).strip()
                plate = str(row.get("Plate_Number", "Unknown")).strip()
                try:
                    conf = float(row.get("Plate_Confidence", 0.0))
                except Exception:
                    conf = 0.0

                key = f"{cat}:{vid}"
                plate_map[key] = {"plate": plate, "confidence": conf}
                # Also index by vid
                plate_map[vid] = {"plate": plate, "confidence": conf}
    except Exception as e:
        print(f"⚠️ Warning loading plate map: {e}")

    return plate_map


def load_seatbelt_detections():
    """Load seatbelt detections from CSV mapped by image_name."""
    sb_csv_map = {}
    if not os.path.exists(SEATBELT_CSV):
        return sb_csv_map

    try:
        with open(SEATBELT_CSV, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image_name", "").strip()
                lbl = row.get("label", "").strip()
                if not img_name:
                    continue
                if img_name not in sb_csv_map:
                    sb_csv_map[img_name] = []
                sb_csv_map[img_name].append(row)
    except Exception as e:
        print(f"⚠️ Warning loading seatbelt CSV: {e}")

    return sb_csv_map


def load_helmet_detections():
    """Load helmet detections from CSV mapped by image_name."""
    h_csv_map = {}
    if not os.path.exists(HELMET_CSV):
        return h_csv_map

    try:
        with open(HELMET_CSV, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image_name", "").strip()
                if not img_name:
                    continue
                if img_name not in h_csv_map:
                    h_csv_map[img_name] = []
                h_csv_map[img_name].append(row)
    except Exception as e:
        print(f"⚠️ Warning loading helmet CSV: {e}")

    return h_csv_map


def build_audit_dataset():
    """Scan disk folders and match with CSV metadata."""
    plate_map = load_plate_map()
    sb_csv_map = load_seatbelt_detections()
    h_csv_map = load_helmet_detections()

    records = []

    # 1. SCAN SEATBELT RESULTS (Cars)
    cars_dir = os.path.join(SEATBELT_DIR, "Cars")
    if os.path.isdir(cars_dir):
        for id_folder in sorted(os.listdir(cars_dir)):
            folder_path = os.path.join(cars_dir, id_folder)
            if not os.path.isdir(folder_path):
                continue

            for fname in sorted(os.listdir(folder_path)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                if "plate" in fname.lower() or "best" in fname.lower():
                    continue

                rel_img_path = f"seatbelt_results/Cars/{id_folder}/{fname}"
                orig_img_name = fname.replace("detected_", "")

                dets = sb_csv_map.get(orig_img_name, [])
                boxes = []
                has_violation = False
                has_seatbelt = False
                highest_conf = 0.0

                for d in dets:
                    lbl = str(d.get("label", "")).strip()
                    try:
                        conf = float(d.get("confidence", 0.0))
                    except Exception:
                        conf = 0.0

                    if conf > highest_conf:
                        highest_conf = conf

                    x1 = d.get("x1", "")
                    y1 = d.get("y1", "")
                    x2 = d.get("x2", "")
                    y2 = d.get("y2", "")

                    if "no" in lbl.lower() and "seatbelt" in lbl.lower():
                        has_violation = True
                    elif "seatbelt" in lbl.lower():
                        has_seatbelt = True

                    boxes.append({
                        "label": lbl,
                        "confidence": round(conf, 3),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })

                # Determine primary status
                if has_violation:
                    status = "No Seatbelt"
                    status_class = "violation"
                elif has_seatbelt:
                    status = "Seatbelt Detected"
                    status_class = "compliant"
                else:
                    status = "No Detection"
                    status_class = "unknown"

                plate_info = plate_map.get(f"Cars:{id_folder}", plate_map.get(id_folder, {"plate": "Unknown", "confidence": 0.0}))

                records.append({
                    "id": f"sb_{len(records)}",
                    "model_type": "seatbelt",
                    "model_name": "YOLOv8 Seatbelt Detector",
                    "category": "Cars",
                    "vehicle_id": id_folder,
                    "filename": orig_img_name,
                    "annotated_filename": fname,
                    "image_rel_path": rel_img_path,
                    "status": status,
                    "status_class": status_class,
                    "confidence": round(highest_conf, 3),
                    "confidence_pct": round(highest_conf * 100, 1),
                    "boxes": boxes,
                    "box_count": len(boxes),
                    "plate_number": plate_info.get("plate", "Unknown"),
                    "plate_confidence": plate_info.get("confidence", 0.0)
                })

    # 2. SCAN HELMET RESULTS (Bikes_Motorcycles)
    bikes_dir = os.path.join(HELMET_DIR, "Bikes_Motorcycles")
    if os.path.isdir(bikes_dir):
        for id_folder in sorted(os.listdir(bikes_dir)):
            folder_path = os.path.join(bikes_dir, id_folder)
            if not os.path.isdir(folder_path):
                continue

            for fname in sorted(os.listdir(folder_path)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                if "plate" in fname.lower() or "best" in fname.lower():
                    continue

                rel_img_path = f"helmet_results/Bikes_Motorcycles/{id_folder}/{fname}"
                orig_img_name = fname.replace("helmet_", "")

                dets = h_csv_map.get(orig_img_name, [])
                boxes = []
                highest_conf = 0.0
                primary_status = "no_detection"

                for d in dets:
                    status_raw = str(d.get("helmet_status", "")).strip().lower()
                    try:
                        conf = float(d.get("confidence", 0.0))
                    except Exception:
                        conf = 0.0

                    if conf > highest_conf:
                        highest_conf = conf

                    x1 = d.get("x1", "")
                    y1 = d.get("y1", "")
                    x2 = d.get("x2", "")
                    y2 = d.get("y2", "")

                    if "with" in status_raw and "without" not in status_raw:
                        primary_status = "with_helmet"
                    elif "without" in status_raw:
                        primary_status = "without_helmet"

                    boxes.append({
                        "label": status_raw,
                        "confidence": round(conf, 3),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })

                # Determine primary status
                if primary_status == "without_helmet":
                    status = "No Helmet"
                    status_class = "violation"
                elif primary_status == "with_helmet":
                    status = "With Helmet"
                    status_class = "compliant"
                else:
                    status = "No Detection (Violation)"
                    status_class = "violation"

                plate_info = plate_map.get(f"Bikes_Motorcycles:{id_folder}", plate_map.get(id_folder, {"plate": "Unknown", "confidence": 0.0}))

                records.append({
                    "id": f"h_{len(records)}",
                    "model_type": "helmet",
                    "model_name": "YOLOv8 Helmet Detector",
                    "category": "Bikes_Motorcycles",
                    "vehicle_id": id_folder,
                    "filename": orig_img_name,
                    "annotated_filename": fname,
                    "image_rel_path": rel_img_path,
                    "status": status,
                    "status_class": status_class,
                    "confidence": round(highest_conf, 3),
                    "confidence_pct": round(highest_conf * 100, 1),
                    "boxes": boxes,
                    "box_count": len(boxes),
                    "plate_number": plate_info.get("plate", "Unknown"),
                    "plate_confidence": plate_info.get("confidence", 0.0)
                })

    return records


def generate_html(records):
    """Generate modern, interactive dark-mode HTML visualizer."""
    data_json = json.dumps(records)

    # Calculate statistics
    total_imgs = len(records)
    sb_records = [r for r in records if r["model_type"] == "seatbelt"]
    h_records = [r for r in records if r["model_type"] == "helmet"]

    sb_total = len(sb_records)
    sb_violations = sum(1 for r in sb_records if r["status_class"] == "violation")
    sb_compliant = sum(1 for r in sb_records if r["status_class"] == "compliant")
    sb_avg_conf = (sum(r["confidence"] for r in sb_records) / sb_total * 100) if sb_total > 0 else 0.0

    h_total = len(h_records)
    h_violations = sum(1 for r in h_records if r["status_class"] == "violation")
    h_compliant = sum(1 for r in h_records if r["status_class"] == "compliant")
    h_avg_conf = (sum(r["confidence"] for r in h_records) / h_total * 100) if h_total > 0 else 0.0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrafficGuard AI — Model Detection Audit & Bounding Box Inspector</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #07090e;
            --bg-secondary: #0f172a;
            --bg-card: rgba(15, 23, 42, 0.75);
            --bg-card-hover: rgba(30, 41, 59, 0.95);
            --border: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(99, 102, 241, 0.45);
            --accent-blue: #38bdf8;
            --accent-indigo: #6366f1;
            --accent-purple: #a855f7;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
            --accent-amber: #f59e0b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, var(--bg-primary) 70%);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, sans-serif;
            min-height: 100vh;
            padding: 2rem;
            line-height: 1.5;
        }}

        /* Header */
        header {{
            max-width: 1440px;
            margin: 0 auto 2rem;
        }}

        .brand-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.8rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .logo-group {{
            display: flex;
            align-items: center;
            gap: 1.2rem;
        }}

        .logo-badge {{
            width: 52px;
            height: 52px;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-purple));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.7rem;
            box-shadow: 0 0 30px rgba(99, 102, 241, 0.5);
        }}

        h1 {{
            font-size: 1.85rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #ffffff, #c084fc, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .subtitle {{
            font-size: 0.88rem;
            color: var(--text-secondary);
        }}

        .header-actions {{
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }}

        .btn-link {{
            padding: 0.55rem 1.1rem;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-primary);
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s ease;
        }}

        .btn-link:hover {{
            background: rgba(99, 102, 241, 0.2);
            border-color: var(--accent-indigo);
            transform: translateY(-2px);
        }}

        /* KPI Banner */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 1.2rem;
            margin-bottom: 2rem;
        }}

        .kpi-card {{
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.25rem 1.4rem;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-3px);
            border-color: var(--border-highlight);
        }}

        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--accent-color, var(--accent-indigo));
        }}

        .kpi-label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            margin-bottom: 0.35rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .kpi-val {{
            font-size: 1.9rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }}

        .kpi-subtext {{
            font-size: 0.76rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }}

        /* Filter Controls */
        .controls-card {{
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.4rem;
            margin-bottom: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
        }}

        .model-tabs {{
            display: flex;
            gap: 0.6rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 1rem;
            overflow-x: auto;
        }}

        .model-tab {{
            padding: 0.65rem 1.25rem;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            font-size: 0.88rem;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            transition: all 0.2s ease;
            white-space: nowrap;
        }}

        .model-tab:hover {{
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-primary);
        }}

        .model-tab.active {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(168, 85, 247, 0.25));
            border-color: var(--accent-indigo);
            color: var(--text-primary);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
        }}

        .tab-badge {{
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            font-size: 0.72rem;
            background: rgba(255, 255, 255, 0.1);
            font-family: 'JetBrains Mono', monospace;
        }}

        .controls-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            align-items: center;
            justify-content: space-between;
        }}

        .search-box {{
            flex: 1;
            min-width: 260px;
            position: relative;
        }}

        .search-box input {{
            width: 100%;
            background: rgba(2, 6, 23, 0.6);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.65rem 1rem 0.65rem 2.5rem;
            color: var(--text-primary);
            font-size: 0.88rem;
            outline: none;
            transition: all 0.2s ease;
        }}

        .search-box input:focus {{
            border-color: var(--accent-indigo);
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
        }}

        .search-icon {{
            position: absolute;
            left: 0.85rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            font-size: 0.95rem;
        }}

        .filter-group {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            flex-wrap: wrap;
        }}

        .select-filter {{
            background: rgba(2, 6, 23, 0.6);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.65rem 1rem;
            color: var(--text-primary);
            font-size: 0.85rem;
            font-weight: 500;
            outline: none;
            cursor: pointer;
            transition: border-color 0.2s;
        }}

        .select-filter:focus {{
            border-color: var(--accent-indigo);
        }}

        .confidence-slider-group {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            background: rgba(2, 6, 23, 0.6);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.45rem 0.9rem;
        }}

        .confidence-slider-group label {{
            font-size: 0.78rem;
            color: var(--text-secondary);
            font-weight: 600;
        }}

        .confidence-slider-group input[type="range"] {{
            width: 100px;
            accent-color: var(--accent-indigo);
            cursor: pointer;
        }}

        .conf-display {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            color: var(--accent-blue);
            min-width: 42px;
        }}

        /* Results Status Bar */
        .results-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            font-size: 0.88rem;
            color: var(--text-secondary);
        }}

        .results-count {{
            font-weight: 600;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }}

        /* Image Cards Grid */
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1.4rem;
            margin-bottom: 2.5rem;
        }}

        .audit-card {{
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.2s ease;
            position: relative;
            cursor: pointer;
        }}

        .audit-card:hover {{
            transform: translateY(-4px);
            border-color: var(--border-highlight);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
        }}

        .image-container {{
            width: 100%;
            height: 220px;
            background: #020617;
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .image-container img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s ease;
        }}

        .audit-card:hover .image-container img {{
            transform: scale(1.04);
        }}

        .card-badges {{
            position: absolute;
            top: 0.75rem;
            left: 0.75rem;
            right: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            pointer-events: none;
        }}

        .status-badge {{
            padding: 0.35rem 0.75rem;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }}

        .status-badge.compliant {{
            background: rgba(16, 185, 129, 0.85);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }}

        .status-badge.violation {{
            background: rgba(244, 63, 94, 0.9);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.3);
            animation: pulse-border 2s infinite;
        }}

        .status-badge.unknown {{
            background: rgba(100, 116, 139, 0.85);
            color: #ffffff;
        }}

        @keyframes pulse-border {{
            0%, 100% {{ box-shadow: 0 0 10px rgba(244, 63, 94, 0.6); }}
            50% {{ box-shadow: 0 0 20px rgba(244, 63, 94, 0.9); }}
        }}

        .conf-pill {{
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: var(--text-primary);
            padding: 0.3rem 0.6rem;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            backdrop-filter: blur(8px);
        }}

        .conf-pill.high {{ color: #4ade80; border-color: rgba(74, 222, 128, 0.4); }}
        .conf-pill.med {{ color: #fbbf24; border-color: rgba(251, 191, 36, 0.4); }}
        .conf-pill.low {{ color: #f87171; border-color: rgba(248, 113, 113, 0.4); }}

        .model-tag {{
            position: absolute;
            bottom: 0.75rem;
            left: 0.75rem;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(6px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 6px;
            padding: 0.25rem 0.55rem;
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}

        /* Card Content */
        .card-content {{
            padding: 1.1rem 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            flex: 1;
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
        }}

        .vehicle-id {{
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
        }}

        .category-tag {{
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 500;
        }}

        .plate-box {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(2, 6, 23, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 0.45rem 0.75rem;
            font-size: 0.82rem;
        }}

        .plate-label {{
            color: var(--text-muted);
            font-size: 0.75rem;
        }}

        .plate-val {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: var(--accent-blue);
            letter-spacing: 0.05em;
        }}

        .boxes-summary {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: 0.2rem;
        }}

        .det-tag {{
            font-size: 0.7rem;
            font-family: 'JetBrains Mono', monospace;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--text-secondary);
        }}

        .card-footer {{
            margin-top: auto;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 0.7rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            color: var(--text-muted);
        }}

        .filename-text {{
            max-width: 190px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-family: 'JetBrains Mono', monospace;
        }}

        .inspect-hint {{
            color: var(--accent-indigo);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }}

        /* Pagination */
        .pagination-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 3rem;
            flex-wrap: wrap;
        }}

        .page-btn {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.55rem 0.9rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .page-btn:hover:not(:disabled) {{
            background: rgba(99, 102, 241, 0.2);
            border-color: var(--accent-indigo);
        }}

        .page-btn.active {{
            background: var(--accent-indigo);
            border-color: var(--accent-indigo);
            color: white;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
        }}

        .page-btn:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}

        /* Modal Lightbox */
        .modal-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(2, 6, 23, 0.88);
            backdrop-filter: blur(12px);
            z-index: 1000;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }}

        .modal-overlay.active {{
            display: flex;
        }}

        .modal-content {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-highlight);
            border-radius: 20px;
            max-width: 1100px;
            width: 100%;
            max-height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 25px 60px rgba(0, 0, 0, 0.7);
            position: relative;
        }}

        .modal-header {{
            padding: 1.25rem 1.6rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .modal-title {{
            font-size: 1.25rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
        }}

        .modal-close {{
            background: rgba(255, 255, 255, 0.08);
            border: none;
            color: var(--text-secondary);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            font-size: 1.2rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }}

        .modal-close:hover {{
            background: var(--accent-rose);
            color: white;
        }}

        .modal-body {{
            display: grid;
            grid-template-columns: 1fr 340px;
            overflow-y: auto;
            flex: 1;
        }}

        @media (max-width: 900px) {{
            .modal-body {{
                grid-template-columns: 1fr;
            }}
        }}

        .modal-image-area {{
            background: #020617;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1.5rem;
            min-height: 400px;
            position: relative;
        }}

        .modal-image-area img {{
            max-width: 100%;
            max-height: 550px;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }}

        .modal-info-panel {{
            padding: 1.6rem;
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
            background: rgba(15, 23, 42, 0.5);
        }}

        .info-section {{
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }}

        .info-label {{
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 700;
        }}

        .info-value {{
            font-size: 0.95rem;
            color: var(--text-primary);
            font-weight: 600;
        }}

        .box-list {{
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            max-height: 220px;
            overflow-y: auto;
        }}

        .box-item {{
            background: rgba(2, 6, 23, 0.6);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
        }}

        .box-item-header {{
            display: flex;
            justify-content: space-between;
            font-weight: 700;
        }}

        .box-coords {{
            color: var(--text-muted);
            font-size: 0.72rem;
        }}

        .modal-nav-btn {{
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: white;
            width: 44px;
            height: 44px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 1.3rem;
            transition: all 0.2s;
            backdrop-filter: blur(8px);
        }}

        .modal-nav-btn:hover {{
            background: var(--accent-indigo);
        }}

        .modal-nav-prev {{ left: 1rem; }}
        .modal-nav-next {{ right: 1rem; }}

        /* Empty state */
        .empty-state {{
            grid-column: 1 / -1;
            text-align: center;
            padding: 4rem 2rem;
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px dashed var(--border);
            color: var(--text-muted);
        }}

        .empty-icon {{
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.6;
        }}
    </style>
</head>
<body>

    <header>
        <div class="brand-bar">
            <div class="logo-group">
                <div class="logo-badge">🔬</div>
                <div>
                    <h1>TrafficGuard AI — Model Detection Audit</h1>
                    <div class="subtitle">Visual inspection of raw bounding boxes & confidences for YOLOv8 Seatbelt & Helmet models</div>
                </div>
            </div>
            <div class="header-actions">
                <a href="challan_audit_visualizer.html" class="btn-link" style="border-color: rgba(245, 158, 11, 0.4); color: #fcd34d;">
                    <span>⚖️ e-Challan & Remarks Audit</span>
                </a>
                <a href="plate_visualizer.html" class="btn-link">
                    <span>📋 Plate & Violation Dashboard</span>
                </a>
            </div>
        </div>

        <!-- KPI Grid -->
        <div class="kpi-grid">
            <div class="kpi-card" style="--accent-color: var(--accent-indigo);">
                <div class="kpi-label">
                    <span>Total Annotated Crops</span>
                    <span>🖼️</span>
                </div>
                <div class="kpi-val">{total_imgs}</div>
                <div class="kpi-subtext">{sb_total} Cars • {h_total} Bikes</div>
            </div>

            <div class="kpi-card" style="--accent-color: var(--accent-blue);">
                <div class="kpi-label">
                    <span>Seatbelt Compliant</span>
                    <span>🟢</span>
                </div>
                <div class="kpi-val">{sb_compliant}</div>
                <div class="kpi-subtext">Avg Conf: {sb_avg_conf:.1f}%</div>
            </div>

            <div class="kpi-card" style="--accent-color: var(--accent-rose);">
                <div class="kpi-label">
                    <span>Seatbelt Violations</span>
                    <span>🚨</span>
                </div>
                <div class="kpi-val">{sb_violations}</div>
                <div class="kpi-subtext">Flagged without seatbelt</div>
            </div>

            <div class="kpi-card" style="--accent-color: var(--accent-emerald);">
                <div class="kpi-label">
                    <span>With Helmet</span>
                    <span>🪖</span>
                </div>
                <div class="kpi-val">{h_compliant}</div>
                <div class="kpi-subtext">Compliant riders</div>
            </div>

            <div class="kpi-card" style="--accent-color: var(--accent-amber);">
                <div class="kpi-label">
                    <span>Helmet Violations</span>
                    <span>🚨</span>
                </div>
                <div class="kpi-val">{h_violations}</div>
                <div class="kpi-subtext">Riders without helmet</div>
            </div>
        </div>

        <!-- Filter Controls -->
        <div class="controls-card">
            <!-- Model Tabs -->
            <div class="model-tabs">
                <button class="model-tab active" data-model="all" onclick="selectModel('all', this)">
                    <span>🌟 All Models</span>
                    <span class="tab-badge" id="tab-all-count">{total_imgs}</span>
                </button>
                <button class="model-tab" data-model="seatbelt" onclick="selectModel('seatbelt', this)">
                    <span>🚗 Seatbelt Model (Cars)</span>
                    <span class="tab-badge" id="tab-sb-count">{sb_total}</span>
                </button>
                <button class="model-tab" data-model="helmet" onclick="selectModel('helmet', this)">
                    <span>🏍️ Helmet Model (Bikes)</span>
                    <span class="tab-badge" id="tab-h-count">{h_total}</span>
                </button>
            </div>

            <!-- Filter Row -->
            <div class="controls-row">
                <div class="search-box">
                    <span class="search-icon">🔍</span>
                    <input type="text" id="searchInput" placeholder="Search Vehicle ID (e.g. id_036), Plate, or Frame..." oninput="handleSearch()">
                </div>

                <div class="filter-group">
                    <select class="select-filter" id="statusFilter" onchange="handleFilter()">
                        <option value="all">Status: All Results</option>
                        <option value="violation">🚨 Violations Only (No Seatbelt / No Helmet)</option>
                        <option value="compliant">🟢 Compliant Only (Seatbelt / Helmet On)</option>
                    </select>

                    <div class="confidence-slider-group">
                        <label for="confSlider">Min Conf:</label>
                        <input type="range" id="confSlider" min="0" max="95" step="5" value="0" oninput="handleConfSlider(this.value)">
                        <span class="conf-display" id="confDisplay">0%</span>
                    </div>

                    <select class="select-filter" id="sortFilter" onchange="handleSort()">
                        <option value="conf_desc">Sort: Confidence (High → Low)</option>
                        <option value="conf_asc">Sort: Confidence (Low → High)</option>
                        <option value="vid_asc">Sort: Vehicle ID (A → Z)</option>
                        <option value="viol_first">Sort: Violations First</option>
                    </select>
                </div>
            </div>
        </div>

        <div class="results-bar">
            <div>Showing <span class="results-count" id="visibleCount">0</span> of <span class="results-count" id="totalFilteredCount">0</span> audited images</div>
            <div id="pageInfoText">Page 1 of 1</div>
        </div>
    </header>

    <!-- Cards Grid -->
    <main>
        <div class="cards-grid" id="cardsGrid">
            <!-- Cards rendered dynamically -->
        </div>

        <!-- Pagination Controls -->
        <div class="pagination-container" id="pagination">
            <!-- Buttons generated dynamically -->
        </div>
    </main>

    <!-- Modal Lightbox -->
    <div class="modal-overlay" id="modalOverlay" onclick="closeModalOnOverlay(event)">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title" id="modalTitle">Vehicle ID</div>
                <button class="modal-close" onclick="closeModal()">✕</button>
            </div>
            <div class="modal-body">
                <div class="modal-image-area">
                    <button class="modal-nav-btn modal-nav-prev" onclick="navigateModal(-1)">❮</button>
                    <img id="modalImg" src="" alt="Detection inspection">
                    <button class="modal-nav-btn modal-nav-next" onclick="navigateModal(1)">❯</button>
                </div>
                <div class="modal-info-panel">
                    <div class="info-section">
                        <div class="info-label">Model Inspected</div>
                        <div class="info-value" id="modalModelName">YOLOv8</div>
                    </div>
                    <div class="info-section">
                        <div class="info-label">Detection Status</div>
                        <div class="info-value" id="modalStatusBadge">Status</div>
                    </div>
                    <div class="info-section">
                        <div class="info-label">Associated License Plate</div>
                        <div class="info-value" id="modalPlateText">Unknown</div>
                    </div>
                    <div class="info-section">
                        <div class="info-label">Source File</div>
                        <div class="info-value" id="modalFilename" style="font-family:'JetBrains Mono', monospace; font-size:0.8rem; word-break:break-all;">frame</div>
                    </div>
                    <div class="info-section">
                        <div class="info-label">Detected Bounding Boxes (<span id="modalBoxCount">0</span>)</div>
                        <div class="box-list" id="modalBoxList">
                            <!-- Box details -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const ALL_RECORDS = {data_json};
        let currentFiltered = [...ALL_RECORDS];
        let currentModel = 'all';
        let currentMinConf = 0;
        let currentPage = 1;
        const ITEMS_PER_PAGE = 36;
        let currentModalIndex = 0;

        function init() {{
            applyFilters();
            setupKeyboardNav();
        }}

        function selectModel(model, btn) {{
            currentModel = model;
            document.querySelectorAll('.model-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentPage = 1;
            applyFilters();
        }}

        function handleSearch() {{
            currentPage = 1;
            applyFilters();
        }}

        function handleFilter() {{
            currentPage = 1;
            applyFilters();
        }}

        function handleConfSlider(val) {{
            currentMinConf = parseInt(val, 10);
            document.getElementById('confDisplay').textContent = currentMinConf + '%';
            currentPage = 1;
            applyFilters();
        }}

        function handleSort() {{
            applyFilters();
        }}

        function applyFilters() {{
            const search = document.getElementById('searchInput').value.trim().toLowerCase();
            const statusFilter = document.getElementById('statusFilter').value;
            const sortVal = document.getElementById('sortFilter').value;

            currentFiltered = ALL_RECORDS.filter(rec => {{
                // Model tab filter
                if (currentModel !== 'all' && rec.model_type !== currentModel) {{
                    return false;
                }}

                // Status filter
                if (statusFilter !== 'all') {{
                    if (statusFilter === 'violation' && rec.status_class !== 'violation') return false;
                    if (statusFilter === 'compliant' && rec.status_class !== 'compliant') return false;
                }}

                // Confidence filter
                if (rec.confidence_pct < currentMinConf) {{
                    return false;
                }}

                // Search filter (Vehicle ID, Plate, Filename)
                if (search) {{
                    const matchVid = rec.vehicle_id.toLowerCase().includes(search);
                    const matchPlate = rec.plate_number.toLowerCase().includes(search);
                    const matchFile = rec.filename.toLowerCase().includes(search);
                    if (!matchVid && !matchPlate && !matchFile) return false;
                }}

                return true;
            }});

            // Sorting
            if (sortVal === 'conf_desc') {{
                currentFiltered.sort((a, b) => b.confidence - a.confidence);
            }} else if (sortVal === 'conf_asc') {{
                currentFiltered.sort((a, b) => a.confidence - b.confidence);
            }} else if (sortVal === 'vid_asc') {{
                currentFiltered.sort((a, b) => a.vehicle_id.localeCompare(b.vehicle_id, undefined, {{ numeric: true }}));
            }} else if (sortVal === 'viol_first') {{
                currentFiltered.sort((a, b) => (b.status_class === 'violation' ? 1 : 0) - (a.status_class === 'violation' ? 1 : 0));
            }}

            document.getElementById('totalFilteredCount').textContent = currentFiltered.length;
            renderCards();
            renderPagination();
        }}

        function renderCards() {{
            const grid = document.getElementById('cardsGrid');
            grid.innerHTML = '';

            if (currentFiltered.length === 0) {{
                grid.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-icon">🔍</div>
                        <h3>No matching model crops found</h3>
                        <p>Try adjusting your search criteria, model tab, or confidence threshold.</p>
                    </div>
                `;
                document.getElementById('visibleCount').textContent = '0';
                return;
            }}

            const startIdx = (currentPage - 1) * ITEMS_PER_PAGE;
            const endIdx = Math.min(startIdx + ITEMS_PER_PAGE, currentFiltered.length);
            const pageRecords = currentFiltered.slice(startIdx, endIdx);

            document.getElementById('visibleCount').textContent = `${{startIdx + 1}}-${{endIdx}}`;

            pageRecords.forEach((rec, idx) => {{
                const absoluteIndex = startIdx + idx;
                const card = document.createElement('div');
                card.className = 'audit-card';
                card.onclick = () => openModal(absoluteIndex);

                let confClass = 'low';
                if (rec.confidence_pct >= 80) confClass = 'high';
                else if (rec.confidence_pct >= 50) confClass = 'med';

                const icon = rec.model_type === 'seatbelt' ? '🚗' : '🏍️';

                let boxesHtml = '';
                if (rec.boxes.length > 0) {{
                    rec.boxes.slice(0, 3).forEach(b => {{
                        boxesHtml += `<span class="det-tag">${{b.label}} (${{Math.round(b.confidence * 100)}}%)</span>`;
                    }});
                    if (rec.boxes.length > 3) {{
                        boxesHtml += `<span class="det-tag">+${{rec.boxes.length - 3}} more</span>`;
                    }}
                }} else {{
                    boxesHtml = `<span class="det-tag" style="color:var(--text-muted);">No detections</span>`;
                }}

                card.innerHTML = `
                    <div class="image-container">
                        <img src="${{rec.image_rel_path}}" alt="${{rec.filename}}" loading="lazy" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'300\\' height=\\'200\\'><rect width=\\'300\\' height=\\'200\\' fill=\\'%231e293b\\'/><text x=\\'50%\\' y=\\'50%\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\' fill=\\'%2364748b\\'>Image not found</text></svg>'">
                        <div class="card-badges">
                            <span class="status-badge ${{rec.status_class}}">${{rec.status}}</span>
                            <span class="conf-pill ${{confClass}}">${{rec.confidence_pct}}%</span>
                        </div>
                        <div class="model-tag">
                            <span>${{icon}}</span>
                            <span>${{rec.model_type.toUpperCase()}}</span>
                        </div>
                    </div>
                    <div class="card-content">
                        <div class="card-header">
                            <div class="vehicle-id">${{rec.vehicle_id}}</div>
                            <div class="category-tag">${{rec.category}}</div>
                        </div>
                        <div class="plate-box">
                            <span class="plate-label">License Plate:</span>
                            <span class="plate-val">${{rec.plate_number !== 'Unknown' ? rec.plate_number : '—'}}</span>
                        </div>
                        <div class="boxes-summary">
                            ${{boxesHtml}}
                        </div>
                        <div class="card-footer">
                            <span class="filename-text" title="${{rec.filename}}">${{rec.filename}}</span>
                            <span class="inspect-hint">Inspect 🔍</span>
                        </div>
                    </div>
                `;

                grid.appendChild(card);
            }});
        }}

        function renderPagination() {{
            const container = document.getElementById('pagination');
            container.innerHTML = '';

            const totalPages = Math.ceil(currentFiltered.length / ITEMS_PER_PAGE) || 1;
            document.getElementById('pageInfoText').textContent = `Page ${{currentPage}} of ${{totalPages}}`;

            if (totalPages <= 1) return;

            // Previous Button
            const prevBtn = document.createElement('button');
            prevBtn.className = 'page-btn';
            prevBtn.innerHTML = '❮ Prev';
            prevBtn.disabled = currentPage === 1;
            prevBtn.onclick = () => {{
                if (currentPage > 1) {{
                    currentPage--;
                    renderCards();
                    renderPagination();
                    window.scrollTo({{ top: 380, behavior: 'smooth' }});
                }}
            }};
            container.appendChild(prevBtn);

            // Page Number Buttons (smart window)
            let startPage = Math.max(1, currentPage - 2);
            let endPage = Math.min(totalPages, currentPage + 2);

            if (startPage > 1) {{
                const firstBtn = document.createElement('button');
                firstBtn.className = 'page-btn';
                firstBtn.textContent = '1';
                firstBtn.onclick = () => goToPage(1);
                container.appendChild(firstBtn);

                if (startPage > 2) {{
                    const dots = document.createElement('span');
                    dots.style.color = 'var(--text-muted)';
                    dots.textContent = '...';
                    container.appendChild(dots);
                }}
            }}

            for (let i = startPage; i <= endPage; i++) {{
                const btn = document.createElement('button');
                btn.className = `page-btn ${{i === currentPage ? 'active' : ''}}`;
                btn.textContent = i;
                btn.onclick = () => goToPage(i);
                container.appendChild(btn);
            }}

            if (endPage < totalPages) {{
                if (endPage < totalPages - 1) {{
                    const dots = document.createElement('span');
                    dots.style.color = 'var(--text-muted)';
                    dots.textContent = '...';
                    container.appendChild(dots);
                }}

                const lastBtn = document.createElement('button');
                lastBtn.className = 'page-btn';
                lastBtn.textContent = totalPages;
                lastBtn.onclick = () => goToPage(totalPages);
                container.appendChild(lastBtn);
            }}

            // Next Button
            const nextBtn = document.createElement('button');
            nextBtn.className = 'page-btn';
            nextBtn.innerHTML = 'Next ❯';
            nextBtn.disabled = currentPage === totalPages;
            nextBtn.onclick = () => {{
                if (currentPage < totalPages) {{
                    currentPage++;
                    renderCards();
                    renderPagination();
                    window.scrollTo({{ top: 380, behavior: 'smooth' }});
                }}
            }};
            container.appendChild(nextBtn);
        }}

        function goToPage(page) {{
            currentPage = page;
            renderCards();
            renderPagination();
            window.scrollTo({{ top: 380, behavior: 'smooth' }});
        }}

        /* Lightbox Modal */
        function openModal(index) {{
            currentModalIndex = index;
            const rec = currentFiltered[index];
            if (!rec) return;

            document.getElementById('modalTitle').textContent = `${{rec.vehicle_id}} • ${{rec.category}}`;
            document.getElementById('modalImg').src = rec.image_rel_path;
            document.getElementById('modalModelName').textContent = rec.model_name;

            const badgeElem = document.getElementById('modalStatusBadge');
            badgeElem.innerHTML = `<span class="status-badge ${{rec.status_class}}">${{rec.status}} (${{rec.confidence_pct}}% Conf)</span>`;

            document.getElementById('modalPlateText').textContent = rec.plate_number !== 'Unknown' 
                ? `${{rec.plate_number}} (${{Math.round(rec.plate_confidence * 100)}}% OCR)`
                : 'Not detected / unavailable';

            document.getElementById('modalFilename').textContent = rec.filename;
            document.getElementById('modalBoxCount').textContent = rec.boxes.length;

            const boxList = document.getElementById('modalBoxList');
            boxList.innerHTML = '';

            if (rec.boxes.length === 0) {{
                boxList.innerHTML = `<div style="color:var(--text-muted); padding:0.5rem 0;">No bounding boxes logged for this image.</div>`;
            }} else {{
                rec.boxes.forEach((b, i) => {{
                    const item = document.createElement('div');
                    item.className = 'box-item';
                    item.innerHTML = `
                        <div class="box-item-header">
                            <span>#${{i + 1}} ${{b.label}}</span>
                            <span style="color:var(--accent-blue);">${{Math.round(b.confidence * 100)}}%</span>
                        </div>
                        <div class="box-coords">Coords: [${{b.x1}}, ${{b.y1}}, ${{b.x2}}, ${{b.y2}}]</div>
                    `;
                    boxList.appendChild(item);
                }});
            }}

            document.getElementById('modalOverlay').classList.add('active');
            document.body.style.overflow = 'hidden';
        }}

        function closeModal() {{
            document.getElementById('modalOverlay').classList.remove('active');
            document.body.style.overflow = '';
        }}

        function closeModalOnOverlay(e) {{
            if (e.target.id === 'modalOverlay') {{
                closeModal();
            }}
        }}

        function navigateModal(dir) {{
            const newIndex = currentModalIndex + dir;
            if (newIndex >= 0 && newIndex < currentFiltered.length) {{
                openModal(newIndex);
            }}
        }}

        function setupKeyboardNav() {{
            document.addEventListener('keydown', (e) => {{
                const modal = document.getElementById('modalOverlay');
                if (modal.classList.contains('active')) {{
                    if (e.key === 'Escape') closeModal();
                    else if (e.key === 'ArrowLeft') navigateModal(-1);
                    else if (e.key === 'ArrowRight') navigateModal(1);
                }}
            }});
        }}

        window.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
"""

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n🎉 Model Detection Audit Dashboard created successfully!")
    print(f"📄 Dashboard Path: {HTML_OUTPUT}")
    print(f"📊 Processed {len(records)} annotated detection images ({sb_total} Seatbelt + {h_total} Helmet).")

    try:
        webbrowser.open(HTML_OUTPUT)
        print(f"🌐 Opening dashboard in browser: {HTML_OUTPUT}")
    except Exception as e:
        print(f"⚠️ Could not auto-open browser: {e}")


def main():
    print("🚀 Generating Model Detection Audit & Bounding Box Inspector...")
    records = build_audit_dataset()
    generate_html(records)


if __name__ == "__main__":
    main()
