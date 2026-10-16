"""
Interactive Plate & Violation Visualizer
Generates a presentation-ready, interactive web dashboard (HTML)
to visually inspect cropped vehicles, their license plates, OCR numbers,
and violation status.
"""

import os
import sys
import csv
import json
import webbrowser

# Fix Windows console encoding for Unicode/emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "../output"))
CROPPED_PHOTOS_DIR = os.path.join(OUTPUT_DIR, "cropped_photos")
MASTER_CSV = os.path.join(OUTPUT_DIR, "master_violation_log.csv")
HTML_OUTPUT = os.path.join(OUTPUT_DIR, "plate_visualizer.html")

# Downstream violation CSVs if they exist
SEATBELT_CSV = os.path.join(OUTPUT_DIR, "seatbelt_results.csv")
HELMET_CSV = os.path.join(OUTPUT_DIR, "helmet_results.csv")
REDLIGHT_CSV = os.path.join(OUTPUT_DIR, "redlight_violations.csv")
CLOSENESS_CSV = os.path.join(OUTPUT_DIR, "vehicle_closeness_check", "violations.csv")


def load_violations():
    """Load downstream violation logs if present."""
    seatbelt_map = {}
    helmet_map = {}
    redlight_set = set()

    if os.path.exists(SEATBELT_CSV):
        try:
            with open(SEATBELT_CSV, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = str(row.get("vehicle_id", "")).strip()
                    lbl = str(row.get("label", "")).lower()
                    if "no" in lbl and "seatbelt" in lbl:
                        seatbelt_map[vid] = "Violation"
                    elif "seatbelt" in lbl:
                        seatbelt_map[vid] = seatbelt_map.get(vid, "Compliant")
        except Exception:
            pass

    if os.path.exists(HELMET_CSV):
        try:
            with open(HELMET_CSV, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = str(row.get("vehicle_id", "")).strip()
                    status = str(row.get("helmet_status", "")).lower()
                    if "without" in status or "no" in status:
                        helmet_map[vid] = "Violation"
                    elif "with" in status:
                        helmet_map[vid] = helmet_map.get(vid, "Compliant")
        except Exception:
            pass

    return seatbelt_map, helmet_map, redlight_set


def build_dashboard_data():
    """Build structured vehicle records with photos, plates, OCR text, and violations."""
    if not os.path.exists(MASTER_CSV):
        print(f"❌ Master violation log not found at: {MASTER_CSV}")
        return []

    seatbelt_map, helmet_map, redlight_set = load_violations()
    records = []

    with open(MASTER_CSV, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("Vehicle_ID", "").strip()
            category = row.get("Category", "Unknown").strip()
            plate_number = row.get("Plate_Number", "Unknown").strip()
            try:
                plate_conf = float(row.get("Plate_Confidence", 0.0))
            except Exception:
                plate_conf = 0.0

            # Find folder on disk
            vehicle_dir = os.path.join(CROPPED_PHOTOS_DIR, category, vid)
            plate_img_rel = ""
            vehicle_img_rel = ""

            if os.path.isdir(vehicle_dir):
                plate_path = os.path.join(vehicle_dir, "BEST_PLATE_CROP.jpg")
                if os.path.exists(plate_path):
                    plate_img_rel = f"cropped_photos/{category}/{vid}/BEST_PLATE_CROP.jpg"

                # Find candidate vehicle full crop (prefer BEST.jpg if exists, else pick highest zone image)
                all_imgs = [f for f in os.listdir(vehicle_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")) and "plate" not in f.lower()]
                if "BEST.jpg" in all_imgs:
                    vehicle_img_rel = f"cropped_photos/{category}/{vid}/BEST.jpg"
                elif all_imgs:
                    # Sort by frame / zone and pick middle or best
                    all_imgs.sort()
                    vehicle_img_rel = f"cropped_photos/{category}/{vid}/{all_imgs[-1]}"

            # Determine violations
            raw_id = vid.replace("id_", "").lstrip("0") or "0"
            sb_status = seatbelt_map.get(raw_id, seatbelt_map.get(vid, "N/A" if category == "Bikes_Motorcycles" else "Compliant"))
            hl_status = helmet_map.get(raw_id, helmet_map.get(vid, "Compliant" if category == "Bikes_Motorcycles" else "N/A"))
            rl_status = "Violation" if (raw_id in redlight_set or vid in redlight_set) else "Compliant"

            records.append({
                "id": vid,
                "category": category,
                "plate_number": plate_number if plate_number != "Unknown" else "",
                "has_plate": bool(plate_img_rel and plate_number and plate_number != "Unknown"),
                "plate_conf": round(plate_conf * 100, 1),
                "plate_img": plate_img_rel,
                "vehicle_img": vehicle_img_rel,
                "seatbelt": sb_status,
                "helmet": hl_status,
                "red_light": rl_status,
                "phone": "Compliant"
            })

    return records


def generate_html(records):
    """Generate self-contained modern HTML presentation dashboard."""
    total_vehicles = len(records)
    plates_detected = sum(1 for r in records if r["plate_img"])
    plates_read = sum(1 for r in records if r["has_plate"])
    avg_conf = round(sum(r["plate_conf"] for r in records if r["has_plate"]) / max(1, plates_read), 1)

    records_json = json.dumps(records)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrafficGuard AI — Plate & Violation Inspector</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-secondary: #111827;
            --bg-card: rgba(17, 24, 39, 0.75);
            --bg-card-hover: rgba(31, 41, 55, 0.9);
            --border: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(99, 102, 241, 0.4);
            --accent-blue: #38bdf8;
            --accent-indigo: #6366f1;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            background: radial-gradient(circle at 50% 0%, #172554 0%, var(--bg-primary) 70%);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, sans-serif;
            min-height: 100vh;
            padding: 2rem;
            line-height: 1.5;
        }}

        /* Header & Brand */
        header {{
            max-width: 1400px;
            margin: 0 auto 2.5rem;
        }}

        .brand-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1.5rem;
        }}

        .logo-group {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .logo-badge {{
            width: 48px;
            height: 48px;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-blue));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.6rem;
            box-shadow: 0 0 25px rgba(99, 102, 241, 0.4);
        }}

        h1 {{
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #ffffff, #93c5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .subtitle {{
            font-size: 0.88rem;
            color: var(--text-secondary);
        }}

        .status-pill {{
            padding: 0.4rem 1rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--accent-emerald);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .status-pill::before {{
            content: '';
            width: 8px;
            height: 8px;
            background: var(--accent-emerald);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-emerald);
        }}

        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }}

        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            backdrop-filter: blur(12px);
            transition: transform 0.2s, border-color 0.2s;
        }}

        .metric-card:hover {{
            transform: translateY(-2px);
            border-color: var(--border-highlight);
        }}

        .metric-label {{
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.4rem;
        }}

        .metric-val {{
            font-size: 2.1rem;
            font-weight: 800;
            color: #fff;
            letter-spacing: -0.03em;
        }}

        .metric-sub {{
            font-size: 0.78rem;
            color: var(--accent-blue);
            margin-top: 0.2rem;
        }}

        /* Filter Controls */
        .controls-panel {{
            max-width: 1400px;
            margin: 0 auto 2rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.25rem 1.5rem;
            backdrop-filter: blur(12px);
            display: flex;
            flex-wrap: wrap;
            gap: 1.25rem;
            align-items: center;
            justify-content: space-between;
        }}

        .filter-group {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        .filter-btn {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 0.45rem 1rem;
            border-radius: 10px;
            font-size: 0.82rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .filter-btn:hover {{
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
        }}

        .filter-btn.active {{
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-blue));
            color: #fff;
            border-color: transparent;
            font-weight: 600;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.35);
        }}

        .search-box {{
            flex: 1;
            min-width: 260px;
            position: relative;
        }}

        .search-box input {{
            width: 100%;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.55rem 1rem 0.55rem 2.4rem;
            color: #fff;
            font-size: 0.85rem;
            outline: none;
            transition: border-color 0.2s;
        }}

        .search-box input:focus {{
            border-color: var(--accent-blue);
        }}

        .search-icon {{
            position: absolute;
            left: 0.8rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            pointer-events: none;
        }}

        /* Vehicle Cards Grid */
        .cards-container {{
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 1.5rem;
        }}

        .vehicle-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 18px;
            overflow: hidden;
            backdrop-filter: blur(12px);
            transition: transform 0.25s, border-color 0.25s, box-shadow 0.25s;
            display: flex;
            flex-direction: column;
        }}

        .vehicle-card:hover {{
            transform: translateY(-4px);
            border-color: var(--border-highlight);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
        }}

        .card-header {{
            padding: 1rem 1.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            background: rgba(0, 0, 0, 0.2);
        }}

        .vid-badge {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 0.95rem;
            color: #fff;
        }}

        .cat-tag {{
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .cat-car {{ background: rgba(56, 189, 248, 0.15); color: var(--accent-blue); border: 1px solid rgba(56, 189, 248, 0.3); }}
        .cat-bike {{ background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }}
        .cat-rickshaw {{ background: rgba(245, 158, 11, 0.15); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.3); }}
        .cat-unknown {{ background: rgba(156, 163, 175, 0.15); color: var(--text-secondary); border: 1px solid var(--border); }}

        /* Imagery Layout */
        .imagery-grid {{
            display: grid;
            grid-template-columns: 1.4fr 1fr;
            gap: 0.75rem;
            padding: 1.25rem;
            background: rgba(0, 0, 0, 0.15);
        }}

        .img-box {{
            position: relative;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            aspect-ratio: 4/3;
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .img-box img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            cursor: pointer;
            transition: transform 0.3s;
        }}

        .img-box img:hover {{
            transform: scale(1.05);
        }}

        .img-caption {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 0.35rem 0.5rem;
            background: linear-gradient(to top, rgba(0,0,0,0.85), transparent);
            font-size: 0.68rem;
            color: var(--text-secondary);
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .no-img-text {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-align: center;
            padding: 1rem;
        }}

        /* Plate Display Banner */
        .plate-banner-box {{
            padding: 1rem 1.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-top: 1px solid var(--border);
            border-bottom: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.02);
        }}

        .pk-plate {{
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            border: 2px solid #94a3b8;
            border-radius: 8px;
            padding: 0.4rem 0.8rem;
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.2);
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.15em;
        }}

        .pk-plate.read-ok {{
            border-color: #38bdf8;
            background: linear-gradient(180deg, #0f172a 0%, #022c22 100%);
            border-color: #10b981;
        }}

        .pk-plate-text {{
            font-weight: 900;
            font-size: 1.15rem;
            color: #ffffff;
            text-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
        }}

        .pk-plate-empty {{
            font-size: 0.85rem;
            font-weight: 600;
            color: #ef4444;
            letter-spacing: 0.05em;
        }}

        .conf-pill {{
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-emerald);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .conf-pill.low {{
            background: rgba(244, 63, 94, 0.15);
            color: var(--accent-rose);
            border-color: rgba(244, 63, 94, 0.3);
        }}

        /* Violation Testing Tags */
        .violations-hub {{
            padding: 1rem 1.25rem;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem;
            margin-top: auto;
            background: rgba(0, 0, 0, 0.2);
        }}

        .v-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.73rem;
            padding: 0.35rem 0.6rem;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            cursor: pointer;
            transition: all 0.15s;
        }}

        .v-item:hover {{
            background: rgba(255, 255, 255, 0.08);
        }}

        .v-label {{
            color: var(--text-secondary);
            font-weight: 500;
        }}

        .v-status {{
            font-weight: 700;
            font-size: 0.7rem;
        }}

        .v-ok {{ color: var(--accent-emerald); }}
        .v-bad {{ color: var(--accent-rose); font-weight: 800; }}
        .v-na {{ color: var(--text-muted); }}

        /* Modal Inspector */
        .modal-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(8px);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            padding: 2rem;
        }}

        .modal-overlay.open {{
            display: flex;
        }}

        .modal-content {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-highlight);
            border-radius: 20px;
            max-width: 900px;
            width: 100%;
            overflow: hidden;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
        }}

        .modal-header {{
            padding: 1.25rem 1.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            background: rgba(0, 0, 0, 0.3);
        }}

        .close-btn {{
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 1.5rem;
            cursor: pointer;
            padding: 0.2rem 0.6rem;
            border-radius: 8px;
        }}

        .close-btn:hover {{
            color: #fff;
            background: rgba(255, 255, 255, 0.1);
        }}

        .modal-body {{
            padding: 1.75rem;
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 1.5rem;
        }}

        .modal-img {{
            width: 100%;
            border-radius: 12px;
            border: 1px solid var(--border);
            object-fit: contain;
            background: #000;
            max-height: 400px;
        }}

        .empty-state {{
            grid-column: 1 / -1;
            text-align: center;
            padding: 4rem;
            color: var(--text-muted);
        }}

        @media (max-width: 800px) {{
            .imagery-grid {{ grid-template-columns: 1fr; }}
            .modal-body {{ grid-template-columns: 1fr; }}
            body {{ padding: 1rem; }}
        }}
    </style>
</head>
<body>

    <header>
        <div class="brand-bar">
            <div class="logo-group">
                <div class="logo-badge">🛡️</div>
                <div>
                    <h1>TrafficGuard AI — Plate & Violation Inspector</h1>
                    <div class="subtitle">Live Vehicle Snapshots, OCR Plate Reading, and Violation Testing Hub</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap;">
                <a href="challan_audit_visualizer.html" style="padding: 0.45rem 1rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; background: rgba(245, 158, 11, 0.2); border: 1px solid rgba(245, 158, 11, 0.4); color: #fcd34d; text-decoration: none; display: flex; align-items: center; gap: 0.4rem;">
                    <span>⚖️ e-Challan & Remarks Audit</span>
                </a>
                <a href="model_audit_visualizer.html" style="padding: 0.45rem 1rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(99, 102, 241, 0.4); color: #818cf8; text-decoration: none; display: flex; align-items: center; gap: 0.4rem;">
                    <span>🔬 Model Detection Audit</span>
                </a>
                <div class="status-pill">System Active</div>
            </div>
        </div>

        <!-- Metrics Dashboard -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Vehicles Tracked</div>
                <div class="metric-val" id="m-total">{total_vehicles}</div>
                <div class="metric-sub">Cropped from Video</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Plates Detected</div>
                <div class="metric-val" id="m-detected" style="color: var(--accent-blue);">{plates_detected}</div>
                <div class="metric-sub">{round((plates_detected / max(1, total_vehicles)) * 100, 1)}% of vehicles</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Plates Read (OCR)</div>
                <div class="metric-val" id="m-read" style="color: var(--accent-emerald);">{plates_read}</div>
                <div class="metric-sub">Avg Confidence: {avg_conf}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Active Violations</div>
                <div class="metric-val" id="m-violations" style="color: var(--accent-rose);">0</div>
                <div class="metric-sub">Interactive Testing Mode</div>
            </div>
        </div>

        <!-- Control / Filter Bar -->
        <div class="controls-panel">
            <div class="filter-group" id="cat-filters">
                <button class="filter-btn active" data-cat="all">All ({total_vehicles})</button>
                <button class="filter-btn" data-cat="Cars">Cars</button>
                <button class="filter-btn" data-cat="Bikes_Motorcycles">Motorcycles</button>
                <button class="filter-btn" data-cat="Rickshaws">Rickshaws</button>
            </div>

            <div class="filter-group" id="status-filters">
                <button class="filter-btn active" data-status="all">All Plates</button>
                <button class="filter-btn" data-status="read">✅ Read ({plates_read})</button>
                <button class="filter-btn" data-status="unread">⚠️ Unread</button>
                <button class="filter-btn" data-status="violation">🚨 Violations Only</button>
            </div>

            <div class="search-box">
                <span class="search-icon">🔍</span>
                <input type="text" id="search-input" placeholder="Search by Vehicle ID or Plate Number...">
            </div>
        </div>
    </header>

    <!-- Cards Grid -->
    <main class="cards-container" id="cards-grid"></main>

    <!-- Modal Inspector -->
    <div class="modal-overlay" id="modal-overlay">
        <div class="modal-content">
            <div class="modal-header">
                <div>
                    <h2 id="modal-title" style="font-size: 1.2rem; font-weight: 800;">Vehicle Details</h2>
                    <div id="modal-subtitle" style="font-size: 0.8rem; color: var(--text-secondary);">Full Verification Inspector</div>
                </div>
                <button class="close-btn" id="modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <div>
                    <div style="font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem; color: var(--text-secondary);">Vehicle View</div>
                    <img id="modal-veh-img" class="modal-img" src="" alt="Vehicle Image">
                </div>
                <div>
                    <div style="font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem; color: var(--text-secondary);">Cropped Plate & OCR</div>
                    <img id="modal-plate-img" class="modal-img" src="" alt="Plate Crop" style="max-height: 180px; margin-bottom: 1rem;">
                    <div class="pk-plate read-ok" style="width: 100%; justify-content: center; margin-bottom: 1rem;">
                        <span class="pk-plate-text" id="modal-plate-text" style="font-size: 1.4rem;"></span>
                    </div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.8;">
                        <div><strong>Confidence:</strong> <span id="modal-conf"></span></div>
                        <div><strong>Category:</strong> <span id="modal-cat"></span></div>
                        <div><strong>Violations:</strong> <span id="modal-violation-summary"></span></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const records = {records_json};
        let currentCat = 'all';
        let currentStatus = 'all';
        let searchQuery = '';

        const grid = document.getElementById('cards-grid');
        const searchInput = document.getElementById('search-input');
        const modalOverlay = document.getElementById('modal-overlay');
        const modalClose = document.getElementById('modal-close');

        function getCategoryTag(cat) {{
            if (cat === 'Cars') return '<span class="cat-tag cat-car">Car / Heavy</span>';
            if (cat === 'Bikes_Motorcycles') return '<span class="cat-tag cat-bike">Motorcycle</span>';
            if (cat === 'Rickshaws') return '<span class="cat-tag cat-rickshaw">Rickshaw</span>';
            return '<span class="cat-tag cat-unknown">Unknown</span>';
        }}

        function renderCards() {{
            grid.innerHTML = '';
            let visibleCount = 0;
            let violationCount = 0;

            const filtered = records.filter(r => {{
                // Category Filter
                if (currentCat !== 'all' && r.category !== currentCat) return false;

                // Status Filter
                const hasViolation = (r.seatbelt === 'Violation' || r.helmet === 'Violation' || r.red_light === 'Violation' || r.phone === 'Violation');
                if (hasViolation) violationCount++;

                if (currentStatus === 'read' && !r.has_plate) return false;
                if (currentStatus === 'unread' && r.has_plate) return false;
                if (currentStatus === 'violation' && !hasViolation) return false;

                // Search Filter
                if (searchQuery) {{
                    const q = searchQuery.toLowerCase();
                    const matchesId = r.id.toLowerCase().includes(q);
                    const matchesPlate = r.plate_number.toLowerCase().includes(q);
                    if (!matchesId && !matchesPlate) return false;
                }}

                return true;
            }});

            document.getElementById('m-violations').textContent = violationCount;

            if (filtered.length === 0) {{
                grid.innerHTML = '<div class="empty-state"><h3>No vehicles match the selected filter</h3><p>Try clearing filters or search query.</p></div>';
                return;
            }}

            filtered.forEach(r => {{
                visibleCount++;
                const card = document.createElement('div');
                card.className = 'vehicle-card';

                const plateDisplay = r.has_plate ? 
                    `<div class="pk-plate read-ok"><span class="pk-plate-text">${{r.plate_number}}</span></div>` :
                    `<div class="pk-plate"><span class="pk-plate-empty">NOT READ</span></div>`;

                const confClass = r.plate_conf >= 70 ? 'conf-pill' : 'conf-pill low';
                const confText = r.has_plate ? `${{r.plate_conf}}% OCR` : 'No Match';

                const vStatus = (val) => {{
                    if (val === 'Violation') return '<span class="v-status v-bad">🚨 ALERT</span>';
                    if (val === 'Compliant') return '<span class="v-status v-ok">✔ PASS</span>';
                    return '<span class="v-status v-na">N/A</span>';
                }};

                card.innerHTML = `
                    <div class="card-header">
                        <span class="vid-badge">${{r.id.toUpperCase()}}</span>
                        ${{getCategoryTag(r.category)}}
                    </div>

                    <div class="imagery-grid">
                        <div class="img-box">
                            ${{r.vehicle_img ? `<img src="${{r.vehicle_img}}" alt="${{r.id}} Vehicle" onclick="openModal('${{r.id}}')">` : '<div class="no-img-text">No vehicle snapshot</div>'}}
                            <div class="img-caption"><span>Vehicle Crop</span><span>🔍 Click</span></div>
                        </div>
                        <div class="img-box">
                            ${{r.plate_img ? `<img src="${{r.plate_img}}" alt="${{r.id}} Plate" onclick="openModal('${{r.id}}')">` : '<div class="no-img-text">No Plate Crop</div>'}}
                            <div class="img-caption"><span>Plate Crop</span></div>
                        </div>
                    </div>

                    <div class="plate-banner-box">
                        ${{plateDisplay}}
                        <div class="${{confClass}}">${{confText}}</div>
                    </div>

                    <div class="violations-hub">
                        <div class="v-item" onclick="toggleViolation('${{r.id}}', 'seatbelt')">
                            <span class="v-label">Seatbelt:</span>
                            ${{vStatus(r.seatbelt)}}
                        </div>
                        <div class="v-item" onclick="toggleViolation('${{r.id}}', 'helmet')">
                            <span class="v-label">Helmet:</span>
                            ${{vStatus(r.helmet)}}
                        </div>
                        <div class="v-item" onclick="toggleViolation('${{r.id}}', 'red_light')">
                            <span class="v-label">Red Light:</span>
                            ${{vStatus(r.red_light)}}
                        </div>
                        <div class="v-item" onclick="toggleViolation('${{r.id}}', 'phone')">
                            <span class="v-label">Phone:</span>
                            ${{vStatus(r.phone)}}
                        </div>
                    </div>
                `;

                grid.appendChild(card);
            }});
        }}

        // Category Filter Buttons
        document.querySelectorAll('#cat-filters .filter-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                document.querySelectorAll('#cat-filters .filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentCat = btn.dataset.cat;
                renderCards();
            }});
        }});

        // Status Filter Buttons
        document.querySelectorAll('#status-filters .filter-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                document.querySelectorAll('#status-filters .filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentStatus = btn.dataset.status;
                renderCards();
            }});
        }});

        // Search Input
        searchInput.addEventListener('input', (e) => {{
            searchQuery = e.target.value.trim();
            renderCards();
        }});

        // Interactive Violation Toggle (for testing demo)
        window.toggleViolation = function(id, type) {{
            const item = records.find(r => r.id === id);
            if (!item) return;

            if (item[type] === 'Violation') {{
                item[type] = 'Compliant';
            }} else if (item[type] === 'Compliant') {{
                item[type] = 'Violation';
            }}
            renderCards();
        }};

        // Modal Inspector
        window.openModal = function(id) {{
            const item = records.find(r => r.id === id);
            if (!item) return;

            document.getElementById('modal-title').textContent = `${{item.id.toUpperCase()}} (${{item.category}})`;
            document.getElementById('modal-subtitle').textContent = item.has_plate ? `Plate: ${{item.plate_number}}` : 'No License Plate Detected';
            document.getElementById('modal-veh-img').src = item.vehicle_img || '';
            document.getElementById('modal-plate-img').src = item.plate_img || '';
            document.getElementById('modal-plate-text').textContent = item.has_plate ? item.plate_number : 'NOT READ';
            document.getElementById('modal-conf').textContent = item.has_plate ? `${{item.plate_conf}}%` : '0.0%';
            document.getElementById('modal-cat').textContent = item.category;

            const violations = [];
            if (item.seatbelt === 'Violation') violations.push('No Seatbelt');
            if (item.helmet === 'Violation') violations.push('No Helmet');
            if (item.red_light === 'Violation') violations.push('Red Light');
            if (item.phone === 'Violation') violations.push('Phone Distraction');

            document.getElementById('modal-violation-summary').textContent = violations.length ? violations.join(', ') : 'None (Compliant)';

            modalOverlay.classList.add('open');
        }};

        modalClose.addEventListener('click', () => modalOverlay.classList.remove('open'));
        modalOverlay.addEventListener('click', (e) => {{
            if (e.target === modalOverlay) modalOverlay.classList.remove('open');
        }});

        // Initial Render
        renderCards();
    </script>
</body>
</html>
"""

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n🎉 Interactive Visualizer Dashboard created successfully!")
    print(f"📄 Dashboard Path: {HTML_OUTPUT}")
    return HTML_OUTPUT


def main():
    print("🚀 Generating Plate & Violation Visualizer Dashboard...")
    records = build_dashboard_data()

    if not records:
        print("⚠️ No records found to visualize. Run plate_reader.py first!")
        return

    html_path = generate_html(records)
    print(f"📊 Processed {len(records)} vehicles with plate and violation status.")
    print(f"🌐 Opening dashboard in browser: {html_path}")

    try:
        webbrowser.open(f"file:///{os.path.abspath(html_path)}")
    except Exception as e:
        print(f"ℹ️ Could not open browser automatically: {e}")
        print(f"🔗 You can open this file directly in any browser: {html_path}")


if __name__ == "__main__":
    main()
