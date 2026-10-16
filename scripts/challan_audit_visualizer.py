"""
e-Challan Preview & Model Quality Audit Generator
Generates an interactive, official-style e-Challan preview dashboard for all 
consolidated traffic violations with a standardized remark/audit system.

Allows auditing whether violations are:
  - 100% Genuine Violations
  - Cropping Overlap / Bleed Issues (neighboring vehicle/rider caught in crop)
  - Model False Positives (person was actually compliant)
  - Model Hallucinations (non-human object detected)
  - Occluded / Low Visibility (tint/glare)
  - Plate Mismatches / Unreadable
"""

import os
import sys
import csv
import json
import webbrowser
import pandas as pd

# Fix Windows console encoding for Unicode/emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../output"))
CONSOLIDATED_CSV = os.path.join(OUTPUT_DIR, "final_consolidated_violations.csv")
MASTER_CSV = os.path.join(OUTPUT_DIR, "master_violation_log.csv")
SEATBELT_DIR = os.path.join(OUTPUT_DIR, "seatbelt_results", "Cars")
HELMET_DIR = os.path.join(OUTPUT_DIR, "helmet_results", "Bikes_Motorcycles")
CROPPED_PHOTOS_DIR = os.path.join(OUTPUT_DIR, "cropped_photos")
HTML_OUTPUT = os.path.join(OUTPUT_DIR, "challan_audit_visualizer.html")


def get_relative_path(abs_path):
    """Convert absolute file path to a relative web path from output directory."""
    if not abs_path or pd.isna(abs_path):
        return ""
    abs_clean = str(abs_path).replace("\\", "/")
    out_clean = OUTPUT_DIR.replace("\\", "/")
    if abs_clean.startswith(out_clean):
        rel = abs_clean[len(out_clean):].lstrip("/")
        return rel
    # Fallback search inside output
    if "output/" in abs_clean:
        return abs_clean.split("output/", 1)[1]
    return abs_clean


def build_challan_dataset():
    """Build structured data for all consolidated violations."""
    if not os.path.exists(CONSOLIDATED_CSV):
        print(f"❌ Error: Consolidated violations CSV not found at: {CONSOLIDATED_CSV}")
        return []

    df = pd.read_csv(CONSOLIDATED_CSV)
    print(f"✅ Loaded {len(df)} consolidated violations from {CONSOLIDATED_CSV}")

    records = []

    for idx, row in df.iterrows():
        vid = str(row.get("Vehicle_ID", f"id_{idx}")).strip()
        category = str(row.get("Category", "Unknown")).strip()
        plate_no = str(row.get("Plate_Number", "Unknown")).strip()
        try:
            plate_conf = float(row.get("Plate_Confidence", 0.0))
        except Exception:
            plate_conf = 0.0

        try:
            fine_amount = int(row.get("Total_Fine_Amount", 2000))
        except Exception:
            fine_amount = 2000

        issue_date = str(row.get("Issue_Date", "2026-09-30")).strip()
        violation_type = str(row.get("Violation_1", "Traffic Violation")).strip()

        # Format challan number (e.g. ECH-SND-2026-004)
        clean_num = vid.replace("id_", "").replace("-", "")
        challan_no = f"ECH-SND-2026-{clean_num.zfill(4)}"

        # Primary vehicle image (BEST.jpg)
        car_img_abs = row.get("Car_Image", "")
        car_img_rel = get_relative_path(car_img_abs) if car_img_abs and os.path.exists(str(car_img_abs)) else ""

        # Plate crop image
        plate_img_abs = row.get("Plate_Image", "")
        plate_img_rel = get_relative_path(plate_img_abs) if plate_img_abs and os.path.exists(str(plate_img_abs)) else ""

        # Primary annotated violation image
        primary_viol_abs = row.get("Violation_Image_1", "")
        primary_viol_rel = get_relative_path(primary_viol_abs) if primary_viol_abs and os.path.exists(str(primary_viol_abs)) else ""

        # Find ALL candidate violation frames in vehicle's output folder
        all_viol_frames = []
        if primary_viol_abs and os.path.exists(os.path.dirname(str(primary_viol_abs))):
            viol_dir = os.path.dirname(str(primary_viol_abs))
            for f in sorted(os.listdir(viol_dir)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")) and "plate" not in f.lower() and "best" not in f.lower():
                    full_p = os.path.join(viol_dir, f)
                    rel_p = get_relative_path(full_p)
                    all_viol_frames.append({
                        "filename": f,
                        "rel_path": rel_p
                    })

        # Traffic section code
        if "seatbelt" in violation_type.lower():
            section_code = "Section 90-B MVO (1965) — Driving Without Fastened Seatbelt"
            law_desc = "Driver or front seat passenger failed to wear a mandatory safety seatbelt while vehicle was in motion."
            icon = "🚗"
        elif "helmet" in violation_type.lower():
            section_code = "Section 89-A MVO (1965) — Operating Two-Wheeler Without Helmet"
            law_desc = "Rider or pillion passenger operating motorcycle/two-wheeler without approved protective safety helmet."
            icon = "🏍️"
        else:
            section_code = "Section 112 MVO (1965) — General Traffic Violation"
            law_desc = "Vehicle operated in violation of provincial traffic regulations."
            icon = "⚠️"

        records.append({
            "challan_no": challan_no,
            "vehicle_id": vid,
            "category": category,
            "category_icon": icon,
            "plate_number": plate_no,
            "plate_confidence": round(plate_conf * 100, 1),
            "fine_amount": fine_amount,
            "issue_date": issue_date,
            "violation_type": violation_type,
            "section_code": section_code,
            "law_desc": law_desc,
            "car_image": car_img_rel,
            "plate_image": plate_img_rel,
            "primary_violation_image": primary_viol_rel,
            "all_violation_frames": all_viol_frames,
            "frame_count": len(all_viol_frames)
        })

    return records


def generate_html(records):
    """Generate the e-Challan Preview & Model Quality Audit HTML interface."""
    data_json = json.dumps(records)
    total_challans = len(records)
    car_count = sum(1 for r in records if r["category"] == "Cars")
    bike_count = sum(1 for r in records if r["category"] == "Bikes_Motorcycles")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sindh Police Traffic Warden Service — e-Challan Preview & Model Quality Audit</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@700;800;900&family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #080c14;
            --bg-secondary: #0f172a;
            --bg-card: rgba(15, 23, 42, 0.85);
            --bg-card-hover: rgba(30, 41, 59, 0.95);
            --border: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(99, 102, 241, 0.4);
            --accent-gold: #f59e0b;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
            --accent-blue: #38bdf8;
            --accent-indigo: #6366f1;
            --accent-purple: #a855f7;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            
            /* Verdict Tag Colors */
            --v-valid: #10b981;
            --v-crop: #f97316;
            --v-falsepos: #ef4444;
            --v-halluc: #a855f7;
            --v-occlude: #64748b;
            --v-plate: #eab308;
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
            max-width: 1480px;
            margin: 0 auto 2rem;
        }}

        .top-nav-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.8rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .brand-header {{
            display: flex;
            align-items: center;
            gap: 1.2rem;
        }}

        .police-crest {{
            width: 58px;
            height: 58px;
            border-radius: 14px;
            background: linear-gradient(135deg, #1e3a8a, #047857);
            border: 2px solid rgba(245, 158, 11, 0.6);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            box-shadow: 0 0 25px rgba(245, 158, 11, 0.35);
        }}

        .brand-titles h1 {{
            font-family: 'Cinzel', serif;
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            background: linear-gradient(to right, #ffffff, #fbbf24, #6ee7b7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .brand-titles .dept-tag {{
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
            font-weight: 600;
        }}

        .nav-actions {{
            display: flex;
            align-items: center;
            gap: 0.8rem;
            flex-wrap: wrap;
        }}

        .btn {{
            padding: 0.55rem 1.1rem;
            border-radius: 10px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s ease;
            text-decoration: none;
            border: 1px solid var(--border);
        }}

        .btn-export {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.3), rgba(5, 150, 105, 0.3));
            border-color: rgba(16, 185, 129, 0.5);
            color: #6ee7b7;
        }}
        .btn-export:hover {{
            background: rgba(16, 185, 129, 0.4);
            transform: translateY(-2px);
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
        }}

        .btn-link {{
            background: rgba(255, 255, 255, 0.04);
            color: var(--text-primary);
        }}
        .btn-link:hover {{
            background: rgba(99, 102, 241, 0.2);
            border-color: var(--accent-indigo);
            transform: translateY(-2px);
        }}

        /* Audit Summary & KPI Bar */
        .audit-dashboard {{
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.6rem;
            margin-bottom: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.4rem;
        }}

        .progress-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.8rem;
        }}

        .progress-title {{
            font-size: 1.1rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}

        .progress-counter {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 800;
            color: var(--accent-blue);
        }}

        .progress-bar-track {{
            width: 100%;
            height: 10px;
            background: rgba(2, 6, 23, 0.6);
            border-radius: 9999px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}

        .progress-bar-fill {{
            height: 100%;
            width: 0%;
            background: linear-gradient(to right, var(--accent-indigo), var(--accent-emerald));
            border-radius: 9999px;
            transition: width 0.4s ease;
        }}

        .verdict-pills-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
        }}

        .verdict-stat-pill {{
            background: rgba(2, 6, 23, 0.5);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
            position: relative;
            overflow: hidden;
        }}

        .verdict-stat-pill::before {{
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background: var(--pill-color, var(--text-muted));
        }}

        .pill-label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            font-weight: 700;
        }}

        .pill-count {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.45rem;
            font-weight: 800;
            color: var(--text-primary);
        }}

        .pill-pct {{
            font-size: 0.74rem;
            color: var(--text-secondary);
        }}

        /* Automated Decision Card */
        .decision-banner {{
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 14px;
            padding: 1rem 1.4rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .decision-icon {{
            font-size: 1.8rem;
        }}

        .decision-text h4 {{
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--accent-gold);
            margin-bottom: 0.2rem;
        }}

        .decision-text p {{
            font-size: 0.84rem;
            color: var(--text-secondary);
        }}

        /* Filter Controls */
        .filter-controls-card {{
            background: var(--bg-card);
            backdrop-filter: blur(14px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 2rem;
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            align-items: center;
            justify-content: space-between;
        }}

        .search-box {{
            flex: 1;
            min-width: 280px;
            position: relative;
        }}

        .search-box input {{
            width: 100%;
            background: rgba(2, 6, 23, 0.6);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.65rem 1rem 0.65rem 2.6rem;
            color: var(--text-primary);
            font-size: 0.88rem;
            outline: none;
            transition: border-color 0.2s;
        }}

        .search-box input:focus {{
            border-color: var(--accent-indigo);
        }}

        .search-icon {{
            position: absolute;
            left: 0.9rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
        }}

        .filters-group {{
            display: flex;
            gap: 0.8rem;
            flex-wrap: wrap;
        }}

        .select-input {{
            background: rgba(2, 6, 23, 0.6);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.65rem 1rem;
            color: var(--text-primary);
            font-size: 0.85rem;
            font-weight: 500;
            outline: none;
            cursor: pointer;
        }}

        /* Challan Cards Grid */
        .challans-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
            gap: 1.6rem;
            margin-bottom: 3rem;
        }}

        @media (max-width: 600px) {{
            .challans-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        /* e-Challan Card Styling */
        .challan-card {{
            background: var(--bg-card);
            backdrop-filter: blur(14px);
            border: 1px solid var(--border);
            border-radius: 18px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            position: relative;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }}

        .challan-card:hover {{
            border-color: var(--border-highlight);
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.5);
        }}

        .challan-card.reviewed {{
            border-left: 5px solid var(--card-verdict-color, var(--accent-emerald));
        }}

        /* Official Challan Header Slip */
        .challan-ticket-header {{
            background: linear-gradient(135deg, #0f172a, #1e293b);
            border-bottom: 2px dashed rgba(255, 255, 255, 0.12);
            padding: 1.1rem 1.4rem;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}

        .ticket-left {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        .ticket-emblem {{
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--accent-gold);
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}

        .ticket-no {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.15rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: 0.02em;
        }}

        .ticket-date {{
            font-size: 0.75rem;
            color: var(--text-muted);
        }}

        .ticket-right {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 0.35rem;
        }}

        .fine-badge {{
            background: rgba(244, 63, 94, 0.2);
            border: 1px solid rgba(244, 63, 94, 0.4);
            color: #fb7185;
            padding: 0.35rem 0.75rem;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 800;
            font-size: 0.95rem;
        }}

        .citation-status-pill {{
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid rgba(245, 158, 11, 0.3);
            color: var(--accent-gold);
        }}

        /* Visual Evidence Split Container */
        .evidence-container {{
            padding: 1.25rem;
            display: grid;
            grid-template-columns: 1.6fr 1fr;
            gap: 1rem;
            background: rgba(2, 6, 23, 0.35);
        }}

        .primary-evidence-box {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .evidence-img-wrap {{
            width: 100%;
            height: 220px;
            background: #020617;
            border-radius: 12px;
            border: 1px solid var(--border);
            overflow: hidden;
            position: relative;
            cursor: pointer;
        }}

        .evidence-img-wrap img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s ease;
        }}

        .evidence-img-wrap:hover img {{
            transform: scale(1.05);
        }}

        .evidence-badge {{
            position: absolute;
            top: 0.6rem;
            left: 0.6rem;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 0.25rem 0.55rem;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 700;
            color: #fb7185;
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}

        .secondary-evidence-box {{
            display: flex;
            flex-direction: column;
            gap: 0.7rem;
        }}

        .side-img-card {{
            flex: 1;
            background: #020617;
            border-radius: 10px;
            border: 1px solid var(--border);
            overflow: hidden;
            position: relative;
            cursor: pointer;
            min-height: 100px;
        }}

        .side-img-card img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s;
        }}

        .side-img-card:hover img {{
            transform: scale(1.06);
        }}

        .side-label {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 0.25rem 0.5rem;
            background: linear-gradient(to top, rgba(0,0,0,0.85), transparent);
            font-size: 0.68rem;
            font-weight: 600;
            color: var(--text-secondary);
        }}

        /* Vehicle Identity Bar (Plate & OCR) */
        .identity-bar {{
            padding: 0.9rem 1.4rem;
            background: rgba(15, 23, 42, 0.6);
            border-top: 1px solid var(--border);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .vehicle-type-info {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}

        .type-icon {{
            font-size: 1.4rem;
        }}

        .type-details {{
            display: flex;
            flex-direction: column;
        }}

        .type-name {{
            font-size: 0.88rem;
            font-weight: 700;
        }}

        .vehicle-id-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-muted);
        }}

        /* Pakistani Registration Plate Display */
        .pak-plate-box {{
            background: #ffffff;
            color: #047857;
            border: 2px solid #065f46;
            border-radius: 8px;
            padding: 0.3rem 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 800;
            font-size: 0.95rem;
            letter-spacing: 0.08em;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }}

        .pak-plate-flag {{
            font-size: 0.85rem;
        }}

        .plate-ocr-pct {{
            font-size: 0.65rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }}

        /* Offense Citation Box */
        .offense-box {{
            padding: 0.9rem 1.4rem;
            background: rgba(244, 63, 94, 0.06);
            border-bottom: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        .offense-code {{
            font-size: 0.78rem;
            font-weight: 700;
            color: #f43f5e;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}

        .offense-desc {{
            font-size: 0.82rem;
            color: var(--text-secondary);
        }}

        /* Alternative Multi-Frame Evidence Strip */
        .frames-strip {{
            padding: 0.6rem 1.4rem;
            background: rgba(2, 6, 23, 0.5);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 0.6rem;
            overflow-x: auto;
        }}

        .strip-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            white-space: nowrap;
            font-weight: 700;
        }}

        .thumb-btn {{
            width: 44px;
            height: 44px;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid var(--border);
            cursor: pointer;
            flex-shrink: 0;
            background: #000;
            padding: 0;
            transition: all 0.2s;
        }}

        .thumb-btn img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}

        .thumb-btn:hover, .thumb-btn.active {{
            border-color: var(--accent-blue);
            transform: scale(1.1);
        }}

        /* Standardized Remarks & Audit Hub */
        .audit-hub {{
            padding: 1.25rem 1.4rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            background: rgba(15, 23, 42, 0.4);
            margin-top: auto;
        }}

        .audit-hub-title {{
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .saved-status-tag {{
            font-size: 0.72rem;
            color: var(--accent-emerald);
            font-weight: 600;
            opacity: 0;
            transition: opacity 0.3s;
        }}

        .saved-status-tag.show {{
            opacity: 1;
        }}

        /* 6 Standard Verdict Chips */
        .verdict-chips-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem;
        }}

        .verdict-chip {{
            padding: 0.55rem 0.75rem;
            border-radius: 8px;
            font-size: 0.76rem;
            font-weight: 700;
            cursor: pointer;
            border: 1px solid var(--border);
            background: rgba(2, 6, 23, 0.5);
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.45rem;
            transition: all 0.2s ease;
            text-align: left;
        }}

        .verdict-chip:hover {{
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-primary);
        }}

        .verdict-chip.selected[data-verdict="VALID_VIOLATION"] {{
            background: rgba(16, 185, 129, 0.2);
            border-color: var(--v-valid);
            color: #6ee7b7;
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.3);
        }}

        .verdict-chip.selected[data-verdict="CROP_BLEED"] {{
            background: rgba(249, 115, 22, 0.2);
            border-color: var(--v-crop);
            color: #fdba74;
            box-shadow: 0 0 12px rgba(249, 115, 22, 0.3);
        }}

        .verdict-chip.selected[data-verdict="FALSE_POSITIVE"] {{
            background: rgba(239, 68, 68, 0.2);
            border-color: var(--v-falsepos);
            color: #fca5a5;
            box-shadow: 0 0 12px rgba(239, 68, 68, 0.3);
        }}

        .verdict-chip.selected[data-verdict="HALLUCINATION"] {{
            background: rgba(168, 85, 247, 0.2);
            border-color: var(--v-halluc);
            color: #d8b4fe;
            box-shadow: 0 0 12px rgba(168, 85, 247, 0.3);
        }}

        .verdict-chip.selected[data-verdict="OCCLUDED"] {{
            background: rgba(100, 116, 139, 0.2);
            border-color: var(--v-occlude);
            color: #cbd5e1;
            box-shadow: 0 0 12px rgba(100, 116, 139, 0.3);
        }}

        .verdict-chip.selected[data-verdict="PLATE_ISSUE"] {{
            background: rgba(234, 179, 8, 0.2);
            border-color: var(--v-plate);
            color: #fde047;
            box-shadow: 0 0 12px rgba(234, 179, 8, 0.3);
        }}

        /* Remarks Textarea */
        .remarks-input {{
            width: 100%;
            height: 64px;
            background: rgba(2, 6, 23, 0.7);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
            color: var(--text-primary);
            font-size: 0.82rem;
            font-family: inherit;
            resize: vertical;
            outline: none;
            transition: border-color 0.2s;
        }}

        .remarks-input:focus {{
            border-color: var(--accent-indigo);
        }}

        /* Standard Guide Modal */
        .guide-modal {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(2, 6, 23, 0.85);
            backdrop-filter: blur(12px);
            z-index: 2000;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }}

        .guide-modal.open {{
            display: flex;
        }}

        .guide-content {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-highlight);
            border-radius: 20px;
            max-width: 850px;
            width: 100%;
            max-height: 90vh;
            overflow-y: auto;
            padding: 2rem;
            box-shadow: 0 25px 60px rgba(0, 0, 0, 0.7);
        }}

        .guide-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1rem;
        }}

        .guide-rule-card {{
            background: rgba(2, 6, 23, 0.6);
            border-radius: 12px;
            padding: 1.1rem;
            margin-bottom: 1rem;
            border-left: 4px solid var(--rule-color);
        }}

        .guide-rule-title {{
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--rule-color);
            margin-bottom: 0.3rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .guide-rule-body {{
            font-size: 0.84rem;
            color: var(--text-secondary);
            line-height: 1.45;
        }}

        .guide-rule-action {{
            margin-top: 0.4rem;
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-primary);
        }}

        /* Lightbox Image Zoom */
        .lightbox-modal {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.92);
            backdrop-filter: blur(12px);
            z-index: 3000;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }}

        .lightbox-modal.open {{
            display: flex;
        }}

        .lightbox-img {{
            max-width: 90vw;
            max-height: 85vh;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.8);
        }}

        .lightbox-close {{
            position: absolute;
            top: 1.5rem;
            right: 2rem;
            background: rgba(255, 255, 255, 0.1);
            border: none;
            color: white;
            font-size: 2rem;
            width: 48px;
            height: 48px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
    </style>
</head>
<body>

    <header>
        <div class="top-nav-bar">
            <div class="brand-header">
                <div class="police-crest">🇵🇰</div>
                <div class="brand-titles">
                    <h1>Sindh Police Traffic Warden Service</h1>
                    <div class="dept-tag">Automated e-Challan Citation & Model Quality Audit System</div>
                </div>
            </div>
            <div class="nav-actions">
                <button class="btn btn-link" onclick="openGuideModal()">
                    <span>📖 Review Standard Guide</span>
                </button>
                <button class="btn btn-export" onclick="exportAuditCSV()">
                    <span>💾 Export Audit & Remarks (CSV)</span>
                </button>
                <a href="model_audit_visualizer.html" class="btn btn-link">
                    <span>🔬 Model Audit Visualizer</span>
                </a>
                <a href="plate_visualizer.html" class="btn btn-link">
                    <span>📋 Plate Dashboard</span>
                </a>
            </div>
        </div>

        <!-- Audit Summary & KPI Dashboard -->
        <div class="audit-dashboard">
            <div class="progress-header">
                <div class="progress-title">
                    <span>Audit Review Progress:</span>
                    <span class="progress-counter" id="progressCount">0 / {total_challans} Reviewed</span>
                    <span id="progressPctText" style="color:var(--text-muted); font-size:0.85rem;">(0%)</span>
                </div>
                <div style="font-size:0.82rem; color:var(--text-muted);">
                    Total Violations: <strong>{total_challans}</strong> ({car_count} Cars • {bike_count} Motorcycles)
                </div>
            </div>

            <div class="progress-bar-track">
                <div class="progress-bar-fill" id="progressBarFill"></div>
            </div>

            <!-- 6 Standard Verdict Counters -->
            <div class="verdict-pills-grid">
                <div class="verdict-stat-pill" style="--pill-color: var(--v-valid);">
                    <div class="pill-label">Valid Violations</div>
                    <div class="pill-count" id="count-VALID_VIOLATION">0</div>
                    <div class="pill-pct" id="pct-VALID_VIOLATION">0% • Model Correct</div>
                </div>

                <div class="verdict-stat-pill" style="--pill-color: var(--v-crop);">
                    <div class="pill-label">Cropping Bleed</div>
                    <div class="pill-count" id="count-CROP_BLEED">0</div>
                    <div class="pill-pct" id="pct-CROP_BLEED">0% • Overlap Issue</div>
                </div>

                <div class="verdict-stat-pill" style="--pill-color: var(--v-falsepos);">
                    <div class="pill-label">Model False Positive</div>
                    <div class="pill-count" id="count-FALSE_POSITIVE">0</div>
                    <div class="pill-pct" id="pct-FALSE_POSITIVE">0% • Compliant Driver/Rider</div>
                </div>

                <div class="verdict-stat-pill" style="--pill-color: var(--v-halluc);">
                    <div class="pill-label">Model Hallucination</div>
                    <div class="pill-count" id="count-HALLUCINATION">0</div>
                    <div class="pill-pct" id="pct-HALLUCINATION">0% • Non-Human Object</div>
                </div>

                <div class="verdict-stat-pill" style="--pill-color: var(--v-occlude);">
                    <div class="pill-label">Occluded / Glare</div>
                    <div class="pill-count" id="count-OCCLUDED">0</div>
                    <div class="pill-pct" id="pct-OCCLUDED">0% • Unverifiable</div>
                </div>

                <div class="verdict-stat-pill" style="--pill-color: var(--v-plate);">
                    <div class="pill-label">Plate Issue</div>
                    <div class="pill-count" id="count-PLATE_ISSUE">0</div>
                    <div class="pill-pct" id="pct-PLATE_ISSUE">0% • Plate Missing/Wrong</div>
                </div>
            </div>

            <!-- Dynamic Decision Banner -->
            <div class="decision-banner" id="decisionBanner">
                <div class="decision-icon" id="decisionIcon">💡</div>
                <div class="decision-text">
                    <h4 id="decisionTitle">Audit Recommendation Engine Active</h4>
                    <p id="decisionDesc">Tag violations with your remarks below. As you review challans, this system will automatically diagnose whether you need new models or if current models are sufficient.</p>
                </div>
            </div>
        </div>

        <!-- Filter Controls -->
        <div class="filter-controls-card">
            <div class="search-box">
                <span class="search-icon">🔍</span>
                <input type="text" id="searchInput" placeholder="Search Challan No (e.g. ECH-SND-2026-0004), Vehicle ID, or Plate..." oninput="handleFilterChange()">
            </div>

            <div class="filters-group">
                <select class="select-input" id="verdictFilter" onchange="handleFilterChange()">
                    <option value="all">Audit Status: All (101)</option>
                    <option value="unreviewed">⏳ Unreviewed Only</option>
                    <option value="reviewed">✅ Reviewed Only</option>
                    <option value="VALID_VIOLATION">🟢 Valid Violations</option>
                    <option value="CROP_BLEED">🟠 Cropping Overlap / Bleed</option>
                    <option value="FALSE_POSITIVE">🔴 Model False Positives</option>
                    <option value="HALLUCINATION">🟣 Model Hallucinations</option>
                    <option value="OCCLUDED">⚪ Occluded / Low Visibility</option>
                    <option value="PLATE_ISSUE">🟡 Plate Issues</option>
                </select>

                <select class="select-input" id="categoryFilter" onchange="handleFilterChange()">
                    <option value="all">Category: All Vehicles</option>
                    <option value="Cars">🚗 Cars Only (21)</option>
                    <option value="Bikes_Motorcycles">🏍️ Motorcycles Only (80)</option>
                </select>

                <select class="select-input" id="violationFilter" onchange="handleFilterChange()">
                    <option value="all">Violation: All Types</option>
                    <option value="seatbelt">No Seatbelt</option>
                    <option value="helmet">No Helmet</option>
                </select>
            </div>
        </div>
    </header>

    <!-- Main Challan Cards Grid -->
    <main>
        <div class="challans-grid" id="challansGrid">
            <!-- Dynamically populated -->
        </div>
    </main>

    <!-- Review Standard Guide Modal -->
    <div class="guide-modal" id="guideModal" onclick="closeGuideModal(event)">
        <div class="guide-content">
            <div class="guide-header">
                <h2>Standard Review Classification Criteria</h2>
                <button style="background:none; border:none; color:var(--text-secondary); font-size:1.5rem; cursor:pointer;" onclick="closeGuideModal(null, true)">✕</button>
            </div>

            <div class="guide-rule-card" style="--rule-color: var(--v-valid);">
                <div class="guide-rule-title">🟢 VALID_VIOLATION — 100% Genuine Violation</div>
                <div class="guide-rule-body">
                    Vehicle crop belongs solely to the target vehicle. The driver is visibly unbelted, or the motorcyclist is visibly without a helmet. Bounding box is placed accurately on the person.
                </div>
                <div class="guide-rule-action">🎯 Model Diagnosis: Model is working as intended. No changes required.</div>
            </div>

            <div class="guide-rule-card" style="--rule-color: var(--v-crop);">
                <div class="guide-rule-title">🟠 CROP_BLEED — Cropping Overlap / Neighboring Entity Bleed</div>
                <div class="guide-rule-body">
                    A car crop captured a motorcycle rider passing right by the car door, causing the seatbelt model to flag the rider; or two motorcycles overlapping closely caused one rider's helmet status to bleed into the other. The model actually detected a human correctly, but the cropping pipeline enclosed outside entities.
                </div>
                <div class="guide-rule-action">🎯 Model Diagnosis: <strong>NO model retraining needed!</strong> Fix this in the pipeline via Windshield/Cabin ROI focus and adjacent-vehicle IoU exclusion.</div>
            </div>

            <div class="guide-rule-card" style="--rule-color: var(--v-falsepos);">
                <div class="guide-rule-title">🔴 FALSE_POSITIVE — Person Compliant (Model Error)</div>
                <div class="guide-rule-body">
                    A person is clearly present and IS wearing a seatbelt or helmet, but the model erroneously classified them as "person-noseatbelt" or "without_helmet" (e.g. black seatbelt against dark shirt, or non-standard helmet shell).
                </div>
                <div class="guide-rule-action">🎯 Model Diagnosis: <strong>Retraining needed.</strong> Needs fine-tuning on local seatbelt/clothing variations.</div>
            </div>

            <div class="guide-rule-card" style="--rule-color: var(--v-halluc);">
                <div class="guide-rule-title">🟣 HALLUCINATION — Model Hallucination / Non-Human Object</div>
                <div class="guide-rule-body">
                    Model drew a bounding box on an empty headrest, car tire, steering wheel, tree branch, or background shadow where no person exists.
                </div>
                <div class="guide-rule-action">🎯 Model Diagnosis: <strong>Confidence threshold adjustment</strong> or negative-sample retraining needed.</div>
            </div>

            <div class="guide-rule-card" style="--rule-color: var(--v-occlude);">
                <div class="guide-rule-title">⚪ OCCLUDED — Glare / Heavy Tint / Unverifiable</div>
                <div class="guide-rule-body">
                    Severe windshield reflection, heavy black window tint, or motion blur prevents a human reviewer from seeing whether a seatbelt/helmet is worn.
                </div>
                <div class="guide-rule-action">🎯 Pipeline Diagnosis: Camera requires circular polarizing lens (CPL) filter or steeper downward mounting angle.</div>
            </div>

            <div class="guide-rule-card" style="--rule-color: var(--v-plate);">
                <div class="guide-rule-title">🟡 PLATE_ISSUE — Violation Real, Plate Unreadable/Wrong</div>
                <div class="guide-rule-body">
                    The seatbelt or helmet violation is real, but the license plate was completely missed or read incorrectly by OCR (e.g. front plate missing on bike, or OCR misread characters).
                </div>
                <div class="guide-rule-action">🎯 Pipeline Diagnosis: License plate detector or PaddleOCR needs attention, not the violation model.</div>
            </div>
        </div>
    </div>

    <!-- Lightbox Zoom Modal -->
    <div class="lightbox-modal" id="lightboxModal" onclick="closeLightbox()">
        <button class="lightbox-close">✕</button>
        <img class="lightbox-img" id="lightboxImg" src="" alt="Enlarged evidence">
    </div>

    <script>
        const CHALLANS = {data_json};
        const STORAGE_KEY = 'trafficguard_challan_audits_v1';
        let userAudits = {{}}; // {{ challan_no: {{ verdict: '', remarks: '', timestamp: '' }} }}

        function init() {{
            loadSavedAudits();
            renderChallans();
            updateAuditSummary();
        }}

        function loadSavedAudits() {{
            try {{
                const saved = localStorage.getItem(STORAGE_KEY);
                if (saved) {{
                    userAudits = JSON.parse(saved);
                }}
            }} catch (e) {{
                console.error("Could not load from localStorage:", e);
            }}
        }}

        function saveAudits() {{
            try {{
                localStorage.setItem(STORAGE_KEY, JSON.stringify(userAudits));
                updateAuditSummary();
            }} catch (e) {{
                console.error("Could not save to localStorage:", e);
            }}
        }}

        function setVerdict(challanNo, verdict) {{
            if (!userAudits[challanNo]) {{
                userAudits[challanNo] = {{ verdict: '', remarks: '', timestamp: '' }};
            }}

            // Toggle if clicked again
            if (userAudits[challanNo].verdict === verdict) {{
                userAudits[challanNo].verdict = '';
            }} else {{
                userAudits[challanNo].verdict = verdict;
            }}
            userAudits[challanNo].timestamp = new Date().toISOString();

            saveAudits();
            updateCardUI(challanNo);
            flashSaved(challanNo);
        }}

        function handleRemarksInput(challanNo, val) {{
            if (!userAudits[challanNo]) {{
                userAudits[challanNo] = {{ verdict: '', remarks: '', timestamp: '' }};
            }}
            userAudits[challanNo].remarks = val;
            userAudits[challanNo].timestamp = new Date().toISOString();
            saveAudits();
            flashSaved(challanNo);
        }}

        function flashSaved(challanNo) {{
            const tag = document.getElementById(`saved-${{challanNo}}`);
            if (tag) {{
                tag.classList.add('show');
                setTimeout(() => tag.classList.remove('show'), 1500);
            }}
        }}

        function updateCardUI(challanNo) {{
            const audit = userAudits[challanNo] || {{}};
            const currentVerdict = audit.verdict || '';

            // Update chips
            document.querySelectorAll(`.chip-${{challanNo}}`).forEach(chip => {{
                if (chip.getAttribute('data-verdict') === currentVerdict) {{
                    chip.classList.add('selected');
                }} else {{
                    chip.classList.remove('selected');
                }}
            }});

            // Update card border & status
            const card = document.getElementById(`card-${{challanNo}}`);
            const statusPill = document.getElementById(`status-pill-${{challanNo}}`);
            if (card) {{
                if (currentVerdict) {{
                    card.classList.add('reviewed');
                    const colorMap = {{
                        'VALID_VIOLATION': 'var(--v-valid)',
                        'CROP_BLEED': 'var(--v-crop)',
                        'FALSE_POSITIVE': 'var(--v-falsepos)',
                        'HALLUCINATION': 'var(--v-halluc)',
                        'OCCLUDED': 'var(--v-occlude)',
                        'PLATE_ISSUE': 'var(--v-plate)'
                    }};
                    card.style.setProperty('--card-verdict-color', colorMap[currentVerdict] || 'var(--accent-emerald)');
                    
                    if (statusPill) {{
                        if (currentVerdict === 'VALID_VIOLATION') {{
                            statusPill.textContent = 'CONFIRMED CITATION';
                            statusPill.style.background = 'rgba(16, 185, 129, 0.2)';
                            statusPill.style.color = '#6ee7b7';
                            statusPill.style.borderColor = 'rgba(16, 185, 129, 0.4)';
                        }} else {{
                            statusPill.textContent = 'VOID / DISMISSED';
                            statusPill.style.background = 'rgba(239, 68, 68, 0.2)';
                            statusPill.style.color = '#fca5a5';
                            statusPill.style.borderColor = 'rgba(239, 68, 68, 0.4)';
                        }}
                    }}
                }} else {{
                    card.classList.remove('reviewed');
                    if (statusPill) {{
                        statusPill.textContent = 'PENDING REVIEW';
                        statusPill.style.background = 'rgba(245, 158, 11, 0.15)';
                        statusPill.style.color = 'var(--accent-gold)';
                        statusPill.style.borderColor = 'rgba(245, 158, 11, 0.3)';
                    }}
                }}
            }}
        }}

        function updateAuditSummary() {{
            const total = CHALLANS.length;
            const counts = {{
                VALID_VIOLATION: 0,
                CROP_BLEED: 0,
                FALSE_POSITIVE: 0,
                HALLUCINATION: 0,
                OCCLUDED: 0,
                PLATE_ISSUE: 0
            }};

            let reviewedCount = 0;
            CHALLANS.forEach(c => {{
                const a = userAudits[c.challan_no];
                if (a && a.verdict) {{
                    reviewedCount++;
                    if (counts[a.verdict] !== undefined) {{
                        counts[a.verdict]++;
                    }}
                }}
            }});

            const pct = total > 0 ? Math.round((reviewedCount / total) * 100) : 0;
            document.getElementById('progressCount').textContent = `${{reviewedCount}} / ${{total}} Reviewed`;
            document.getElementById('progressPctText').textContent = `(${{pct}}%)`;
            document.getElementById('progressBarFill').style.width = `${{pct}}%`;

            // Update verdict counters
            for (const [key, count] of Object.entries(counts)) {{
                const countElem = document.getElementById(`count-${{key}}`);
                const pctElem = document.getElementById(`pct-${{key}}`);
                if (countElem) countElem.textContent = count;
                if (pctElem) {{
                    const subPct = reviewedCount > 0 ? Math.round((count / reviewedCount) * 100) : 0;
                    pctElem.textContent = `${{subPct}}% of reviewed`;
                }}
            }}

            // Dynamic Decision Engine
            const iconElem = document.getElementById('decisionIcon');
            const titleElem = document.getElementById('decisionTitle');
            const descElem = document.getElementById('decisionDesc');

            if (reviewedCount < 10) {{
                iconElem.textContent = '💡';
                titleElem.textContent = 'Audit Recommendation Engine Gathering Data';
                descElem.textContent = `You have reviewed ${{reviewedCount}} of ${{total}} challans. Review at least 10–15 challans to trigger automatic data/model diagnostics.`;
            }} else {{
                const cropPct = (counts.CROP_BLEED / reviewedCount) * 100;
                const falsePosPct = (counts.FALSE_POSITIVE / reviewedCount) * 100;
                const validPct = (counts.VALID_VIOLATION / reviewedCount) * 100;
                const hallucPct = (counts.HALLUCINATION / reviewedCount) * 100;

                if (cropPct >= 30) {{
                    iconElem.textContent = '⚡';
                    titleElem.textContent = `Diagnosis: Pipeline Cropping Bleed is the Primary Issue (${{Math.round(cropPct)}}%)`;
                    titleElem.style.color = 'var(--v-crop)';
                    descElem.innerHTML = `<strong>You do NOT need to retrain the YOLO detection models.</strong> The models are correctly detecting humans without helmets/seatbelts, but dense vehicle overlap is causing outside riders to be captured in the vehicle bounding box. <br>👉 <em>Solution: Add Windshield/Cabin ROI slicing for cars and adjacent-bike IoU exclusion.</em>`;
                }} else if (falsePosPct >= 25) {{
                    iconElem.textContent = '⚠️';
                    titleElem.textContent = `Diagnosis: Model False Positives Detected (${{Math.round(falsePosPct)}}%)`;
                    titleElem.style.color = 'var(--v-falsepos)';
                    descElem.innerHTML = `<strong>Model retraining is recommended.</strong> The model is struggling to recognize compliant drivers/riders (e.g. seatbelts blending into clothing or unique helmet designs). <br>👉 <em>Solution: Annotate 150–200 local vehicle crops and fine-tune Models/seatbelt.pt or helmet.pt.</em>`;
                }} else if (validPct >= 70) {{
                    iconElem.textContent = '🎉';
                    titleElem.textContent = `Diagnosis: Models Are Production-Ready (${{Math.round(validPct)}}% Valid)`;
                    titleElem.style.color = 'var(--v-valid)';
                    descElem.innerHTML = `<strong>Current models are performing strongly.</strong> Over 70% of issued challans are verified genuine violations. No retraining is required at this time.`;
                }} else if (hallucPct >= 20) {{
                    iconElem.textContent = '🟣';
                    titleElem.textContent = `Diagnosis: Model Hallucinations on Inanimate Objects (${{Math.round(hallucPct)}}%)`;
                    titleElem.style.color = 'var(--v-halluc)';
                    descElem.innerHTML = `The model is triggering on car headrests, luggage, or shadows. <br>👉 <em>Solution: Raise confidence threshold from 0.35 to 0.50 before considering retraining.</em>`;
                }} else {{
                    iconElem.textContent = '📊';
                    titleElem.textContent = `Audit Progress: ${{reviewedCount}} Challans Analyzed`;
                    titleElem.style.color = 'var(--accent-blue)';
                    descElem.textContent = `Valid: ${{counts.VALID_VIOLATION}} | Cropping Bleed: ${{counts.CROP_BLEED}} | False Positives: ${{counts.FALSE_POSITIVE}} | Plate Issues: ${{counts.PLATE_ISSUE}}.`;
                }}
            }}
        }}

        function handleFilterChange() {{
            renderChallans();
        }}

        function renderChallans() {{
            const grid = document.getElementById('challansGrid');
            grid.innerHTML = '';

            const search = document.getElementById('searchInput').value.trim().toLowerCase();
            const verdictFilter = document.getElementById('verdictFilter').value;
            const catFilter = document.getElementById('categoryFilter').value;
            const violFilter = document.getElementById('violationFilter').value;

            const filtered = CHALLANS.filter(c => {{
                // Search filter
                if (search) {{
                    const matchChallan = c.challan_no.toLowerCase().includes(search);
                    const matchVid = c.vehicle_id.toLowerCase().includes(search);
                    const matchPlate = c.plate_number.toLowerCase().includes(search);
                    if (!matchChallan && !matchVid && !matchPlate) return false;
                }}

                // Category filter
                if (catFilter !== 'all' && c.category !== catFilter) return false;

                // Violation filter
                if (violFilter !== 'all') {{
                    if (violFilter === 'seatbelt' && !c.violation_type.toLowerCase().includes('seatbelt')) return false;
                    if (violFilter === 'helmet' && !c.violation_type.toLowerCase().includes('helmet')) return false;
                }}

                // Verdict filter
                const audit = userAudits[c.challan_no] || {{}};
                const currentVerdict = audit.verdict || '';

                if (verdictFilter === 'unreviewed' && currentVerdict !== '') return false;
                if (verdictFilter === 'reviewed' && currentVerdict === '') return false;
                if (verdictFilter !== 'all' && verdictFilter !== 'unreviewed' && verdictFilter !== 'reviewed') {{
                    if (currentVerdict !== verdictFilter) return false;
                }}

                return true;
            }});

            if (filtered.length === 0) {{
                grid.innerHTML = `
                    <div style="grid-column: 1 / -1; text-align: center; padding: 4rem 2rem; background: var(--bg-card); border-radius: 16px; border: 1px dashed var(--border); color: var(--text-muted);">
                        <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">🔍</div>
                        <h3>No matching e-Challan records found</h3>
                        <p style="font-size:0.85rem; margin-top:0.4rem;">Try adjusting your filter selection or search query.</p>
                    </div>
                `;
                return;
            }}

            filtered.forEach(c => {{
                const audit = userAudits[c.challan_no] || {{ verdict: '', remarks: '' }};
                const card = document.createElement('div');
                card.className = `challan-card ${{audit.verdict ? 'reviewed' : ''}}`;
                card.id = `card-${{c.challan_no}}`;

                const plateDisplay = c.plate_number !== 'Unknown' 
                    ? `<div class="pak-plate-box">
                         <span class="pak-plate-flag">🇵🇰</span>
                         <span>${{c.plate_number}}</span>
                       </div>`
                    : `<div style="font-size:0.75rem; color:var(--text-muted); font-style:italic;">No Plate Detected</div>`;

                let thumbsHtml = '';
                if (c.all_violation_frames && c.all_violation_frames.length > 1) {{
                    c.all_violation_frames.forEach((f, idx) => {{
                        thumbsHtml += `
                            <button class="thumb-btn ${{idx === 0 ? 'active' : ''}}" onclick="switchFrame('${{c.challan_no}}', '${{f.rel_path}}', this)" title="${{f.filename}}">
                                <img src="${{f.rel_path}}" alt="frame ${{idx+1}}">
                            </button>
                        `;
                    }});
                }}

                card.innerHTML = `
                    <!-- Header Slip -->
                    <div class="challan-ticket-header">
                        <div class="ticket-left">
                            <div class="ticket-emblem">
                                <span>⚖️ Sindh Police</span>
                                <span>•</span>
                                <span>e-Challan</span>
                            </div>
                            <div class="ticket-no">${{c.challan_no}}</div>
                            <div class="ticket-date">Issued: ${{c.issue_date}} • Zone Surveillance Grid</div>
                        </div>
                        <div class="ticket-right">
                            <div class="fine-badge">PKR ${{c.fine_amount.toLocaleString()}}</div>
                            <span class="citation-status-pill" id="status-pill-${{c.challan_no}}">
                                ${{audit.verdict === 'VALID_VIOLATION' ? 'CONFIRMED CITATION' : (audit.verdict ? 'VOID / DISMISSED' : 'PENDING REVIEW')}}
                            </span>
                        </div>
                    </div>

                    <!-- Visual Evidence Box -->
                    <div class="evidence-container">
                        <div class="primary-evidence-box">
                            <div class="evidence-img-wrap" onclick="openLightbox('${{c.primary_violation_image || c.car_image}}')">
                                <img id="main-evidence-${{c.challan_no}}" src="${{c.primary_violation_image || c.car_image}}" alt="Violation evidence" loading="lazy" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'300\\' height=\\'200\\'><rect width=\\'300\\' height=\\'200\\' fill=\\'%231e293b\\'/><text x=\\'50%\\' y=\\'50%\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\' fill=\\'%2364748b\\'>Image not found</text></svg>'">
                                <span class="evidence-badge">🚨 Model Bounding Box Crop</span>
                            </div>
                        </div>
                        <div class="secondary-evidence-box">
                            <div class="side-img-card" onclick="openLightbox('${{c.car_image}}')">
                                <img src="${{c.car_image}}" alt="Vehicle BEST crop" loading="lazy">
                                <span class="side-label">Vehicle Context</span>
                            </div>
                            <div class="side-img-card" onclick="openLightbox('${{c.plate_image || c.car_image}}')">
                                <img src="${{c.plate_image || c.car_image}}" alt="Plate crop" loading="lazy" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'150\\' height=\\'80\\'><rect width=\\'150\\' height=\\'80\\' fill=\\'%230f172a\\'/><text x=\\'50%\\' y=\\'50%\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\' fill=\\'%23475569\\'>No Plate</text></svg>'">
                                <span class="side-label">Plate Crop</span>
                            </div>
                        </div>
                    </div>

                    <!-- Multi-Frame Strip (if multiple frames available) -->
                    ${{thumbsHtml ? `
                    <div class="frames-strip">
                        <span class="strip-label">Violation Frames (${{c.frame_count}}):</span>
                        ${{thumbsHtml}}
                    </div>` : ''}}

                    <!-- Vehicle Identity Bar -->
                    <div class="identity-bar">
                        <div class="vehicle-type-info">
                            <span class="type-icon">${{c.category_icon}}</span>
                            <div class="type-details">
                                <span class="type-name">${{c.category === 'Cars' ? 'Motor Car / Automobile' : 'Motorcycle / Two-Wheeler'}}</span>
                                <span class="vehicle-id-tag">Tracker ID: ${{c.vehicle_id}}</span>
                            </div>
                        </div>
                        <div>
                            ${{plateDisplay}}
                            ${{c.plate_number !== 'Unknown' ? `<div class="plate-ocr-pct" style="text-align:right;">OCR Conf: ${{c.plate_confidence}}%</div>` : ''}}
                        </div>
                    </div>

                    <!-- Offense Citation Law Description -->
                    <div class="offense-box">
                        <span class="offense-code">${{c.section_code}}</span>
                        <p class="offense-desc">${{c.law_desc}}</p>
                    </div>

                    <!-- Standardized Audit & Remarks Hub -->
                    <div class="audit-hub">
                        <div class="audit-hub-title">
                            <span>Standard Audit Verdict</span>
                            <span class="saved-status-tag" id="saved-${{c.challan_no}}">✓ Auto-saved</span>
                        </div>

                        <!-- 6 Standard Verdict Chips -->
                        <div class="verdict-chips-grid">
                            <button class="verdict-chip chip-${{c.challan_no}} ${{audit.verdict === 'VALID_VIOLATION' ? 'selected' : ''}}" data-verdict="VALID_VIOLATION" onclick="setVerdict('${{c.challan_no}}', 'VALID_VIOLATION')">
                                <span>🟢</span>
                                <span>100% Valid</span>
                            </button>

                            <button class="verdict-chip chip-${{c.challan_no}} ${{audit.verdict === 'CROP_BLEED' ? 'selected' : ''}}" data-verdict="CROP_BLEED" onclick="setVerdict('${{c.challan_no}}', 'CROP_BLEED')">
                                <span>🟠</span>
                                <span>Cropping Bleed</span>
                            </button>

                            <button class="verdict-chip chip-${{c.challan_no}} ${{audit.verdict === 'FALSE_POSITIVE' ? 'selected' : ''}}" data-verdict="FALSE_POSITIVE" onclick="setVerdict('${{c.challan_no}}', 'FALSE_POSITIVE')">
                                <span>🔴</span>
                                <span>False Positive</span>
                            </button>

                            <button class="verdict-chip chip-${{c.challan_no}} ${{audit.verdict === 'HALLUCINATION' ? 'selected' : ''}}" data-verdict="HALLUCINATION" onclick="setVerdict('${{c.challan_no}}', 'HALLUCINATION')">
                                <span>🟣</span>
                                <span>Hallucination</span>
                            </button>

                            <button class="verdict-chip chip-${{c.challan_no}} ${{audit.verdict === 'OCCLUDED' ? 'selected' : ''}}" data-verdict="OCCLUDED" onclick="setVerdict('${{c.challan_no}}', 'OCCLUDED')">
                                <span>⚪</span>
                                <span>Occluded / Glare</span>
                            </button>

                            <button class="verdict-chip chip-${{c.challan_no}} ${{audit.verdict === 'PLATE_ISSUE' ? 'selected' : ''}}" data-verdict="PLATE_ISSUE" onclick="setVerdict('${{c.challan_no}}', 'PLATE_ISSUE')">
                                <span>🟡</span>
                                <span>Plate Failed</span>
                            </button>
                        </div>

                        <!-- Custom User Remarks Textarea -->
                        <div>
                            <textarea class="remarks-input" id="remarks-${{c.challan_no}}" placeholder="Enter specific observations (e.g. adjacent motorcycle rider caught in car door crop; windshield dark tint...)" oninput="handleRemarksInput('${{c.challan_no}}', this.value)">${{audit.remarks || ''}}</textarea>
                        </div>
                    </div>
                `;

                grid.appendChild(card);
            }});
        }}

        function switchFrame(challanNo, relPath, btn) {{
            const img = document.getElementById(`main-evidence-${{challanNo}}`);
            if (img) {{
                img.src = relPath;
            }}
            const parent = btn.parentElement;
            parent.querySelectorAll('.thumb-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        }}

        /* Lightbox functions */
        function openLightbox(src) {{
            if (!src) return;
            document.getElementById('lightboxImg').src = src;
            document.getElementById('lightboxModal').classList.add('open');
        }}

        function closeLightbox() {{
            document.getElementById('lightboxModal').classList.remove('open');
        }}

        /* Guide Modal */
        function openGuideModal() {{
            document.getElementById('guideModal').classList.add('open');
        }}

        function closeGuideModal(e, force = false) {{
            if (force || (e && e.target.id === 'guideModal')) {{
                document.getElementById('guideModal').classList.remove('open');
            }}
        }}

        /* Export Audit to CSV */
        function exportAuditCSV() {{
            let csvContent = "data:text/csv;charset=utf-8,";
            csvContent += "Challan_No,Vehicle_ID,Category,Plate_Number,Violation_Type,Fine_PKR,Audit_Verdict,User_Remarks,Reviewed_Timestamp\\n";

            CHALLANS.forEach(c => {{
                const a = userAudits[c.challan_no] || {{ verdict: 'UNREVIEWED', remarks: '', timestamp: '' }};
                const verdict = a.verdict || 'UNREVIEWED';
                const remarks = (a.remarks || '').replace(/"/g, '""').replace(/\\n/g, ' ');
                const row = [
                    `"${{c.challan_no}}"`,
                    `"${{c.vehicle_id}}"`,
                    `"${{c.category}}"`,
                    `"${{c.plate_number}}"`,
                    `"${{c.violation_type}}"`,
                    c.fine_amount,
                    `"${{verdict}}"`,
                    `"${{remarks}}"`,
                    `"${{a.timestamp || ''}}"`
                ].join(',');
                csvContent += row + "\\n";
            }});

            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `sindh_echallan_model_audit_${{new Date().toISOString().slice(0,10)}}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}

        window.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
"""

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n🎉 e-Challan Audit Dashboard created successfully!")
    print(f"📄 Dashboard Path: {HTML_OUTPUT}")
    print(f"📊 Processed {total_challans} violation citations ready for standard review.")

    try:
        webbrowser.open(HTML_OUTPUT)
        print(f"🌐 Opening dashboard in browser: {HTML_OUTPUT}")
    except Exception as e:
        print(f"⚠️ Could not auto-open browser: {e}")


def main():
    print("🚀 Generating Sindh Police e-Challan Preview & Model Quality Audit Dashboard...")
    records = build_challan_dataset()
    generate_html(records)


if __name__ == "__main__":
    main()
