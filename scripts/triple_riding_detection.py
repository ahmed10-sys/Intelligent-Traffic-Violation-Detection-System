"""
Triple Riding & Rider Count Detector for Motorcycles
-----------------------------------------------------
1. Scans all cropped bike images in output/cropped_photos/Bikes_Motorcycles/
2. Runs YOLOv8 pretrained COCO model (class 0 = person) to count riders per bike
3. Generates annotated bounding box images with rider count banners
4. Exports CSV log: output/triple_riding_results.csv
5. Generates interactive HTML audit tool: output/triple_riding_visualizer.html
   with standard rating criteria (Accurate, Over-count, Under-count, Bleed, Ambiguous)
   and auto-saving remarks.
"""

import os
import sys
import json
import csv
import re
import cv2
import pandas as pd
from ultralytics import YOLO

# Fix Windows console encoding for Unicode/emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "Models", "yolov8m.pt")
BIKE_DIR = os.path.join(PROJECT_ROOT, "output", "cropped_photos", "Bikes_Motorcycles")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "triple_riding_results")
CSV_PATH = os.path.join(PROJECT_ROOT, "output", "triple_riding_results.csv")
HTML_PATH = os.path.join(PROJECT_ROOT, "output", "triple_riding_visualizer.html")

CONF_THRES = 0.25  # Person detection threshold
PERSON_CLASS = 0   # COCO class 0 = person

os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_rider_annotations(img, detections):
    """Draw colorful bounding boxes and summary header banner on bike crop."""
    annotated = img.copy()
    h, w = annotated.shape[:2]
    count = len(detections)

    # Color scheme for riders
    colors = [
        (0, 255, 0),    # Rider 1 (Green)
        (255, 165, 0),  # Rider 2 (Cyan/Blue)
        (0, 0, 255),    # Rider 3 (Red)
        (255, 0, 255),  # Rider 4 (Magenta)
        (0, 255, 255),  # Rider 5 (Yellow)
    ]

    for idx, (x1, y1, x2, y2, conf) in enumerate(detections):
        color = colors[idx % len(colors)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = f"Rider #{idx+1} ({conf*100:.0f}%)"
        
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - 25)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Header Banner overlay
    banner_h = 42
    if count >= 3:
        # Red violation banner
        banner_bg = (0, 0, 200)
        text = f"TRIPLE RIDING VIOLATION ({count} PERSONS DETECTED)"
    elif count == 2:
        # Blue pillion banner
        banner_bg = (180, 100, 20)
        text = f"DOUBLE RIDING ({count} PERSONS - PILLION)"
    elif count == 1:
        # Emerald solo banner
        banner_bg = (30, 140, 30)
        text = f"SOLO RIDER (1 PERSON)"
    else:
        # Gray banner
        banner_bg = (70, 70, 70)
        text = "NO PERSON DETECTED"

    cv2.rectangle(annotated, (0, 0), (w, banner_h), banner_bg, -1)
    cv2.putText(annotated, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated

def generate_html_visualizer(results_data):
    """Generate modern, rich, interactive HTML audit visualizer."""
    total_crops = len(results_data)
    unique_bikes = len(set(r["vehicle_id"] for r in results_data))
    triple_count = sum(1 for r in results_data if r["person_count"] >= 3)
    double_count = sum(1 for r in results_data if r["person_count"] == 2)
    solo_count = sum(1 for r in results_data if r["person_count"] == 1)
    zero_count = sum(1 for r in results_data if r["person_count"] == 0)

    # Prepare JSON data for client-side filtering and rendering
    json_data = json.dumps(results_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Motorcycle Triple Riding & Rider Count Audit</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-primary: #0a0e17;
      --bg-card: rgba(18, 24, 38, 0.75);
      --bg-card-hover: rgba(28, 38, 60, 0.9);
      --border-color: rgba(255, 255, 255, 0.08);
      --border-highlight: rgba(99, 102, 241, 0.3);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --danger: #ef4444;
      --danger-glow: rgba(239, 68, 68, 0.25);
      --warning: #f59e0b;
      --success: #10b981;
      --info: #3b82f6;
      --purple: #8b5cf6;
      --cyan: #06b6d4;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: radial-gradient(circle at 50% 0%, #171f38 0%, var(--bg-primary) 70%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 24px 32px;
      line-height: 1.5;
    }}

    /* Top Header */
    .header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 20px;
      padding: 24px;
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-color);
      border-radius: 20px;
      margin-bottom: 24px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }}

    .header-title {{
      display: flex;
      align-items: center;
      gap: 16px;
    }}

    .header-badge {{
      background: linear-gradient(135deg, #ef4444, #f97316);
      width: 52px;
      height: 52px;
      border-radius: 14px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 28px;
      box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    }}

    .header h1 {{
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(to right, #ffffff, #cbd5e1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .header p {{
      color: var(--text-muted);
      font-size: 14px;
      margin-top: 2px;
    }}

    /* Action buttons in header */
    .header-actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}

    .btn {{
      padding: 10px 18px;
      border-radius: 10px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s ease;
      border: 1px solid transparent;
      outline: none;
    }}

    .btn-primary {{
      background: linear-gradient(135deg, #4f46e5, #6366f1);
      color: white;
      box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
    }}
    .btn-primary:hover {{
      background: linear-gradient(135deg, #4338ca, #4f46e5);
      transform: translateY(-1px);
    }}

    .btn-secondary {{
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-main);
      border-color: var(--border-color);
    }}
    .btn-secondary:hover {{
      background: rgba(255, 255, 255, 0.12);
    }}

    /* KPI Grid */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}

    .kpi-card {{
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      position: relative;
      overflow: hidden;
      transition: transform 0.2s, border-color 0.2s;
    }}

    .kpi-card:hover {{
      transform: translateY(-2px);
      border-color: var(--border-highlight);
    }}

    .kpi-card::after {{
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 4px;
      height: 100%;
    }}

    .kpi-card.total::after {{ background: var(--cyan); }}
    .kpi-card.bikes::after {{ background: var(--purple); }}
    .kpi-card.triple::after {{ background: var(--danger); }}
    .kpi-card.double::after {{ background: var(--info); }}
    .kpi-card.solo::after {{ background: var(--success); }}

    .kpi-label {{
      font-size: 13px;
      font-weight: 500;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}

    .kpi-val {{
      font-size: 28px;
      font-weight: 800;
      margin-top: 6px;
      font-family: 'JetBrains Mono', monospace;
    }}

    .kpi-val.danger {{ color: #f87171; text-shadow: 0 0 16px var(--danger-glow); }}
    .kpi-val.info {{ color: #60a5fa; }}
    .kpi-val.success {{ color: #34d399; }}

    /* Standard Rating Guide Box */
    .guide-banner {{
      background: rgba(30, 41, 59, 0.6);
      border: 1px dashed rgba(255, 255, 255, 0.15);
      border-radius: 14px;
      padding: 16px 20px;
      margin-bottom: 24px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 16px;
      font-size: 13px;
    }}

    .guide-title {{
      font-weight: 700;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .guide-chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      flex: 1;
    }}

    .guide-chip {{
      padding: 4px 10px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: #cbd5e1;
    }}

    /* Filter Controls Bar */
    .controls-bar {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 16px 20px;
      margin-bottom: 24px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }}

    .filter-tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}

    .tab-btn {{
      padding: 8px 16px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.04);
      color: var(--text-muted);
      border: 1px solid var(--border-color);
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      transition: all 0.2s;
    }}

    .tab-btn:hover {{
      background: rgba(255, 255, 255, 0.1);
      color: var(--text-main);
    }}

    .tab-btn.active {{
      background: #4f46e5;
      color: white;
      border-color: #6366f1;
      box-shadow: 0 0 12px rgba(79, 70, 229, 0.4);
    }}

    .tab-btn.active.triple {{
      background: #ef4444;
      border-color: #f87171;
      box-shadow: 0 0 12px rgba(239, 68, 68, 0.4);
    }}

    .search-inputs {{
      display: flex;
      gap: 10px;
      flex: 1;
      max-width: 400px;
    }}

    .search-box {{
      width: 100%;
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 8px 14px;
      color: white;
      font-size: 13px;
      outline: none;
      transition: border-color 0.2s;
    }}
    .search-box:focus {{
      border-color: #6366f1;
    }}

    .select-box {{
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 8px 12px;
      color: white;
      font-size: 13px;
      outline: none;
      cursor: pointer;
    }}

    /* Gallery Grid */
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
      transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
      position: relative;
    }}

    .card:hover {{
      transform: translateY(-4px);
      border-color: var(--border-highlight);
      box-shadow: 0 12px 28px rgba(0,0,0,0.4);
    }}

    .card.is-triple {{
      border-color: rgba(239, 68, 68, 0.4);
      background: linear-gradient(180deg, rgba(239, 68, 68, 0.08) 0%, var(--bg-card) 40%);
    }}

    .card-img-wrap {{
      position: relative;
      width: 100%;
      height: 270px;
      background: #000;
      overflow: hidden;
      cursor: zoom-in;
    }}

    .card-img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      transition: transform 0.3s ease;
    }}

    .card-img-wrap:hover .card-img {{
      transform: scale(1.03);
    }}

    .card-badge {{
      position: absolute;
      top: 12px;
      right: 12px;
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 700;
      backdrop-filter: blur(8px);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
      gap: 6px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }}

    .badge-triple {{
      background: rgba(239, 68, 68, 0.9);
      color: white;
      border: 1px solid #fca5a5;
      animation: pulseGlow 2s infinite;
    }}

    @keyframes pulseGlow {{
      0%, 100% {{ box-shadow: 0 0 10px rgba(239, 68, 68, 0.6); }}
      50% {{ box-shadow: 0 0 20px rgba(239, 68, 68, 0.9); }}
    }}

    .badge-double {{
      background: rgba(59, 130, 246, 0.85);
      color: white;
      border: 1px solid #93c5fd;
    }}

    .badge-solo {{
      background: rgba(16, 185, 129, 0.85);
      color: white;
      border: 1px solid #6ee7b7;
    }}

    .badge-zero {{
      background: rgba(100, 116, 139, 0.85);
      color: white;
    }}

    .card-body {{
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      flex: 1;
    }}

    .card-meta {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
    }}

    .meta-id {{
      font-weight: 700;
      color: #e2e8f0;
      font-family: 'JetBrains Mono', monospace;
      background: rgba(255, 255, 255, 0.06);
      padding: 3px 8px;
      border-radius: 6px;
    }}

    .meta-file {{
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      max-width: 180px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .detections-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}

    .det-pill {{
      font-size: 11px;
      font-family: 'JetBrains Mono', monospace;
      padding: 3px 8px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: #94a3b8;
    }}

    /* Rating & Review Section */
    .review-section {{
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 10px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-top: auto;
    }}

    .review-label {{
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: flex;
      justify-content: space-between;
    }}

    .status-indicator {{
      font-size: 11px;
      color: var(--success);
      font-weight: 500;
      display: none;
    }}

    .rating-chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}

    .rate-btn {{
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: var(--text-muted);
      transition: all 0.15s;
    }}

    .rate-btn:hover {{
      background: rgba(255, 255, 255, 0.12);
      color: white;
    }}

    .rate-btn.selected-accurate {{
      background: rgba(16, 185, 129, 0.2);
      border-color: #10b981;
      color: #34d399;
    }}
    .rate-btn.selected-over {{
      background: rgba(245, 158, 11, 0.2);
      border-color: #f59e0b;
      color: #fbbf24;
    }}
    .rate-btn.selected-under {{
      background: rgba(239, 68, 68, 0.2);
      border-color: #ef4444;
      color: #f87171;
    }}
    .rate-btn.selected-bleed {{
      background: rgba(139, 92, 246, 0.2);
      border-color: #8b5cf6;
      color: #a78bfa;
    }}
    .rate-btn.selected-ambiguous {{
      background: rgba(100, 116, 139, 0.2);
      border-color: #64748b;
      color: #cbd5e1;
    }}

    .remark-input {{
      width: 100%;
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 6px;
      padding: 6px 10px;
      color: white;
      font-size: 12px;
      outline: none;
      transition: border-color 0.2s;
    }}
    .remark-input:focus {{
      border-color: #6366f1;
    }}

    /* Lightbox Modal */
    .modal {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.88);
      backdrop-filter: blur(8px);
      z-index: 9999;
      justify-content: center;
      align-items: center;
      padding: 24px;
    }}

    .modal.active {{
      display: flex;
    }}

    .modal-content {{
      max-width: 90vw;
      max-height: 90vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
    }}

    .modal-img {{
      max-width: 100%;
      max-height: 80vh;
      border-radius: 12px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.8);
      border: 1px solid rgba(255, 255, 255, 0.15);
    }}

    .modal-close {{
      position: absolute;
      top: 24px;
      right: 32px;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: white;
      font-size: 24px;
      width: 44px;
      height: 44px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    .empty-state {{
      grid-column: 1 / -1;
      text-align: center;
      padding: 60px;
      background: var(--bg-card);
      border-radius: 16px;
      color: var(--text-muted);
    }}
  </style>
</head>
<body>

  <!-- Top Header -->
  <header class="header">
    <div class="header-title">
      <div class="header-badge">🏍️</div>
      <div>
        <h1>Motorcycle Triple Riding & Occupancy Audit</h1>
        <p>Pretrained YOLO Person Detection benchmark across {total_crops} bike crops ({unique_bikes} unique motorcycles)</p>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn btn-secondary" onclick="exportAuditJSON()">📥 Export Audit JSON</button>
      <button class="btn btn-secondary" onclick="exportAuditCSV()">📊 Export CSV Report</button>
      <button class="btn btn-primary" onclick="window.scrollTo({{top: 0, behavior: 'smooth'}})">⬆️ Top</button>
    </div>
  </header>

  <!-- KPI Cards -->
  <section class="kpi-grid">
    <div class="kpi-card total">
      <span class="kpi-label">Total Bike Crops</span>
      <span class="kpi-val">{total_crops}</span>
    </div>
    <div class="kpi-card bikes">
      <span class="kpi-label">Tracked Motorcycles</span>
      <span class="kpi-val">{unique_bikes}</span>
    </div>
    <div class="kpi-card triple">
      <span class="kpi-label">🚨 Triple Riding Candidates</span>
      <span class="kpi-val danger" id="kpi-triple-val">{triple_count}</span>
    </div>
    <div class="kpi-card double">
      <span class="kpi-label">👥 Double Riders (Pillion)</span>
      <span class="kpi-val info">{double_count}</span>
    </div>
    <div class="kpi-card solo">
      <span class="kpi-label">👤 Solo Riders</span>
      <span class="kpi-val success">{solo_count}</span>
    </div>
  </section>

  <!-- Standard Rating Guide -->
  <div class="guide-banner">
    <span class="guide-title">📋 Rating Standard:</span>
    <div class="guide-chips">
      <span class="guide-chip"><strong>✅ Accurate Count:</strong> Model correctly isolated the exact rider count (1, 2, or 3+).</span>
      <span class="guide-chip"><strong>⚠️ Over-count:</strong> Model falsely detected luggage, spare tire, or background as a person.</span>
      <span class="guide-chip"><strong>⚠️ Under-count:</strong> Model missed a child or pillion passenger.</span>
      <span class="guide-chip"><strong>🔀 Neighbor Bleed:</strong> Model included a rider from an adjacent bike lane.</span>
      <span class="guide-chip"><strong>❓ Ambiguous:</strong> Image is too blurry or occluded to verify with naked eye.</span>
    </div>
  </div>

  <!-- Filter & Search Controls -->
  <div class="controls-bar">
    <div class="filter-tabs">
      <button class="tab-btn active" onclick="setCountFilter('all', this)">All Crops ({total_crops})</button>
      <button class="tab-btn triple" onclick="setCountFilter('triple', this)">🚨 Triple Riding ({triple_count})</button>
      <button class="tab-btn" onclick="setCountFilter('double', this)">👥 Double Riders ({double_count})</button>
      <button class="tab-btn" onclick="setCountFilter('solo', this)">👤 Solo ({solo_count})</button>
      <button class="tab-btn" onclick="setCountFilter('zero', this)">❓ No Person ({zero_count})</button>
    </div>

    <div class="search-inputs">
      <select class="select-box" id="reviewFilter" onchange="applyFilters()">
        <option value="all">All Review Statuses</option>
        <option value="unreviewed">Unreviewed Only</option>
        <option value="Accurate">✅ Accurate</option>
        <option value="Over-count">⚠️ Over-count</option>
        <option value="Under-count">⚠️ Under-count</option>
        <option value="Neighbor Bleed">🔀 Neighbor Bleed</option>
        <option value="Ambiguous">❓ Ambiguous</option>
      </select>
      <input type="text" class="search-box" id="searchBox" placeholder="Search Vehicle ID (e.g. 84)..." oninput="applyFilters()">
    </div>
  </div>

  <!-- Cards Grid -->
  <main class="grid" id="cardsGrid">
    <!-- Populated by JavaScript -->
  </main>

  <!-- Image Lightbox Modal -->
  <div class="modal" id="modal" onclick="closeModal()">
    <button class="modal-close" onclick="closeModal()">&times;</button>
    <div class="modal-content" onclick="event.stopPropagation()">
      <img id="modalImg" class="modal-img" src="" alt="Enlarged view">
      <div id="modalCaption" style="font-size: 14px; color: #cbd5e1; font-family: 'JetBrains Mono', monospace;"></div>
    </div>
  </div>

  <script>
    const DATA = {json_data};
    let currentFilter = 'all';

    // LocalStorage helper for ratings and remarks
    const STORAGE_KEY = 'TRIPLE_RIDING_AUDIT_RATINGS';
    let userRatings = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');

    function saveUserRatings() {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(userRatings));
    }}

    function setRating(imgKey, ratingValue) {{
      if (!userRatings[imgKey]) userRatings[imgKey] = {{}};
      userRatings[imgKey].rating = ratingValue;
      userRatings[imgKey].updatedAt = new Date().toISOString();
      saveUserRatings();

      // Update UI buttons on that card
      const card = document.getElementById('card-' + imgKey);
      if (card) {{
        card.querySelectorAll('.rate-btn').forEach(b => {{
          b.className = 'rate-btn';
        }});
        const activeBtn = card.querySelector(`[data-val="${{ratingValue}}"]`);
        if (activeBtn) {{
          if (ratingValue === 'Accurate') activeBtn.classList.add('selected-accurate');
          else if (ratingValue === 'Over-count') activeBtn.classList.add('selected-over');
          else if (ratingValue === 'Under-count') activeBtn.classList.add('selected-under');
          else if (ratingValue === 'Neighbor Bleed') activeBtn.classList.add('selected-bleed');
          else if (ratingValue === 'Ambiguous') activeBtn.classList.add('selected-ambiguous');
        }}
        const indicator = card.querySelector('.status-indicator');
        if (indicator) {{
          indicator.style.display = 'inline';
          indicator.textContent = '💾 Saved';
          setTimeout(() => {{ indicator.style.display = 'none'; }}, 1500);
        }}
      }}
    }}

    function saveRemark(imgKey, text) {{
      if (!userRatings[imgKey]) userRatings[imgKey] = {{}};
      userRatings[imgKey].remark = text;
      userRatings[imgKey].updatedAt = new Date().toISOString();
      saveUserRatings();

      const card = document.getElementById('card-' + imgKey);
      if (card) {{
        const indicator = card.querySelector('.status-indicator');
        if (indicator) {{
          indicator.style.display = 'inline';
          indicator.textContent = '💾 Remark Saved';
          setTimeout(() => {{ indicator.style.display = 'none'; }}, 1500);
        }}
      }}
    }}

    function setCountFilter(filterType, element) {{
      currentFilter = filterType;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      element.classList.add('active');
      applyFilters();
    }}

    function applyFilters() {{
      const search = document.getElementById('searchBox').value.toLowerCase().trim();
      const reviewFilter = document.getElementById('reviewFilter').value;
      const grid = document.getElementById('cardsGrid');

      const filtered = DATA.filter(item => {{
        // Count filter
        if (currentFilter === 'triple' && item.person_count < 3) return false;
        if (currentFilter === 'double' && item.person_count !== 2) return false;
        if (currentFilter === 'solo' && item.person_count !== 1) return false;
        if (currentFilter === 'zero' && item.person_count !== 0) return false;

        // Search box filter
        if (search) {{
          const matchesId = item.vehicle_id.toString().toLowerCase().includes(search) || 
                            `id_${{item.vehicle_id}}`.includes(search);
          const matchesFile = item.filename.toLowerCase().includes(search);
          if (!matchesId && !matchesFile) return false;
        }}

        // Review filter
        const key = item.key;
        const review = userRatings[key] || {{}};
        if (reviewFilter === 'unreviewed' && review.rating) return false;
        if (reviewFilter !== 'all' && reviewFilter !== 'unreviewed' && review.rating !== reviewFilter) return false;

        return true;
      }});

      renderGrid(filtered);
    }}

    function renderGrid(items) {{
      const grid = document.getElementById('cardsGrid');
      if (items.length === 0) {{
        grid.innerHTML = '<div class="empty-state"><h3>No matching motorcycle crops found</h3><p>Try resetting the search filters</p></div>';
        return;
      }}

      grid.innerHTML = items.map(item => {{
        const key = item.key;
        const userReview = userRatings[key] || {{}};
        const currentRating = userReview.rating || '';
        const currentRemark = userReview.remark || '';

        // Badge styling
        let badgeClass = 'badge-zero';
        let badgeText = '❓ 0 Persons';
        if (item.person_count >= 3) {{
          badgeClass = 'badge-triple';
          badgeText = `🚨 ${{item.person_count}} PERSONS (TRIPLE)`;
        }} else if (item.person_count === 2) {{
          badgeClass = 'badge-double';
          badgeText = `👥 2 PERSONS (PILLION)`;
        }} else if (item.person_count === 1) {{
          badgeClass = 'badge-solo';
          badgeText = `👤 1 PERSON (SOLO)`;
        }}

        // Det pills
        const pillsHtml = item.detections.map((d, i) => 
          `<span class="det-pill">Rider #${{i+1}}: ${{(d.conf * 100).toFixed(0)}}%</span>`
        ).join('');

        return `
          <div class="card ${{item.person_count >= 3 ? 'is-triple' : ''}}" id="card-${{key}}">
            <div class="card-img-wrap" onclick="openModal('${{item.annotated_url}}', 'Vehicle ID: ${{item.vehicle_id}} | ${{item.filename}}')">
              <img class="card-img" src="${{item.annotated_url}}" alt="${{item.filename}}" loading="lazy">
              <span class="card-badge ${{badgeClass}}">${{badgeText}}</span>
            </div>

            <div class="card-body">
              <div class="card-meta">
                <span class="meta-id">ID: ${{item.vehicle_id}}</span>
                <span class="meta-file" title="${{item.filename}}">${{item.filename}}</span>
              </div>

              <div class="detections-pills">
                ${{pillsHtml || '<span class="det-pill">No detections</span>'}}
              </div>

              <!-- Review & Rating Box -->
              <div class="review-section">
                <div class="review-label">
                  <span>Audit Rating Standard</span>
                  <span class="status-indicator"></span>
                </div>

                <div class="rating-chips">
                  <button class="rate-btn ${{currentRating === 'Accurate' ? 'selected-accurate' : ''}}" 
                          data-val="Accurate" onclick="setRating('${{key}}', 'Accurate')">✅ Accurate</button>
                  <button class="rate-btn ${{currentRating === 'Over-count' ? 'selected-over' : ''}}" 
                          data-val="Over-count" onclick="setRating('${{key}}', 'Over-count')">⚠️ Over-count</button>
                  <button class="rate-btn ${{currentRating === 'Under-count' ? 'selected-under' : ''}}" 
                          data-val="Under-count" onclick="setRating('${{key}}', 'Under-count')">⚠️ Under-count</button>
                  <button class="rate-btn ${{currentRating === 'Neighbor Bleed' ? 'selected-bleed' : ''}}" 
                          data-val="Neighbor Bleed" onclick="setRating('${{key}}', 'Neighbor Bleed')">🔀 Neighbor Bleed</button>
                  <button class="rate-btn ${{currentRating === 'Ambiguous' ? 'selected-ambiguous' : ''}}" 
                          data-val="Ambiguous" onclick="setRating('${{key}}', 'Ambiguous')">❓ Ambiguous</button>
                </div>

                <input type="text" class="remark-input" placeholder="Add remarks or notes..." 
                       value="${{currentRemark.replace(/"/g, '&quot;')}}" 
                       onchange="saveRemark('${{key}}', this.value)">
              </div>
            </div>
          </div>
        `;
      }}).join('');
    }}

    function openModal(imgUrl, caption) {{
      document.getElementById('modalImg').src = imgUrl;
      document.getElementById('modalCaption').textContent = caption;
      document.getElementById('modal').classList.add('active');
    }}

    function closeModal() {{
      document.getElementById('modal').classList.remove('active');
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape') closeModal();
    }});

    function exportAuditJSON() {{
      const exportData = DATA.map(item => ({{
        vehicle_id: item.vehicle_id,
        filename: item.filename,
        detected_persons: item.person_count,
        detections: item.detections,
        audit_rating: (userRatings[item.key] || {{}}).rating || 'Unreviewed',
        audit_remark: (userRatings[item.key] || {{}}).remark || '',
      }}));

      const blob = new Blob([JSON.stringify(exportData, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'triple_riding_audit_review.json';
      a.click();
      URL.revokeObjectURL(url);
    }}

    function exportAuditCSV() {{
      const headers = ['vehicle_id', 'filename', 'detected_persons', 'is_triple_riding', 'audit_rating', 'audit_remark'];
      const rows = DATA.map(item => {{
        const r = userRatings[item.key] || {{}};
        return [
          item.vehicle_id,
          `"${{item.filename}}"`,
          item.person_count,
          item.person_count >= 3 ? 'YES' : 'NO',
          `"${{r.rating || 'Unreviewed'}}"`,
          `"${{(r.remark || '').replace(/"/g, '""')}}"`
        ].join(',');
      }});

      const csvContent = [headers.join(','), ...rows].join('\\n');
      const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'triple_riding_audit_results.csv';
      a.click();
      URL.revokeObjectURL(url);
    }}

    // Initial render
    applyFilters();
  </script>
</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n🎉 Triple Riding Audit Visualizer generated: {HTML_PATH}")

def main():
    print("=" * 65)
    print("🏍️ TRIPLE RIDING & OCCUPANCY DETECTION BENCHMARK")
    print("=" * 65)
    print(f"Model       : {MODEL_PATH}")
    print(f"Crops Dir   : {BIKE_DIR}")
    print(f"Output Dir  : {OUTPUT_DIR}")
    print(f"Conf Thres  : {CONF_THRES}")

    if not os.path.exists(BIKE_DIR):
        print(f"❌ Error: Bike crops directory not found: {BIKE_DIR}")
        return

    # Load YOLOv8 model
    print(f"Loading pretrained YOLO model...")
    model = YOLO(MODEL_PATH)

    id_folders = sorted([f for f in os.listdir(BIKE_DIR) if os.path.isdir(os.path.join(BIKE_DIR, f)) and f.startswith("id_")])
    if not id_folders:
        print(f"❌ No id_XXX folders found in {BIKE_DIR}")
        return

    print(f"Found {len(id_folders)} motorcycle IDs to evaluate.\n")

    csv_rows = []
    html_items = []
    total_crops_processed = 0
    triple_riding_candidates = 0

    for id_folder in id_folders:
        folder_path = os.path.join(BIKE_DIR, id_folder)
        try:
            vehicle_id = int(id_folder.replace("id_", ""))
        except:
            vehicle_id = id_folder

        id_out_dir = os.path.join(OUTPUT_DIR, id_folder)
        os.makedirs(id_out_dir, exist_ok=True)

        img_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png")) and "plate" not in f.lower()])

        for filename in img_files:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            total_crops_processed += 1

            # Run person detection (class 0)
            results = model.predict(img, conf=CONF_THRES, classes=[PERSON_CLASS], verbose=False)
            boxes = results[0].boxes

            detections = []
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    conf = float(b.conf[0])
                    detections.append((x1, y1, x2, y2, conf))

            # Sort detections from top to bottom or left to right
            detections.sort(key=lambda d: d[0])
            person_count = len(detections)

            if person_count >= 3:
                status = "triple_riding"
                triple_riding_candidates += 1
            elif person_count == 2:
                status = "double_riding"
            elif person_count == 1:
                status = "solo_rider"
            else:
                status = "no_person"

            # Draw annotations
            annotated_img = draw_rider_annotations(img, detections)
            out_filename = f"annotated_{filename}"
            out_path = os.path.join(id_out_dir, out_filename)
            cv2.imwrite(out_path, annotated_img)

            # Store in CSV records
            conf_str = ";".join([f"{d[4]:.2f}" for d in detections])
            box_str = ";".join([f"{d[0]},{d[1]},{d[2]},{d[3]}" for d in detections])
            csv_rows.append([
                vehicle_id, filename, person_count, status, conf_str, box_str, out_path
            ])

            # Store in HTML records
            # Relative URL for portability
            rel_annotated = f"triple_riding_results/{id_folder}/{out_filename}".replace("\\", "/")
            key = f"v{vehicle_id}_{filename.replace('.', '_')}"

            html_items.append({
                "key": key,
                "vehicle_id": vehicle_id,
                "filename": filename,
                "person_count": person_count,
                "status": status,
                "annotated_url": rel_annotated,
                "detections": [{"conf": round(d[4], 2), "box": [d[0], d[1], d[2], d[3]]} for d in detections]
            })

        print(f"  ID {id_folder}: Processed {len(img_files)} crops")

    # Write CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "vehicle_id", "filename", "person_count", "status", "confidences", "boxes", "annotated_path"
        ])
        writer.writerows(csv_rows)

    print(f"\n✅ CSV saved to: {CSV_PATH}")
    print(f"Total crops evaluated: {total_crops_processed}")
    print(f"Triple riding candidates: {triple_riding_candidates}")

    # Generate HTML
    generate_html_visualizer(html_items)

if __name__ == "__main__":
    main()
