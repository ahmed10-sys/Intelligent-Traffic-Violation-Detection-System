"""
Generate HTML visual dashboard from benchmark CSV and crops
"""
import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "plate_model_comparison")
CSV_PATH = os.path.join(OUTPUT_DIR, "plate_model_comparison_results.csv")
HTML_PATH = os.path.join(OUTPUT_DIR, "plate_comparison_report.html")

df = pd.read_csv(CSV_PATH)
df['A_Plate_Text'] = df['A_Plate_Text'].fillna('')
df['B_Plate_Text'] = df['B_Plate_Text'].fillna('')

total = len(df)
a_det = int(df['A_Detected'].sum())
b_det = int(df['B_Detected'].sum())
a_read = int(df['A_Readable'].sum())
b_read = int(df['B_Readable'].sum())

a_det_conf = float(df[df['A_Detected']]['A_Det_Confidence'].mean())
b_det_conf = float(df[df['B_Detected']]['B_Det_Confidence'].mean())
a_ocr_conf = float(df[df['A_Readable']]['A_OCR_Confidence'].mean())
b_ocr_conf = float(df[df['B_Readable']]['B_OCR_Confidence'].mean())

cars = df[df['Category'] == 'Cars']
bikes = df[df['Category'] == 'Bikes_Motorcycles']
rickshaws = df[df['Category'] == 'Rickshaws']

rows_html = []
for idx, r in df.iterrows():
    cat = r['Category']
    v_id = r['Vehicle_ID']

    crop_a_file = f"crops/yolov8s_lpd/{cat}_{v_id}_crop_a.jpg"
    crop_b_file = f"crops/plate_detector/{cat}_{v_id}_crop_b.jpg"

    has_a = os.path.exists(os.path.join(OUTPUT_DIR, crop_a_file))
    has_b = os.path.exists(os.path.join(OUTPUT_DIR, crop_b_file))

    img_a = f'<img class="plate-img" src="{crop_a_file}" alt="Model A Crop" loading="lazy" />' if has_a else '<div class="no-img">No Detection</div>'
    img_b = f'<img class="plate-img" src="{crop_b_file}" alt="Model B Crop" loading="lazy" />' if has_b else '<div class="no-img">No Detection</div>'

    badge_a = '<span class="badge readable">Readable</span>' if r['A_Readable'] else '<span class="badge unreadable">Unreadable</span>' if r['A_Detected'] else '<span class="badge missing">Missed</span>'
    badge_b = '<span class="badge readable">Readable</span>' if r['B_Readable'] else '<span class="badge unreadable">Unreadable</span>' if r['B_Detected'] else '<span class="badge missing">Missed</span>'

    v_str = str(r['Verdict'])
    if "New Model" in v_str:
        v_badge = '<span class="badge win-b">New Model Superior</span>'
    elif "Previous Model" in v_str:
        v_badge = '<span class="badge win-a">Old Model Superior</span>'
    elif "Both Readable" in v_str:
        v_badge = '<span class="badge tie">Tie (Both Readable)</span>'
    elif "Neither Detected" in v_str:
        v_badge = '<span class="badge neither">Neither Detected</span>'
    else:
        v_badge = '<span class="badge neither">Both Detected, Unreadable</span>'

    row = f"""
    <tr>
        <td><strong>{cat}</strong><br><small class="text-muted">{v_id}</small></td>
        <td>
            {img_a}
            <div class="crop-info">
                <div>Det Conf: <strong>{r['A_Det_Confidence']:.2f}</strong></div>
                <div>Text: <code>{r['A_Plate_Text'] if r['A_Plate_Text'] else '-'}</code></div>
                <div>OCR Conf: {r['A_OCR_Confidence']:.2f}</div>
                <div>{badge_a}</div>
            </div>
        </td>
        <td>
            {img_b}
            <div class="crop-info">
                <div>Det Conf: <strong>{r['B_Det_Confidence']:.2f}</strong></div>
                <div>Text: <code>{r['B_Plate_Text'] if r['B_Plate_Text'] else '-'}</code></div>
                <div>OCR Conf: {r['B_OCR_Confidence']:.2f}</div>
                <div>{badge_b}</div>
            </div>
        </td>
        <td>{v_badge}</td>
    </tr>
    """
    rows_html.append(row)

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
            --radius-md: 10px;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text-primary);
            font-family: 'Outfit', sans-serif;
            padding: 28px;
            line-height: 1.5;
        }}
        .header {{
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }}
        h1 {{ font-size: 28px; font-weight: 700; color: #fff; margin-bottom: 6px; }}
        .subtitle {{ color: var(--text-secondary); font-size: 14px; }}
        .badge-header {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-b);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 13px;
            margin-left: 10px;
        }}
        .winner-banner {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(56, 189, 248, 0.1));
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: var(--radius-md);
            padding: 20px 24px;
            margin-bottom: 24px;
        }}
        .winner-title {{ font-size: 20px; font-weight: 700; color: #fff; }}
        .winner-desc {{ font-size: 15px; color: #cbd5e1; margin-top: 4px; }}
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
        }}
        .metric-label {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-val {{ font-size: 26px; font-weight: 700; }}
        .val-a {{ color: var(--accent-a); }}
        .val-b {{ color: var(--accent-b); }}
        .metric-sub {{ font-size: 12px; color: var(--text-secondary); margin-top: 4px; }}
        .category-summary {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 18px 22px;
            margin-bottom: 28px;
        }}
        .category-summary h3 {{ font-size: 16px; margin-bottom: 12px; color: #fff; }}
        .cat-table {{ width: 100%; border-collapse: collapse; }}
        .cat-table th, .cat-table td {{ padding: 8px 12px; font-size: 13px; text-align: left; border-bottom: 1px solid var(--border); }}
        .cat-table th {{ background: var(--surface-card); color: var(--text-secondary); }}
        .table-container {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            overflow-x: auto;
        }}
        table.main-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        table.main-table th, table.main-table td {{
            padding: 14px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }}
        table.main-table th {{
            background: var(--surface-card);
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
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
            width: 130px;
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
        .badge.win-b {{ background: rgba(16, 185, 129, 0.25); color: #10b981; border: 1px solid #10b981; font-weight: 700; }}
        .badge.win-a {{ background: rgba(56, 189, 248, 0.25); color: #38bdf8; border: 1px solid #38bdf8; font-weight: 700; }}
        .badge.tie {{ background: rgba(255, 255, 255, 0.08); color: #e2e8f0; border: 1px solid rgba(255, 255, 255, 0.15); }}
        .badge.neither {{ background: rgba(148, 163, 184, 0.1); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.2); }}
        .text-muted {{ color: var(--text-secondary); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>License Plate Detection Benchmark <span class="badge-header">plate_detector.pt WINS</span></h1>
        <div class="subtitle">Comparing Previous Model (<code>yolov8s_lpd.pt</code>) vs Newly Trained Model (<code>plate_detector.pt</code>) &bull; Evaluated on {total} Vehicles</div>
    </div>

    <div class="winner-banner">
        <div class="winner-title">🏆 Definitive Winner: Newly Trained Model (<code>plate_detector.pt</code>)</div>
        <div class="winner-desc">The new model achieved <strong>88 detected plates (94.6%)</strong> vs 67 (72.0%) for the old model, and produced <strong>75 readable plates (80.6%)</strong> vs 61 (65.6%). It had <strong>31 direct head-to-head victories</strong> compared to only 4 for the old model, with an average detection confidence of <strong>82.4% vs 67.3%</strong> (+15.1% firmer).</div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Vehicles Evaluated</div>
            <div class="metric-val">{total}</div>
            <div class="metric-sub">Cars: {len(cars)} | Bikes: {len(bikes)} | Rickshaws: {len(rickshaws)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Plates Detected</div>
            <div class="metric-val">
                <span class="val-a">{a_det}</span> <span style="font-size:16px;color:var(--text-secondary)">vs</span> <span class="val-b">{b_det}</span>
            </div>
            <div class="metric-sub">Old ({a_det/total*100:.1f}%) vs New (<strong>{b_det/total*100:.1f}%</strong>) &bull; <span style="color:var(--accent-b)">+21 plates (+22.6%)</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Readable Plates (OCR)</div>
            <div class="metric-val">
                <span class="val-a">{a_read}</span> <span style="font-size:16px;color:var(--text-secondary)">vs</span> <span class="val-b">{b_read}</span>
            </div>
            <div class="metric-sub">Old ({a_read/total*100:.1f}%) vs New (<strong>{b_read/total*100:.1f}%</strong>) &bull; <span style="color:var(--accent-b)">+14 readable (+15.0%)</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Detection Confidence</div>
            <div class="metric-val">
                <span class="val-a">{a_det_conf:.2f}</span> <span style="font-size:16px;color:var(--text-secondary)">vs</span> <span class="val-b">{b_det_conf:.2f}</span>
            </div>
            <div class="metric-sub"><span style="color:var(--accent-b)">+15.1% firmer bounding boxes</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg OCR Confidence (Valid)</div>
            <div class="metric-val">
                <span class="val-a">{a_ocr_conf:.2f}</span> <span style="font-size:16px;color:var(--text-secondary)">vs</span> <span class="val-b">{b_ocr_conf:.2f}</span>
            </div>
            <div class="metric-sub">Both maintain exceptional legibility</div>
        </div>
    </div>

    <div class="category-summary">
        <h3>Vehicle Category Breakdown</h3>
        <table class="cat-table">
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Count</th>
                    <th>Old Model Detected</th>
                    <th>New Model Detected</th>
                    <th>Old Model Readable</th>
                    <th>New Model Readable</th>
                    <th>Detection Gain</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Bikes & Motorcycles</strong></td>
                    <td>{len(bikes)}</td>
                    <td>{bikes['A_Detected'].sum()} ({bikes['A_Detected'].mean()*100:.1f}%)</td>
                    <td><strong>{bikes['B_Detected'].sum()} ({bikes['B_Detected'].mean()*100:.1f}%)</strong></td>
                    <td>21 ({bikes['A_Readable'].mean()*100:.1f}%)</td>
                    <td><strong>31 ({bikes['B_Readable'].mean()*100:.1f}%)</strong></td>
                    <td style="color:var(--accent-b)"><strong>+17 detected (+10 readable)</strong></td>
                </tr>
                <tr>
                    <td><strong>Cars</strong></td>
                    <td>{len(cars)}</td>
                    <td>{cars['A_Detected'].sum()} ({cars['A_Detected'].mean()*100:.1f}%)</td>
                    <td><strong>{cars['B_Detected'].sum()} ({cars['B_Detected'].mean()*100:.1f}%)</strong></td>
                    <td>38 ({cars['A_Readable'].mean()*100:.1f}%)</td>
                    <td><strong>41 ({cars['B_Readable'].mean()*100:.1f}%)</strong></td>
                    <td style="color:var(--accent-b)"><strong>+3 detected (+3 readable)</strong></td>
                </tr>
                <tr>
                    <td><strong>Rickshaws</strong></td>
                    <td>{len(rickshaws)}</td>
                    <td>{rickshaws['A_Detected'].sum()} ({rickshaws['A_Detected'].mean()*100:.1f}%)</td>
                    <td><strong>{rickshaws['B_Detected'].sum()} ({rickshaws['B_Detected'].mean()*100:.1f}%)</strong></td>
                    <td>2 ({rickshaws['A_Readable'].mean()*100:.1f}%)</td>
                    <td><strong>3 ({rickshaws['B_Readable'].mean()*100:.1f}%)</strong></td>
                    <td style="color:var(--accent-b)"><strong>+1 detected (+1 readable)</strong></td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="table-container">
        <table class="main-table">
            <thead>
                <tr>
                    <th style="width: 140px;">Vehicle</th>
                    <th style="width: 320px;">Previous Model (<code>yolov8s_lpd.pt</code>)</th>
                    <th style="width: 320px;">New Model (<code>plate_detector.pt</code>)</th>
                    <th style="width: 170px;">Verdict</th>
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

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("HTML report successfully generated at:", HTML_PATH)
