"""
Build Kaggle-Ready License Plate YOLO Dataset
=============================================
This script consolidates labeled annotations and clean background/negative crops
from Label Studio and the vehicle extraction pipeline into a standard Ultralytics
YOLO dataset format, split 80% train / 20% validation, and creates a Kaggle-ready .zip.

Features:
- Decodes Label Studio exported label filenames back to clean image stems.
- Includes all verified positive license plate annotations (cars, motorcycles, rickshaws, trucks).
- Selects high-quality background samples (~10% negative images) from skipped crops:
  * Filters out extreme close-ups, abnormal aspect ratios, and tiny crops.
  * Ranks by image sharpness (Laplacian variance) to avoid blurry frames.
  * Diversifies across vehicle track IDs to avoid over-sampling a single vehicle.
- Track-aware Grouped Train/Val Split (80/20) to prevent data leakage across video tracks.
- Produces clean YOLO directories (images/train, images/val, labels/train, labels/val).
- Generates data.yaml, dataset README.md with Kaggle quickstart, and metadata.csv.
- Automatically archives to output/license_plate_yolo_dataset.zip.
"""

import os
import sys
import re
import cv2
import yaml
import shutil
import zipfile
import urllib.parse
import pandas as pd
import numpy as np
from collections import defaultdict

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CROPS_DIR = os.path.join(OUTPUT_DIR, "plate_dataset_crops")
MANIFEST_PATH = os.path.join(CROPS_DIR, "dataset_manifest.csv")
LABELS_SRC_DIR = os.path.join(OUTPUT_DIR, "plate_dataset_labels", "labels")

DATASET_OUT_DIR = os.path.join(OUTPUT_DIR, "license_plate_yolo_dataset")
ZIP_OUT_PATH = os.path.join(OUTPUT_DIR, "license_plate_yolo_dataset.zip")

SPLIT_RATIO = 0.80  # 80% Train, 20% Val
BG_RATIO = 0.10     # ~10% Background samples (standard YOLO best practice)
RANDOM_SEED = 42

def clean_label_filename(filename):
    """Strip 8-hex hash and unquote URL characters to retrieve true image stem."""
    clean = re.sub(r'^[0-9a-fA-F]{8}-?', '', filename)
    clean = re.sub(r'^[0-9a-fA-F]{8}__', '', clean)
    clean = urllib.parse.unquote(clean)
    clean = os.path.basename(clean.replace('/', '\\'))
    stem = os.path.splitext(clean)[0]
    return stem

def calculate_sharpness(img_path):
    """Compute variance of Laplacian to score image sharpness."""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return 0.0

def main():
    print("=" * 75)
    print("🚀 BUILDING KAGGLE-READY LICENSE PLATE YOLO DATASET")
    print("=" * 75)
    print(f"📁 Crops Source:          {CROPS_DIR}")
    print(f"📁 Exported Labels:       {LABELS_SRC_DIR}")
    print(f"📁 Dataset Output:        {DATASET_OUT_DIR}")
    print(f"📦 Zip Output:            {ZIP_OUT_PATH}")
    print(f"📊 Train/Val Split:       {int(SPLIT_RATIO * 100)}% / {int((1 - SPLIT_RATIO) * 100)}%")
    print(f"🎯 Target Background:    ~{int(BG_RATIO * 100)}% clean negative samples")
    print("=" * 75)

    if not os.path.isdir(CROPS_DIR) or not os.path.isdir(LABELS_SRC_DIR):
        print("❌ Error: Required input directories not found!")
        return

    np.random.seed(RANDOM_SEED)

    # 1. Load manifest and match exported labels
    print("\n[1/6] Indexing annotations and manifest...")
    manifest_df = pd.read_csv(MANIFEST_PATH)
    manifest_df['stem'] = manifest_df['filename'].apply(lambda x: os.path.splitext(x)[0])

    exported_files = os.listdir(LABELS_SRC_DIR)
    label_stem_map = {}  # stem -> source label file path
    label_box_counts = {}

    for f in exported_files:
        fp = os.path.join(LABELS_SRC_DIR, f)
        stem = clean_label_filename(f)
        with open(fp, 'r', encoding='utf-8') as src:
            lines = [l.strip() for l in src if l.strip()]
        label_stem_map[stem] = fp
        label_box_counts[stem] = len(lines)

    manifest_df['is_annotated'] = manifest_df['stem'].isin(label_stem_map)
    manifest_df['box_count'] = manifest_df['stem'].map(label_box_counts).fillna(0).astype(int)

    # Positive samples: images annotated with at least 1 box
    positive_df = manifest_df[(manifest_df['is_annotated']) & (manifest_df['box_count'] > 0)].copy()
    num_positive = len(positive_df)
    total_boxes = positive_df['box_count'].sum()
    print(f"  ✓ Found {num_positive} positive images with {total_boxes} license plate bounding boxes.")

    # 2. Select Clean Negative / Background Samples (~10% ratio)
    print("\n[2/6] Selecting clean background images from skipped crops...")
    # Explicit empty annotated images (0 boxes)
    explicit_empty_df = manifest_df[(manifest_df['is_annotated']) & (manifest_df['box_count'] == 0)].copy()
    skipped_df = manifest_df[~manifest_df['is_annotated']].copy()

    # Calculate target total background count
    target_bg_total = int(num_positive * BG_RATIO)
    needed_from_skipped = max(0, target_bg_total - len(explicit_empty_df))
    print(f"  * Target background images: {target_bg_total}")
    print(f"  * Explicit empty annotations: {len(explicit_empty_df)}")
    print(f"  * Needed from skipped images: {needed_from_skipped} (from {len(skipped_df)} candidates)")

    # Filter out extreme closeups, tiny crops, and abnormal aspect ratios
    # Vehicles that are reasonable distance and well-proportioned
    clean_candidates = skipped_df[
        (skipped_df['crop_w'] >= 600) & (skipped_df['crop_w'] <= 1350) &
        (skipped_df['crop_h'] >= 900) & (skipped_df['crop_h'] <= 1700) &
        (skipped_df['box_w'] >= 180) & (skipped_df['box_w'] <= 850) &
        (skipped_df['box_h'] >= 350) & (skipped_df['box_h'] <= 1050)
    ].copy()

    print(f"  ✓ Filtered {len(skipped_df) - len(clean_candidates)} extreme closeups/garbage.")
    print(f"  ✓ {len(clean_candidates)} clean candidates available.")

    # Rank by image sharpness
    print("  * Calculating sharpness (Laplacian variance) for clean candidates...")
    sharpness_scores = []
    for _, row in clean_candidates.iterrows():
        img_path = os.path.join(CROPS_DIR, row['filename'])
        sharpness_scores.append(calculate_sharpness(img_path))
    clean_candidates['sharpness'] = sharpness_scores

    # Filter out blurry candidates (top 80% sharpness)
    sharp_cutoff = clean_candidates['sharpness'].quantile(0.20)
    sharp_candidates = clean_candidates[clean_candidates['sharpness'] >= sharp_cutoff].copy()

    # Group by vehicle track to ensure diverse sampling
    sharp_candidates = sharp_candidates.sample(frac=1.0, random_state=RANDOM_SEED)  # shuffle
    # Pick at most 2 background crops per track
    diverse_bg = sharp_candidates.groupby(['video_source', 'track_id']).head(2)

    if len(diverse_bg) >= needed_from_skipped:
        selected_skipped = diverse_bg.sample(n=needed_from_skipped, random_state=RANDOM_SEED)
    else:
        selected_skipped = sharp_candidates.head(needed_from_skipped)

    bg_df = pd.concat([explicit_empty_df, selected_skipped], ignore_index=True)
    num_bg = len(bg_df)
    print(f"  ✓ Selected {num_bg} high-quality background images ({len(explicit_empty_df)} empty-labeled + {len(selected_skipped)} skipped).")

    # 3. Track-Aware Grouped Train/Val Split (80/20)
    print("\n[3/6] Performing track-aware Train/Val split (80/20)...")
    positive_df['is_background'] = 0
    positive_df['track_key'] = positive_df['video_source'] + "_trk" + positive_df['track_id'].astype(str)

    bg_df['is_background'] = 1
    bg_df['track_key'] = bg_df['video_source'] + "_trk" + bg_df['track_id'].astype(str)

    combined_df = pd.concat([positive_df, bg_df], ignore_index=True)

    # Split tracks to prevent data leakage
    # We split positive tracks and background tracks independently to balance both sets
    def split_tracks(df_subset, split_ratio, seed):
        tracks = df_subset['track_key'].unique()
        rng = np.random.RandomState(seed)
        rng.shuffle(tracks)
        
        # Target item count
        target_train_items = int(len(df_subset) * split_ratio)
        train_tracks = set()
        val_tracks = set()
        current_train_items = 0

        # Track to item count
        track_counts = df_subset.groupby('track_key').size().to_dict()

        for t in tracks:
            count = track_counts[t]
            if current_train_items + count <= target_train_items or len(train_tracks) == 0:
                train_tracks.add(t)
                current_train_items += count
            else:
                val_tracks.add(t)

        return train_tracks, val_tracks

    pos_train_trk, pos_val_trk = split_tracks(positive_df, SPLIT_RATIO, RANDOM_SEED)
    bg_train_trk, bg_val_trk = split_tracks(bg_df, SPLIT_RATIO, RANDOM_SEED + 1)

    all_train_tracks = pos_train_trk.union(bg_train_trk)
    combined_df['split'] = combined_df['track_key'].apply(lambda x: 'train' if x in all_train_tracks else 'val')

    train_df = combined_df[combined_df['split'] == 'train']
    val_df = combined_df[combined_df['split'] == 'val']

    print(f"  ✓ Train Set: {len(train_df)} images ({len(train_df[train_df['is_background'] == 0])} pos, {len(train_df[train_df['is_background'] == 1])} bg, {train_df['box_count'].sum()} boxes)")
    print(f"  ✓ Val Set:   {len(val_df)} images ({len(val_df[val_df['is_background'] == 0])} pos, {len(val_df[val_df['is_background'] == 1])} bg, {val_df['box_count'].sum()} boxes)")
    print(f"  ✓ Actual Train Split: {len(train_df) / len(combined_df) * 100:.1f}%")

    # 4. Create directory structure and copy files
    print("\n[4/6] Creating YOLO directory structure and copying files...")
    dirs_to_create = [
        os.path.join(DATASET_OUT_DIR, "images", "train"),
        os.path.join(DATASET_OUT_DIR, "images", "val"),
        os.path.join(DATASET_OUT_DIR, "labels", "train"),
        os.path.join(DATASET_OUT_DIR, "labels", "val")
    ]
    # Clean previous output if exists
    if os.path.exists(DATASET_OUT_DIR):
        shutil.rmtree(DATASET_OUT_DIR)

    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

    copied_images = 0
    written_labels = 0

    for idx, row in combined_df.iterrows():
        split = row['split']
        img_name = row['filename']
        stem = row['stem']
        src_img_path = os.path.join(CROPS_DIR, img_name)
        dst_img_path = os.path.join(DATASET_OUT_DIR, "images", split, img_name)

        # Copy image
        shutil.copy2(src_img_path, dst_img_path)
        copied_images += 1

        # Copy/Create label file
        dst_lbl_path = os.path.join(DATASET_OUT_DIR, "labels", split, f"{stem}.txt")
        if row['is_background'] == 1 or row['box_count'] == 0:
            # Empty label file for background sample
            with open(dst_lbl_path, 'w', encoding='utf-8') as f:
                pass
        else:
            # Positive sample: copy bounding boxes
            src_lbl_path = label_stem_map[stem]
            shutil.copy2(src_lbl_path, dst_lbl_path)

        written_labels += 1

    print(f"  ✓ Successfully copied {copied_images} images and {written_labels} label files.")

    # 5. Generate data.yaml, README.md, metadata.csv
    print("\n[5/6] Generating configuration files and documentation...")

    # data.yaml
    data_yaml_content = {
        'path': '.',
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'license_plate'
        }
    }
    yaml_path = os.path.join(DATASET_OUT_DIR, "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml_content, f, sort_keys=False, default_flow_style=False)
    print(f"  ✓ Created: {yaml_path}")

    # metadata.csv
    meta_path = os.path.join(DATASET_OUT_DIR, "metadata.csv")
    combined_df[['filename', 'split', 'vehicle_type', 'track_id', 'video_source', 'is_background', 'box_count', 'crop_w', 'crop_h']].to_csv(meta_path, index=False)
    print(f"  ✓ Created: {meta_path}")

    # README.md
    readme_path = os.path.join(DATASET_OUT_DIR, "README.md")
    readme_content = f"""# Vehicle License Plate Detection Dataset (Kaggle Edition)

A high-resolution, traffic-monitoring vehicle license plate detection dataset curated for real-world traffic surveillance and violation enforcement.

## 📊 Dataset Summary

- **Total Images**: {len(combined_df):,}
- **Train Set**: {len(train_df):,} images ({len(train_df[train_df['is_background'] == 0]):,} positive + {len(train_df[train_df['is_background'] == 1]):,} clean background)
- **Validation Set**: {len(val_df):,} images ({len(val_df[val_df['is_background'] == 0]):,} positive + {len(val_df[val_df['is_background'] == 1]):,} clean background)
- **Total License Plate Bounding Boxes**: {total_boxes:,}
- **Classes**: 1 class (`0: license_plate`)
- **Background Negative Ratio**: ~{len(bg_df) / len(combined_df) * 100:.1f}% (standard YOLO best practice to minimize false positives)
- **Vehicle Diversity**: Motorcycles, Cars, Rickshaws, Trucks
- **Data Leakage Prevention**: Split is strictly grouped by vehicle tracking ID so that all crops of a specific vehicle belong entirely to either Train or Val.

---

## 📁 Directory Structure

```
license_plate_yolo_dataset/
├── data.yaml
├── README.md
├── metadata.csv
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## 🚀 Quickstart Training on Kaggle

You can train any Ultralytics YOLO model (YOLOv8, YOLOv9, YOLOv10, YOLOv11) directly in Kaggle:

```python
!pip install ultralytics

from ultralytics import YOLO

# Load model (e.g. YOLOv8s or YOLOv11s)
model = YOLO('yolov8s.pt')

# Train model
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    plots=True
)

# Validate model
metrics = model.val()
print(f"mAP50: {{metrics.box.map50:.4f}}")
print(f"mAP50-95: {{metrics.box.map:.4f}}")
```

---

## 🏷️ Annotation Format

Annotations follow the standard normalized YOLO format:
`<class_id> <x_center> <y_center> <width> <height>`

Coordinates are normalized between `0.0` and `1.0`.
Background/negative images have empty `.txt` label files.
"""
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"  ✓ Created: {readme_path}")

    # 6. Create Kaggle-ready ZIP archive
    print("\n[6/6] Packaging into Kaggle-ready ZIP archive...")
    if os.path.exists(ZIP_OUT_PATH):
        os.remove(ZIP_OUT_PATH)

    with zipfile.ZipFile(ZIP_OUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(DATASET_OUT_DIR):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, DATASET_OUT_DIR)
                zf.write(abs_path, rel_path)

    zip_size_mb = os.path.getsize(ZIP_OUT_PATH) / (1024 * 1024)
    print(f"  ✓ Created ZIP archive: {ZIP_OUT_PATH} ({zip_size_mb:.2f} MB)")

    print("\n" + "=" * 75)
    print("🎉 DATASET CREATION COMPLETED SUCCESSFULLY!")
    print(f"📦 Output ZIP: {ZIP_OUT_PATH}")
    print(f"📁 Extracted Dataset: {DATASET_OUT_DIR}")
    print("=" * 75)

if __name__ == "__main__":
    main()
