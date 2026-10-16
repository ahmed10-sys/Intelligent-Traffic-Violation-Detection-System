"""
Pre-label License Plate Dataset for Label Studio
Runs the existing YOLO plate detection model (yolov8s_lpd.pt) on cropped vehicle images,
generates normalized YOLO-format prediction .txt files in a separate output directory,
and creates a Label Studio import JSON matching import.py format.
"""

import os
import sys
import glob
import json
import argparse
import cv2
from ultralytics import YOLO

# Fix Windows console encoding for Unicode/emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

# ----------------------------------------------------------------------
# 🗂️ DEFAULT CONFIGURATION
# ----------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "Models", "yolov8s_lpd.pt")
if not os.path.exists(DEFAULT_MODEL_PATH):
    DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8s_lpd.pt")

DEFAULT_IMAGES_DIR = os.path.join(PROJECT_ROOT, "output", "plate_dataset_crops")
DEFAULT_LABELS_DIR = os.path.join(PROJECT_ROOT, "output", "plate_dataset_yolo_labels")
DEFAULT_JSON_OUTPUT = os.path.join(PROJECT_ROOT, "output", "label_studio_plate_import.json")
DEFAULT_ZERO_DETECTIONS_OUTPUT = os.path.join(PROJECT_ROOT, "output", "plate_dataset_zero_detections.txt")

# Label Studio Local Files URL prefix pattern (matching import.py: /data/local-files/?d=<Folder>%5C)
DEFAULT_URL_PREFIX = "/data/local-files/?d=plate_dataset_crops%5C"
CONFIDENCE_THRESHOLD = 0.35
LABEL_NAME = "license_plate"
CLASS_ID = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-label cropped images with existing license plate model.")
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR,
                        help="Path to the directory containing cropped vehicle images.")
    parser.add_argument("--labels_dir", type=str, default=DEFAULT_LABELS_DIR,
                        help="NEW separate directory where YOLO .txt prediction files will be saved.")
    parser.add_argument("--json_output", type=str, default=DEFAULT_JSON_OUTPUT,
                        help="Path to the output Label Studio import JSON file.")
    parser.add_argument("--zero_output", type=str, default=DEFAULT_ZERO_DETECTIONS_OUTPUT,
                        help="Path to text file where zero-detection image filenames will be saved.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to YOLO license plate detection model weights.")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                        help="Confidence threshold for keeping plate detections (default: 0.35).")
    parser.add_argument("--prefix", type=str, default=DEFAULT_URL_PREFIX,
                        help="Label Studio storage URL prefix (e.g. /data/local-files/?d=Folder%%5C).")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("🏷️  PRE-LABELING LICENSE PLATE DATASET FOR LABEL STUDIO")
    print("=" * 70)
    print(f"📁 Images Source:       {args.images_dir}")
    print(f"📁 YOLO Labels Output:  {args.labels_dir}")
    print(f"📄 Label Studio JSON:   {args.json_output}")
    print(f"🧠 Plate Model:         {args.model_path}")
    print(f"🎯 Confidence Cutoff:   {args.conf}")
    print(f"🔗 URL Prefix:          {args.prefix}")
    print("=" * 70)

    if not os.path.isdir(args.images_dir):
        print(f"❌ Error: Images directory does not exist: {args.images_dir}")
        return

    if not os.path.isfile(args.model_path):
        print(f"❌ Error: Plate model not found at: {args.model_path}")
        return

    # Create separate output directories without touching images folder
    os.makedirs(args.labels_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json_output)), exist_ok=True)

    # Load YOLO plate detection model
    print("🔁 Loading existing license plate detection model...")
    model = YOLO(args.model_path)
    print("✅ Model loaded successfully.")

    # Find candidate images
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = sorted([
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith(valid_exts)
    ])

    total_images = len(image_files)
    if total_images == 0:
        print(f"❌ No image files found in: {args.images_dir}")
        return

    print(f"🔍 Found {total_images} cropped vehicle images to process.\n")

    images_with_predictions = 0
    total_plate_boxes = 0
    zero_detection_images = []
    tasks = []

    for idx, filename in enumerate(image_files, 1):
        img_path = os.path.join(args.images_dir, filename)

        # Run model prediction
        try:
            results = model.predict(img_path, conf=args.conf, verbose=False)
        except Exception as e:
            print(f"⚠️ Error running prediction on {filename}: {e}")
            zero_detection_images.append(filename)
            continue

        detected_boxes_yolo = []  # List of (class_id, x_center, y_center, w, h, conf)
        predictions_label_studio = []

        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            xywhn_boxes = boxes.xywhn.cpu().numpy()

            for xywhn, conf in zip(xywhn_boxes, confs):
                if conf < args.conf:
                    continue

                xc, yc, w, h = map(float, xywhn)

                # Ensure values are within [0, 1]
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                detected_boxes_yolo.append((CLASS_ID, xc, yc, w, h))

                # Label Studio percentage coordinates (0-100) matching import.py
                # Top-left x and y as percentages
                ls_x = max(0.0, min(100.0, (xc - w / 2.0) * 100.0))
                ls_y = max(0.0, min(100.0, (yc - h / 2.0) * 100.0))
                ls_w = max(0.0, min(100.0, w * 100.0))
                ls_h = max(0.0, min(100.0, h * 100.0))

                predictions_label_studio.append({
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [LABEL_NAME],
                        "x": ls_x,
                        "y": ls_y,
                        "width": ls_w,
                        "height": ls_h
                    }
                })

        # Save YOLO-format .txt prediction file if plates detected
        if len(detected_boxes_yolo) > 0:
            images_with_predictions += 1
            total_plate_boxes += len(detected_boxes_yolo)

            stem = os.path.splitext(filename)[0]
            txt_path = os.path.join(args.labels_dir, f"{stem}.txt")

            with open(txt_path, "w", encoding="utf-8") as f:
                for cls_id, xc, yc, w, h in detected_boxes_yolo:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        else:
            zero_detection_images.append(filename)

        # Build Label Studio task structure (matching import.py exactly)
        task_entry = {
            "data": {
                "image": args.prefix + filename
            },
            "predictions": [{
                "model_version": "v1",
                "result": predictions_label_studio
            }]
        }
        tasks.append(task_entry)

        # Progress reporting
        if idx % 250 == 0 or idx == total_images:
            pct = (idx / total_images) * 100.0
            print(f"  ⏳ Processed {idx}/{total_images} ({pct:.1f}%) | "
                  f"With Plates: {images_with_predictions} | Boxes: {total_plate_boxes} | "
                  f"Zero Detections: {len(zero_detection_images)}")

    # Write Label Studio JSON file
    print("\n💾 Writing Label Studio JSON import file...")
    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

    # Write zero detection filenames
    with open(args.zero_output, "w", encoding="utf-8") as f:
        for fname in zero_detection_images:
            f.write(f"{fname}\n")

    zero_count = len(zero_detection_images)
    detection_rate = (images_with_predictions / total_images * 100.0) if total_images > 0 else 0.0

    print("\n" + "=" * 70)
    print("🎉 PRE-LABELING SUMMARY")
    print("=" * 70)
    print(f"📊 Total images processed:            {total_images}")
    print(f"✅ Images with at least 1 plate:      {images_with_predictions} ({detection_rate:.1f}%)")
    print(f"📦 Total plate bounding boxes found:  {total_plate_boxes}")
    print(f"❌ Images with zero detections:       {zero_count} ({100.0 - detection_rate:.1f}%)")
    print(f"📁 YOLO .txt labels directory:        {args.labels_dir}")
    print(f"📄 Label Studio import JSON:          {args.json_output}")
    print(f"📝 Zero-detection filename list:      {args.zero_output}")
    print("=" * 70)

    # Display preview of zero-detection images
    if zero_count > 0:
        print(f"\n🔍 Spot-check sample of zero-detection images (first 15 of {zero_count}):")
        for sample_fname in zero_detection_images[:15]:
            print(f"   • {sample_fname}")
        if zero_count > 15:
            print(f"   ... and {zero_count - 15} more listed in {args.zero_output}")
    else:
        print("\n🎉 All processed images had at least one plate detected!")


if __name__ == "__main__":
    main()
