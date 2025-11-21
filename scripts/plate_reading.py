import os
import csv
import re
import cv2
import numpy as np
import easyocr

# -------------------------
# PATHS AND SETTINGS
# -------------------------
BASE_DIR = "../output/cropped_photos"
OUTPUT_DIR = "../output/license_plate_results"
CSV_PATH = os.path.join(OUTPUT_DIR, "license_plate_results.csv")

BEST_FILENAME = "BEST.jpg"
MIN_CONFIDENCE = 0.35
MIN_PLATE_LENGTH = 4

# By default, license plates are most relevant for Cars/Trucks,
# but you can include Bikes_Motorcycles if needed.
CATEGORIES = ["Cars"]


# -------------------------
# HELPERS
# -------------------------
def normalize_plate_text(text: str) -> str:
    """Strip spaces/punctuation and uppercase common plate characters."""
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def annotate_detection(image: np.ndarray, bbox, label: str, confidence: float) -> np.ndarray:
    """Draw detection polygon plus label."""
    annotated = image.copy()
    points = np.array(bbox, dtype=int)

    for i in range(len(points)):
        start_pt = tuple(points[i])
        end_pt = tuple(points[(i + 1) % len(points)])
        cv2.line(annotated, start_pt, end_pt, (0, 255, 0), 2)

    x, y = points[0]
    cv2.putText(
        annotated,
        f"{label} ({confidence:.2f})",
        (x, max(y - 10, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return annotated


# -------------------------
# MAIN PROCESSING LOGIC
# -------------------------
def main():
    print("🚦 Starting license plate reading on BEST.jpg crops...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reader = easyocr.Reader(["en"], gpu=False)

    with open(CSV_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "vehicle_id",
            "category",
            "image_name",
            "raw_text",
            "normalized_plate",
            "confidence",
            "image_path",
            "annotated_image_path",
        ])

        total_scanned = 0
        total_plates = 0

        for category in CATEGORIES:
            category_dir = os.path.join(BASE_DIR, category)
            if not os.path.exists(category_dir):
                print(f"⚠️  Category folder not found: {category_dir}")
                continue

            print(f"\n📁 Processing category: {category}")
            id_folders = [
                f
                for f in sorted(os.listdir(category_dir))
                if os.path.isdir(os.path.join(category_dir, f)) and f.startswith("id_")
            ]

            if not id_folders:
                print(f"  No ID folders found in {category}")
                continue

            for id_folder in id_folders:
                vehicle_id = id_folder.replace("id_", "")
                best_image_path = os.path.join(category_dir, id_folder, BEST_FILENAME)

                if not os.path.isfile(best_image_path):
                    continue

                image = cv2.imread(best_image_path)
                if image is None:
                    print(f"  ⚠️  Could not read image: {best_image_path}")
                    continue

                results = reader.readtext(best_image_path, detail=1)
                best_detection = None

                for bbox, text, confidence in results:
                    normalized = normalize_plate_text(text)
                    if len(normalized) < MIN_PLATE_LENGTH:
                        continue
                    if confidence < MIN_CONFIDENCE:
                        continue

                    if best_detection is None or confidence > best_detection["confidence"]:
                        best_detection = {
                            "bbox": bbox,
                            "raw_text": text,
                            "normalized": normalized,
                            "confidence": confidence,
                        }

                total_scanned += 1

                if best_detection:
                    total_plates += 1
                    annotated = annotate_detection(
                        image,
                        best_detection["bbox"],
                        best_detection["normalized"],
                        best_detection["confidence"],
                    )

                    out_category_dir = os.path.join(OUTPUT_DIR, category, id_folder)
                    os.makedirs(out_category_dir, exist_ok=True)
                    annotated_name = f"plate_{best_detection['normalized']}.jpg"
                    annotated_path = os.path.join(out_category_dir, annotated_name)
                    cv2.imwrite(annotated_path, annotated)

                    writer.writerow([
                        vehicle_id,
                        category,
                        BEST_FILENAME,
                        best_detection["raw_text"],
                        best_detection["normalized"],
                        f"{best_detection['confidence']:.4f}",
                        best_image_path,
                        annotated_path,
                    ])

                    print(
                        f"  ✅ ID {vehicle_id}: plate {best_detection['normalized']} "
                        f"(conf {best_detection['confidence']:.2f})"
                    )
                else:
                    writer.writerow([
                        vehicle_id,
                        category,
                        BEST_FILENAME,
                        "",
                        "",
                        "",
                        best_image_path,
                        "",
                    ])
                    print(f"  ❌ ID {vehicle_id}: no plate detected")

        print("\n" + "=" * 50)
        print("📄 License plate reading complete")
        print("=" * 50)
        print(f"Vehicles scanned: {total_scanned}")
        print(f"Plates detected: {total_plates}")
        print(f"CSV saved at: {CSV_PATH}")
        print(f"Annotated images saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

