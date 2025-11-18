from ultralytics import YOLO
import cv2
import os
import csv

# -------------------------
# PATHS
# -------------------------
MODEL_PATH = "../models/seatbelt.pt"
BASE_DIR = "../output/cropped_photos"
OUTPUT_DIR = "../output/seatbelt_results"
CSV_PATH = "../output/seatbelt_results.csv"

CONF_THRES = 0.35  # confidence threshold

# -------------------------
# SETUP
# -------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

csv_file = open(CSV_PATH, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["vehicle_id", "image_name", "label", "confidence", "x1", "y1", "x2", "y2"])

# -------------------------
# MAIN LOOP - ONLY PROCESS CARS FOLDER
# -------------------------
print("🚀 Starting seatbelt detection...")
print(f"Base directory: {BASE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print("💡 Only processing Cars folder (bikes/motorcycles don't have seatbelts)")

# ONLY process Cars folder (vehicles that actually have seatbelts)
category = "Cars"
category_path = os.path.join(BASE_DIR, category)

if not os.path.exists(category_path):
    print(f"❌ Cars folder not found: {category_path}")
    exit()

print(f"\n📁 Processing category: {category}")

# Get all ID folders in the Cars category
id_folders = [f for f in os.listdir(category_path) 
              if os.path.isdir(os.path.join(category_path, f)) and f.startswith("id_")]

if not id_folders:
    print(f"  No ID folders found in {category}")
    exit()

total_processed = 0
total_detections = 0
category_processed = 0
category_detections = 0

for id_folder in id_folders:
    folder_path = os.path.join(category_path, id_folder)
    
    # Extract vehicle ID from folder name (id_001 → 1)
    try:
        vehicle_id = int(id_folder.replace("id_", ""))
    except:
        vehicle_id = id_folder  # fallback to original name
    
    # Process ALL images in the ID folder
    images_processed = 0
    images_with_detections = 0
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"  ⚠ Could not load image: {id_folder}/{filename}")
            continue

        # Run seatbelt detection
        results = model.predict(img, conf=CONF_THRES, verbose=False)
        detections = results[0].boxes

        annotated = img.copy()
        has_detections = False

        # If no detections → write no_detection
        if detections is None or len(detections) == 0:
            writer.writerow([vehicle_id, filename, "no_detection", 0, None, None, None, None])
        else:
            # Process every detected box
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                # Save CSV entry
                writer.writerow([vehicle_id, filename, label, conf, x1, y1, x2, y2])
                
                # Draw annotation
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                
                has_detections = True
                category_detections += 1
                total_detections += 1

        # Save annotated image (only if we have detections to save space)
        if has_detections:
            # Create organized output structure
            category_out_dir = os.path.join(OUTPUT_DIR, category)
            id_out_dir = os.path.join(category_out_dir, id_folder)
            os.makedirs(id_out_dir, exist_ok=True)
            
            out_path = os.path.join(id_out_dir, f"detected_{filename}")
            cv2.imwrite(out_path, annotated)
            images_with_detections += 1

        images_processed += 1
        total_processed += 1
        category_processed += 1

    # Progress for this vehicle
    if images_processed > 0:
        detection_rate = (images_with_detections / images_processed) * 100
        print(f"  ✅ {id_folder}: {images_processed} images, {images_with_detections} with detections ({detection_rate:.1f}%)")

csv_file.close()

print(f"\n{'='*50}")
print("✅ SEATBELT DETECTION COMPLETE!")
print(f"{'='*50}")
print(f"Category processed: {category}")
print(f"Total images processed: {total_processed}")
print(f"Total detections found: {total_detections}")
print(f"CSV saved at: {CSV_PATH}")
print(f"Annotated images saved in: {OUTPUT_DIR}")

print(f"\n📁 Output structure:")
print(f"{OUTPUT_DIR}/")
print("└── Cars/")
print("    ├── id_001/")
print("    │   ├── detected_frame000123_zone0_car_right.jpg")
print("    │   └── ...")
print("    ├── id_002/")
print("    └── ...")

print(f"\n💡 Only processed Cars folder (bikes/motorcycles don't have seatbelts)")
print(f"💡 Only images with detections are saved to save space.")