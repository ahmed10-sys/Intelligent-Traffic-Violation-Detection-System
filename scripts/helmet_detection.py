from ultralytics import YOLO
import cv2
import os
import csv

# -------------------------
# PATHS
# -------------------------
MODEL_PATH = "../models/helmet.pt"
BASE_DIR = "../output/cropped_photos"
OUTPUT_DIR = "../output/helmet_results"
CSV_PATH = "../output/helmet_results.csv"

CONF_THRES = 0.35  # confidence threshold

# -------------------------
# SETUP
# -------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load pre-trained helmet detection model
print("🚀 Loading helmet detection model...")
model = YOLO(MODEL_PATH)

csv_file = open(CSV_PATH, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["vehicle_id", "image_name", "helmet_status", "confidence", "x1", "y1", "x2", "y2"])

# -------------------------
# MAIN LOOP - ONLY PROCESS Bikes_Motorcycles FOLDER
# -------------------------
print("🔍 Starting helmet detection...")
print("💡 Only processing Bikes_Motorcycles folder")

category = "Bikes_Motorcycles"
category_path = os.path.join(BASE_DIR, category)

if not os.path.exists(category_path):
    print(f"❌ Bikes_Motorcycles folder not found: {category_path}")
    exit()

print(f"\n📁 Processing category: {category}")

# Get all ID folders in Bikes_Motorcycles
id_folders = [f for f in os.listdir(category_path) 
              if os.path.isdir(os.path.join(category_path, f)) and f.startswith("id_")]

if not id_folders:
    print(f"  No ID folders found in {category}")
    exit()

total_processed = 0
total_violations = 0  # No helmet detected

for id_folder in id_folders:
    folder_path = os.path.join(category_path, id_folder)
    
    # Extract vehicle ID
    try:
        vehicle_id = int(id_folder.replace("id_", ""))
    except:
        vehicle_id = id_folder
    
    # Process ALL images in the ID folder
    images_processed = 0
    helmet_detected_count = 0
    no_helmet_count = 0
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"  ⚠ Could not load image: {id_folder}/{filename}")
            continue

        # Run helmet detection
        results = model.predict(img, conf=CONF_THRES, verbose=False)
        detections = results[0].boxes

        annotated = img.copy()
        helmet_detected = False
        best_confidence = 0

        if detections is not None and len(detections) > 0:
            # Process detections
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]  # "with_helmet" or "without_helmet"
                
                # Track best detection
                if conf > best_confidence:
                    best_confidence = conf
                    helmet_detected = (label == "with_helmet")
                
                # Draw annotations
                color = (0, 255, 0) if label == "with_helmet" else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
            
            # Save CSV entry for best detection
            status = "with_helmet" if helmet_detected else "without_helmet"
            writer.writerow([vehicle_id, filename, status, best_confidence, x1, y1, x2, y2])
            
            if helmet_detected:
                helmet_detected_count += 1
            else:
                no_helmet_count += 1
                total_violations += 1
        else:
            # No detection - assume no helmet (violation)
            writer.writerow([vehicle_id, filename, "no_detection", 0, None, None, None, None])
            no_helmet_count += 1
            total_violations += 1

        # Save annotated image if violation or for documentation
        if not helmet_detected or helmet_detected_count == 0:  # Save if violation or first helmet
            category_out_dir = os.path.join(OUTPUT_DIR, category)
            id_out_dir = os.path.join(category_out_dir, id_folder)
            os.makedirs(id_out_dir, exist_ok=True)
            
            out_path = os.path.join(id_out_dir, f"helmet_{filename}")
            cv2.imwrite(out_path, annotated)

        images_processed += 1
        total_processed += 1

    # Progress for this vehicle
    if images_processed > 0:
        helmet_rate = (helmet_detected_count / images_processed) * 100
        print(f"  ✅ {id_folder}: {images_processed} images | 🪖 With helmet: {helmet_detected_count} | 🚨 No helmet: {no_helmet_count} ({helmet_rate:.1f}% compliance)")

csv_file.close()

print(f"\n{'='*50}")
print("✅ HELMET DETECTION COMPLETE!")
print(f"{'='*50}")
print(f"Total images processed: {total_processed}")
print(f"Total helmet violations: {total_violations}")
print(f"CSV saved at: {CSV_PATH}")
print(f"Annotated images saved in: {OUTPUT_DIR}")