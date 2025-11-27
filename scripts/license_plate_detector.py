import os
import cv2
import math
import logging
from ultralytics import YOLO

# ----------------------------------------------------------------------
# 🗂️ CONFIGURATION
# ----------------------------------------------------------------------
BASE_DIR = "../output/cropped_photos"
LPD_MODEL_PATH = "../models/yolov8s_lpd.pt" 

CONFIDENCE_THRESHOLD = 0.40
CENTER_TOLERANCE = 0.40      # 40% - Max distance from center allowed (0.0=center, 0.5=edge)
SAVE_ALL_PLATES = False      

# ----------------------------------------------------------------------
# ⚙️ HELPER: CENTER BIAS CALCULATION
# ----------------------------------------------------------------------
def is_centered(box, img_width, img_height, tolerance):
    """
    Returns True if the box center is within 'tolerance' % of the image center.
    """
    x1, y1, x2, y2 = box
    box_cx = (x1 + x2) / 2
    box_cy = (y1 + y2) / 2
    
    img_cx = img_width / 2
    img_cy = img_height / 2
    
    # Calculate normalized distance from center (0 to 1)
    # 1.0 would mean the box is at the very corner of the image
    dist_x = abs(box_cx - img_cx) / img_width
    dist_y = abs(box_cy - img_cy) / img_height
    
    # Euclidean distance normalized
    distance = math.sqrt(dist_x**2 + dist_y**2)
    
    return distance < tolerance

# ----------------------------------------------------------------------
# ⚙️ MAIN SCRIPT
# ----------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print(f"🚀 Starting Smart License Plate Extraction...")
    print(f"   (Filtering out background plates based on position)")
    
    try:
        model = YOLO(LPD_MODEL_PATH)
        print(f"✅ Loaded LPD Model")
    except Exception as e:
        print(f"🚨 Error loading model: {e}")
        return

    total_vehicles = 0
    total_plates_found = 0

    for category in os.listdir(BASE_DIR):
        cat_path = os.path.join(BASE_DIR, category)
        if not os.path.isdir(cat_path): continue

        print(f"\n--- Category: {category} ---")

        for vehicle_id in sorted(os.listdir(cat_path)):
            vehicle_path = os.path.join(cat_path, vehicle_id)
            if not os.path.isdir(vehicle_path): continue

            total_vehicles += 1
            
            # Gather candidate images
            images = [f for f in os.listdir(vehicle_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                      and "plate" not in f.lower() and "best" not in f.lower()]

            if not images: continue

            best_conf_for_id = -1
            best_plate_crop = None
            best_source_filename = ""

            for img_file in images:
                img_path = os.path.join(vehicle_path, img_file)
                frame_img = cv2.imread(img_path)
                if frame_img is None: continue
                
                h, w = frame_img.shape[:2]

                # Run Inference
                results = model.predict(frame_img, conf=CONFIDENCE_THRESHOLD, verbose=False)
                if not results: continue

                for result in results:
                    for box in result.boxes:
                        # 1. Check Confidence
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        
                        # 2. Check Position (Must be somewhat central)
                        if not is_centered(xyxy, w, h, CENTER_TOLERANCE):
                            # Skip this plate, it's likely a background car
                            continue
                            
                        # 3. If it's central AND better than previous best, keep it
                        if conf > best_conf_for_id:
                            best_conf_for_id = conf
                            best_source_filename = img_file
                            
                            x1, y1, x2, y2 = map(int, xyxy)
                            # Add padding
                            x1, y1 = max(0, x1 - 5), max(0, y1 - 5)
                            x2, y2 = min(w, x2 + 5), min(h, y2 + 5)
                            
                            best_plate_crop = frame_img[y1:y2, x1:x2]

            # Save the winner
            if best_plate_crop is not None:
                output_path = os.path.join(vehicle_path, "BEST_PLATE_CROP.jpg")
                cv2.imwrite(output_path, best_plate_crop)
                total_plates_found += 1
                print(f"  ✅ ID {vehicle_id}: Found plate (Conf: {best_conf_for_id:.2f})")
            else:
                print(f"  ❌ ID {vehicle_id}: No valid central plate found.")

    print("\n" + "="*50)
    print(f"🎉 Processed {total_vehicles} vehicles. Found {total_plates_found} plates.")

if __name__ == "__main__":
    main()