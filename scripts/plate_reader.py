import os
import sys
import cv2
import csv
import re
import logging
import numpy as np
from paddleocr import PaddleOCR

# Fix Windows console encoding for Unicode/emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ==========================================
# 🗂️ CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../output/cropped_photos"))
OUTPUT_CSV = os.path.normpath(os.path.join(SCRIPT_DIR, "../output/master_violation_log.csv"))
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# 🛠️ DEBUG TOGGLE: Save processed images named after detected text
SAVE_DEBUG_IMAGES = False
DEBUG_OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../output/debug_plates_check"))

# Disable Paddle Logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

# ==========================================
# 🧠 PAKISTANI LOGIC LAYER
# ==========================================
def apply_pakistan_correction(text):
    if not isinstance(text, str) or not text:
        return ""

    # Rule 0: Remove Border Noise
    if len(text) > 6 and text[-1] in ['1', 'I', '|', 'l', ']', ')']:
        text = text[:-1]
    
    # Rule 1: Clean (Keep Hyphens for splitting, then remove later)
    text = re.sub(r'[^A-Z0-9-]', '', text.upper())

    # Rule 2: Split Alpha/Numeric
    if "-" in text:
        parts = text.split('-')
        alpha_part = parts[0]
        num_part = parts[1] if len(parts) > 1 else ""
    else:
        # Smart split based on first digit
        match = re.search(r'\d', text)
        if match:
            idx = match.start()
            alpha_part = text[:idx]
            num_part = text[idx:]
        else:
            alpha_part, num_part = text, ""

    # Rule 3: Character Swapping
    dict_to_letters = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B', '6': 'G'}
    dict_to_numbers = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'A': '4', 'Z': '7', 'G': '6', 'D': '0', 'Q': '0'} 

    new_alpha = "".join([dict_to_letters.get(c, c) for c in alpha_part])
    new_num = "".join([dict_to_numbers.get(c, c) for c in num_part])

    final_text = new_alpha + new_num
    
    # FINAL CLEANUP: Remove any remaining hyphens for clean DB storage
    return final_text.replace('-', '')

# ==========================================
# 🖼️ PREPROCESSING & EMBLEM REMOVAL
# ==========================================
def remove_red_emblem(img):
    """
    Automatically detects and inpaints the red/maroon provincial emblem 
    located in the top-left section of Pakistani plates (e.g. motorcycles, rickshaws).
    """
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red hue spans 0-18 and 160-180 in OpenCV HSV
    lower_red1 = np.array([0, 35, 35])
    upper_red1 = np.array([18, 255, 255])
    lower_red2 = np.array([160, 35, 35])
    upper_red2 = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    # The emblem is strictly in top-left (left 35% width, top 55% height)
    mask[:, int(w * 0.35):] = 0
    mask[int(h * 0.55):, :] = 0
    
    # Paint over red pixels with the plate background color
    if np.any(mask):
        bg_color = [int(x) for x in np.median(img[int(h * 0.6):, :], axis=(0, 1))]
        cleaned = img.copy()
        cleaned[mask > 0] = bg_color
        return cleaned
    return img

def preprocess_for_rec(img):
    if img is None: return None
    # 1. Strip red emblem artifact so it doesn't get read as a letter
    img = remove_red_emblem(img)
    # 2. 2x upscale for clarity
    scale = 200 
    w = int(img.shape[1] * scale / 100)
    h = int(img.shape[0] * scale / 100)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return img

# ==========================================
# 💾 DEBUG FUNCTION (New)
# ==========================================
def save_verification_image(img, text, v_id):
    """
    Saves the processed image with the detected text as the filename.
    Format: [PlateNumber]_[VehicleID].jpg
    """
    if not SAVE_DEBUG_IMAGES or img is None:
        return

    if not os.path.exists(DEBUG_OUTPUT_DIR):
        os.makedirs(DEBUG_OUTPUT_DIR)
        print(f"📂 Created Debug Folder: {DEBUG_OUTPUT_DIR}")

    # Clean text for filename (remove illegal chars)
    safe_text = re.sub(r'[^\w\-]', '', text)
    if not safe_text: safe_text = "Unknown"

    filename = f"{safe_text}_{v_id}.jpg"
    save_path = os.path.join(DEBUG_OUTPUT_DIR, filename)
    
    cv2.imwrite(save_path, img)

def main():
    print(f"🚀 Starting Final Plate Reading (PaddleOCR)...")
    print(f"📂 Reading from: {BASE_DIR}")
    print(f"💾 Saving to: {OUTPUT_CSV}")

    if SAVE_DEBUG_IMAGES:
        print(f"📸 Debug Images will be saved to: {DEBUG_OUTPUT_DIR}")

    # Init Paddle (Stable Config)
    try:
        ocr = PaddleOCR(lang='en', det=False, use_angle_cls=False, show_log=False)
    except:
        ocr = PaddleOCR(lang='en', det=False, use_textline_orientation=False)

    headers = [
        "Vehicle_ID", "Category", "Plate_Number", "Plate_Confidence", 
        "Helmet_Status", "Seatbelt_Status", "Phone_Status", "Plate_Image_Path"
    ]

    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        total_read = 0

        # Traverse folders
        for category in os.listdir(BASE_DIR):
            cat_path = os.path.join(BASE_DIR, category)
            if not os.path.isdir(cat_path): continue
            
            print(f"\n--- Category: {category} ---")

            for vehicle_id in sorted(os.listdir(cat_path)):
                vehicle_path = os.path.join(cat_path, vehicle_id)
                plate_image_path = os.path.join(vehicle_path, "BEST_PLATE_CROP.jpg")

                # Defaults
                plate_text = "Unknown"
                plate_conf = "0.00"
                
                if os.path.exists(plate_image_path):
                    img = cv2.imread(plate_image_path)
                    processed = preprocess_for_rec(img)
                    
                    try:
                        result = ocr.ocr(processed)
                        
                        # Universal Parser
                        raw_text = ""
                        current_conf = 0.0
                        
                        if result:
                            stack = [result]
                            while stack:
                                current = stack.pop(0)
                                if isinstance(current, list):
                                    stack.extend(current)
                                elif isinstance(current, tuple) and len(current) == 2:
                                    if isinstance(current[0], str):
                                        raw_text += current[0]
                                        current_conf = current[1]

                        # Apply Logic
                        final_text = apply_pakistan_correction(raw_text)

                        if len(final_text) >= 3:
                            plate_text = final_text
                            plate_conf = f"{current_conf:.2f}"
                            print(f"  ✅ ID {vehicle_id}: {plate_text} (Conf: {plate_conf})")
                            
                            # 📸 SAVE DEBUG IMAGE HERE
                            save_verification_image(processed, plate_text, vehicle_id)
                            
                            total_read += 1
                        else:
                            print(f"  ⚠️ ID {vehicle_id}: Read '{raw_text}' -> Filtered (Too short)")
                            
                    except Exception as e:
                        print(f"  ❌ ID {vehicle_id}: Error {e}")
                else:
                    print(f"  ⏭️ ID {vehicle_id}: No plate crop.")

                # Write to CSV
                writer.writerow([
                    vehicle_id, category, plate_text, plate_conf, 
                    "Pending", "Pending", "Pending", plate_image_path
                ])

    print("\n" + "="*50)
    print(f"🎉 MASTER CSV CREATED: {OUTPUT_CSV}")
    print(f"Total Plates Read: {total_read}")
    if SAVE_DEBUG_IMAGES:
        print(f"🖼️  Check '{DEBUG_OUTPUT_DIR}' to verify plate readings!")
    print("🌐 View visual plate & violation dashboard: python scripts/plate_visualizer.py")
    print("="*50)

if __name__ == "__main__":
    main()