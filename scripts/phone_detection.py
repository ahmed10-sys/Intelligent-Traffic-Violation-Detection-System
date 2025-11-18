from ultralytics import YOLO
import cv2
import os
import csv
from datetime import datetime

# -------------------------
# PATHS
# -------------------------
MODEL_PATH = "../models/phone_distractions.pt"
OUTPUT_DIR = r"C:\Users\syedm\OneDrive\Documents\Traffic Violation System\phone_detection_test"
CSV_PATH = os.path.join(OUTPUT_DIR, "phone_detection_results.csv")

# -------------------------
# SETTINGS
# -------------------------
CONF_THRES = 0.3
PHONE_CLASS = 67

# -------------------------
# SETUP
# -------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
print("🚀 Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Initialize CSV
csv_file = open(CSV_PATH, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["timestamp", "image_name", "phone_detected", "confidence", "x1", "y1", "x2", "y2", "image_width", "image_height"])

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def is_supported_image(filename):
    """Check if file is a supported image format"""
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.jfif', '.webp')
    return filename.lower().endswith(supported_extensions)

def convert_avif_to_jpg(avif_path):
    """Convert AVIF to JPG using OpenCV (if OpenCV supports AVIF)"""
    try:
        img = cv2.imread(avif_path)
        if img is not None:
            jpg_path = avif_path.replace('.avif', '_converted.jpg').replace('.AVIF', '_converted.jpg')
            cv2.imwrite(jpg_path, img)
            return jpg_path
    except:
        pass
    return None

def detect_phones_in_image(image_path, output_dir):
    """Detect phones in a single image and save results"""
    image_name = os.path.basename(image_path)
    
    # Handle AVIF files by converting them
    if image_path.lower().endswith('.avif'):
        print(f"   🔄 Converting AVIF to JPG: {image_name}")
        converted_path = convert_avif_to_jpg(image_path)
        if converted_path:
            image_path = converted_path
            image_name = os.path.basename(converted_path)
        else:
            print(f"   ❌ Could not convert AVIF file: {image_name}")
            return False, 0
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return False, 0
    
    original_img = img.copy()
    h, w = img.shape[:2]
    
    print(f"\n🔍 Processing: {image_name} ({w}x{h})")
    
    # Run detection
    results = model.predict(img, conf=CONF_THRES, verbose=False)
    detections = results[0].boxes
    
    phone_detected = False
    phone_confidence = 0
    phone_boxes = []
    
    if detections is not None and len(detections) > 0:
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Check if it's a phone
            if cls == PHONE_CLASS:
                phone_detected = True
                phone_boxes.append((x1, y1, x2, y2, conf))
                phone_confidence = max(phone_confidence, conf)
                
                # Draw phone detection
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img, f"PHONE {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                print(f"   📱 Phone detected! Confidence: {conf:.2f}")
    
    # Add result overlay
    status = "PHONE DETECTED" if phone_detected else "No phone"
    status_color = (0, 0, 255) if phone_detected else (0, 255, 0)
    
    cv2.putText(img, f"Status: {status}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    cv2.putText(img, f"Image: {image_name}", (20, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save annotated image
    output_path = os.path.join(output_dir, f"detected_{image_name}")
    cv2.imwrite(output_path, img)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if phone_detected:
        for x1, y1, x2, y2, conf in phone_boxes:
            writer.writerow([timestamp, image_name, "YES", conf, x1, y1, x2, y2, w, h])
    else:
        writer.writerow([timestamp, image_name, "NO", 0, None, None, None, None, w, h])
    
    print(f"   💾 Saved: {output_path}")
    return phone_detected, len(phone_boxes)

# -------------------------
# MAIN FUNCTION
# -------------------------
def main():
    print("📱 PHONE DETECTION TEST SCRIPT")
    print("=" * 50)
    print("This script detects phones in individual images")
    print("Supported formats: JPG, JPEG, PNG, BMP, TIFF, JFIF, WEBP, AVIF")
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 50)
    
    # Create test images folder
    test_images_dir = os.path.join(OUTPUT_DIR, "test_images")
    os.makedirs(test_images_dir, exist_ok=True)
    
    print(f"\n📁 Place your test images in: {test_images_dir}")
    print("💡 Supported formats: JPG, JPEG, PNG, BMP, TIFF, JFIF, WEBP, AVIF")
    input("Press Enter when you have images in the folder...")
    
    # Get all supported images from test folder
    test_images = [f for f in os.listdir(test_images_dir) 
                   if is_supported_image(f) or f.lower().endswith('.avif')]
    
    if not test_images:
        print("❌ No supported images found in test_images folder!")
        print(f"Please add images to: {test_images_dir}")
        return
    
    print(f"\n🖼️ Found {len(test_images)} images to process:")
    for img in test_images:
        print(f"   - {img}")
    
    input("\nPress Enter to start detection...")
    
    # Process each image
    total_processed = 0
    phones_found = 0
    
    for image_file in test_images:
        image_path = os.path.join(test_images_dir, image_file)
        phone_detected, num_phones = detect_phones_in_image(image_path, OUTPUT_DIR)
        
        total_processed += 1
        if phone_detected:
            phones_found += 1
    
    # Close CSV
    csv_file.close()
    
    # Summary
    print(f"\n{'='*50}")
    print("✅ PHONE DETECTION COMPLETED!")
    print(f"{'='*50}")
    print(f"📊 Images processed: {total_processed}")
    print(f"📱 Images with phones: {phones_found}")
    print(f"📄 Results CSV: {CSV_PATH}")
    print(f"🖼️ Annotated images: {OUTPUT_DIR}")
    
    if phones_found > 0:
        print(f"\n🚨 VIOLATIONS DETECTED: {phones_found} images show phone use!")
    else:
        print(f"\n✅ NO VIOLATIONS: No phones detected in any images")

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    main()