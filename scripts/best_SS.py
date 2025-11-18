import cv2
import os
import numpy as np

# ----------------------------------------------------
# SETTINGS
# ----------------------------------------------------
BASE_DIR = "../output/cropped_photos"
DELETE_OTHERS = False   # set to True if you want to delete other images after selecting best

# ----------------------------------------------------
# SCORING FUNCTIONS
# ----------------------------------------------------
def sharpness_score(img):
    """Variance of Laplacian = sharpness."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def contrast_score(img):
    """Standard deviation of pixel intensities = contrast."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return np.std(gray)

def size_score(img):
    """Image area = vehicle size with padding."""
    h, w = img.shape[:2]
    return w * h

def closeness_score(img, filename):
    """
    Estimate closeness using the frame number from filename.
    More recent frames often mean closer to camera.
    Example file: frame000512_zone0_car→.jpeg
    """
    # Extract frame number from filename like "frame000512_zone0_car→.jpeg"
    try:
        # Get the part after "frame" and before first "_"
        frame_part = filename.split('_')[0].replace("frame", "")
        num = int(frame_part)
        return num
    except:
        return 0

def brightness_score(img):
    """Average brightness - avoid over/under exposed images."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return np.mean(gray)

# ----------------------------------------------------
# COMBINED SCORE
# ----------------------------------------------------
def total_score(img, filename):
    s1 = size_score(img)           # Image size
    s2 = sharpness_score(img)      # Sharpness
    s3 = contrast_score(img)       # Contrast
    s4 = closeness_score(img, filename)  # Frame number (closeness)
    s5 = brightness_score(img)     # Brightness
    
    # Normalize brightness score (ideal around 127)
    brightness_penalty = abs(s5 - 127) / 127  # 0 = perfect, 1 = worst
    
    # weighted scoring (tuned for vehicle images)
    score = (
        s1 * 0.35 +     # biggest image (most important)
        s2 * 0.25 +     # sharpness
        s3 * 0.20 +     # contrast
        s4 * 0.10 +     # closeness (frame number)
        (1 - brightness_penalty) * 0.10  # brightness quality
    )
    return score

# ----------------------------------------------------
# MAIN BEST SELECTOR
# ----------------------------------------------------
def process_category(category_dir):
    """Process all ID folders within a category directory"""
    category_name = os.path.basename(category_dir)
    print(f"\n📁 Processing category: {category_name}")
    
    id_folders = [f for f in os.listdir(category_dir) 
                  if os.path.isdir(os.path.join(category_dir, f)) and f.startswith("id_")]
    
    if not id_folders:
        print(f"  No ID folders found in {category_name}")
        return
    
    total_processed = 0
    for id_folder in id_folders:
        folder_path = os.path.join(category_dir, id_folder)
        
        scores = []
        files = [f for f in os.listdir(folder_path) 
                if f.lower().endswith((".jpg", ".jpeg", ".png")) and f != "BEST.jpg"]
        
        if len(files) <= 1:
            if files:
                print(f"  {id_folder}: Only one image, keeping as is.")
            continue

        # Score all images
        for fname in files:
            full_path = os.path.join(folder_path, fname)
            img = cv2.imread(full_path)
            if img is None:
                continue

            score = total_score(img, fname)
            scores.append((score, fname, full_path))
        
        if not scores:
            continue

        # Sort descending → best score first
        scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_file, best_path = scores[0]
        
        # Get runner-up for comparison
        runner_up_score = scores[1][0] if len(scores) > 1 else best_score
        
        # Rename best to BEST.jpg
        new_path = os.path.join(folder_path, "BEST.jpg")
        
        # Remove existing BEST.jpg if it exists
        if os.path.exists(new_path):
            os.remove(new_path)
        
        os.rename(best_path, new_path)
        
        # Delete others if enabled
        if DELETE_OTHERS:
            for _, fname, file_path in scores[1:]:
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        total_processed += 1
        score_diff = best_score - runner_up_score
        print(f"  ✅ {id_folder}: {best_file} → BEST.jpg (score: {best_score:.1f}, diff: {score_diff:+.1f})")
    
    print(f"  📊 Processed {total_processed} vehicles in {category_name}")

# ----------------------------------------------------
# EXECUTION
# ----------------------------------------------------
print("🚀 Starting Best Screenshot Selection...")
print(f"Base directory: {BASE_DIR}")
print(f"Delete others: {DELETE_OTHERS}")

# Process each category folder
categories = ["Bikes_Motorcycles", "Cars"]
total_categories_processed = 0

for category in categories:
    category_path = os.path.join(BASE_DIR, category)
    if os.path.exists(category_path):
        process_category(category_path)
        total_categories_processed += 1
    else:
        print(f"⚠️  Category folder not found: {category}")

print(f"\n{'='*50}")
print("🎉 BEST SCREENSHOT SELECTION COMPLETE!")
print(f"{'='*50}")
print(f"Categories processed: {total_categories_processed}/2")
print(f"Base directory: {BASE_DIR}")

if DELETE_OTHERS:
    print("🗑️  Other images have been DELETED (only BEST.jpg remains)")
else:
    print("💾 All original images preserved (BEST.jpg created alongside)")

print("\n📁 Folder structure:")
print("Bikes_Motorcycles/")
print("├── id_001/")
print("│   ├── BEST.jpg        ← Best screenshot")
print("│   └── [other images]  ← Only if DELETE_OTHERS=False")
print("└── id_002/")
print("Cars/")
print("├── id_003/")
print("│   ├── BEST.jpg        ← Best screenshot") 
print("│   └── [other images]  ← Only if DELETE_OTHERS=False")
print("└── id_004/")