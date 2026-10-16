import cv2
import os
import sys
import pandas as pd

# Fix Windows console encoding for Unicode/emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

# --- Paths ---
VIDEO_PATH = os.environ.get("VIDEO_PATH", os.path.join(PROJECT_ROOT, "input", "test2.mp4"))
CSV_PATH = os.path.join(PROJECT_ROOT, "output", "vehicle_positions.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "cropped_photos")
# --- Setup ---
df = pd.read_csv(CSV_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Settings ---
max_per_id = 10      # total max crops per vehicle
zones_per_id = 3     # number of vertical ROI zones
max_per_zone = 3     # max crops allowed in EACH zone

# Absolute padding (pixels) - ADJUSTED FOR BETTER FRONT VIEW
padding_x_front = 400   # More padding in front direction
padding_x_back = 200    # Less padding in back direction  
padding_y = 450

# Dynamic ROI detection from CSV (where in_roi == 1)
roi_rows = df[df["in_roi"] == 1]
if not roi_rows.empty:
    roi_top = int(roi_rows["center_y"].min())
    roi_bottom = int(roi_rows["center_y"].max())
else:
    roi_top = 0
    roi_bottom = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 2160

# Vehicle class mapping (New scheme: 0=person, 1=car, 2=bus, 3=truck, 4=motorcycle, 5=rickshaw)
BIKE_CLASSES = [4]         # 4: motorcycle
RICKSHAW_CLASSES = [5]     # 5: rickshaw
CAR_CLASSES = [1, 2, 3]    # 1: car, 2: bus, 3: truck

CLASS_NAMES = {
    0: "person",
    1: "car",
    2: "bus",
    3: "truck",
    4: "motorcycle",
    5: "rickshaw"
}

# Create main category folders
BIKE_DIR = os.path.join(OUTPUT_DIR, "Bikes_Motorcycles")
RICKSHAW_DIR = os.path.join(OUTPUT_DIR, "Rickshaws")
CAR_DIR = os.path.join(OUTPUT_DIR, "Cars")
UNKNOWN_DIR = os.path.join(OUTPUT_DIR, "Unknown")
os.makedirs(BIKE_DIR, exist_ok=True)
os.makedirs(RICKSHAW_DIR, exist_ok=True)
os.makedirs(CAR_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

saved_counts = {}       # overall count per ID
zone_counts = {}        # COUNT PER ZONE per ID
frame_idx = 0
last_box = {}           # key: track_id → last (x1,y1,x2,y2)
vehicle_directions = {} # Track vehicle movement direction
crop_metadata_records = []  # Metadata mapping crops to full-frame coordinates

print("Starting video cropping...")
print(f"ROI automatically inferred from CSV: Y={roi_top} to Y={roi_bottom}")
print(f"Vehicle Categories:")
print(f"  🚴 Bikes/Motorcycles: Classes {BIKE_CLASSES}")
print(f"  🛺 Rickshaws: Classes {RICKSHAW_CLASSES}")
print(f"  🚗 Cars & Heavy Vehicles: Classes {CAR_CLASSES}")
print(f"  📐 Front-focused padding: {padding_x_front}px front, {padding_x_back}px back")

# ---------------------------------------------------------
# Helper: clamp bounding box to frame
# ---------------------------------------------------------
def sanitize_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    
    # Ensure valid box dimensions
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
        
    return x1, y1, x2, y2

# ---------------------------------------------------------
# Get vehicle category from class ID
# ---------------------------------------------------------
def get_vehicle_category(class_id):
    if class_id in BIKE_CLASSES:
        return "bike"
    elif class_id in RICKSHAW_CLASSES:
        return "rickshaw"
    elif class_id in CAR_CLASSES:
        return "car"
    else:
        return "unknown"

# ---------------------------------------------------------
# Get output directory for vehicle category
# ---------------------------------------------------------
def get_category_dir(category):
    if category == "bike":
        return BIKE_DIR
    elif category == "rickshaw":
        return RICKSHAW_DIR
    elif category == "car":
        return CAR_DIR
    else:
        return UNKNOWN_DIR

# ---------------------------------------------------------
# Determine vehicle direction and apply smart padding
# ---------------------------------------------------------
def get_smart_padding(x1, y1, x2, y2, track_id, w, h, category="car", other_boxes=None):
    center_x = (x1 + x2) // 2
    
    # Track vehicle direction
    if track_id not in vehicle_directions:
        vehicle_directions[track_id] = []
    
    vehicle_directions[track_id].append(center_x)
    
    # Determine direction based on recent positions
    if len(vehicle_directions[track_id]) >= 3:
        recent_positions = vehicle_directions[track_id][-3:]
        movement = recent_positions[-1] - recent_positions[0]
        
        if movement > 10:  # Moving right → front is on right
            front_direction = "right"
        elif movement < -10:  # Moving left → front is on left
            front_direction = "left"
        else:  # Not moving much, use default
            front_direction = "right"
    else:
        front_direction = "right"
    
    if category == "bike":
        # 🚴 Adaptive Tight Cropping for Motorcycles:
        # Clamp horizontal padding to minimal margins (10-25px) to prevent bleed from neighboring bikes
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = min(25, max(10, int(box_w * 0.10)))
        x1p = max(0, x1 - pad_x)
        x2p = min(w, x2 + pad_x)
        
        # Vertical padding: upper focus (head region needs slight headroom, bottom road needs zero padding)
        pad_y_top = min(35, max(15, int(box_h * 0.15)))
        pad_y_bottom = min(15, max(5, int(box_h * 0.05)))
        y1p = max(0, y1 - pad_y_top)
        y2p = min(h, y2 + pad_y_bottom)
    else:
        # 🚗 Cars, Rickshaws & Heavy Vehicles
        # Apply asymmetric padding based on front direction
        if front_direction == "right":
            x1p = max(0, x1 - padding_x_back)
            x2p = min(w, x2 + padding_x_front)
        else:
            x1p = max(0, x1 - padding_x_front)
            x2p = min(w, x2 + padding_x_back)
        
        y1p = max(0, y1 - padding_y)
        y2p = min(h, y2 + padding_y)
        
        # 🛡️ Cross-Detection Masking / Occlusion Padding Clamping:
        # If an adjacent motorcycle/rickshaw is in the same frame, clamp car padding so it does NOT envelop them!
        if other_boxes:
            for ob in other_boxes:
                ox1, oy1, ox2, oy2, ocls = ob
                if ocls in [4, 5]:  # Motorcycle (4) or Rickshaw (5)
                    # Check vertical overlap
                    if max(y1p, oy1) < min(y2p, oy2):
                        # If motorcycle is to the right of car, clamp right padding
                        if ox1 >= x2 and x2p > ox1:
                            x2p = max(x2, ox1 - 5)
                        # If motorcycle is to the left of car, clamp left padding
                        if ox2 <= x1 and x1p < ox2:
                            x1p = min(x1, ox2 + 5)
    
    return x1p, y1p, x2p, y2p, front_direction

# ---------------------------------------------------------
# Get clean filename without special characters
# ---------------------------------------------------------
def get_clean_filename(frame_idx, zone_index, vehicle_type, front_direction):
    """Create clean ASCII filename without special characters"""
    # Use text instead of arrow symbols
    direction_text = "right" if front_direction == "right" else "left"
    
    filename = f"frame{frame_idx:06d}_zone{zone_index}_{vehicle_type}_{direction_text}.jpeg"
    return filename

# ---------------------------------------------------------
# Compute zone boundaries
# ---------------------------------------------------------
roi_height = max(1, roi_bottom - roi_top)
zone_size = max(1, roi_height // zones_per_id)    # size of each zone

print(f"ROI: {roi_top}-{roi_bottom} (height: {roi_height})")
print(f"Zone size: {zone_size} pixels")
print(f"Smart padding: {padding_x_front}px front, {padding_x_back}px back, {padding_y}px vertical")

# ========== Main Loop ==========
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    frame_data = df[df["frame"] == frame_idx]
    
    if frame is None:
        continue
        
    h, w = frame.shape[:2]

    for _, row in frame_data.iterrows():
        # Skip if not in ROI (based on CSV data)
        if row["in_roi"] == 0:
            continue
            
        track_id = int(row["id"])
        class_id = int(row["class"])
        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])

        # --- Clean bounding box ---
        x1, y1, x2, y2 = sanitize_bbox(x1, y1, x2, y2, w, h)

        # Skip if box is too small
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue

        # --- Compute center ---
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # --- ROI Check (double check with CSV data) ---
        if center_y < roi_top or center_y > roi_bottom:
            continue

        # --- Determine ROI ZONE ---
        zone_index = (center_y - roi_top) // zone_size
        zone_index = int(max(0, min(zone_index, zones_per_id - 1)))

        # Get vehicle category
        vehicle_category = get_vehicle_category(class_id)

        # Initialize zone counts if new ID
        if track_id not in zone_counts:
            zone_counts[track_id] = {z: 0 for z in range(zones_per_id)}
            saved_counts[track_id] = 0

        # --- Movement Filter ---
        if track_id in last_box:
            old_x1, old_y1, old_x2, old_y2 = last_box[track_id]
            old_center_x = (old_x1 + old_x2) // 2
            old_center_y = (old_y1 + old_y2) // 2
            
            movement = abs(center_x - old_center_x) + abs(center_y - old_center_y)
            size_change = abs((x2-x1) - (old_x2-old_x1)) + abs((y2-y1) - (old_y2-old_y1))

            # Skip if very similar to previous detection
            if movement < 40 and size_change < 40:
                continue

        # Update last box
        last_box[track_id] = (x1, y1, x2, y2)

        # --- Limit per-zone AND per-ID ---
        if zone_counts[track_id][zone_index] >= max_per_zone:
            continue

        # --- Other vehicles in frame for cross-detection masking ---
        other_boxes = []
        for _, other_row in frame_data.iterrows():
            if int(other_row["id"]) != track_id:
                other_boxes.append((
                    int(other_row["x1"]),
                    int(other_row["y1"]),
                    int(other_row["x2"]),
                    int(other_row["y2"]),
                    int(other_row["class"])
                ))

        # --- Apply SMART PADDING (Front-focused + Adaptive Tight Crop + Clamping) ---
        x1p, y1p, x2p, y2p, front_direction = get_smart_padding(
            x1, y1, x2, y2, track_id, w, h, category=vehicle_category, other_boxes=other_boxes
        )

        # Ensure padded box is valid
        if x2p <= x1p or y2p <= y1p:
            continue

        # Extract crop
        crop = frame[y1p:y2p, x1p:x2p]

        if crop.size == 0:
            continue

        # --- Save inside category folder with ID subfolder ---
        category_dir = get_category_dir(vehicle_category)
        id_dir = os.path.join(category_dir, f"id_{track_id:03d}")
        os.makedirs(id_dir, exist_ok=True)

        # Get vehicle type name for filename
        vehicle_type = CLASS_NAMES.get(class_id, "unknown")

        # Use clean filename without special characters
        clean_filename = get_clean_filename(frame_idx, zone_index, vehicle_type, front_direction)
        crop_path = os.path.join(id_dir, clean_filename)
        
        success = cv2.imwrite(crop_path, crop)
        
        if success:
            # --- Update counters ---
            saved_counts[track_id] = saved_counts.get(track_id, 0) + 1
            zone_counts[track_id][zone_index] += 1
            
            crop_metadata_records.append({
                "vehicle_id": track_id,
                "filename": clean_filename,
                "frame_idx": frame_idx,
                "class_id": class_id,
                "category": vehicle_category,
                "orig_x1": x1, "orig_y1": y1, "orig_x2": x2, "orig_y2": y2,
                "crop_x1": x1p, "crop_y1": y1p, "crop_x2": x2p, "crop_y2": y2p
            })
            
            print(f"✅ Saved: {vehicle_category.upper()} ID {track_id}, Frame {frame_idx}, Zone {zone_index}, "
                  f"Dir: {front_direction}, Total: {saved_counts[track_id]}/{max_per_id}")
        else:
            print(f"❌ Failed to save: {clean_filename}")

    # Progress logging
    if frame_idx % 500 == 0:
        total_saved = sum(saved_counts.values())
        unique_vehicles = len(saved_counts)
        
        # Count by category
        bike_count = sum(1 for tid in saved_counts if get_vehicle_category(df[df["id"] == tid]["class"].iloc[0]) == "bike")
        rickshaw_count = sum(1 for tid in saved_counts if get_vehicle_category(df[df["id"] == tid]["class"].iloc[0]) == "rickshaw")
        car_count = sum(1 for tid in saved_counts if get_vehicle_category(df[df["id"] == tid]["class"].iloc[0]) == "car")
        unknown_count = unique_vehicles - (bike_count + rickshaw_count + car_count)
        
        print(f"📊 Frame {frame_idx}: {total_saved} crops for {unique_vehicles} vehicles "
              f"(🚴: {bike_count}, 🛺: {rickshaw_count}, 🚗: {car_count}, ❓: {unknown_count})")

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"⏹️ Stopped by user at frame {frame_idx}")
        break

# Final statistics
print("\n" + "="*50)
print("📈 CROPPING COMPLETE - FINAL STATISTICS")
print("="*50)
total_crops = sum(saved_counts.values())
total_vehicles = len(saved_counts)

# Count vehicles by category
category_vehicles = {"bike": [], "rickshaw": [], "car": [], "unknown": []}

for track_id in saved_counts.keys():
    class_id = df[df["id"] == track_id]["class"].iloc[0]
    category = get_vehicle_category(class_id)
    category_vehicles.setdefault(category, []).append(track_id)

category_crops = {cat: sum(saved_counts[tid] for tid in tids) for cat, tids in category_vehicles.items()}

print(f"Total frames processed: {frame_idx}")
print(f"Total vehicles detected: {total_vehicles}")
print(f"Total crops saved: {total_crops}")
print(f"Output directory: {OUTPUT_DIR}")

print(f"\n📁 Category Breakdown:")
print(f"  🚴 Bikes/Motorcycles: {len(category_vehicles['bike'])} vehicles, {category_crops['bike']} crops")
print(f"  🛺 Rickshaws: {len(category_vehicles['rickshaw'])} vehicles, {category_crops['rickshaw']} crops")
print(f"  🚗 Cars & Heavy Vehicles: {len(category_vehicles['car'])} vehicles, {category_crops['car']} crops")
if category_vehicles["unknown"]:
    print(f"  ❓ Unknown: {len(category_vehicles['unknown'])} vehicles, {category_crops['unknown']} crops")

# Show per-vehicle statistics by category
for cat_name, label in [("bike", "Bikes/Motorcycles"), ("rickshaw", "Rickshaws"), ("car", "Cars & Heavy Vehicles"), ("unknown", "Unknown")]:
    v_list = category_vehicles.get(cat_name, [])
    if v_list:
        print(f"\n📋 {label} breakdown:")
        for track_id in sorted(v_list):
            count = saved_counts[track_id]
            class_id = df[df["id"] == track_id]["class"].iloc[0]
            v_type = CLASS_NAMES.get(class_id, f"class_{class_id}")
            zones = zone_counts[track_id]
            zone_str = ", ".join([f"Zone{z}:{c}" for z, c in zones.items()])
            print(f"  ID {track_id:03d} ({v_type}): {count} crops [{zone_str}]")

# Save crop metadata for downstream occlusion filtering
if crop_metadata_records:
    metadata_path = os.path.normpath(os.path.join(OUTPUT_DIR, "..", "crop_metadata.csv"))
    pd.DataFrame(crop_metadata_records).to_csv(metadata_path, index=False)
    print(f"💾 Crop metadata saved to: {metadata_path}")

cap.release()
cv2.destroyAllWindows()