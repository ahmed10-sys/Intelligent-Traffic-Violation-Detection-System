import cv2
import os
import pandas as pd

# --- Paths ---
VIDEO_PATH = "../input/bridge_back.mp4" 
CSV_PATH = "../output/vehicle_positions.csv"
OUTPUT_DIR = "../output/cropped_photos"
# --- Settings ---
roi_top = 876
roi_bottom = 1824

max_per_id = 10      # total max crops per vehicle
zones_per_id = 3     # number of vertical ROI zones
max_per_zone = 3     # max crops allowed in EACH zone

# Absolute padding (pixels) - ADJUSTED FOR BETTER FRONT VIEW
padding_x_front = 400   # More padding in front direction
padding_x_back = 200    # Less padding in back direction  
padding_y = 450

# Vehicle class mapping (from YOLO)
BIKE_CLASSES = [1, 3]  # 1: bicycle, 3: motorcycle
CAR_CLASSES = [2, 5, 7]  # 2: car, 5: bus, 7: truck (all merged together)

# --- Setup ---
df = pd.read_csv(CSV_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create main category folders
BIKE_DIR = os.path.join(OUTPUT_DIR, "Bikes_Motorcycles")
CAR_DIR = os.path.join(OUTPUT_DIR, "Cars")  # Now includes all non-bike vehicles
os.makedirs(BIKE_DIR, exist_ok=True)
os.makedirs(CAR_DIR, exist_ok=True)

saved_counts = {}       # overall count per ID
zone_counts = {}        # COUNT PER ZONE per ID
frame_idx = 0
last_box = {}           # key: track_id → last (x1,y1,x2,y2)
vehicle_directions = {} # Track vehicle movement direction

print("Starting video cropping...")
print(f"Vehicle Categories:")
print(f"  🚴 Bikes/Motorcycles: Classes {BIKE_CLASSES}")
print(f"  🚗 Cars & Other Vehicles: Classes {CAR_CLASSES}")
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
    else:
        return "car"  # Everything else goes to Cars folder

# ---------------------------------------------------------
# Get output directory for vehicle category
# ---------------------------------------------------------
def get_category_dir(category):
    if category == "bike":
        return BIKE_DIR
    else:
        return CAR_DIR

# ---------------------------------------------------------
# Determine vehicle direction and apply smart padding
# ---------------------------------------------------------
def get_smart_padding(x1, y1, x2, y2, track_id, w, h):
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
            front_direction = "right"  # Default assumption
    else:
        front_direction = "right"  # Default assumption
    
    # Apply asymmetric padding based on front direction
    if front_direction == "right":
        # More padding on right (front), less on left (back)
        x1p = max(0, x1 - padding_x_back)
        x2p = min(w, x2 + padding_x_front)
    else:  # front_direction == "left"
        # More padding on left (front), less on right (back)
        x1p = max(0, x1 - padding_x_front)
        x2p = min(w, x2 + padding_x_back)
    
    # Symmetric Y padding
    y1p = max(0, y1 - padding_y)
    y2p = min(h, y2 + padding_y)
    
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
roi_height = roi_bottom - roi_top
zone_size = roi_height // zones_per_id    # size of each zone

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

        if saved_counts.get(track_id, 0) >= max_per_id:
            continue

        # --- Apply SMART PADDING (Front-focused) ---
        x1p, y1p, x2p, y2p, front_direction = get_smart_padding(x1, y1, x2, y2, track_id, w, h)

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
        if class_id == 1:
            vehicle_type = "bicycle"
        elif class_id == 2:
            vehicle_type = "car"
        elif class_id == 3:
            vehicle_type = "motorcycle"
        elif class_id == 5:
            vehicle_type = "bus"
        elif class_id == 7:
            vehicle_type = "truck"
        else:
            vehicle_type = "unknown"

        # Use clean filename without special characters
        clean_filename = get_clean_filename(frame_idx, zone_index, vehicle_type, front_direction)
        crop_path = os.path.join(id_dir, clean_filename)
        
        success = cv2.imwrite(crop_path, crop)
        
        if success:
            # --- Update counters ---
            saved_counts[track_id] = saved_counts.get(track_id, 0) + 1
            zone_counts[track_id][zone_index] += 1
            
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
        car_count = unique_vehicles - bike_count
        
        print(f"📊 Frame {frame_idx}: {total_saved} crops for {unique_vehicles} vehicles "
              f"(🚴: {bike_count}, 🚗: {car_count})")

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
bike_vehicles = []
car_vehicles = []

for track_id in saved_counts.keys():
    class_id = df[df["id"] == track_id]["class"].iloc[0]
    category = get_vehicle_category(class_id)
    
    if category == "bike":
        bike_vehicles.append(track_id)
    else:
        car_vehicles.append(track_id)

bike_crops = sum(saved_counts[tid] for tid in bike_vehicles)
car_crops = sum(saved_counts[tid] for tid in car_vehicles)

print(f"Total frames processed: {frame_idx}")
print(f"Total vehicles detected: {total_vehicles}")
print(f"Total crops saved: {total_crops}")
print(f"Output directory: {OUTPUT_DIR}")

print(f"\n📁 Category Breakdown:")
print(f"  🚴 Bikes/Motorcycles: {len(bike_vehicles)} vehicles, {bike_crops} crops")
print(f"  🚗 Cars & Other Vehicles: {len(car_vehicles)} vehicles, {car_crops} crops")

# Show per-vehicle statistics by category
if bike_vehicles:
    print(f"\n📋 Bikes/Motorcycles breakdown:")
    for track_id in sorted(bike_vehicles):
        count = saved_counts[track_id]
        class_id = df[df["id"] == track_id]["class"].iloc[0]
        vehicle_type = "bicycle" if class_id == 1 else "motorcycle"
        zones = zone_counts[track_id]
        zone_str = ", ".join([f"Zone{z}:{c}" for z, c in zones.items()])
        print(f"  ID {track_id:03d} ({vehicle_type}): {count} crops [{zone_str}]")

if car_vehicles:
    print(f"\n📋 Cars & Other Vehicles breakdown:")
    for track_id in sorted(car_vehicles):
        count = saved_counts[track_id]
        class_id = df[df["id"] == track_id]["class"].iloc[0]
        if class_id == 2:
            vehicle_type = "car"
        elif class_id == 5:
            vehicle_type = "bus"
        elif class_id == 7:
            vehicle_type = "truck"
        else:
            vehicle_type = "unknown"
        zones = zone_counts[track_id]
        zone_str = ", ".join([f"Zone{z}:{c}" for z, c in zones.items()])
        print(f"  ID {track_id:03d} ({vehicle_type}): {count} crops [{zone_str}]")

cap.release()
cv2.destroyAllWindows()