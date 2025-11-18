import cv2
import pandas as pd
import numpy as np
import os
from collections import defaultdict

# -------------------------
# PATHS
# -------------------------
VIDEO_PATH = "../input/bridge_back.mp4"
CSV_PATH = "../output/vehicle_positions.csv"
OUTPUT_DIR = "../output/vehicle_closeness_check"
# -------------------------
# SETTINGS
# -------------------------
DISTANCE_THRESHOLD = 500  # pixels - adjusted for realistic vehicle distances
INCLUDE_BIKES = True      # Toggle: set to False to exclude bikes (class 1, 3)
BIKE_CLASSES = [1, 3]     # COCO: bicycle=1, motorcycle=3
CAR_CLASSES = [2, 5, 7]   # COCO: car=2, bus=5, truck=7
DISPLAY_SCALE = 0.5       # Scale down for display (0.5 = 50% of original size)
ROI_TOP = 876             # Your ROI settings from detection
ROI_BOTTOM = 1824

# -------------------------
# SETUP
# -------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CSV
print("📂 Loading vehicle positions CSV...")
df = pd.read_csv(CSV_PATH)

# Filter only vehicles in ROI
df = df[df['in_roi'] == 1]
print(f"📊 Loaded {len(df)} vehicle detections in ROI")

# Load video
print("🎬 Loading video...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Error: Could not open video!")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, "vehicle_closeness_output.mp4"), 
                      fourcc, fps, (width, height))

# CSV output files
violations_file = open(os.path.join(OUTPUT_DIR, "violations.csv"), "w", newline="")
all_pairs_file = open(os.path.join(OUTPUT_DIR, "all_pairs.csv"), "w", newline="")

violations_file.write("frame,vehicle_id_1,vehicle_id_2,class_1,class_2,class_name_1,class_name_2,distance_pixels,center_x_1,center_y_1,center_x_2,center_y_2\n")
all_pairs_file.write("frame,vehicle_id_1,vehicle_id_2,class_1,class_2,class_name_1,class_name_2,distance_pixels,violation,center_x_1,center_y_1,center_x_2,center_y_2\n")

# Class name mapping
CLASS_NAMES = {
    1: "bicycle",
    2: "car", 
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def is_bike(class_id):
    return class_id in BIKE_CLASSES

def get_class_name(class_id):
    return CLASS_NAMES.get(class_id, f"unknown_{class_id}")

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def get_vehicles_in_frame(frame_num):
    """Get all vehicles in a specific frame"""
    frame_data = df[df['frame'] == frame_num]
    return frame_data.to_dict('records')

def get_vehicle_color(class_id):
    """Get color based on vehicle class"""
    if class_id == 1:  # bicycle
        return (255, 255, 0)  # Cyan
    elif class_id == 3:  # motorcycle
        return (255, 0, 255)  # Magenta
    elif class_id == 2:  # car
        return (0, 255, 0)    # Green
    else:  # bus/truck
        return (0, 165, 255)  # Orange

# -------------------------
# MAIN LOOP
# -------------------------
print("\n🔍 Processing frames...")
print(f"📏 Distance threshold: {DISTANCE_THRESHOLD} pixels")
print(f"🚴 Include bikes: {INCLUDE_BIKES}")
print("Press ESC to cancel at any time\n")

frame_count = 0
total_violations = 0
violation_frames = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # Get all vehicles in this frame
    vehicles = get_vehicles_in_frame(frame_num)
    
    # Initialize violation_pairs for every frame
    violation_pairs = []
    
    if len(vehicles) > 0:
        # Check all pairs of vehicles
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1 = vehicles[i]
                v2 = vehicles[j]
                
                # Skip bikes if INCLUDE_BIKES is False
                if not INCLUDE_BIKES:
                    if is_bike(v1['class']) or is_bike(v2['class']):
                        continue
                
                # Calculate distance between vehicle centers
                distance = calculate_distance(v1['center_x'], v1['center_y'], 
                                            v2['center_x'], v2['center_y'])
                
                is_violation = distance < DISTANCE_THRESHOLD
                
                # Get class names
                class_name_1 = get_class_name(v1['class'])
                class_name_2 = get_class_name(v2['class'])
                
                # Log to all_pairs CSV
                all_pairs_file.write(f"{frame_num},{v1['id']},{v2['id']},{v1['class']},{v2['class']},{class_name_1},{class_name_2},{distance:.2f},{int(is_violation)},{v1['center_x']},{v1['center_y']},{v2['center_x']},{v2['center_y']}\n")
                
                # Log violations
                if is_violation:
                    violation_pairs.append((v1, v2, distance))
                    violations_file.write(f"{frame_num},{v1['id']},{v2['id']},{v1['class']},{v2['class']},{class_name_1},{class_name_2},{distance:.2f},{v1['center_x']},{v1['center_y']},{v2['center_x']},{v2['center_y']}\n")
                    total_violations += 1
                    violation_frames.add(frame_num)
        
        # Draw on frame
        for vehicle in vehicles:
            x1, y1, x2, y2 = int(vehicle['x1']), int(vehicle['y1']), int(vehicle['x2']), int(vehicle['y2'])
            
            # Determine if this vehicle is in a violation pair
            in_violation = any(v[0]['id'] == vehicle['id'] or v[1]['id'] == vehicle['id'] 
                              for v in violation_pairs)
            
            # Get vehicle-specific color
            vehicle_color = get_vehicle_color(vehicle['class'])
            
            # Use RED if in violation, vehicle-specific color if safe
            color = (0, 0, 255) if in_violation else vehicle_color
            
            # Draw bounding box
            thickness = 3 if in_violation else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw vehicle ID and class
            class_name = get_class_name(vehicle['class'])
            label = f"ID:{int(vehicle['id'])} {class_name}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw center point
            center_x, center_y = int(vehicle['center_x']), int(vehicle['center_y'])
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
        
        # Draw distance lines between violation pairs
        for v1, v2, distance in violation_pairs:
            x1, y1 = int(v1['center_x']), int(v1['center_y'])
            x2, y2 = int(v2['center_x']), int(v2['center_y'])
            
            # Draw line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw distance text at midpoint
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(frame, f"{distance:.0f}px", (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw ROI boundaries
    cv2.line(frame, (0, ROI_TOP), (width, ROI_TOP), (0, 255, 255), 2)
    cv2.line(frame, (0, ROI_BOTTOM), (width, ROI_BOTTOM), (0, 255, 255), 2)
    cv2.putText(frame, "ROI", (10, ROI_TOP - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Frame info overlay
    cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Vehicles: {len(vehicles)}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Violations: {len(violation_pairs)}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if len(violation_pairs) > 0 else (0, 255, 0), 2)
    cv2.putText(frame, f"Threshold: {DISTANCE_THRESHOLD}px", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, "Press ESC to cancel", (10, height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Write frame
    out.write(frame)
    
    # Display frame in real-time
    display_frame = cv2.resize(frame, (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)))
    cv2.imshow('Vehicle Closeness Detection', display_frame)
    
    # ESC key to cancel
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        print("\n⛔ Process cancelled by user!")
        break

    frame_count += 1
    
    if frame_count % 30 == 0:
        print(f"  ✓ Frame {frame_num}: {len(vehicles)} vehicles, {len(violation_pairs)} violations")

# -------------------------
# CLEANUP
# -------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
violations_file.close()
all_pairs_file.close()

print("\n" + "="*50)
print("✅ VEHICLE CLOSENESS DETECTION COMPLETED!")
print("="*50)
print(f"📊 Total frames processed: {frame_count}")
print(f"🚨 Total violation pairs detected: {total_violations}")
print(f"🎯 Frames with violations: {len(violation_frames)}")
print(f"📹 Output video: {os.path.join(OUTPUT_DIR, 'vehicle_closeness_output.mp4')}")
print(f"📄 Violations CSV: {os.path.join(OUTPUT_DIR, 'violations.csv')}")
print(f"📄 All pairs CSV: {os.path.join(OUTPUT_DIR, 'all_pairs.csv')}")

# Summary statistics
if total_violations > 0:
    print(f"\n📈 Violation Summary:")
    violations_df = pd.read_csv(os.path.join(OUTPUT_DIR, "violations.csv"))
    print(f"   Average distance: {violations_df['distance_pixels'].mean():.1f} pixels")
    print(f"   Minimum distance: {violations_df['distance_pixels'].min():.1f} pixels")
    print(f"   Maximum distance: {violations_df['distance_pixels'].max():.1f} pixels")