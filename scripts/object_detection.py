from ultralytics import YOLO
import cv2
import csv
import os
import numpy as np
import time

# ========== CONFIGURATION ==========
MODEL_PATH = "../models/yolov8m.pt"
VIDEO_PATH = "../input/bridge_back.mp4"
CSV_PATH = "../output/vehicle_positions.csv"
OUTPUT_VIDEO = "../output/object_detection_roi_output.mp4"

# ROI (will be set automatically)
ROI_TOP = None
ROI_BOTTOM = None

# VEHICLE CLASSES: bicycle, car, motorcycle, bus, truck (IDs depend on your model)
VEHICLE_CLASSES = [1, 2, 3, 5, 7]

# SPEED OPTIMIZATIONS AND FLAGS
SPEED_OPTIMIZATIONS = {
    "frame_skip": 2,               # Process every Nth frame (e.g., 2 -> process frames where frame_count % 2 == 0)
    "resize_ratio": 0.4,           # Resize frames to 40% for detection speed (set to 1.0 to disable)
    "disable_csv_logging": False,  # If True, CSV logging is disabled
    "disable_video_output": False, # If True, won't write an output video
    "disable_display": False,      # If True, won't show preview window
    "roi_only_detection": False,   # If True, perform detection on ROI crop only
    "csv_flush_interval": 50       # Flush CSV every N processed frames
}

CONF_THRESHOLD = 0.2  # detection confidence threshold
TRACKER_CONFIG = "bytetrack.yaml"  # tracker config for model.track (leave if works in your env)

# ========== UTILITIES ==========

def ensure_output_dirs(path_list):
    for p in path_list:
        d = os.path.dirname(p)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

def select_roi(video_path, display_scale=0.25):
    """
    Interactive ROI selection: click top and bottom Y positions (on scaled preview).
    Returns (roi_top, roi_bottom).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video for ROI selection.")
        return 0, 0

    roi_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            scale = param["scale"]
            y_orig = int(y / scale)
            roi_points.append(y_orig)
            print(f"Clicked Y (original frame): {y_orig}")

    print("🎯 Select ROI - Click TWO points (top and bottom) then press ESC")
    cv2.namedWindow("Select ROI - Click top and bottom points", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_display = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
        cv2.imshow("Select ROI - Click top and bottom points", frame_display)
        cv2.setMouseCallback("Select ROI - Click top and bottom points", click_event, {"scale": display_scale})

        # Draw instructions
        cv2.putText(frame_display, "Click TOP then BOTTOM of ROI", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_display, "Press ESC when done", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or len(roi_points) >= 2:  # ESC
            break

    cap.release()
    cv2.destroyWindow("Select ROI - Click top and bottom points")

    if len(roi_points) >= 2:
        roi_top, roi_bottom = sorted(roi_points[:2])
        print(f"✅ ROI Selected: top={roi_top}, bottom={roi_bottom}")
        return roi_top, roi_bottom
    else:
        print("❌ ROI selection failed or skipped.")
        return None, None

def is_vehicle_in_roi(box, roi_top, roi_bottom):
    """Return True if the box center Y lies inside the ROI."""
    x1, y1, x2, y2 = box
    center_y = (y1 + y2) // 2
    return roi_top <= center_y <= roi_bottom

# ========== MAIN DETECTION FUNCTION ==========

def run_detection_with_roi():
    global ROI_TOP, ROI_BOTTOM

    # Ensure output dirs exist
    ensure_output_dirs([CSV_PATH, OUTPUT_VIDEO])

    # Load model (let ultralytics choose device, or force CPU/GPU if desired)
    print("🔁 Loading model...")
    model = YOLO(MODEL_PATH)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Error: Could not open video!")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # ROI selection (interactive). If user cancels, fallback to full frame.
    selected_top, selected_bottom = select_roi(VIDEO_PATH)
    if selected_top is None or selected_bottom is None:
        ROI_TOP, ROI_BOTTOM = 0, height
        print(f"ℹ️ Using full frame as ROI: top={ROI_TOP}, bottom={ROI_BOTTOM}")
    else:
        ROI_TOP, ROI_BOTTOM = max(0, selected_top), min(height, selected_bottom)

    # Prepare video writer if needed
    out = None
    if not SPEED_OPTIMIZATIONS["disable_video_output"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
        if not out.isOpened():
            print("⚠️ Warning: Could not open VideoWriter. Video output disabled.")
            out = None

    # Prepare CSV writer (open once)
    csv_file = None
    csv_writer = None
    if not SPEED_OPTIMIZATIONS["disable_csv_logging"]:
        try:
            csv_file = open(CSV_PATH, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "frame", "id", "class", "confidence",
                "x1", "y1", "x2", "y2",
                "center_x", "center_y",
                "in_roi"
            ])
        except Exception as e:
            print(f"❌ Error opening CSV for writing: {e}")
            csv_file = None
            csv_writer = None

    # Print configuration
    print("🚀 Starting ULTRA-FAST CPU-optimized detection...")
    print(f"📐 ROI: Y={ROI_TOP} to Y={ROI_BOTTOM}")
    print(f"⚡ Optimizations: Frame skip={SPEED_OPTIMIZATIONS['frame_skip']}, Resize={SPEED_OPTIMIZATIONS['resize_ratio']}")
    print(f"⚡ CSV Logging: {'ENABLED' if csv_writer else 'DISABLED'}")
    print(f"⚡ Video Output: {'ENABLED' if out else 'DISABLED'}")
    print(f"⚡ Display: {'ENABLED' if not SPEED_OPTIMIZATIONS['disable_display'] else 'DISABLED'}")
    print(f"⚡ ROI-Only Detection: {'ENABLED' if SPEED_OPTIMIZATIONS['roi_only_detection'] else 'DISABLED'}")
    print("Processing... Press Ctrl+C or 'q' in window to stop\n")

    # Counters
    frame_count = 0
    processed_frames = 0
    total_detections = 0
    roi_detections = 0
    bike_detections = 0
    car_detections = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # FRAME SKIPPING: process only frames where frame_count % frame_skip == 0
            if SPEED_OPTIMIZATIONS["frame_skip"] > 1 and (frame_count % SPEED_OPTIMIZATIONS["frame_skip"] != 0):
                continue

            processed_frames += 1

            # Prepare original_frame if we will display or save video (always define to avoid NameError)
            original_frame = frame.copy() if (not SPEED_OPTIMIZATIONS["disable_display"] or out is not None) else None

            # Draw ROI visualization on original_frame if available
            if original_frame is not None and not SPEED_OPTIMIZATIONS["disable_display"]:
                roi_overlay = original_frame.copy()
                cv2.rectangle(roi_overlay, (0, ROI_TOP), (width, ROI_BOTTOM), (0, 255, 255), -1)
                cv2.addWeighted(roi_overlay, 0.1, original_frame, 0.9, 0, original_frame)
                cv2.line(original_frame, (0, ROI_TOP), (width, ROI_TOP), (0, 255, 255), 2)
                cv2.line(original_frame, (0, ROI_BOTTOM), (width, ROI_BOTTOM), (0, 255, 255), 2)

            # Prepare detection_frame (ROI crop or full frame), then resize if requested
            if SPEED_OPTIMIZATIONS["roi_only_detection"]:
                roi_height = ROI_BOTTOM - ROI_TOP
                if roi_height > 0:
                    roi_frame = frame[ROI_TOP:ROI_BOTTOM, 0:width]
                    if SPEED_OPTIMIZATIONS["resize_ratio"] < 1.0:
                        new_width = int(width * SPEED_OPTIMIZATIONS["resize_ratio"])
                        new_height = int(roi_height * SPEED_OPTIMIZATIONS["resize_ratio"])
                        detection_frame = cv2.resize(roi_frame, (new_width, new_height))
                    else:
                        detection_frame = roi_frame
                        new_width, new_height = width, roi_height
                else:
                    detection_frame = frame
                    new_width, new_height = width, height
            else:
                if SPEED_OPTIMIZATIONS["resize_ratio"] < 1.0:
                    new_width = int(width * SPEED_OPTIMIZATIONS["resize_ratio"])
                    new_height = int(height * SPEED_OPTIMIZATIONS["resize_ratio"])
                    detection_frame = cv2.resize(frame, (new_width, new_height))
                else:
                    detection_frame = frame.copy()
                    new_width, new_height = width, height

            # Run detection + tracking
            results = model.track(
                detection_frame,
                tracker=TRACKER_CONFIG,
                persist=True,
                conf=CONF_THRESHOLD,
                iou=0.5,
                classes=VEHICLE_CLASSES,
                verbose=False
            )

            # Per-frame counters
            frame_detections = 0
            frame_roi_detections = 0
            frame_bike_detections = 0
            frame_car_detections = 0

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                # Try to fetch tracking IDs; fall back to -1 if missing
                try:
                    if results[0].boxes.id is not None:
                        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    else:
                        track_ids = np.array([-1] * len(boxes))
                except Exception:
                    track_ids = np.array([-1] * len(boxes))

                for i_det, (box, cls, conf, track_id) in enumerate(zip(boxes, classes, confidences, track_ids)):
                    if conf < CONF_THRESHOLD:
                        continue

                    # SCALE coordinates back to original frame size
                    if SPEED_OPTIMIZATIONS["resize_ratio"] < 1.0:
                        scale_x = width / new_width
                        scale_y = (ROI_BOTTOM - ROI_TOP) / new_height if SPEED_OPTIMIZATIONS["roi_only_detection"] else height / new_height
                        # box is [x1, y1, x2, y2] in detection_frame coords
                        x1 = int(box[0] * scale_x)
                        y1 = int(box[1] * scale_y)
                        x2 = int(box[2] * scale_x)
                        y2 = int(box[3] * scale_y)
                    else:
                        x1, y1, x2, y2 = map(int, box)

                    # If ROI-only detection was used, add ROI top offset back
                    if SPEED_OPTIMIZATIONS["roi_only_detection"]:
                        y1 += ROI_TOP
                        y2 += ROI_TOP

                    # Check whether center is in ROI (if not using ROI-only mode)
                    in_roi = True if SPEED_OPTIMIZATIONS["roi_only_detection"] else is_vehicle_in_roi([x1, y1, x2, y2], ROI_TOP, ROI_BOTTOM)

                    if in_roi:
                        frame_roi_detections += 1
                        roi_detections += 1

                        # Count types
                        if cls == 1 or cls == 3:  # bicycle or motorcycle
                            bike_detections += 1
                            frame_bike_detections += 1
                        elif cls == 2:  # car
                            car_detections += 1
                            frame_car_detections += 1

                    # Center coordinates
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # FAST CSV logging (single open file)
                    if csv_writer is not None:
                        csv_writer.writerow([
                            frame_count, int(track_id), cls, float(conf),
                            x1, y1, x2, y2,
                            center_x, center_y,
                            int(in_roi)
                        ])

                    # Visualization
                    if original_frame is not None and not SPEED_OPTIMIZATIONS["disable_display"]:
                        # Color selection
                        if cls == 1:
                            color = (255, 255, 0)  # bicycle - cyan
                        elif cls == 3:
                            color = (255, 0, 255)  # motorcycle - magenta
                        elif cls == 2:
                            color = (0, 255, 0)    # car - green
                        else:
                            color = (0, 165, 255)  # bus/truck - orange

                        thickness = 3 if in_roi else 1
                        cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, thickness)

                        label = f"{model.names[cls]} {conf:.2f}" if hasattr(model, "names") else f"cls:{cls} {conf:.2f}"
                        status = "ROI" if in_roi else "OUT"
                        label = f"ID:{track_id} | {status} | {label}"

                        # Clamp text y so it doesn't draw above frame
                        text_y = max(10, y1 - 10)
                        cv2.putText(original_frame, label, (x1, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.circle(original_frame, (center_x, center_y), 4, (0, 0, 255), -1)

                    frame_detections += 1
                    total_detections += 1

            # Periodically flush CSV to reduce data loss (and release buffered data)
            if csv_file is not None and (processed_frames % SPEED_OPTIMIZATIONS["csv_flush_interval"] == 0):
                try:
                    csv_file.flush()
                except Exception:
                    pass

            # Write output video if enabled and writer is available
            if out is not None and original_frame is not None:
                out.write(original_frame)

            # Display frame (if enabled)
            if original_frame is not None and not SPEED_OPTIMIZATIONS["disable_display"]:
                # Overlay info
                cv2.putText(original_frame, f"Frame: {frame_count}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(original_frame, f"Detections: {frame_detections}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(original_frame, f"ROI Detections: {frame_roi_detections}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(original_frame, f"Bikes: {frame_bike_detections} | Cars: {frame_car_detections}", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(original_frame, f"CPU Mode | Skip: {SPEED_OPTIMIZATIONS['frame_skip']}x", (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                display_frame = cv2.resize(original_frame, None, fx=0.4, fy=0.4)
                cv2.imshow(f"ULTRA-FAST CPU Mode (Skip {SPEED_OPTIMIZATIONS['frame_skip']}x)", display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n⏹️ Stopped by user")
                    break

            # Progress log
            if processed_frames % 10 == 0:
                efficiency = (roi_detections / total_detections * 100) if total_detections > 0 else 0
                print(f"⚡ Frame: {frame_count} | Processed: {processed_frames} | ROI: {roi_detections}/{total_detections} | Bikes: {bike_detections}")

    except KeyboardInterrupt:
        print("\n⏹️ Processing stopped by user")

    finally:
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        if not SPEED_OPTIMIZATIONS["disable_display"]:
            cv2.destroyAllWindows()

        # Close CSV
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass

        # Final statistics
        efficiency = (roi_detections / total_detections * 100) if total_detections > 0 else 0
        print(f"\n✅ ULTRA-FAST detection complete!")
        print(f"📊 Total Frames: {frame_count} | Processed: {processed_frames}")
        print(f"🎯 Total Detections: {total_detections}")
        print(f"🎯 ROI Detections: {roi_detections} ({efficiency:.1f}% efficiency)")
        print(f"🚴 Bicycle/Motorcycle: {bike_detections}")
        print(f"🚗 Cars: {car_detections}")
        if csv_writer is not None:
            print(f"📄 Data saved to: {CSV_PATH}")


if __name__ == "__main__":
    run_detection_with_roi()



# from ultralytics import YOLO
# import cv2
# import csv
# import os
# import numpy as np

# # ========== CONFIGURATION ==========

# MODEL_PATH = "../models/yolov8m.pt"  # <-- CHANGED: Model is now in '../models/'
# VIDEO_PATH = "../input/bridge_back.mp4" # <-- CHANGED: Input video is now in '../input/'
# CSV_PATH = "../output/vehicle_positions.csv" # <-- CHANGED: Output CSV is now in '../output/'
# OUTPUT_VIDEO = "../output/object_detection_roi_output.mp4" # <-- Suggested: Save the output video here

# # ========== GPU CONFIGURATION ==========
# USE_GPU = False  # ⚡ Set to True for GPU, False for CPU
# GPU_DEVICE = 0   # Which GPU to use (0 = first GPU)

# # ROI(will be set automatically)
# ROI_TOP = None
# ROI_BOTTOM = None

# # ========== VEHICLE CLASSES ==========
# VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

# # ========== SPEED OPTIMIZATION SETTINGS ==========
# SPEED_OPTIMIZATIONS = {
#     'frame_skip': 2,               # ⚡ Process every 3rd frame
#     'resize_ratio': 0.4,           # ⚡ Resize frames to 40% for faster processing
#     'disable_csv_logging': False,   # ⚡ Set to True to disable CSV logging for max speed
#     'disable_video_output': False,   # ⚡ DISABLED video output for max speed
#     'disable_display': False,        # ⚡ DISABLED preview window for max speed
#     'roi_only_detection': False,     # ⚡ Only detect in ROI region for max speed
# }

# # Simple confidence thresholds
# CONF_THRESHOLD = 0.2  # ⚡ Single threshold for all classes

# # ========== GPU CHECK FUNCTION ==========
# def check_gpu_availability():
#     """Check if GPU is available and properly configured"""
#     if torch.cuda.is_available():
#         gpu_count = torch.cuda.device_count()
#         print(f"✅ {gpu_count} GPU(s) detected:")
#         for i in range(gpu_count):
#             print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
#             print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
#         return True
#     else:
#         print("❌ CUDA not available - using CPU")
#         print("Possible reasons:")
#         print("  - PyTorch not installed with CUDA support")
#         print("  - NVIDIA drivers not installed")
#         print("  - GPU not compatible with CUDA")
#         return False

# # Use this in your main function
# gpu_available = check_gpu_availability()
# # ========== ROI SELECTION FUNCTION ==========
# def select_roi(video_path):
#     """Run your existing ROI selection script"""
#     cap = cv2.VideoCapture(video_path)
#     roi_points = []
    
#     def click_event(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             scale = param["scale"]
#             y_orig = int(y / scale)
#             roi_points.append(y_orig)
#             print(f"Clicked Y (original frame): {y_orig}")
    
#     display_scale = 0.25
#     print("🎯 Select ROI - Click TWO points (top and bottom) then press ESC")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_display = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
#         cv2.imshow("Select ROI - Click top and bottom points", frame_display)
#         cv2.setMouseCallback("Select ROI - Click top and bottom points", click_event, {"scale": display_scale})

#         # Draw instructions
#         cv2.putText(frame_display, "Click TOP then BOTTOM of ROI", (10, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#         cv2.putText(frame_display, "Press ESC when done", (10, 60),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#         if cv2.waitKey(1) & 0xFF == 27 or len(roi_points) >= 2:
#             break

#     cap.release()
#     cv2.destroyAllWindows()
    
#     if len(roi_points) >= 2:
#         roi_top, roi_bottom = sorted(roi_points)
#         print(f"✅ ROI Selected: top={roi_top}, bottom={roi_bottom}")
#         return roi_top, roi_bottom
#     else:
#         print("❌ ROI selection failed, using full frame")
#         return 0, 1080  # Default height

# # ========== VEHICLE INSIDE ROI CHECK ==========
# def is_vehicle_in_roi(box, roi_top, roi_bottom):
#     """Fast ROI check - simplified"""
#     x1, y1, x2, y2 = box
#     center_y = (y1 + y2) // 2
#     return roi_top <= center_y <= roi_bottom

# # ========== MAIN DETECTION FUNCTION ==========
# def run_detection_with_roi():
#     global ROI_TOP, ROI_BOTTOM
    
#     # Check GPU availability and load model accordingly
#     gpu_available = check_gpu_availability()
    
#     if USE_GPU and gpu_available:
#         try:
#             model = YOLO(MODEL_PATH)
#             device_name = f"GPU {GPU_DEVICE}"
#             print(f"🚀 Model loaded on {device_name}")
#         except Exception as e:
#             print(f"❌ Failed to load model on GPU: {e}")
#             print("🔄 Falling back to CPU...")
#             model = YOLO(MODEL_PATH)
#             device_name = "CPU"
#             USE_GPU = False
#     else:
#         model = YOLO(MODEL_PATH)
#         device_name = "CPU"
#         if USE_GPU and not gpu_available:
#             print("⚠️  GPU requested but not available, using CPU")
    
#     cap = cv2.VideoCapture(VIDEO_PATH)
    
#     if not cap.isOpened():
#         print("❌ Error: Could not open video!")
#         return

#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     # Select ROI first
#     ROI_TOP, ROI_BOTTOM = select_roi(VIDEO_PATH)
    
#     # Output video (only if enabled)
#     if not SPEED_OPTIMIZATIONS['disable_video_output']:
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

#     # Initialize CSV (only if enabled)
#     if not SPEED_OPTIMIZATIONS['disable_csv_logging']:
#         with open(CSV_PATH, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(["frame", "id", "class", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y", "in_roi"])

#     print("🚀 Starting ULTRA-FAST detection...")
#     print(f"📐 ROI: Y={ROI_TOP} to Y={ROI_BOTTOM}")
#     print(f"⚡ Device: {device_name}")
#     print(f"⚡ Optimizations: Frame skip={SPEED_OPTIMIZATIONS['frame_skip']}, Resize={SPEED_OPTIMIZATIONS['resize_ratio']}")
#     print(f"⚡ CSV Logging: {'ENABLED' if not SPEED_OPTIMIZATIONS['disable_csv_logging'] else 'DISABLED'}")
#     print(f"⚡ Video Output: {'ENABLED' if not SPEED_OPTIMIZATIONS['disable_video_output'] else 'DISABLED'}")
#     print(f"⚡ Display: {'ENABLED' if not SPEED_OPTIMIZATIONS['disable_display'] else 'DISABLED'}")
#     print(f"⚡ ROI-Only Detection: {'ENABLED' if SPEED_OPTIMIZATIONS['roi_only_detection'] else 'DISABLED'}")
#     print("Processing... Press Ctrl+C to stop\n")

#     frame_count = 0
#     processed_frames = 0
#     total_detections = 0
#     roi_detections = 0
#     bike_detections = 0
#     car_detections = 0

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_count += 1
            
#             # ⚡ FRAME SKIPPING - Process every Nth frame
#             if frame_count % SPEED_OPTIMIZATIONS['frame_skip'] != 0:
#                 continue
                
#             processed_frames += 1
            
#             # For visualization only (if display enabled)
#             if not SPEED_OPTIMIZATIONS['disable_display']:
#                 original_frame = frame.copy()
#                 # Draw ROI area
#                 roi_overlay = original_frame.copy()
#                 cv2.rectangle(roi_overlay, (0, ROI_TOP), (width, ROI_BOTTOM), (0, 255, 255), -1)
#                 cv2.addWeighted(roi_overlay, 0.1, original_frame, 0.9, 0, original_frame)
#                 # Draw ROI boundaries
#                 cv2.line(original_frame, (0, ROI_TOP), (width, ROI_TOP), (0, 255, 255), 2)
#                 cv2.line(original_frame, (0, ROI_BOTTOM), (width, ROI_BOTTOM), (0, 255, 255), 2)
            
#             # ========== ULTRA-FAST ROI-ONLY DETECTION ==========
#             if SPEED_OPTIMIZATIONS['roi_only_detection']:
#                 # ⚡ CROP to ROI region only for detection
#                 roi_height = ROI_BOTTOM - ROI_TOP
#                 if roi_height > 0:  # Ensure valid ROI
#                     roi_frame = frame[ROI_TOP:ROI_BOTTOM, 0:width]
                    
#                     # ⚡ RESIZE ROI frame for faster processing
#                     if SPEED_OPTIMIZATIONS['resize_ratio'] < 1.0:
#                         new_width = int(width * SPEED_OPTIMIZATIONS['resize_ratio'])
#                         new_height = int(roi_height * SPEED_OPTIMIZATIONS['resize_ratio'])
#                         detection_frame = cv2.resize(roi_frame, (new_width, new_height))
#                     else:
#                         detection_frame = roi_frame
#                         new_width, new_height = width, roi_height
#                 else:
#                     detection_frame = frame
#                     new_width, new_height = width, height
#             else:
#                 # Fallback to full frame detection
#                 if SPEED_OPTIMIZATIONS['resize_ratio'] < 1.0:
#                     new_width = int(width * SPEED_OPTIMIZATIONS['resize_ratio'])
#                     new_height = int(height * SPEED_OPTIMIZATIONS['resize_ratio'])
#                     detection_frame = cv2.resize(frame, (new_width, new_height))
#                 else:
#                     detection_frame = frame.copy()
#                     new_width, new_height = width, height
            
#             # Run detection on optimized frame with GPU/CPU selection
#             results = model.track(
#                 detection_frame,
#                 tracker="bytetrack.yaml",
#                 persist=True,
#                 conf=CONF_THRESHOLD,
#                 iou=0.5,
#                 classes=VEHICLE_CLASSES,
#                 verbose=False,
#                 device=GPU_DEVICE if USE_GPU else 'cpu'  # ⚡ GPU/CPU selection
#             )
            
#             frame_detections = 0
#             frame_roi_detections = 0
#             frame_bike_detections = 0
#             frame_car_detections = 0
            
#             if results[0].boxes is not None and len(results[0].boxes) > 0:
#                 boxes = results[0].boxes.xyxy.cpu().numpy()
#                 classes = results[0].boxes.cls.cpu().numpy().astype(int)
#                 confidences = results[0].boxes.conf.cpu().numpy()
                
#                 # Get tracking IDs
#                 if results[0].boxes.id is not None:
#                     track_ids = results[0].boxes.id.cpu().numpy().astype(int)
#                 else:
#                     track_ids = np.array([-1] * len(boxes))
                
#                 for i, (box, cls, conf, track_id) in enumerate(zip(boxes, classes, confidences, track_ids)):
#                     if conf >= CONF_THRESHOLD:
#                         # ⚡ Scale coordinates back to original size
#                         if SPEED_OPTIMIZATIONS['resize_ratio'] < 1.0:
#                             scale_x = width / new_width
#                             scale_y = (ROI_BOTTOM - ROI_TOP) / new_height if SPEED_OPTIMIZATIONS['roi_only_detection'] else height / new_height
#                             x1, y1, x2, y2 = [int(coord * scale_x) if i % 2 == 0 else int(coord * scale_y) 
#                                              for i, coord in enumerate(box)]
#                         else:
#                             x1, y1, x2, y2 = map(int, box)
                        
#                         # Adjust Y coordinates if using ROI-only detection
#                         if SPEED_OPTIMIZATIONS['roi_only_detection']:
#                             y1 += ROI_TOP
#                             y2 += ROI_TOP
                        
#                         # All detections are in ROI when using ROI-only mode
#                         in_roi = True if SPEED_OPTIMIZATIONS['roi_only_detection'] else is_vehicle_in_roi([x1, y1, x2, y2], ROI_TOP, ROI_BOTTOM)
                        
#                         if in_roi:
#                             frame_roi_detections += 1
#                             roi_detections += 1
                            
#                             # Count specific vehicle types
#                             if cls == 1 or cls == 3:  # bicycle or motorcycle
#                                 bike_detections += 1
#                                 frame_bike_detections += 1
#                             elif cls == 2:  # car
#                                 car_detections += 1
#                                 frame_car_detections += 1
                        
#                         # Calculate center
#                         center_x = (x1 + x2) // 2
#                         center_y = (y1 + y2) // 2
                        
#                         # ⚡ CSV logging (only if enabled)
#                         if not SPEED_OPTIMIZATIONS['disable_csv_logging']:
#                             with open(CSV_PATH, "a", newline="") as f:
#                                 writer = csv.writer(f)
#                                 writer.writerow([
#                                     frame_count, int(track_id), cls, float(conf),
#                                     x1, y1, x2, y2, center_x, center_y, int(in_roi)
#                                 ])
                        
#                         # Visualization only if display enabled
#                         if not SPEED_OPTIMIZATIONS['disable_display']:
#                             # Choose color based on vehicle type
#                             if cls == 1:  # bicycle
#                                 color = (255, 255, 0)  # Cyan
#                             elif cls == 3:  # motorcycle
#                                 color = (255, 0, 255)  # Magenta
#                             elif cls == 2:  # car
#                                 color = (0, 255, 0)    # Green
#                             else:  # bus/truck
#                                 color = (0, 165, 255)  # Orange
                            
#                             thickness = 3 if in_roi else 1
                            
#                             cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, thickness)
                            
#                             # Draw label
#                             label = f"{model.names[cls]} {conf:.2f}"
#                             status = "ROI" if in_roi else "OUT"
#                             label = f"ID:{track_id} | {status} | {label}"
                            
#                             cv2.putText(original_frame, label, (x1, y1 - 10),
#                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            
#                             # Draw center point
#                             cv2.circle(original_frame, (center_x, center_y), 4, (0, 0, 255), -1)
                        
#                         frame_detections += 1
#                         total_detections += 1
            
#             # ⚡ Video output (only if enabled)
#             if not SPEED_OPTIMIZATIONS['disable_video_output'] and not SPEED_OPTIMIZATIONS['disable_display']:
#                 out.write(original_frame)
            
#             # Display frame (only if enabled)
#             if not SPEED_OPTIMIZATIONS['disable_display']:
#                 # Add frame info
#                 cv2.putText(original_frame, f"Frame: {frame_count}", (20, 40),
#                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#                 cv2.putText(original_frame, f"Detections: {frame_detections}", (20, 70),
#                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#                 cv2.putText(original_frame, f"ROI Detections: {frame_roi_detections}", (20, 100),
#                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 cv2.putText(original_frame, f"Bikes: {frame_bike_detections} | Cars: {frame_car_detections}", (20, 130),
#                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#                 cv2.putText(original_frame, f"Device: {device_name} | Skip: {SPEED_OPTIMIZATIONS['frame_skip']}x", (20, 160),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
#                 display_frame = cv2.resize(original_frame, None, fx=0.4, fy=0.4)
#                 cv2.imshow(f"ULTRA-FAST {device_name} Mode (Skip {SPEED_OPTIMIZATIONS['frame_skip']}x)", display_frame)
                
#                 # Exit on 'Q' press
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     print("\n⏹️ Stopped by user")
#                     break
            
#             # Progress update
#             if processed_frames % 10 == 0:
#                 efficiency = (roi_detections / total_detections * 100) if total_detections > 0 else 0
#                 print(f"⚡ Frame: {frame_count} | Processed: {processed_frames} | ROI: {roi_detections}/{total_detections} | Bikes: {bike_detections} | Device: {device_name}")

#     except KeyboardInterrupt:
#         print("\n⏹️ Processing stopped by user")

#     # Cleanup
#     cap.release()
#     if not SPEED_OPTIMIZATIONS['disable_video_output']:
#         out.release()
#     if not SPEED_OPTIMIZATIONS['disable_display']:
#         cv2.destroyAllWindows()
    
#     # Final statistics
#     efficiency = (roi_detections / total_detections * 100) if total_detections > 0 else 0
    
#     print(f"\n✅ ULTRA-FAST detection complete!")
#     print(f"📊 Total Frames: {frame_count} | Processed: {processed_frames}")
#     print(f"🎯 Total Detections: {total_detections}")
#     print(f"🎯 ROI Detections: {roi_detections} ({efficiency:.1f}% efficiency)")
#     print(f"🚴 Bicycle/Motorcycle: {bike_detections}")
#     print(f"🚗 Cars: {car_detections}")
#     print(f"⚡ Processing Device: {device_name}")
#     if not SPEED_OPTIMIZATIONS['disable_csv_logging']:
#         print(f"📄 Data saved to: {CSV_PATH}")

# # ========== RUN THE DETECTION ==========
# if __name__ == "__main__":
#     run_detection_with_roi()