Intelligent Traffic Violation Detection System
=============================================

Overview
--------
This project chains multiple computer-vision tasks to detect key traffic violations from camera footage. 
A YOLOv8-based detector tracks vehicles inside a configurable region of interest (ROI), filters the footage down to the most useful crops, 
and runs specialized classifiers for specific violations:
- **Helmet compliance** for bikes/motorcycles
- **Seatbelt usage** for cars and other enclosed vehicles
- **Phone distraction** in still images
- **Red light jumping** using per-frame signal simulation
- **Unsafe following distance / vehicle closeness**

Each step logs CSV evidence, writes annotated imagery/video, and keeps intermediate artifacts 
so you can audit or re-run downstream analyses.

Repository Layout
-----------------
- `scripts/object_detection.py` – entry point for YOLOv8 vehicle tracking with ROI selection, CSV logging, and optional annotated-video export.
- `scripts/detected_photos_crop.py` – consumes `vehicle_positions.csv` plus the source video to crop per-vehicle snapshots (organized by ID under `output/cropped_photos`).
- `scripts/best_SS.py` – ranks crops per vehicle using sharpness/contrast/size heuristics and renames the best image to `BEST.jpg`.
- `scripts/helmet_detection.py` – runs the helmet model on `Bikes_Motorcycles/id_xxx` crops, saves annotated violations, and logs `helmet_results.csv`.
- `scripts/seatbelt_detection.py` – runs the seatbelt model on `Cars/id_xxx` crops, writes `seatbelt_results.csv`, and saves annotated detections.
- `scripts/vehicle_closeness.py` – replays the video with CSV positions to highlight unsafe proximity pairs, exporting video plus `violations.csv` / `all_pairs.csv`.
- `scripts/red_light.py` – simulates signal cycles and flags vehicles crossing the stop line during the red phase, saving an annotated video.
- `scripts/phone_detection.py` – standalone utility for spotting phone usage in arbitrary photos (drop images in the instructed folder).
- `Models/` – YOLO checkpoints (`yolov8m.pt`, `helmet.pt`, `seatbelt.pt`, `phone_distractions.pt`) already referenced by the scripts.
- `output/` – working directory for CSV logs, cropped imagery, and rendered videos (e.g., `vehicle_positions.csv`, `object_detection_roi_output.mp4`, `vehicle_closeness_check/*`).
- `requirements.txt` – pinned Conda environment (Python 3.10.19, torch 2.9.1+cpu, OpenCV 4.12, Ultralytics 8.3.228, pandas/polars/matplotlib, etc.).

End-to-End Pipeline
-------------------
1. **Vehicle Detection & Tracking**
   - Run `python scripts/object_detection.py`.
   - The script prompts you to click the ROI top/bottom on the input video (`input/bridge_back.mp4` by default).
   - It tracks COCO vehicle classes (1,2,3,5,7), logs every detection to `output/vehicle_positions.csv`, and optionally writes an annotated video.
2. **Targeted Cropping**
   - `python scripts/detected_photos_crop.py`
   - Reads the detection CSV plus the source video to extract multiple crops per track ID, with asymmetric padding biased toward the direction of travel.
   - Stores crops under `output/cropped_photos/Bikes_Motorcycles/id_###` and `.../Cars/id_###`.
3. **Best Screenshot Selection (optional)**
   - `python scripts/best_SS.py`
   - Scores each crop on size, sharpness, contrast, frame recency, and brightness to retain the single “BEST.jpg” per vehicle (and optionally delete the others).
4. **Violation-Specific Models**
   - **Helmet** – `python scripts/helmet_detection.py`; outputs annotated violations in `output/helmet_results/Bikes_Motorcycles` plus `helmet_results.csv`.
   - **Seatbelt** – `python scripts/seatbelt_detection.py`; outputs annotated detections in `output/seatbelt_results/Cars` plus `seatbelt_results.csv`.
   - **Phone Distraction** – `python scripts/phone_detection.py`; waits for you to drop still images into `phone_detection_test/test_images`, then annotates and logs results in that folder.
5. **Scenario Analyses**
   - **Red Light** – `python scripts/red_light.py` replays the video, simulates configurable red/green durations, and highlights vehicles crossing the stop line while red.
   - **Vehicle Closeness** – `python scripts/vehicle_closeness.py` uses per-frame centers to find pairs within `DISTANCE_THRESHOLD` pixels, writing both violation-only and all-pairs CSVs plus a video overlay.

Configuration Highlights
------------------------
- Paths at the top of each script define input/output locations. Defaults assume:
  - Videos live in `input/`
  - Models live in `Models/`
  - All outputs go under `output/`
- `object_detection.py` exposes a `SPEED_OPTIMIZATIONS` dictionary for frame skipping, resize ratio, ROI-only inference, and display / logging toggles.
- `detected_photos_crop.py` controls crop frequency (`max_per_id`, `zones_per_id`, `max_per_zone`) and asymmetric padding to prefer front-facing captures.
- `helmet_detection.py` and `seatbelt_detection.py` treat non-detections as violations by default; adjust logic if you need stricter confidence handling.
- `vehicle_closeness.py` can exclude bikes (`INCLUDE_BIKES=False`) or adjust `DISTANCE_THRESHOLD` to match camera geometry.
- `phone_detection.py` requires manual image placement and supports AVIF-to-JPG conversion fallback.

Running the Project
-------------------
1. **Create the environment**
   ```bash
   conda env create -n itvds -f requirements.txt
   conda activate itvds
   ```
2. **Place assets**
   - Put the raw surveillance video in `input/`.
   - Ensure the pretrained models exist in `Models/` (already included).
3. **Execute modules**
   - Detection/tracking → cropping → (optional) best screenshot → violation-specific analyses, as outlined above.
4. **Review outputs**
   - CSV logs (`output/*.csv`) for structured evidence.
   - Annotated videos/images inside `output/` subdirectories for visual proof.

Extending / Customizing
-----------------------
- Swap `MODEL_PATH` references with finetuned YOLO weights if you retrain on local data.
- Adjust ROI coordinates or add polygon-based ROI filtering if the scene changes.
- Integrate the CSV outputs into downstream dashboards, ticketing systems, or legal evidence packs.
- Expand the pipeline with additional Ultralytics tasks (e.g., license-plate OCR) by following the existing script conventions for path handling and logging.

Support Files
-------------
- `Models/seatbelt.pt`, `Models/helmet.pt`, `Models/phone_distractions.pt`, `Models/yolov8m.pt`
- `output/vehicle_positions.csv`, `output/helmet_detection_results.csv`, `output/seatbelt_all_results.csv`, plus rendered videos like `object_detection_roi_output.mp4`, `redlight_out.mp4`, and `vehicle_closeness_check/vehicle_closeness_output.mp4`.

With the above workflow, you can run only the modules relevant to your study (e.g., red-light violations) or execute the entire chain for a multi-violation audit.
