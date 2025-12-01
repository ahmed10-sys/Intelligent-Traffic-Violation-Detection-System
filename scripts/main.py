import os
import sys
import subprocess
from datetime import datetime


# ============================================
# CONFIGURATION FLAGS
# ============================================

# Toggle individual steps here (set to False to skip)
ENABLE_OBJECT_DETECTION = True    # 1) object_detection.py
ENABLE_CROPPING = True                # 2) detected_photos_crop.py
ENABLE_BEST_SS = True               # 3) best_SS.py
ENABLE_LICENSE_PLATE_DETECTOR = True   # 4) license_plate_detector.py
ENABLE_PLATE_READING = True            # 5) plate_reader.py
ENABLE_SEATBELT_DETECTION = True         # 5) seatbelt_detection.py
ENABLE_HELMET_DETECTION = True        # 6) helmet_detection.py
ENABLE_RED_LIGHT = False               # 7) red_light.py
ENABLE_VEHICLE_CLOSENESS = False      # 8) vehicle_closeness.py (distance)
ENABLE_PHONE_DETECTION = False        # 9) phone_detection.py (image-based, interactive)
ENABLE_CONSOLIDATION = True           # 10) consolidate_results.py


# Known outputs (for quick reference / summary only)
OUTPUT_MAP = {
    "object_detection.py": "../output/vehicle_positions.csv",
    "detected_photos_crop.py": "../output/cropped_photos/",
    "best_SS.py": "../output/cropped_photos/**/BEST.jpg",
    "license_plate_detector.py": "../output/cropped_photos/**/BEST_PLATE_CROP.jpg",
    "plate_reader.py": "../output/master_violation_log.csv",
    "seatbelt_detection.py": "../output/seatbelt_results.csv",
    "helmet_detection.py": "../output/helmet_results.csv",
    "red_light.py": "../output/redlight_out.mp4",
    "vehicle_closeness.py": "../output/vehicle_closeness_check/violations.csv",
    "phone_detection.py": "C:/Users/syedm/OneDrive/Documents/Traffic Violation System/phone_detection_test/phone_detection_results.csv",
    "consolidate_results.py": "../output/final_consolidated_violations.csv",
}


def run_script(label: str, filename: str, enabled: bool = True) -> None:
    """
    Run another script in this folder as a subprocess.
    Uses the same Python executable that launched main.py.
    """
    if not enabled:
        print(f"\n[SKIP] {label} ({filename}) is disabled in main.py")
        return

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(scripts_dir, filename)

    if not os.path.exists(script_path):
        print(f"\n[WARN] {label}: script not found at {script_path}")
        return

    print("\n" + "=" * 70)
    print(f"▶ Starting: {label}  [{filename}]")
    print("=" * 70)

    try:
        subprocess.run([sys.executable, script_path], check=True)
        print(f"✅ Finished: {label}")
    except subprocess.CalledProcessError as e:
        print(f"❌ {label} failed with exit code {e.returncode}")
    except Exception as e:  # pragma: no cover - defensive
        print(f"❌ {label} failed with unexpected error: {e}")


def write_run_summary(start_time: datetime, end_time: datetime) -> None:
    """Write a simple summary file listing main outputs."""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.normpath(os.path.join(scripts_dir, "../output/run_summary.txt"))
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    lines = [
        "Intelligent Traffic Violation - Pipeline Run Summary",
        "-" * 60,
        f"Started: {start_time.isoformat(timespec='seconds')}",
        f"Finished: {end_time.isoformat(timespec='seconds')}",
        "",
        "Main outputs (relative to scripts folder unless absolute):",
    ]

    for script, out_path in OUTPUT_MAP.items():
        lines.append(f"- {script:25s} -> {out_path}")

    lines.append("")
    lines.append("Note: Files are only created if the corresponding script ran successfully.")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n📄 Run summary written to: {summary_path}")


def main() -> None:
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    start_time = datetime.now()

    print("=" * 70)
    print("🚦 Intelligent Traffic Violation Detection - Main Pipeline")
    print("=" * 70)
    print(f"Scripts directory : {scripts_dir}")
    print(f"Python executable : {sys.executable}")
    print(f"Start time        : {start_time.isoformat(timespec='seconds')}")
    print("")
    print("Execution order:")
    print("  1) object_detection.py        -> vehicle_positions.csv")
    print("  2) detected_photos_crop.py    -> cropped_photos/*")
    print("  3) best_SS.py                 -> BEST.jpg per vehicle")
    print("  4) license_plate_detector.py  -> BEST_PLATE_CROP.jpg")
    print("  5) plate_reader.py            -> master_violation_log.csv")
    print("  6) seatbelt_detection.py      -> seatbelt_results.csv")
    print("  7) helmet_detection.py        -> helmet_results.csv")
    print("  8) red_light.py               -> redlight_out.mp4")
    print("  9) vehicle_closeness.py       -> violations.csv (toggle)")
    print("  10) phone_detection.py        -> phone_detection_results.csv (toggle, interactive)")
    print("  11) consolidate_results.py    -> final_consolidated_violations.csv")
    print("")
    print("Step toggles:")
    print(f"  Object detection      : {ENABLE_OBJECT_DETECTION}")
    print(f"  Cropping              : {ENABLE_CROPPING}")
    print(f"  Best screenshot (BEST): {ENABLE_BEST_SS}")
    print(f"  License Plate Detector: {ENABLE_LICENSE_PLATE_DETECTOR}")
    print(f"  Plate reading         : {ENABLE_PLATE_READING}")
    print(f"  Seatbelt detection    : {ENABLE_SEATBELT_DETECTION}")
    print(f"  Helmet detection      : {ENABLE_HELMET_DETECTION}")
    print(f"  Red light detection   : {ENABLE_RED_LIGHT}")
    print(f"  Vehicle closeness     : {ENABLE_VEHICLE_CLOSENESS}")
    print(f"  Phone detection       : {ENABLE_PHONE_DETECTION}")
    print(f"  Consolidation         : {ENABLE_CONSOLIDATION}")
    print("=" * 70)

    # 1) Detection & tracking → vehicle_positions.csv
    run_script(
        "Object detection + tracking (CSV)",
        "object_detection.py",
        enabled=ENABLE_OBJECT_DETECTION,
    )

    # 2) Crop vehicles from video using vehicle_positions.csv
    run_script(
        "Crop detected vehicles",
        "detected_photos_crop.py",
        enabled=ENABLE_CROPPING,
    )

    # 3) Select BEST.jpg per vehicle (for better plate/seatbelt/helmet detection)
    run_script(
        "Best screenshot selection (BEST.jpg per vehicle)",
        "best_SS.py",
        enabled=ENABLE_BEST_SS,
    )

    # 4) License plate detection (Crops plate from BEST.jpg)
    run_script(
        "License plate detection",
        "license_plate_detector.py",
        enabled=ENABLE_LICENSE_PLATE_DETECTOR,
    )

    # 5) License plate reading on BEST_PLATE_CROP.jpg
    run_script(
        "License plate reading",
        "plate_reader.py",
        enabled=ENABLE_PLATE_READING,
    )

    # 5) Seatbelt detection on cropped Cars
    run_script(
        "Seatbelt detection",
        "seatbelt_detection.py",
        enabled=ENABLE_SEATBELT_DETECTION,
    )

    # 6) Helmet detection on cropped Bikes/Motorcycles
    run_script(
        "Helmet detection",
        "helmet_detection.py",
        enabled=ENABLE_HELMET_DETECTION,
    )

    # 7) Red-light violation detection using vehicle_positions.csv
    run_script(
        "Red light violation detection",
        "red_light.py",
        enabled=ENABLE_RED_LIGHT,
    )

    # 8) Optional: Vehicle closeness analysis using vehicle_positions.csv
    run_script(
        "Vehicle closeness detection",
        "vehicle_closeness.py",
        enabled=ENABLE_VEHICLE_CLOSENESS,
    )

    # 9) Optional: Phone detection demo (uses its own test images folder)
    run_script(
        "Phone detection (image-based demo)",
        "phone_detection.py",
        enabled=ENABLE_PHONE_DETECTION,
    )

    # 10) Consolidate all results into one master CSV
    run_script(
        "Consolidate Results",
        "consolidate_results.py",
        enabled=ENABLE_CONSOLIDATION,
    )

    end_time = datetime.now()
    write_run_summary(start_time, end_time)

    print("\n" + "=" * 70)
    print("🎉 Main pipeline finished")
    print("=" * 70)


if __name__ == "__main__":
    main()


