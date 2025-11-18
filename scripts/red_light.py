import cv2
import pandas as pd
from typing import Set

def process_red_light_violations(
    video_path: str,
    csv_path: str,
    output_path: str,
    line_y: int = 300,
    signal_x: int = 50,
    signal_y: int = 50,
    signal_size: int = 50,
    red_duration: int = 150,
    green_duration: int = 150,
    display_width: int = 1280  # NEW: Max width for the cv2.imshow window
) -> int:
    """
    Processes a video to detect red-light violations based on vehicle tracking data.
    The frames are resized for display to handle large resolutions like 4K.

    Args:
        video_path: Path to the input video file.
        csv_path: Path to the input CSV file containing vehicle positions.
        output_path: Path for the output video file.
        line_y: The y-coordinate of the stop line.
        signal_x: The x-coordinate of the top-left corner of the traffic signal box.
        signal_y: The y-coordinate of the top-left corner of the traffic signal box.
        signal_size: The side length of the square traffic signal box.
        red_duration: The duration (in frames) the signal is RED.
        green_duration: The duration (in frames) the signal is GREEN.
        display_width: The maximum width to use for the cv2.imshow window.

    Returns:
        The total number of unique vehicles that committed a violation.
    """
    
    # Load data
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Vehicle position CSV not found at {csv_path}")
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file at {video_path}")
        return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate height for display while maintaining aspect ratio
    display_height = int(height * display_width / width)
    
    # Create the VideoWriter for the output file (uses original resolution)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_no = 0
    violations: Set[int] = set()

    # Create the display window and set it to be resizable
    cv2.namedWindow("Red Light Violation Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Red Light Violation Detection", display_width, display_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        # Simulated traffic signal
        cycle = red_duration + green_duration
        if (frame_no % cycle) < red_duration:
            signal_state = "RED"
            color = (0, 0, 255)  # BGR Red
        else:
            signal_state = "GREEN"
            color = (0, 255, 0)  # BGR Green
            
        cv2.rectangle(frame, (signal_x, signal_y), (signal_x + signal_size, signal_y + signal_size), color, -1)
        cv2.putText(frame, f"Signal: {signal_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Draw Stop line
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2) # BGR Blue

        # Filter Vehicle for current frame
        frame_data = data[data["frame"] == frame_no]

        for _, row in frame_data.iterrows():
            vid = int(row["id"])
            cx, cy = int(row["center_x"]), int(row["center_y"])

            # Draw vehicle Center
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1) # BGR Yellow

            # Check Violation: vehicle center y-coordinate is past the stop line AND signal is RED
            if cy > line_y and signal_state == "RED":
                violations.add(vid)
                cv2.putText(frame, "🚨 Violation", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display violation count
        cv2.putText(frame, f"Violations : {len(violations)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 1. Write the full-resolution frame to the output file
        out.write(frame)
        
        # 2. Resize the frame for a monitor-friendly display
        display_frame = cv2.resize(frame, (display_width, display_height))
        
        # 3. Show the resized frame
        cv2.imshow("Red Light Violation Detection", display_frame)
        
        if cv2.waitKey(1) & 0xff == 27: # Press 'Esc' to exit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return len(violations)

if __name__ == "__main__":
    # --- Configuration for Standalone Execution ---
    VIDEO_PATH = "../input/bridge_back.mp4"
    CSV_PATH = "../output/vehicle_positions.csv"
    OUTPUT_PATH = "../output/redlight_out.mp4"
    
    # Run the main processing function
    violations_count = process_red_light_violations(
        video_path=VIDEO_PATH,
        csv_path=CSV_PATH,
        output_path=OUTPUT_PATH,
        display_width=1280 # Use 1280p width for viewing, change this value if needed
    )

    # Final output messages
    print(f"✅ Done! {violations_count} violations detected.")
    print(f"🎥 Output saved as {OUTPUT_PATH}")