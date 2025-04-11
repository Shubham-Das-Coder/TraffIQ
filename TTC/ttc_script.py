import os
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from sort import Sort
from tqdm import tqdm

################################################################################
# Helper Function: Generate Unique Output Paths
################################################################################
def get_unique_path(path):
    """
    If 'path' already exists, return a new path with ' (1)', ' (2)', etc. appended before the extension.
    Example:
      'file.csv' -> 'file (1).csv' -> 'file (2).csv'
    """
    if not os.path.exists(path):
        return path  # no conflict, just use this path
    
    base, ext = os.path.splitext(path)
    i = 1
    new_path = f"{base} ({i}){ext}"
    while os.path.exists(new_path):
        i += 1
        new_path = f"{base} ({i}){ext}"
    return new_path

################################################################################
# Helper Functions: IoU, Speed/Distance, TTC
################################################################################
def iou(boxA, boxB):
    """
    Compute Intersection-over-Union for two boxes in [x1, y1, x2, y2] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    if (boxAArea + boxBArea - interArea) == 0:
        return 0
    return interArea / float(boxAArea + boxBArea - interArea)

def estimate_speed_distance(prev_positions, current_positions, fps):
    """
    Estimate speed (km/h) and distance (m) based on movement of object centers
    between consecutive frames.
    """
    speeds = {}
    distances = {}
    for obj_id, curr_pos in current_positions.items():
        if obj_id in prev_positions:
            prev_pos = prev_positions[obj_id]
            pixel_dist = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
            # Example calibration: 1 pixel ~ 0.035 m
            real_world_dist = pixel_dist * 0.035
            # Convert m/s -> km/h (fps frames/sec)
            speed_kmh = real_world_dist * fps * 3.6
            speeds[obj_id] = speed_kmh
            distances[obj_id] = real_world_dist
    return speeds, distances

def compute_ttc(distance_m, speed_kmh):
    """
    Compute Time to Collision (TTC) in seconds, given distance (m) and speed (km/h).
    """
    if speed_kmh > 0:
        speed_m_s = speed_kmh / 3.6  # convert km/h to m/s
        return distance_m / speed_m_s
    return float('inf')

################################################################################
# Main Script
################################################################################

# 1) Check device (GPU or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 2) Load YOLO model
yolo_model = YOLO('yolov8n.pt')
yolo_model.to(device)  # move model to GPU if available

# 3) Initialize SORT tracker
tracker = Sort()

# 4) Directories
input_folder = r"D:\TTC_Model\Input_Videos"
output_folder = r"D:\TTC_Model\Output_Videos"
csv_folder = r"D:\TTC_Model\Output_Csv"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)

# 5) Gather all .mp4 or .mov files
video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.mov', '.avi'))]

for video_file in video_files:
    video_path = os.path.join(input_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
        continue

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Build default output paths
    csv_output_path = os.path.join(
        csv_folder,
        video_file.lower().replace('.mp4', '.csv').replace('.mov', '.csv')
    )
    out_video_path = os.path.join(
        output_folder,
        video_file.lower().replace('.mp4', '_out.mp4').replace('.mov', '_out.mp4')
    )

    # 6) Ensure uniqueness if files exist
    csv_output_path = get_unique_path(csv_output_path)
    out_video_path = get_unique_path(out_video_path)

    print(f"\nProcessing video: {video_file}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total Frames: {total_frames}")
    print(f"CSV output: {csv_output_path}")
    print(f"Video output: {out_video_path}")

    # 7) Create VideoWriter for annotated output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width, frame_height))

    prev_positions = {}
    data_records = []
    frame_count = 0

    pbar = tqdm(total=total_frames, desc=f"Processing {video_file}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_s = frame_count / fps

        # 8) Run YOLO detection on GPU
        results = yolo_model(frame, device=device)

        # 9) Parse YOLO results -> [x1, y1, x2, y2, conf]
        detections = []
        detection_classes = []
        for r in results:
            if hasattr(r, 'boxes'):
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])  # YOLO class index
                    detections.append([x1, y1, x2, y2, conf])
                    detection_classes.append(cls_id)

        # 10) Convert to np.array for SORT
        if len(detections) == 0:
            dets_np = np.empty((0, 5))
        else:
            dets_np = np.array(detections)

        # 11) Update SORT
        tracked_objects = tracker.update(dets_np)

        # Collect current positions for speed/distance
        current_positions = {}
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            current_positions[obj_id] = (center_x, center_y)

        # 12) Speeds, distances, TTC
        speeds, distances = estimate_speed_distance(prev_positions, current_positions, fps)

        # 13) Match tracked objects to best IoU detection -> class name
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            tracked_box = [x1, y1, x2, y2]

            best_iou = 0
            best_class_id = -1
            for i, det in enumerate(detections):
                box_det = det[:4]
                iou_val = iou(tracked_box, box_det)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_class_id = detection_classes[i]

            # YOLO class name
            if best_class_id >= 0:
                class_name = yolo_model.names[best_class_id]
            else:
                class_name = "unknown"

            speed_kmh = speeds.get(obj_id, 0)
            distance_m = distances.get(obj_id, 0)
            ttc_s = compute_ttc(distance_m, speed_kmh)

            # 14) Annotate frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_y = y1 - 40
            cv2.putText(frame, f"{class_name} (ID:{obj_id})",
                        (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            label_y += 20
            cv2.putText(frame, f"Speed: {speed_kmh:.1f} km/h",
                        (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            label_y += 20
            cv2.putText(frame, f"Dist: {distance_m:.1f} m",
                        (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            label_y += 20
            cv2.putText(frame, f"TTC: {ttc_s:.1f} s",
                        (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)

            # 15) Record data for CSV
            data_records.append([
                timestamp_s,
                obj_id,
                class_name,
                speed_kmh,
                distance_m,
                ttc_s
            ])

        # Write annotated frame
        out_video.write(frame)

        # Prepare next iteration
        prev_positions = current_positions.copy()
        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out_video.release()

    # 16) Save CSV
    df = pd.DataFrame(
        data_records,
        columns=["Time_s", "Object_ID", "Class", "Speed_kmh", "Distance_m", "TTC_s"]
    )
    df.to_csv(csv_output_path, index=False)

    print(f"\nSaved annotated video to: {out_video_path}")
    print(f"Saved CSV to: {csv_output_path}")

cv2.destroyAllWindows()
print("\nAll videos processed. TTC data saved to CSV and annotated videos saved.")
