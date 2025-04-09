# Relative Speed

import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Define video path
video_path = "E:/Shubham/Data/IMG_1328.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Set lower resolution for better performance
new_width = 640
new_height = 360

# Scaling factor (meters per pixel) - Adjust based on calibration
scale_m_per_px = 0.04  # Example scale (4 cm per pixel, adjust as needed)

# Get YOLO class names for filtering
yolo_classes = model.names

# Define allowed classes (traffic-related)
allowed_classes = [
    "car", "truck", "bus", "motorbike", "bicycle", "person", "traffic light"
]

# Define ROI Adjustments
top_y = int(0.4 * new_height)  # 60% from the bottom
mid_x = int(0.5 * new_width)   # Middle of the screen

# Adjusted bottom points
bottom_y = int(0.85 * new_height)
left_x = int(0.15 * new_width)
right_x = int(0.85 * new_width)

# Define new triangular ROI
triangle_roi = np.array([
    [left_x, bottom_y],
    [right_x, bottom_y],
    [mid_x, top_y]
], np.int32)

# Video writer for processed output
out = cv2.VideoWriter('output_resized.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (new_width, new_height))

# Store previous frame positions for speed calculation
previous_positions = {}  # {object_id: (x, y, timestamp)}

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (new_width, new_height))
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Create a mask for the ROI
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [triangle_roi], (255, 255, 255))
    roi_frame = cv2.bitwise_and(frame, mask)

    results = model(roi_frame)
    current_time = time.time()
    
    for result in results:
        if result.boxes is not None:
            for box, cls_id in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls_id.item())
                class_name = yolo_classes[class_id]

                if class_name in allowed_classes:
                    if cv2.pointPolygonTest(triangle_roi, (x1, y1), False) >= 0 or \
                       cv2.pointPolygonTest(triangle_roi, (x2, y2), False) >= 0:
                        
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        object_id = f"{class_name}_{x1}_{y1}"

                        if object_id in previous_positions:
                            prev_x, prev_y, prev_time = previous_positions[object_id]
                            dx = center_x - prev_x
                            dy = center_y - prev_y
                            pixel_speed = np.sqrt(dx**2 + dy**2) / (current_time - prev_time)
                            real_speed = pixel_speed * scale_m_per_px
                            cv2.putText(frame, f"Speed: {real_speed:.2f} m/s", 
                                        (center_x, center_y - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        previous_positions[object_id] = (center_x, center_y, current_time)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.polylines(frame, [triangle_roi], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.imshow("Traffic Analysis", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
