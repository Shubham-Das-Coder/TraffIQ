# Relative Speed in Pixels per Second

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

# Get YOLO class names for filtering
yolo_classes = model.names

# Define allowed classes (traffic-related)
allowed_classes = [
    "car", "truck", "bus", "motorbike", "bicycle", "person", "traffic light"
]

# Define ROI Adjustments
top_y = int(0.4 * new_height)  # 60% from the bottom (100% - 60% = 40% from top)
mid_x = int(0.5 * new_width)   # Middle of the screen

# Adjusted bottom points at 10% height from bottom (raising them)
bottom_y = int(0.85 * new_height)  
left_x = int(0.15 * new_width)  # Shift left point inward
right_x = int(0.85 * new_width) # Shift right point inward

# Define new triangular ROI with adjusted bottom vertices
triangle_roi = np.array([
    [left_x, bottom_y],  # Raised bottom-left
    [right_x, bottom_y], # Raised bottom-right
    [mid_x, top_y]       # Top-center (60% from bottom)
], np.int32)

# Video writer for processed output
out = cv2.VideoWriter('output_resized.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (new_width, new_height))

# Store previous frame positions for speed calculation
previous_positions = {}  # {object_id: (x, y, time)}

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster computation
    frame = cv2.resize(frame, (new_width, new_height))

    # Rotate frame if upside down
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Create a mask for the ROI
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [triangle_roi], (255, 255, 255))  # White triangle

    # Apply the mask to get only the ROI area
    roi_frame = cv2.bitwise_and(frame, mask)

    # Detect vehicles only within the ROI
    results = model(roi_frame)

    current_time = time.time()

    for result in results:
        if result.boxes is not None:  # Ensure detections exist
            for box, cls_id in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls_id.item())  # Get class ID
                class_name = yolo_classes[class_id]  # Get class name

                # Filter only traffic-related objects
                if class_name in allowed_classes:
                    # Check if the bounding box is inside the triangular ROI
                    if cv2.pointPolygonTest(triangle_roi, (x1, y1), False) >= 0 or \
                       cv2.pointPolygonTest(triangle_roi, (x2, y2), False) >= 0:
                        # Calculate center of the bounding box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Calculate speed by comparing with previous frame positions
                        object_id = f"{class_name}_{x1}_{y1}"  # A unique ID based on the object
                        if object_id in previous_positions:
                            prev_x, prev_y, prev_time = previous_positions[object_id]

                            # Calculate displacement (dx, dy) between current and previous frame
                            dx = center_x - prev_x
                            dy = center_y - prev_y

                            # Calculate speed (pixels per second)
                            speed = np.sqrt(dx**2 + dy**2) / (current_time - prev_time)  # pixels/sec
                            cv2.putText(frame, f"Speed: {speed:.2f} px/s", (center_x, center_y - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Update the position and timestamp for the object
                        previous_positions[object_id] = (center_x, center_y, current_time)
                        
                        # Draw bounding box and label on the frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                        cv2.putText(frame, f"{class_name}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw the triangular ROI on the frame (for visualization)
    cv2.polylines(frame, [triangle_roi], isClosed=True, color=(255, 0, 0), thickness=2)

    # Display video with ROI, detections, and speed
    cv2.imshow("Traffic Analysis", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
