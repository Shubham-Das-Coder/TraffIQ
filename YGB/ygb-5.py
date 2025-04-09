import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")  

# Define video path
video_path = "E:/Shubham/Data/IMG_1328.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define new lower resolution for better performance
new_width = 640
new_height = 360

# Video writer for processed output
out = cv2.VideoWriter('output_resized.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (new_width, new_height))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster computation
    frame = cv2.resize(frame, (new_width, new_height))

    # Rotate frame if upside down
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Define triangular ROI
    triangle_roi = np.array([
        [0, new_height],  # Bottom-left corner
        [new_width, new_height],  # Bottom-right corner
        [new_width // 2, new_height // 2]  # Middle-top vertex
    ], np.int32)

    # Create a mask for the ROI
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [triangle_roi], (255, 255, 255))  # White triangle

    # Apply the mask to get only the ROI area
    roi_frame = cv2.bitwise_and(frame, mask)

    # Detect vehicles only within the ROI
    results = model(roi_frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Only process bounding boxes inside the triangular ROI
            if cv2.pointPolygonTest(triangle_roi, (x1, y1), False) >= 0 or \
               cv2.pointPolygonTest(triangle_roi, (x2, y2), False) >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

    # Draw the triangular ROI on the frame (for visualization)
    cv2.polylines(frame, [triangle_roi], isClosed=True, color=(255, 0, 0), thickness=2)

    # Display video with ROI and detections
    cv2.imshow("Traffic Analysis", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
