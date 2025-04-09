import cv2
import torch
import time
from scipy.spatial.distance import euclidean

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load video
cap = cv2.VideoCapture("E:/Shubham/Data/IMG_1328.MOV")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Reference scale (in meters per pixel, needs calibration with a real-world object)
scale_factor = 0.04  # Example: 1 pixel = 0.04 meters

# Initialize previous frame information
prev_frame_time = None
prev_position = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate frame if it's upside down (180 degrees)
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Perform object detection
    results = model(frame)
    
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        # Get bounding box center
        x_center = (xyxy[0] + xyxy[2]) / 2
        y_center = (xyxy[1] + xyxy[3]) / 2
        
        # Calculate displacement and speed if there's previous data
        if prev_position is not None and prev_frame_time is not None:
            # Calculate distance traveled (displacement)
            displacement = euclidean(prev_position, (x_center, y_center))
            
            # Time difference between frames (in seconds)
            current_time = time.time()
            time_diff = current_time - prev_frame_time
            
            # Avoid division by zero if time_diff is too small
            if time_diff > 0:
                # Calculate speed (in meters per second)
                speed = (displacement * scale_factor) / time_diff
                # Print speed
                print(f"Speed: {speed:.2f} meters per second")
            else:
                print("Skipping speed calculation due to zero time difference")

        # Update previous frame data
        prev_position = (x_center, y_center)
        prev_frame_time = time.time()

    # Display frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
