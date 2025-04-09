import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
from norfair import Tracker, Detection
from transformers import pipeline

# Load YOLOv8-Tiny model (lightweight and fast)
model = YOLO("yolov8n.pt")

# Load video
cap = cv2.VideoCapture("E:/Shubham/Data/IMG_1328.MOV")
fps = int(cap.get(cv2.CAP_PROP_FPS))
scale_factor = 0.04  # Pixels to meters conversion

# Initialize tracker (ByteTrack)
tracker = Tracker(distance_function="euclidean", distance_threshold=30)

# Load NLP Model for Traffic Insights (Using DistilBERT)
nlp_model = pipeline("text-generation", model="distilgpt2")

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate frame if needed
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Perform YOLOv8 detection
    results = model(frame, stream=True)
    
    detections = []
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box.astype(int)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # ✅ FIX: Use np.array([[x, y]]) instead of tuple
            detections.append(Detection(points=np.array([[center_x, center_y]])))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Update tracker
    tracked_objects = tracker.update(detections=detections)

    for obj in tracked_objects:
        obj_id = obj.id
        x, y = obj.estimate[0]

        # Speed calculation (meters per second)
        if hasattr(obj, "prev_position"):
            displacement = np.linalg.norm(np.array([x, y]) - np.array(obj.prev_position))
            time_diff = time.time() - obj.prev_time
            speed = (displacement * scale_factor) / time_diff if time_diff > 0 else 0

            # Display speed
            cv2.putText(frame, f"Speed: {speed:.2f} m/s", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Time to Collision (TTC)
            if speed > 0:
                ttc = abs(y - frame.shape[0]) / speed
                cv2.putText(frame, f"TTC: {ttc:.2f} s", (int(x), int(y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            obj.prev_speed = speed
        obj.prev_position = (x, y)
        obj.prev_time = time.time()

    # LLM Analysis (Using DistilBERT instead of Together AI)
    prompt = "Analyze traffic patterns based on vehicle speed and collision risks."
    analysis = nlp_model(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]

    # Display AI-generated insights
    cv2.putText(frame, analysis[:50], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display frame
    cv2.imshow("Traffic Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
