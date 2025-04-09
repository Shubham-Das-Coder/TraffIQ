import cv2
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
import math

# Initialize YOLOv8 (object detection)
model = YOLO("yolov8n.pt")  # Lightweight model

# Initialize LLM (for traffic insights)
llm_pipeline = pipeline("text-generation", model="distilgpt2")  # Free & fast

# Traffic parameters
vehicle_classes = [2, 3, 5, 7]  # COCO: car, bike, bus, truck
track_history = {}  # Stores vehicle tracks for speed estimation
vehicle_count = 0

def estimate_speed(p1, p2, fps=30, px_to_m=0.1):
    """Estimate speed in km/h using pixel movement."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    distance_px = math.sqrt(dx**2 + dy**2)
    distance_m = distance_px * px_to_m  # Calibrate for your camera
    speed_mps = distance_m * fps
    speed_kmh = speed_mps * 3.6
    return speed_kmh

def query_llm(prompt):
    """Get traffic insights from a free LLM."""
    response = llm_pipeline(
        prompt,
        max_length=100,
        temperature=0.7,
    )
    return response[0]['generated_text']

def process_video(video_path):
    global vehicle_count
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Rotate frame 180° if upside-down
        frame = cv2.rotate(frame, cv2.ROTATE_180)  # Fixes upside-down video
        
        # Detect & track vehicles
        results = model.track(frame, persist=True, classes=vehicle_classes)
        
        # Draw bounding boxes & track IDs
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Speed estimation
                if track_id in track_history:
                    prev_pos = track_history[track_id]
                    speed = estimate_speed(prev_pos, (x1, y1))
                    cv2.putText(frame, f"{speed:.1f} km/h", (x1, y1 - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                track_history[track_id] = (x1, y1)
        
        # Update vehicle count
        vehicle_count = len(track_history)
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow("Traffic Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Generate LLM report
    prompt = f"Analyze traffic with {vehicle_count} vehicles. Provide insights."
    report = query_llm(prompt)
    print("\n🚦 LLM Traffic Report:")
    print(report)

if __name__ == "__main__":
    video_path = "E:/Shubham/Data/IMG_1328.MOV"  # Your video path
    process_video(video_path)