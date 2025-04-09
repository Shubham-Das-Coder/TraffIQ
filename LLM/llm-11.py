import cv2
import torch
from ultralytics import YOLO
from transformers import pipeline

# Load the vehicle detection model (YOLOv8)
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano for efficiency

# Load the LLM for traffic analysis
llm = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

# Define video path
video_path = "E:/Shubham/Data/IMG_1328.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Video writer to save processed video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

# Process video frame by frame
frame_count = 0
vehicle_count = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate frame if upside down
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Detect vehicles
    results = model(frame)
    detected_vehicles = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_vehicles += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Append vehicle count per frame
    vehicle_count.append(detected_vehicles)
    frame_count += 1

    # Display video
    cv2.imshow("Traffic Analysis", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Generate Traffic Insights using LLM
analysis_prompt = f"Analyze the traffic from a video where {sum(vehicle_count)/frame_count:.2f} vehicles are detected per frame."
traffic_analysis = llm(analysis_prompt, max_length=150)

print("Traffic Analysis Report:")
print(traffic_analysis[0]["generated_text"])
