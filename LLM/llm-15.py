import cv2
from ultralytics import YOLO
from transformers import pipeline

# Load the vehicle detection model (YOLOv8)
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano for efficiency

# Load the DistilGPT-2 model for traffic analysis (general-purpose text generation)
traffic_model = pipeline("text-generation", model="distilgpt2")

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

# Define dimensions for ROI (1/6th of the screen, lower middle section)
roi_width = frame_width // 3
roi_height = frame_height // 2

# Process video frame by frame
frame_count = 0
vehicle_count = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate frame if upside down
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Draw the division lines on the video (6 parts)
    # Upper half (horizontal line)
    cv2.line(frame, (0, frame_height // 2), (frame_width, frame_height // 2), (0, 255, 0), 2)

    # Three subdivisions in the upper half (vertical lines)
    cv2.line(frame, (frame_width // 3, 0), (frame_width // 3, frame_height // 2), (0, 255, 0), 2)
    cv2.line(frame, (2 * frame_width // 3, 0), (2 * frame_width // 3, frame_height // 2), (0, 255, 0), 2)

    # Lower half (horizontal line)
    cv2.line(frame, (0, frame_height), (frame_width, frame_height), (0, 255, 0), 2)

    # Three subdivisions in the lower half (vertical lines)
    cv2.line(frame, (frame_width // 3, frame_height // 2), (frame_width // 3, frame_height), (0, 255, 0), 2)
    cv2.line(frame, (2 * frame_width // 3, frame_height // 2), (2 * frame_width // 3, frame_height), (0, 255, 0), 2)

    # Draw a box for the lower middle section (1/6th of the video)
    roi_x1 = frame_width // 3
    roi_y1 = frame_height // 2
    roi_x2 = roi_x1 + roi_width
    roi_y2 = roi_y1 + roi_height
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

    # Crop the ROI (lower middle section) for analysis
    roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # Detect vehicles in the ROI
    results = model(roi_frame)
    detected_vehicles = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_vehicles += 1
            cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Append vehicle count per frame
    vehicle_count.append(detected_vehicles)
    frame_count += 1

    # Generate Traffic Insights using DistilGPT-2
    average_vehicles = sum(vehicle_count) / frame_count if frame_count > 0 else 0
    analysis_prompt = f"Analyze the traffic from a video where {average_vehicles:.2f} vehicles are detected per frame."
    traffic_analysis = traffic_model(analysis_prompt, max_length=150)

    # Get the generated text from the model
    analysis_text = traffic_analysis[0]["generated_text"]

    # Overlay the analysis text on the video
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Traffic Analysis: {analysis_text}", (20, frame_height - 40), font, 0.8, (255, 255, 255), 2)

    # Display the video frame with the text
    cv2.imshow("Traffic Analysis", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
