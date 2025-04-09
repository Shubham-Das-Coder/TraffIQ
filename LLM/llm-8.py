import cv2
import torch
import time
from scipy.spatial.distance import euclidean
from together import Together

# Initialize Together client for LLM
client = Together(api_key="tgp_v1_SX55lua8c1mUbcpSooZoF6Sinto3hHNm0j7rVjgiII8")  # Replace with your actual API key

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load video
cap = cv2.VideoCapture("E:/Shubham/Data/IMG_1328.MOV")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Reference scale (in meters per pixel, needs calibration with a real-world object)
scale_factor = 0.04  # Example: 1 pixel = 0.04 meters

# Initialize previous frame information
prev_frame_time = None
prev_positions = {}  # Store previous positions per vehicle
prev_speeds = {}  # Store previous speeds per vehicle

# Set colors
box_color = (0, 255, 0)  # Green for bounding box
text_color = (0, 255, 255)  # Yellow for text

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate frame if it's upside down
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Perform object detection
    results = model(frame)
    
    # Store new frame positions
    current_positions = {}

    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        # Get bounding box center
        x_center = (xyxy[0] + xyxy[2]) / 2
        y_center = (xyxy[1] + xyxy[3]) / 2
        vehicle_id = f"{x_center}-{y_center}"  # Assign an ID (needs a better tracking system)
        
        # Draw bounding box
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), box_color, 2)

        # Check previous position to calculate speed
        if vehicle_id in prev_positions and prev_frame_time is not None:
            displacement = euclidean(prev_positions[vehicle_id], (x_center, y_center))
            current_time = time.time()
            time_diff = current_time - prev_frame_time

            # Avoid division by zero
            if time_diff > 0:
                speed = (displacement * scale_factor) / time_diff  # Speed in meters per second
                prev_speeds[vehicle_id] = speed  # Store speed

                # Display speed
                cv2.putText(frame, f"Speed: {speed:.2f} m/s", (int(xyxy[0]), int(xyxy[1]) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

                # Calculate Time to Collision (TTC)
                if vehicle_id in prev_speeds:
                    distance_to_next_vehicle = abs(x_center - prev_positions[vehicle_id][0]) * scale_factor
                    relative_speed = abs(prev_speeds[vehicle_id] - speed)

                    if relative_speed > 0:
                        ttc = distance_to_next_vehicle / relative_speed
                        cv2.putText(frame, f"TTC: {ttc:.2f} s", (int(xyxy[0]), int(xyxy[1]) - 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        # Store current position
        current_positions[vehicle_id] = (x_center, y_center)

        # LLM Analysis
        message = f"Analyze this traffic scenario: Vehicle detected at ({x_center:.2f}, {y_center:.2f}) with speed {prev_speeds.get(vehicle_id, 0):.2f} m/s."
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[{"role": "user", "content": message}],
        )

        # Get LLM output and display it
        analysis = response.choices[0].message.content
        cv2.putText(frame, analysis[:50], (10, min(int(xyxy[1]) - 60, frame.shape[0] - 50)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # Update previous frame data
    prev_positions = current_positions
    prev_frame_time = time.time()

    # Display frame
    cv2.imshow("Traffic Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
