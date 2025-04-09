import cv2
import torch
from together import Together

# Initialize Together client (API key must be set)
client = Together(api_key='tgp_v1_SX55lua8c1mUbcpSooZoF6Sinto3hHNm0j7rVjgiII8')  # Replace with your actual API key

# Load object detection model (YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load video
cap = cv2.VideoCapture("E:/Shubham/Data/IMG_1328.MOV")

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (frame_width, frame_height))

# Open a text file to save LLM messages
log_file = open("traffic_analysis.txt", "w")

# Start processing the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate the frame 180 degrees to correct orientation
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Object Detection
    results = model(frame)

    # Process detections and overlay on frame
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        
        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Create the message for Llama-Vision-Free
        message = f"Analyze this traffic scenario: {label}"

        # Send request to Llama-Vision-Free
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[{"role": "user", "content": message}],
        )

        # Get the analysis from the response
        analysis = response.choices[0].message.content
        print(analysis)

        # Save LLM output to a text file
        log_file.write(f"{analysis}\n\n")

        # Overlay LLM analysis message on the video
        y_offset = min(int(xyxy[1]) - 30, frame_height - 50)  # Ensure text is within frame
        cv2.putText(frame, analysis[:50], (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to output video
    out.write(frame)

    # Display the frame
    cv2.imshow("Traffic Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()
