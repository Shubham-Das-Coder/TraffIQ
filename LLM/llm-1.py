import cv2
import torch
from transformers import pipeline

# Load object detection model (YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load LLM model using Hugging Face (change "gpt2" to another HF-supported model if needed)
llm = pipeline("text-generation", model="gpt2")

# Load video
cap = cv2.VideoCapture("E:/Shubham/Data/IMG_1328.MOV")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, 20.0, (640, 480))

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
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # LLM Traffic Analysis (GPT-2)
    description = llm(f"Analyze this traffic scenario: {label}")
    print(description[0]["generated_text"])

    # Write frame to output video
    out.write(frame)

    cv2.imshow("Traffic Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
