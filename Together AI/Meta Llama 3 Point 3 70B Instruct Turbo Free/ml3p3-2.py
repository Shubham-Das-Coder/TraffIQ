import cv2
import os
from ultralytics import YOLO
from dotenv import load_dotenv
from together import Together
from collections import defaultdict, Counter
import time

# Load API key
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in .env file")

client = Together(api_key=api_key)

# Load YOLO model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy if you have GPU

def get_llm_comment(detection_counts):
    if not detection_counts:
        prompt = "There are no vehicles or people visible in the frame. What could be inferred about the traffic?"
    else:
        items = ", ".join([f"{v} {k}(s)" for k, v in detection_counts.items()])
        prompt = f"The current frame shows: {items}. Give a brief comment on the traffic situation."

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"

def process_video_live(video_path, target_fps=5, analysis_interval=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(video_fps / target_fps)
    frame_idx = 0

    detection_buffer = []
    last_analysis_time = time.time()
    llm_feedback = "Analyzing..."

    print(f"[INFO] Running real-time analysis at ~{target_fps} FPS")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            results = model(frame, verbose=False)[0]
            annotated_frame = results.plot()

            current_frame_detections = []
            for cls_id in results.boxes.cls:
                class_name = model.names[int(cls_id)]
                current_frame_detections.append(class_name)

            detection_buffer.extend(current_frame_detections)

            # Every few frames, run LLM for feedback
            if len(detection_buffer) >= analysis_interval:
                counts = Counter(detection_buffer)
                llm_feedback = get_llm_comment(counts)
                detection_buffer.clear()

            # Display LLM feedback on frame
            cv2.rectangle(annotated_frame, (10, 5), (640, 60), (0, 0, 0), -1)
            y_offset = 25
            for line in llm_feedback.split('\n'):
                cv2.putText(annotated_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                y_offset += 20

            # Show the frame
            cv2.imshow("Real-Time Traffic Analysis (Press Q to exit)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = r"D:/Shubham/1101-143052492.mp4"
    process_video_live(video_path, target_fps=5, analysis_interval=10)

if __name__ == "__main__":
    main()
