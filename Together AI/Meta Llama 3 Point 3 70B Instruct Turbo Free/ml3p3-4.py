import cv2
import os
from ultralytics import YOLO
from dotenv import load_dotenv
from together import Together
from collections import Counter

# Load API key from .env file
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in .env file")

# Initialize Together client
client = Together(api_key=api_key)

# Load YOLO model
model = YOLO("yolov8n.pt")  # Change to 'yolov8s.pt' or better if you want more accuracy and speed is okay

# Function to get response from LLM
def get_llm_frame_analysis(detections: list):
    if not detections:
        prompt = (
            "There are no objects detected in this frame from a car dashboard camera. "
            "Is the road empty or could this be due to poor visibility? Give your observation."
        )
    else:
        counts = Counter(detections)
        detail = ", ".join([f"{v} {k}(s)" for k, v in counts.items()])
        prompt = (
            f"This dashboard camera frame shows: {detail}. "
            "Is there any risk of collision or dangerous traffic situation? Give a safety comment."
        )

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}]
        )
        comment = response.choices[0].message.content.strip()
        print(f"[LLM] {comment}")
        return comment
    except Exception as e:
        error_msg = f"[LLM ERROR] {str(e)}"
        print(error_msg)
        return error_msg

# Process the video with detection + LLM
def analyze_video_real_time(video_path, target_fps=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(int(video_fps / target_fps), 1)
    frame_idx = 0

    print(f"\n[INFO] Processing at {target_fps} FPS (every {frame_skip} frames). Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            results = model(frame, verbose=False)[0]
            annotated_frame = results.plot()

            detected_objects = [
                model.names[int(cls_id)] for cls_id in results.boxes.cls
            ]

            feedback = get_llm_frame_analysis(detected_objects)

            # Draw feedback on frame
            cv2.rectangle(annotated_frame, (10, 5), (640, 70), (0, 0, 0), -1)
            y_offset = 25
            for line in feedback.split('\n'):
                cv2.putText(annotated_frame, line, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            cv2.imshow("Real-Time Traffic Analysis with LLM", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Stopped by user.")
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")

def main():
    video_path = r"D:/Shubham/1101-143052492.mp4"
    analyze_video_real_time(video_path, target_fps=5)

if __name__ == "__main__":
    main()
