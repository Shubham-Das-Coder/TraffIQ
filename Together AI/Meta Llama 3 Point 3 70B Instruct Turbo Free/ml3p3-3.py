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

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can switch to yolov8s.pt if your GPU supports it

# Function to send frame-specific detection summary to LLM
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

    print(f"\n[LLM Prompt] {prompt}")

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}]
        )
        comment = response.choices[0].message.content.strip()
        print(f"[LLM Feedback] {comment}")
        return comment
    except Exception as e:
        error_msg = f"[LLM ERROR] {str(e)}"
        print(error_msg)
        return error_msg

# Process video frame-by-frame and get real-time feedback
def analyze_video_real_time(video_path, target_fps=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(video_fps / target_fps)
    frame_idx = 0

    print(f"\n[INFO] Processing video at {target_fps} FPS (skipping every {frame_skip} frames)")
    print("[INFO] Press 'q' to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # Run YOLO detection
            results = model(frame, verbose=False)[0]
            annotated_frame = results.plot()

            current_detections = []
            for cls_id in results.boxes.cls:
                class_name = model.names[int(cls_id)]
                current_detections.append(class_name)

            # Log detections
            if current_detections:
                print(f"[Frame {frame_idx}] Detections: {current_detections}")
            else:
                print(f"[Frame {frame_idx}] No objects detected")

            # Get feedback from LLM for this frame
            feedback = get_llm_frame_analysis(current_detections)

            # Display feedback on video frame
            cv2.rectangle(annotated_frame, (10, 5), (640, 70), (0, 0, 0), -1)
            y_offset = 25
            for line in feedback.split('\n'):
                cv2.putText(annotated_frame, line, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            # Show frame
            cv2.imshow("Real-Time Traffic & Collision Detection", annotated_frame)

            # Exit on key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Exit requested by user.")
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Video processing finished.")

def main():
    video_path = r"D:/Shubham/1101-143052492.mp4"
    analyze_video_real_time(video_path, target_fps=15)

if __name__ == "__main__":
    main()
