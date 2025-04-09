import cv2
import os
from ultralytics import YOLO
from dotenv import load_dotenv
from together import Together
from collections import defaultdict
from tqdm import tqdm

# Load API Key from .env
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in .env file")

client = Together(api_key=api_key)

# Load YOLOv8 Model (nano version for speed)
model = YOLO("yolov8n.pt")

def analyze_video(video_path, target_fps=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(video_fps / target_fps))

    print(f"[INFO] Video FPS: {video_fps:.2f}, Analyzing every {skip} frames (~{target_fps} FPS)")
    frame_idx = 0
    detections = defaultdict(int)

    pbar = tqdm(total=total_frames // skip, desc="Analyzing video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            # Run YOLO detection
            results = model(frame)[0]
            annotated_frame = results.plot()

            # Count each detected object
            for cls_id in results.boxes.cls:
                class_name = model.names[int(cls_id)]
                detections[class_name] += 1

            # Show annotated frame
            cv2.imshow("Traffic Detection (Press Q to exit)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            pbar.update(1)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    pbar.close()
    return detections

def summarize_detections(detections):
    if not detections:
        return "No traffic-related objects were detected in the video."

    summary = ", ".join(f"{count} {name}(s)" for name, count in detections.items())
    return f"The video contained the following objects: {summary}."

def analyze_with_llm(summary):
    prompt = (
        f"{summary} Based on these objects, provide a detailed analysis of the traffic situation. "
        "Discuss congestion, safety concerns, and any notable environmental or urban context."
    )

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content  # fixed here

def main():
    video_path = r"D:/Shubham/1101-143052492.mp4"
    print(f"[INFO] Starting video analysis on: {video_path}")

    detections = analyze_video(video_path, target_fps=5)
    summary = summarize_detections(detections)

    print("\nYOLOv8 Detection Summary:")
    print(summary)

    print("\nSending summary to LLM for deeper traffic insight...")
    llm_response = analyze_with_llm(summary)

    print("\nLLM Analysis:")
    print(llm_response)

if __name__ == "__main__":
    main()
