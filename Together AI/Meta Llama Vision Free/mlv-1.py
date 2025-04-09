from together import Together
from dotenv import load_dotenv
import os
import cv2
import base64

# Load variables from .env file
load_dotenv()
api_key = os.getenv("API_KEY")

# Initialize the client
client = Together(api_key=api_key)

# Load video and extract a representative frame (middle frame)
def extract_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError("Could not read frame from video")

    # Save to temp and encode as base64
    temp_path = "frame.jpg"
    cv2.imwrite(temp_path, frame)

    with open(temp_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    return image_base64

# Provide the path to your traffic video
video_path = "dashboard_traffic_video.mp4"
image_base64 = extract_frame(video_path)

# Create vision prompt
vision_prompt = {
    "type": "image_url",
    "image_url": {
        "url": f"data:image/jpeg;base64,{image_base64}"
    }
}

# Define the full prompt
response = client.chat.completions.create(
    model="meta-llama/Llama-Vision-Free",
    messages=[
        {
            "role": "user",
            "content": [
                vision_prompt,
                {"type": "text", "text": "Describe the traffic situation and surroundings in this image from a car's dashboard."}
            ]
        }
    ],
)

# Print the response
print(response.choices[0].message.content)
