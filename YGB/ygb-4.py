import cv2
from ultralytics import YOLO

# Define COCO class labels (for YOLOv5/YOLOv8 models)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'N/A', 'dining table', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load the vehicle detection model (YOLOv8)
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano for efficiency

# Define video path
video_path = "E:/Shubham/Data/IMG_1328.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define new lower resolution for video (e.g., 640x360)
new_width = 640
new_height = 360

# Video writer to save processed video (output resolution matches the resized frames)
out = cv2.VideoWriter('output_resized.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (new_width, new_height))

# Define dimensions for ROI (1/6th of the screen, lower middle section)
roi_width = new_width // 3
roi_height = new_height // 2

# Process video frame by frame
frame_count = 0
vehicle_count = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to reduce resolution
    frame = cv2.resize(frame, (new_width, new_height))

    # Rotate frame if upside down
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Draw the division lines on the video (6 parts)
    # Upper half (horizontal line)
    cv2.line(frame, (0, new_height // 2), (new_width, new_height // 2), (0, 255, 0), 2)

    # Three subdivisions in the upper half (vertical lines)
    cv2.line(frame, (new_width // 3, 0), (new_width // 3, new_height // 2), (0, 255, 0), 2)
    cv2.line(frame, (2 * new_width // 3, 0), (2 * new_width // 3, new_height // 2), (0, 255, 0), 2)

    # Lower half (horizontal line)
    cv2.line(frame, (0, new_height), (new_width, new_height), (0, 255, 0), 2)

    # Three subdivisions in the lower half (vertical lines)
    cv2.line(frame, (new_width // 3, new_height // 2), (new_width // 3, new_height), (0, 255, 0), 2)
    cv2.line(frame, (2 * new_width // 3, new_height // 2), (2 * new_width // 3, new_height), (0, 255, 0), 2)

    # Draw a box for the lower middle section (1/6th of the video)
    roi_x1 = new_width // 3
    roi_y1 = new_height // 2
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
            # Extracting the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box without class name or confidence score
            cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            detected_vehicles += 1

    # Append vehicle count per frame
    vehicle_count.append(detected_vehicles)
    frame_count += 1

    # Display the video frame with detected bounding boxes and ROI
    cv2.imshow("Traffic Analysis", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
