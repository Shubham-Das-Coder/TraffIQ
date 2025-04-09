import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("E:/Shubham/Data/IMG_1328.MOV")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_straight_lanes.mp4", fourcc, fps, (frame_width, frame_height))

def region_of_interest(image):
    """Applies a mask to focus only on the lanes."""
    height, width = image.shape[:2]
    
    # Define a polygon mask (focus on bottom half of frame)
    mask = np.zeros_like(image)
    polygon = np.array([[
        (50, height),
        (width // 2 - 100, height // 2 + 50),
        (width // 2 + 100, height // 2 + 50),
        (width - 50, height)
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)

def is_near_straight_line(x1, y1, x2, y2):
    """Checks if a line is nearly vertical (between 70° - 110°)."""
    angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi  # Convert radians to degrees
    return 70 <= angle <= 110  # Allow more tilted lanes

def detect_lanes(frame):
    """Detects only near-straight lanes in a given frame."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Apply Region of Interest Mask
    masked_edges = region_of_interest(edges)

    # Use Hough Transform to detect lanes
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    # Draw detected lanes
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Filter only near-straight lines
            if is_near_straight_line(x1, y1, x2, y2):
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return frame

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate frame if needed
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Detect lanes
    processed_frame = detect_lanes(frame)

    # Write the processed frame to output video
    out.write(processed_frame)

    # Display frame
    cv2.imshow("Straight Lane Detection", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
