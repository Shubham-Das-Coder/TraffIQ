"""
Canny Edge Detection
Region of Interest (ROI) Masking
Hough Line Transform
Slope and Intercept Calculation
Averaging of Lane Lines
"""

import cv2
import numpy as np
import screeninfo  # To detect screen size

# Set your video path here
video_path = "D:/Shubham/Data/IMG_1338.MOV"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Detect screen resolution automatically
screen = screeninfo.get_monitors()[0]
screen_width = screen.width
screen_height = screen.height

def preprocess_frame(frame):
    """Preprocess the frame: grayscale -> blur -> Canny edge detection -> mask ROI."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Define a region of interest (ROI) polygon
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.6)),
        (0, int(height * 0.6))
    ]])

    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    return masked_edges

def average_slope_intercept(lines):
    """Separate left and right lane lines based on slope and average them."""
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:  # avoid division by zero
            continue

        slope = (y2 - y1) / (x2 - x1)

        if abs(slope) < 0.5:  # ignore near-horizontal lines
            continue

        intercept = y1 - slope * x1

        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    left_lane = np.mean(left_lines, axis=0) if left_lines else None
    right_lane = np.mean(right_lines, axis=0) if right_lines else None

    return left_lane, right_lane

def make_line_points(y1, y2, line):
    """Convert slope and intercept into pixel points."""
    if line is None:
        return None

    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return ((x1, y1), (x2, y2))

def detect_and_draw_lanes(edges, frame):
    """Detect lanes using Hough Line Transform and draw averaged left/right lanes."""
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    if lines is None:
        return

    left_lane, right_lane = average_slope_intercept(lines)

    height = frame.shape[0]
    y1 = height
    y2 = int(height * 0.6)

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    if left_line is not None:
        cv2.line(frame, left_line[0], left_line[1], (0, 255, 0), 8)

    if right_line is not None:
        cv2.line(frame, right_line[0], right_line[1], (0, 255, 0), 8)

# Create a named window and set it fullscreen
cv2.namedWindow('Corrected and Cleaned Lane Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Corrected and Cleaned Lane Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🧹 Flip the frame vertically and horizontally (180 degree rotate)
    frame = cv2.flip(frame, -1)

    # Preprocess the frame
    edges = preprocess_frame(frame)

    # Detect lanes and draw them
    detect_and_draw_lanes(edges, frame)

    # Draw ROI polygon for visualization
    height, width, _ = frame.shape
    roi_polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.6)),
        (0, int(height * 0.6))
    ]])
    cv2.polylines(frame, roi_polygon, isClosed=True, color=(255, 0, 0), thickness=2)

    # Resize frame to full screen
    resized_frame = cv2.resize(frame, (screen_width, screen_height))

    # Show the resized frame
    cv2.imshow('Corrected and Cleaned Lane Detection', resized_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
