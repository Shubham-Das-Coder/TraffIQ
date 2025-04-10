# enhanced_lane_detection.py
import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    # Define a triangular ROI (assuming camera is front-facing on a car)
    polygon = np.array([
        [(int(0.1 * width), height),
         (int(0.9 * width), height),
         (int(0.55 * width), int(0.6 * height)),
         (int(0.45 * width), int(0.6 * height))]
    ])
    cv2.fillPoly(mask, polygon, 255)

    # Return masked image
    return cv2.bitwise_and(img, mask)

def filter_lines_by_angle(lines):
    filtered = []
    if lines is None:
        return filtered

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:  # vertical line
            slope = float('inf')
        else:
            slope = dy / dx

        angle = np.arctan(slope) * 180 / np.pi

        # Keep lines within ~45° left or right from vertical (slope between ~1 and ~infinity)
        if 0.8 <= abs(slope) <= 10:  # avoid horizontal or extremely steep lines
            filtered.append((x1, y1, x2, y2))
    
    return filtered

# Load the video
cap = cv2.VideoCapture("your_video.mp4")  # Replace with your video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color masks
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask_combined = cv2.bitwise_or(mask_white, mask_yellow)
    filtered = cv2.bitwise_and(frame, frame, mask=mask_combined)

    # Convert to grayscale, blur, and detect edges
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Apply ROI mask
    cropped_edges = region_of_interest(edges)

    # Hough Line Detection
    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    filtered_lines = filter_lines_by_angle(lines)

    # Draw lines
    for x1, y1, x2, y2 in filtered_lines:
        color = (0, 255, 0) if (x2 - x1) > 0 else (255, 0, 0)  # green for right, blue for left
        cv2.line(frame, (x1, y1), (x2, y2), color, 3)

    cv2.imshow("Enhanced Lane Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
