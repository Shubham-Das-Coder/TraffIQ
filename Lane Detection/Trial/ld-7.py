import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    # Focus area where lanes usually appear (bottom half triangle)
    polygon = np.array([[
        (int(0.1 * width), height),
        (int(0.9 * width), height),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.45 * width), int(0.6 * height))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def filter_lines_by_angle(lines):
    filtered = []
    if lines is None:
        return filtered

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            slope = float('inf')
        else:
            slope = dy / dx

        # Only keep lines within ±45° to vertical (slope between ~1 and ~infinity)
        if 1 <= abs(slope) <= 10:
            filtered.append((x1, y1, x2, y2))
    
    return filtered

# Load video
cap = cv2.VideoCapture("D:/Shubham/1101-143052492.mp4")  # Replace with your actual video path

if not cap.isOpened():
    print("Error opening video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color thresholds
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask_combined = cv2.bitwise_or(mask_white, mask_yellow)
    masked_img = cv2.bitwise_and(frame, frame, mask=mask_combined)

    # Preprocessing
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    roi_edges = region_of_interest(edges)

    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)
    filtered_lines = filter_lines_by_angle(lines)

    # Draw filtered lines
    for x1, y1, x2, y2 in filtered_lines:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imshow("Lane Detection - Press Q to quit", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
