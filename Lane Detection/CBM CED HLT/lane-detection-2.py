import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def is_valid_line(x1, y1, x2, y2, angle_thresh=25):
    """Filter out nearly horizontal lines based on angle threshold."""
    angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
    return abs(angle) > angle_thresh

def cluster_and_filter_lines(lines, eps=60, min_samples=1):
    """
    Group lines by their midpoints using DBSCAN clustering.
    This helps to retain well-separated lane lines.
    """
    if len(lines) == 0:
        return []

    # Compute midpoints of each line
    midpoints = np.array([[(line[0][0] + line[0][2]) // 2] for line in lines])

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(midpoints)
    labels = clustering.labels_

    # For each cluster, keep the longest line
    filtered_lines = []
    for label in set(labels):
        cluster_lines = [lines[i] for i in range(len(lines)) if labels[i] == label]
        longest_line = max(cluster_lines, key=lambda l: np.linalg.norm([l[0][2] - l[0][0], l[0][3] - l[0][1]]))
        filtered_lines.append(longest_line)

    return filtered_lines

cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 0)  # Flip vertically if needed

    height, width = frame.shape[:2]
    roi = frame[int(height * 0.5):, :]  # Bottom half of the frame only

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define white and yellow lane color masks
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    result = cv2.bitwise_and(roi, roi, mask=combined_mask)

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    if lines is not None:
        # Step 1: Filter out nearly horizontal lines
        valid_lines = [line for line in lines if is_valid_line(*line[0])]

        # Step 2: Cluster and filter lines based on midpoints (x-coordinates)
        final_lines = cluster_and_filter_lines(valid_lines)

        # Step 3: Draw final filtered lines
        for line in final_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1 + height // 2), (x2, y2 + height // 2), (0, 255, 0), 3)

    # Show result
    cv2.imshow("Enhanced Lane Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
