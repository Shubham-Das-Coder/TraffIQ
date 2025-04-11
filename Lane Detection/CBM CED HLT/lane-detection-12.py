import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def is_valid_lane_line(x1, y1, x2, y2, min_angle=20, max_angle=160):
    """Filter out nearly-horizontal or vertical-ish lines based on angle."""
    angle = abs(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
    return min_angle < angle < max_angle

def cluster_and_filter_lines(lines, eps=70, min_samples=1):
    """
    Use DBSCAN to group line midpoints and return the most representative lines.
    """
    if not lines:
        return []

    midpoints = np.array([[(line[0][0] + line[0][2]) // 2] for line in lines])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(midpoints)

    filtered_lines = []
    for label in set(clustering.labels_):
        clustered_lines = [lines[i] for i in range(len(lines)) if clustering.labels_[i] == label]
        if clustered_lines:
            # Select the longest line in the cluster
            longest = max(clustered_lines, key=lambda l: np.linalg.norm([l[0][2] - l[0][0], l[0][3] - l[0][1]]))
            filtered_lines.append(longest)
    return filtered_lines

# Load video
cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 0)  # Flip vertically if needed
    height, width = frame.shape[:2]

    # Draw red overlay on upper half (ignored area)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, int(height * 0.4)), (0, 0, 255), -1)  # Red transparent overlay
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Define Region of Interest (bottom 60% of the frame)
    roi_offset = int(height * 0.4)
    roi = frame[roi_offset:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Lane color range - white only
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Only white lane mask used
    masked_image = cv2.bitwise_and(roi, roi, mask=mask_white)

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    base_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 120)

    # Contrast filter: detect if neighborhood around edge has a strong brightness difference
    contrast_edges = np.zeros_like(edges)
    for y in range(1, edges.shape[0] - 1):
        for x in range(1, edges.shape[1] - 1):
            if edges[y, x] != 0:
                patch = base_gray[y-1:y+2, x-1:x+2]
                if patch.size == 9:
                    stddev = np.std(patch)
                    if stddev > 25:  # Adaptive for grey roads
                        contrast_edges[y, x] = 255

    edges = contrast_edges

    # Hough line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=60, maxLineGap=50)

    if lines is not None:
        angled_lines = [line for line in lines if is_valid_lane_line(*line[0])]
        final_lines = cluster_and_filter_lines(angled_lines)

        for line in final_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1 + roi_offset), (x2, y2 + roi_offset), (0, 0, 255), 3)  # Red line

    cv2.imshow("Contrast-Aware Lane Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
