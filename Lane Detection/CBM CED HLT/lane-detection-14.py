import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def is_valid_lane_line(x1, y1, x2, y2, min_angle=20, max_angle=160):
    angle = abs(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
    return min_angle < angle < max_angle

def cluster_and_filter_lines(lines, eps=70, min_samples=1):
    if not lines:
        return []

    midpoints = np.array([[(line[0][0] + line[0][2]) // 2] for line in lines])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(midpoints)

    filtered_lines = []
    for label in set(clustering.labels_):
        clustered_lines = [lines[i] for i in range(len(lines)) if clustering.labels_[i] == label]
        if clustered_lines:
            longest = max(clustered_lines, key=lambda l: np.linalg.norm([l[0][2] - l[0][0], l[0][3] - l[0][1]]))
            filtered_lines.append(longest)
    return filtered_lines

# Load video
cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 0)
    height, width = frame.shape[:2]

    roi_offset = int(height * 0.4)
    roi = frame[roi_offset:, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Binary threshold: white lanes on dark road
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Invert the binary image (for negative visualization)
    negative = cv2.bitwise_not(binary)

    # Convert to BGR for overlaying red lines later
    negative_bgr = cv2.cvtColor(negative, cv2.COLOR_GRAY2BGR)

    # Detect edges on the binary image (not negative)
    edges = cv2.Canny(binary, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=60, maxLineGap=50)

    if lines is not None:
        angled_lines = [line for line in lines if is_valid_lane_line(*line[0])]
        final_lines = cluster_and_filter_lines(angled_lines)

        for line in final_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(negative_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Stack upper part (ignored) + lower part (processed)
    upper = frame[:roi_offset]
    line_marked = cv2.line(upper.copy(), (0, roi_offset-1), (width, roi_offset-1), (0, 0, 255), 2)
    final_frame = np.vstack((line_marked, negative_bgr))

    cv2.imshow("Negative Lane Visualization", final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
