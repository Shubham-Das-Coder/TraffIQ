import cv2
import numpy as np

def is_valid_line(x1, y1, x2, y2, angle_thresh=30):
    angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
    return abs(angle) > angle_thresh  # Filter out nearly horizontal lines

def filter_close_lines(lines, min_dist=50):
    """Remove lines that are too close to each other based on x-coordinates"""
    filtered = []
    x_coords = []

    for line in lines:
        x1, _, x2, _ = line[0]
        avg_x = (x1 + x2) // 2
        if all(abs(avg_x - x) > min_dist for x in x_coords):
            filtered.append(line)
            x_coords.append(avg_x)
    return filtered

cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 0)  # Flip vertically if needed

    height, width = frame.shape[:2]
    roi = frame[int(height/2):, :]  # Bottom half of the frame only

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # White and Yellow mask
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 55, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    result = cv2.bitwise_and(roi, roi, mask=combined_mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    if lines is not None:
        # Step 1: Filter by angle (to remove horizontal)
        valid_lines = [line for line in lines if is_valid_line(*line[0])]

        # Step 2: Filter lines that are too close to each other
        final_lines = filter_close_lines(valid_lines)

        for line in final_lines:
            x1, y1, x2, y2 = line[0]
            # Offset y-coordinates because we used only ROI
            cv2.line(frame, (x1, y1 + height//2), (x2, y2 + height//2), (0, 255, 0), 3)

    cv2.imshow("Filtered Lane Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
