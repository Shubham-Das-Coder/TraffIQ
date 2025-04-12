# Lane Change Detection

import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

# Track the previous lane side (LEFT or RIGHT)
prev_side = None
lane_change_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 0)
    height, width = frame.shape[:2]
    center_x = width // 2

    # Triangle ROI vertices
    y_bottom = int(0.85 * height)
    top_vertex = (center_x, height // 2)
    pts = np.array([
        [0, y_bottom],
        [width, y_bottom],
        top_vertex
    ], np.int32)

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Triangle mask
    mask = np.zeros_like(binary)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(binary, mask)

    # Morph operations
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    rect_blocks = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel_rect, iterations=1)
    solid_lines = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel_line, iterations=2)
    combined = cv2.bitwise_or(rect_blocks, solid_lines)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find lane side based on contour position
    left_points = 0
    right_points = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue

        # Draw lane
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), thickness=cv2.FILLED)

        for point in cnt:
            px, py = point[0]
            if py >= height // 2 and py <= y_bottom:  # Only consider triangle height
                if px < center_x:
                    left_points += 1
                else:
                    right_points += 1

    # Determine current side
    current_side = None
    if left_points > right_points and left_points > 50:
        current_side = "LEFT"
    elif right_points > left_points and right_points > 50:
        current_side = "RIGHT"

    # Detect lane change
    if prev_side and current_side and current_side != prev_side:
        lane_change_detected = True
    else:
        lane_change_detected = False

    # Draw ROI and center line
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.line(frame, (center_x, height // 2), (center_x, y_bottom), (255, 0, 0), 2)

    # Show lane side and lane change status
    if current_side:
        cv2.putText(frame, f"Lane Side: {current_side}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    if lane_change_detected:
        cv2.putText(frame, "Lane Change Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    prev_side = current_side  # Update for next frame

    # Display
    cv2.imshow("Lane Change Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
