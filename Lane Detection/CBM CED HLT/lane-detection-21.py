# Lane Change Detection

import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("D:/Shubham/Data/IMG_1328.MOV")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 0)
    height, width = frame.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Binary threshold to isolate white markings
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Triangle vertices
    y_bottom = int(0.85 * height)
    top_vertex = (width // 2, height // 2)
    pts = np.array([
        [0, y_bottom],          # bottom-left
        [width, y_bottom],      # bottom-right
        top_vertex              # top
    ], np.int32)

    # Create triangle mask
    mask = np.zeros_like(binary)
    cv2.fillPoly(mask, [pts], 255)

    # Apply mask
    masked = cv2.bitwise_and(binary, mask)

    # Morphological operations
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    rect_blocks = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel_rect, iterations=1)
    solid_lines = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel_line, iterations=2)
    combined = cv2.bitwise_or(rect_blocks, solid_lines)

    # Contour detection
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lane_change_detected = False
    center_x = width // 2
    center_line_top_y = height // 2
    center_line_bottom_y = y_bottom

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue

        # Draw the contour
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), thickness=cv2.FILLED)

        # Check for intersection with the vertical center line
        for point in cnt:
            px, py = point[0]
            if abs(px - center_x) <= 5 and center_line_top_y <= py <= center_line_bottom_y:
                lane_change_detected = True
                break

    # Draw the triangle ROI
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Draw the vertical reference line (center of triangle)
    cv2.line(frame, (center_x, center_line_top_y), (center_x, center_line_bottom_y), (255, 0, 0), 2)

    # Display message if lane change is detected
    if lane_change_detected:
        cv2.putText(frame, "Lane Change Detected!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Show result
    cv2.imshow("Lane Markings Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
